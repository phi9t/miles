#!/bin/bash

# Zephyr container launcher for Miles RL training
set -eu -o pipefail

SCRIPT_DIR="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
readonly SCRIPT_DIR
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
readonly PROJECT_ROOT

MILES_HOME="${MILES_HOME:-${PROJECT_ROOT}}"
readonly MILES_HOME

PROJECT_ID="${MILES_PROJECT_ID:-miles}"
readonly PROJECT_ID

# ── Zephyr cache layout ────────────────────────────────────────────────────
readonly ZEPHYR_CACHE_ROOT="${ZEPHYR_CACHE_ROOT:-/mnt/data_infra/zephyr_container_infra}"
readonly ZEPHYR_SHARED_ROOT="${ZEPHYR_SHARED_ROOT:-${ZEPHYR_CACHE_ROOT}/shared}"
readonly ZEPHYR_PROJECTS_ROOT="${ZEPHYR_PROJECTS_ROOT:-${ZEPHYR_CACHE_ROOT}/projects}"
readonly ZEPHYR_PROJECT_ROOT="${ZEPHYR_PROJECT_ROOT:-${ZEPHYR_PROJECTS_ROOT}/${PROJECT_ID}}"
readonly ZEPHYR_META_ROOT="${ZEPHYR_META_ROOT:-${ZEPHYR_CACHE_ROOT}/meta}"

# ── Host directories ───────────────────────────────────────────────────────
readonly HOST_OUTPUT_ROOT="${ZEPHYR_SHARED_OUTPUT_ROOT:-${ZEPHYR_PROJECT_ROOT}/outputs}"
readonly HOST_RUNS_DIR="${ZEPHYR_PROJECT_ROOT}/runs"
readonly HOST_LEASE_DIR="${ZEPHYR_PROJECT_ROOT}/leases"
readonly HOST_LOGS_DIR="${ZEPHYR_PROJECT_ROOT}/logs"

# ── Shared caches ──────────────────────────────────────────────────────────
readonly HOST_HF_CACHE="${ZEPHYR_SHARED_HF_CACHE:-${ZEPHYR_SHARED_ROOT}/hf_cache}"
readonly HOST_TORCH_CACHE="${ZEPHYR_SHARED_TORCH_CACHE:-${ZEPHYR_SHARED_ROOT}/torch_cache}"
readonly HOST_TRITON_CACHE="${ZEPHYR_SHARED_TRITON_CACHE:-${ZEPHYR_SHARED_ROOT}/triton_cache}"
readonly HOST_NV_COMPUTE_CACHE="${ZEPHYR_SHARED_NV_COMPUTE_CACHE:-${ZEPHYR_SHARED_ROOT}/nv_compute_cache}"

# ── Miles data directories ─────────────────────────────────────────────────
readonly MILES_MODELS_DIR="${MILES_MODELS_DIR:-/mnt/data_infra/miles/models}"
readonly MILES_DATASETS_DIR="${MILES_DATASETS_DIR:-/mnt/data_infra/miles/datasets}"
readonly MILES_CHECKPOINTS_DIR="${MILES_CHECKPOINTS_DIR:-/mnt/data_infra/miles/checkpoints}"

# ── Container paths ────────────────────────────────────────────────────────
readonly CONTAINER_HOME="/root"
readonly CONTAINER_OUTPUT_ROOT="${CONTAINER_HOME}/outputs"
readonly CONTAINER_HF_CACHE="/opt/hf_cache"
readonly CONTAINER_TORCH_CACHE="/opt/torch_cache"
readonly CONTAINER_TRITON_CACHE="/opt/triton_cache"
readonly CONTAINER_NV_COMPUTE_CACHE="/opt/nv_compute_cache"
readonly CONTAINER_WORKSPACE="/workspace"
readonly CONTAINER_ENTRYPOINT_DIR="/opt/container_entrypoints"

# ── Container config ───────────────────────────────────────────────────────
CONTAINER_IMAGE="${MILES_IMAGE:-sygaldry/zephyr:sglang-20260226}"
readonly REQUIRED_CUDA_VERSION="${MILES_REQUIRED_CUDA_VERSION:-12.4}"
readonly CONTAINER_NET="${MILES_NET:-host}"
readonly CONTAINER_IPC="${MILES_IPC:-host}"
readonly MILES_SHM_SIZE="${MILES_SHM_SIZE:-32g}"
readonly MILES_PRIVILEGED="${MILES_PRIVILEGED:-true}"
readonly MILES_DOCKER_USER="${MILES_DOCKER_USER:-0:0}"
readonly EXTRA_DOCKER_ARGS="${MILES_EXTRA_DOCKER_ARGS:-}"
DEFAULT_CACHE_PROFILE="${ZEPHYR_CACHE_PROFILE:-shared}"
readonly DEFAULT_CACHE_PROFILE
readonly LEASE_MODE_DEFAULT="${ZEPHYR_LEASE_MODE:-warn}"

# ═══════════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════════

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [launch:${BASH_LINENO[0]}] $*" >&2
}

error() {
    log "ERROR: $*"
    exit 1
}

version_lt() {
    local a="$1"
    local b="$2"
    local a_major="${a%%.*}"
    local a_minor="${a#*.}"
    local b_major="${b%%.*}"
    local b_minor="${b#*.}"
    if [[ "${a_major}" -lt "${b_major}" ]]; then return 0; fi
    if [[ "${a_major}" -gt "${b_major}" ]]; then return 1; fi
    if [[ "${a_minor:-0}" -lt "${b_minor:-0}" ]]; then return 0; fi
    return 1
}

detect_host_cuda_version() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi
    local cuda_line
    if command -v rg >/dev/null 2>&1; then
        cuda_line="$(nvidia-smi 2>/dev/null | rg -o "CUDA Version: [0-9]+\\.[0-9]+" -m 1 || true)"
    else
        cuda_line="$(nvidia-smi 2>/dev/null | grep -Eo "CUDA Version: [0-9]+\\.[0-9]+" | head -n 1 || true)"
    fi
    if [[ -z "${cuda_line}" ]]; then
        return 1
    fi
    echo "${cuda_line##*CUDA Version: }"
}

resolve_mount_path() {
    local path="$1"
    local parent_dir
    parent_dir="$(dirname "${path}")"
    if [[ ! -d "${parent_dir}" ]]; then
        mkdir -p "${parent_dir}"
    fi
    if [[ ! -e "${path}" ]]; then
        mkdir -p "${path}"
    fi
    realpath "${path}"
}

check_requirements() {
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
    fi
    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running or not accessible"
    fi
    if ! docker info 2>/dev/null | grep -q nvidia; then
        error "NVIDIA Docker runtime not detected. This is a GPU-only container infrastructure."
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Lease system
# ═══════════════════════════════════════════════════════════════════════════

lease_file_for() {
    local lease_dir="$1"
    local resource="$2"
    echo "${lease_dir}/${resource}.lease"
}

read_lease_expiry() {
    local lease_file="$1"
    if [[ ! -f "${lease_file}" ]]; then
        echo ""
        return
    fi
    awk -F= '$1=="expires_epoch"{print $2}' "${lease_file}" 2>/dev/null || true
}

acquire_lease() {
    local lease_mode="$1"
    local lease_dir="$2"
    local resource="$3"
    local owner="$4"
    local ttl_s="$5"
    local run_id="$6"

    mkdir -p "${lease_dir}"
    local lease_file
    lease_file="$(lease_file_for "${lease_dir}" "${resource}")"
    local now
    now=$(date +%s)

    if [[ -f "${lease_file}" ]]; then
        local expiry
        expiry="$(read_lease_expiry "${lease_file}")"
        if [[ -n "${expiry}" && "${expiry}" -ge "${now}" ]]; then
            local msg="Resource lease exists (${resource}) at ${lease_file}"
            if [[ "${lease_mode}" == "enforce" ]]; then
                error "${msg}"
            fi
            if [[ "${lease_mode}" == "warn" ]]; then
                log "WARNING: ${msg}"
            fi
        fi
    fi

    local expires
    expires=$((now + ttl_s))
    cat > "${lease_file}" <<EOF_LEASE
resource=${resource}
owner=${owner}
run_id=${run_id}
pid=$$
created_epoch=${now}
expires_epoch=${expires}
EOF_LEASE
    echo "${lease_file}"
}

release_lease() {
    local lease_file="$1"
    if [[ -n "${lease_file}" && -f "${lease_file}" ]]; then
        rm -f "${lease_file}"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# Host directory setup
# ═══════════════════════════════════════════════════════════════════════════

setup_host_directories() {
    local dirs=(
        "${ZEPHYR_SHARED_ROOT}"
        "${ZEPHYR_PROJECT_ROOT}"
        "${ZEPHYR_META_ROOT}"
        "${HOST_OUTPUT_ROOT}"
        "${HOST_RUNS_DIR}"
        "${HOST_LEASE_DIR}"
        "${HOST_LOGS_DIR}"
        "${HOST_HF_CACHE}"
        "${HOST_TORCH_CACHE}"
        "${HOST_TRITON_CACHE}"
        "${HOST_NV_COMPUTE_CACHE}"
        "${MILES_MODELS_DIR}"
        "${MILES_DATASETS_DIR}"
        "${MILES_CHECKPOINTS_DIR}"
    )
    local dir
    for dir in "${dirs[@]}"; do
        [[ -d "${dir}" ]] || mkdir -p "${dir}"
    done

    cat > "${ZEPHYR_META_ROOT}/layout_version.json" <<'JSON'
{"layout_version":2,"layout_name":"unified-shared-cache-project-isolation"}
JSON
}

resolve_common_mount_paths() {
    RESOLVED_HF_CACHE="$(resolve_mount_path "${HOST_HF_CACHE}")"
    RESOLVED_OUTPUT_ROOT="$(resolve_mount_path "${HOST_OUTPUT_ROOT}")"
    RESOLVED_TORCH_CACHE="$(resolve_mount_path "${HOST_TORCH_CACHE}")"
    RESOLVED_TRITON_CACHE="$(resolve_mount_path "${HOST_TRITON_CACHE}")"
    RESOLVED_NV_COMPUTE_CACHE="$(resolve_mount_path "${HOST_NV_COMPUTE_CACHE}")"
    RESOLVED_MODELS_DIR="$(resolve_mount_path "${MILES_MODELS_DIR}")"
    RESOLVED_DATASETS_DIR="$(resolve_mount_path "${MILES_DATASETS_DIR}")"
    RESOLVED_CHECKPOINTS_DIR="$(resolve_mount_path "${MILES_CHECKPOINTS_DIR}")"
}

# ═══════════════════════════════════════════════════════════════════════════
# Image management
# ═══════════════════════════════════════════════════════════════════════════

ensure_image() {
    if docker image inspect "${CONTAINER_IMAGE}" >/dev/null 2>&1; then
        log "Image ${CONTAINER_IMAGE} found locally"
        return 0
    fi
    log "Image ${CONTAINER_IMAGE} not found locally; pulling..."
    docker pull "${CONTAINER_IMAGE}" || error "Failed to pull image: ${CONTAINER_IMAGE}"
}

# ═══════════════════════════════════════════════════════════════════════════
# Docker argument builder
# ═══════════════════════════════════════════════════════════════════════════

build_docker_args() {
    local entrypoint_path="$1"
    local run_id="$2"
    local lease_mode="$3"
    local cache_profile="$4"

    local docker_args=()
    docker_args+=("--rm" "--init")
    docker_args+=("--name=miles-${PROJECT_ID}-${run_id}")
    if [[ -t 0 ]]; then
        docker_args+=("--interactive" "--tty")
    fi
    if [[ -n "${MILES_DOCKER_USER}" ]]; then
        docker_args+=("--user=${MILES_DOCKER_USER}")
    fi

    # Network and IPC
    [[ -n "${CONTAINER_NET}" ]] || error "MILES_NET must be non-empty"
    [[ -n "${CONTAINER_IPC}" ]] || error "MILES_IPC must be non-empty"
    docker_args+=("--net=${CONTAINER_NET}" "--ipc=${CONTAINER_IPC}")

    # Resource limits
    docker_args+=(
        "--shm-size=${MILES_SHM_SIZE}"
        "--memory=0"
        "--memory-swap=0"
        "--ulimit" "memlock=-1"
        "--ulimit" "stack=67108864"
        "--ulimit" "nofile=65535:65535"
    )
    if [[ "${MILES_PRIVILEGED}" == "true" ]]; then
        docker_args+=("--privileged")
    fi

    # GPU runtime
    local host_cuda_version
    host_cuda_version="$(detect_host_cuda_version || true)"
    log "Host CUDA version: ${host_cuda_version}"
    log "Required CUDA version: ${REQUIRED_CUDA_VERSION}"

    if [[ -n "${host_cuda_version}" ]] && version_lt "${host_cuda_version}" "${REQUIRED_CUDA_VERSION}"; then
        error "Host CUDA ${host_cuda_version} < required ${REQUIRED_CUDA_VERSION}"
    fi

    docker_args+=("--runtime=nvidia" "--gpus=all")

    # Volume mounts: project workspace
    local resolved_project_root
    resolved_project_root="$(resolve_mount_path "${MILES_HOME}")"
    docker_args+=(
        "--volume=${resolved_project_root}:${CONTAINER_WORKSPACE}"
        "--workdir=${CONTAINER_WORKSPACE}"
    )

    # Volume mounts: outputs
    docker_args+=("--volume=${RESOLVED_OUTPUT_ROOT}:${CONTAINER_OUTPUT_ROOT}")

    # Volume mounts: shared caches
    docker_args+=(
        "--volume=${RESOLVED_HF_CACHE}:${CONTAINER_HF_CACHE}"
        "--volume=${RESOLVED_TORCH_CACHE}:${CONTAINER_TORCH_CACHE}"
        "--volume=${RESOLVED_TRITON_CACHE}:${CONTAINER_TRITON_CACHE}"
        "--volume=${RESOLVED_NV_COMPUTE_CACHE}:${CONTAINER_NV_COMPUTE_CACHE}"
    )

    # Volume mounts: Miles data directories (matching CI: -v .../models:/root/models)
    docker_args+=(
        "--volume=${RESOLVED_MODELS_DIR}:/root/models"
        "--volume=${RESOLVED_DATASETS_DIR}:/root/datasets"
        "--volume=${RESOLVED_CHECKPOINTS_DIR}:/root/checkpoints"
    )

    # Volume mounts: /tmp passthrough
    docker_args+=("--volume=/tmp:/tmp")

    # Volume mounts: entrypoints from host repo
    docker_args+=("--volume=${SCRIPT_DIR}/entrypoints:${CONTAINER_ENTRYPOINT_DIR}:ro")

    # Entrypoint
    docker_args+=("--entrypoint=${entrypoint_path}")

    # Container environment
    docker_args+=(
        "--env=MILES_IN_CONTAINER=1"
        "--env=MILES_ROOT=${CONTAINER_WORKSPACE}"
        "--env=MILES_PROJECT_ID=${PROJECT_ID}"
        "--env=MILES_RUN_ID=${run_id}"
        "--env=ZEPHYR_LEASE_MODE=${lease_mode}"
        "--env=ZEPHYR_CACHE_PROFILE=${cache_profile}"
        "--env=HOME=${CONTAINER_HOME}"
        "--env=HF_HOME=${CONTAINER_HF_CACHE}"
        "--env=TORCH_HOME=${CONTAINER_TORCH_CACHE}"
        "--env=TRITON_CACHE_DIR=${CONTAINER_TRITON_CACHE}"
        "--env=CUDA_CACHE_PATH=${CONTAINER_NV_COMPUTE_CACHE}"
    )

    # Passthrough env vars: standard
    local env_vars=(
        "TERM"
        "LANG"
        "LC_ALL"
        "WANDB_API_KEY"
        "WANDB_PROJECT"
        "WANDB_ENTITY"
        "RAY_ADDRESS"
        "MASTER_ADDR"
        "MASTER_PORT"
        "CUDA_VISIBLE_DEVICES"
        "SYGALDRY_MLSYS_ENV"
        "SYGALDRY_MLSYS_VENV_ROOT"
    )
    local var
    for var in "${env_vars[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            docker_args+=("--env=${var}=${!var}")
        fi
    done

    # Passthrough env vars: NCCL_* wildcard
    while IFS='=' read -r name value; do
        if [[ "${name}" == NCCL_* ]]; then
            docker_args+=("--env=${name}=${value}")
        fi
    done < <(env)

    # Extra docker args
    if [[ -n "${EXTRA_DOCKER_ARGS}" ]]; then
        local extra_args=()
        read -r -a extra_args <<<"${EXTRA_DOCKER_ARGS}"
        docker_args+=("${extra_args[@]}")
    fi

    printf '%s\n' "${docker_args[@]}"
}

# ═══════════════════════════════════════════════════════════════════════════
# Config printer
# ═══════════════════════════════════════════════════════════════════════════

print_effective_config() {
    local run_id="$1"
    local lease_mode="$2"
    local cache_profile="$3"
    log "Effective config:"
    log "  Project ID: ${PROJECT_ID}"
    log "  Run ID: ${run_id}"
    log "  Image: ${CONTAINER_IMAGE}"
    log "  Lease mode: ${lease_mode}"
    log "  Cache profile: ${cache_profile}"
    log "  Cache root: ${ZEPHYR_CACHE_ROOT}"
    log "  Shared root: ${ZEPHYR_SHARED_ROOT}"
    log "  Project root: ${ZEPHYR_PROJECT_ROOT}"
    log "  HF cache: ${HOST_HF_CACHE}"
    log "  Torch cache: ${HOST_TORCH_CACHE}"
    log "  Triton cache: ${HOST_TRITON_CACHE}"
    log "  NV compute cache: ${HOST_NV_COMPUTE_CACHE}"
    log "  Models dir: ${MILES_MODELS_DIR}"
    log "  Datasets dir: ${MILES_DATASETS_DIR}"
    log "  Checkpoints dir: ${MILES_CHECKPOINTS_DIR}"
    log "  Network: ${CONTAINER_NET}  IPC: ${CONTAINER_IPC}"
    log "  SHM size: ${MILES_SHM_SIZE}  Privileged: ${MILES_PRIVILEGED}"
    log "  Docker user: ${MILES_DOCKER_USER}"
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

main() {
    log "Starting Miles container launcher..."

    local entrypoint_name="${MILES_ENTRYPOINT:-default}"
    local passthrough_args=()
    local run_id="${MILES_RUN_ID:-}"
    local lease_mode="${LEASE_MODE_DEFAULT}"
    local cache_profile="${DEFAULT_CACHE_PROFILE}"
    local print_config=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --entrypoint=*) entrypoint_name="${1#*=}"; shift ;;
            --entrypoint|-e) entrypoint_name="${2:-}"; [[ -n "${entrypoint_name}" ]] || error "Missing value for --entrypoint"; shift 2 ;;
            --shell) entrypoint_name="shell"; shift ;;
            --run-id=*) run_id="${1#*=}"; shift ;;
            --run-id) run_id="${2:-}"; [[ -n "${run_id}" ]] || error "Missing value for --run-id"; shift 2 ;;
            --lease-mode=*) lease_mode="${1#*=}"; shift ;;
            --lease-mode) lease_mode="${2:-}"; [[ -n "${lease_mode}" ]] || error "Missing value for --lease-mode"; shift 2 ;;
            --cache-profile=*) cache_profile="${1#*=}"; shift ;;
            --cache-profile) cache_profile="${2:-}"; [[ -n "${cache_profile}" ]] || error "Missing value for --cache-profile"; shift 2 ;;
            --image=*)
                CONTAINER_IMAGE="${1#*=}"
                [[ -n "${CONTAINER_IMAGE}" ]] || error "Missing value for --image"
                shift
                ;;
            --image)
                CONTAINER_IMAGE="${2:-}"
                [[ -n "${CONTAINER_IMAGE}" ]] || error "Missing value for --image"
                shift 2
                ;;
            --print-effective-config) print_config=1; shift ;;
            --) shift; passthrough_args+=("$@"); break ;;
            *) passthrough_args+=("$1"); shift ;;
        esac
    done

    entrypoint_name="${entrypoint_name%.sh}"
    if [[ -z "${run_id}" ]]; then
        run_id="run-$(date +%Y%m%d-%H%M%S)-$$"
    fi

    if [[ "${lease_mode}" != "off" && "${lease_mode}" != "warn" && "${lease_mode}" != "enforce" ]]; then
        error "Invalid --lease-mode: ${lease_mode} (expected off|warn|enforce)"
    fi
    if [[ "${cache_profile}" != "shared" && "${cache_profile}" != "isolated" && "${cache_profile}" != "hybrid" ]]; then
        error "Invalid --cache-profile: ${cache_profile} (expected shared|isolated|hybrid)"
    fi

    if [[ ${print_config} -eq 1 ]]; then
        print_effective_config "${run_id}" "${lease_mode}" "${cache_profile}"
        exit 0
    fi

    check_requirements
    setup_host_directories
    ensure_image

    resolve_common_mount_paths

    local entrypoint_path="${CONTAINER_ENTRYPOINT_DIR}/${entrypoint_name}.sh"

    # Validate entrypoint exists on host
    if [[ ! -f "${SCRIPT_DIR}/entrypoints/${entrypoint_name}.sh" ]]; then
        error "Entrypoint not found: ${SCRIPT_DIR}/entrypoints/${entrypoint_name}.sh"
    fi

    # Acquire GPU lease
    local lease_file=""
    if [[ "${lease_mode}" != "off" ]]; then
        lease_file="$(acquire_lease "${lease_mode}" "${HOST_LEASE_DIR}" "gpu-all" "${PROJECT_ID}" 21600 "${run_id}")"
    fi

    # Build docker arguments
    readarray -t docker_args < <(build_docker_args \
        "${entrypoint_path}" \
        "${run_id}" \
        "${lease_mode}" \
        "${cache_profile}" \
    )

    log "Launching container: ${CONTAINER_IMAGE}"
    log "  Entrypoint: ${entrypoint_path}"
    log "  Run ID: ${run_id}"

    set +e
    if [[ ${#passthrough_args[@]} -gt 0 ]]; then
        docker run "${docker_args[@]}" "${CONTAINER_IMAGE}" "${passthrough_args[@]}"
    else
        docker run "${docker_args[@]}" "${CONTAINER_IMAGE}"
    fi
    local rc=$?
    set -e

    release_lease "${lease_file}"
    exit ${rc}
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
