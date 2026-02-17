#!/bin/bash
# Training entrypoint for Miles containers.
# Sets up CUDA, validates GPUs, manages Ray head node, runs training,
# and cleans up Ray/SGLang processes on exit.
set -euo pipefail

# ── CUDA setup ──────────────────────────────────────────────────────────────
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="/usr/local/cuda"

activate_mlsys_venv() {
    local venv_root="${SYGALDRY_MLSYS_VENV_ROOT:-/opt/mlsys-envs}"
    local env_name="${SYGALDRY_MLSYS_ENV:-}"
    local activate_path=""
    if [[ -z "${env_name}" && -f "${venv_root}/.default-env" ]]; then
        env_name="$(head -n1 "${venv_root}/.default-env" 2>/dev/null || true)"
    fi
    if [[ -n "${env_name}" ]]; then
        if [[ -f "${venv_root}/${env_name}/bin/activate" ]]; then
            activate_path="${venv_root}/${env_name}/bin/activate"
        else
            activate_path="$(find "${venv_root}/${env_name}" -maxdepth 3 -path '*/bin/activate' -type f 2>/dev/null | head -n1 || true)"
        fi
    fi
    if [[ -n "${activate_path}" && -f "${activate_path}" ]]; then
        # shellcheck disable=SC1090
        source "${activate_path}"
        echo "[train] MLSys venv activated: ${VIRTUAL_ENV:-unknown}"
    fi
}

activate_mlsys_venv

# ── Megatron PYTHONPATH ─────────────────────────────────────────────────────
if [[ -d "/root/Megatron-LM" ]]; then
    export PYTHONPATH="/root/Megatron-LM:${PYTHONPATH:-}"
elif [[ -d "/opt/megatron" ]]; then
    export PYTHONPATH="/opt/megatron:${PYTHONPATH:-}"
elif [[ -d "${MILES_ROOT:-}/third_party/Megatron-LM" ]]; then
    export PYTHONPATH="${MILES_ROOT}/third_party/Megatron-LM:${PYTHONPATH:-}"
fi

# ── GPU validation ──────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[train] ERROR: nvidia-smi not found" >&2
    exit 1
fi

gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "[train] Detected ${gpu_count} GPU(s)"

python3 -c "
import torch
n = torch.cuda.device_count()
assert n > 0, 'No CUDA devices available for training'
print(f'[train] PyTorch CUDA devices: {n}')
" || { echo "[train] ERROR: PyTorch cannot access CUDA devices" >&2; exit 1; }

# ── Ray head node auto-start ────────────────────────────────────────────────
RAY_STARTED_HERE=0
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    echo "[train] RAY_ADDRESS not set; starting local Ray head node..."
    ray start --head --num-gpus="${gpu_count}" --disable-usage-stats 2>&1 | sed 's/^/[ray] /'
    export RAY_ADDRESS="127.0.0.1:6379"
    RAY_STARTED_HERE=1
    echo "[train] Ray head started at ${RAY_ADDRESS}"
else
    echo "[train] Using existing Ray cluster at ${RAY_ADDRESS}"
fi

# ── Cleanup trap ────────────────────────────────────────────────────────────
cleanup() {
    local rc=$?
    echo "[train] Cleaning up (exit code: ${rc})..."

    # Stop SGLang server processes
    pkill -f "sglang" 2>/dev/null || true

    # Stop Ray if we started it
    if [[ "${RAY_STARTED_HERE}" -eq 1 ]]; then
        ray stop --force 2>/dev/null || true
        echo "[train] Ray stopped"
    fi

    exit "${rc}"
}
trap cleanup EXIT INT TERM

# ── Select training script ──────────────────────────────────────────────────
TRAIN_SCRIPT="train.py"
TRAIN_ARGS=()

# Parse our flags, pass everything else through
for arg in "$@"; do
    case "${arg}" in
        --async)
            TRAIN_SCRIPT="train_async.py"
            ;;
        *)
            TRAIN_ARGS+=("${arg}")
            ;;
    esac
done

echo "[train] Using ${TRAIN_SCRIPT} with ${#TRAIN_ARGS[@]} args"
echo "[train] Starting training..."

if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    python3 "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
else
    python3 "${TRAIN_SCRIPT}"
fi
