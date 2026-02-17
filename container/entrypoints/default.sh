#!/bin/bash
# Interactive shell entrypoint for Miles containers.
# Sets up CUDA, validates GPUs, configures Megatron, and drops into a shell.
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
        echo "[info] MLSys venv activated: ${VIRTUAL_ENV:-unknown}"
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
validate_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[warn] nvidia-smi not found; GPU features may not work" >&2
        return 0
    fi
    local gpu_count
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
    if [[ "${gpu_count}" -eq 0 ]]; then
        echo "[warn] No GPUs detected by nvidia-smi" >&2
        return 0
    fi
    echo "[info] Detected ${gpu_count} GPU(s)"

    python3 -c "
import torch
n = torch.cuda.device_count()
if n == 0:
    print('[warn] PyTorch sees 0 CUDA devices')
else:
    print(f'[info] PyTorch CUDA devices: {n}')
" 2>/dev/null || echo "[warn] PyTorch GPU check failed" >&2
}

validate_gpu

# ── Convenience functions (exported so they survive exec bash -i) ────────────
gpu-test() {
    python3 -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs: {n}')
for i in range(n):
    print(f'  {i}: {torch.cuda.get_device_name(i)}')
"
}
export -f gpu-test

ray-status() {
    python3 -c "import ray; ray.init(ignore_reinit_error=True); print(ray.cluster_resources())"
}
export -f ray-status

sglang-check() {
    python3 -c "import sglang; print(f'SGLang {sglang.__version__}')" 2>/dev/null || echo "SGLang not available"
}
export -f sglang-check

# ── Welcome banner ──────────────────────────────────────────────────────────
cat <<'BANNER'
╔══════════════════════════════════════════════════════════════╗
║  Miles RL Training Container                                ║
║                                                             ║
║  Commands:                                                  ║
║    gpu-test      – Show available GPUs                      ║
║    ray-status    – Show Ray cluster resources                ║
║    sglang-check  – Check SGLang availability                ║
║                                                             ║
║  Training:                                                  ║
║    python train.py --help                                   ║
║    python train_async.py --help                              ║
║                                                             ║
║  Data Mounts:                                               ║
║    /root/models  /root/datasets  /root/checkpoints          ║
║    /root/outputs                                            ║
╚══════════════════════════════════════════════════════════════╝
BANNER

echo "[info] Project: ${MILES_PROJECT_ID:-miles}  Run: ${MILES_RUN_ID:-interactive}"

# ── Exec args or interactive shell ──────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    exec "$@"
fi

exec bash -i
