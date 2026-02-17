#!/bin/bash
# Evaluation entrypoint for Miles containers.
# Sets up CUDA, validates GPUs, then execs the provided command.
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
        echo "[eval] MLSys venv activated: ${VIRTUAL_ENV:-unknown}"
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
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
    echo "[eval] Detected ${gpu_count} GPU(s)"
    python3 -c "
import torch
n = torch.cuda.device_count()
assert n > 0, 'No CUDA devices available for evaluation'
print(f'[eval] PyTorch CUDA devices: {n}')
" || { echo "[eval] ERROR: PyTorch cannot access CUDA devices" >&2; exit 1; }
else
    echo "[eval] WARNING: nvidia-smi not found; proceeding without GPU validation" >&2
fi

# ── Exec command ────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "[eval] No command provided. Usage: eval.sh <command> [args...]" >&2
    exit 1
fi

echo "[eval] Running: $*"
exec "$@"
