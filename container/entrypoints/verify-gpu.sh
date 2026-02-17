#!/bin/bash
# GPU diagnostics entrypoint.
# Checks nvidia-smi, nvcc, PyTorch CUDA, SGLang, Megatron-LM, and Ray.
set -euo pipefail

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

SEP="────────────────────────────────────────────────────────────────"

section() { echo -e "\n${SEP}\n  $1\n${SEP}"; }

# ── nvidia-smi ──────────────────────────────────────────────────────────────
section "nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi: not found"
fi

# ── nvcc ────────────────────────────────────────────────────────────────────
section "nvcc --version"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc: not found"
fi

# ── PyTorch CUDA ────────────────────────────────────────────────────────────
section "PyTorch CUDA"
python3 -c "
import torch
print(f'PyTorch version:  {torch.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
print(f'CUDA version:     {torch.version.cuda}')
print(f'cuDNN version:    {torch.backends.cudnn.version()}')
n = torch.cuda.device_count()
print(f'Device count:     {n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem  = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    print(f'  GPU {i}: {name}  ({mem:.1f} GiB)')
    # Quick smoke test on each device
    t = torch.randn(64, 64, device=f'cuda:{i}')
    assert t.sum().is_cuda, f'GPU {i} compute failed'
print('All GPU compute checks passed.')
" 2>&1 || echo "PyTorch CUDA check failed"

# ── SGLang ──────────────────────────────────────────────────────────────────
section "SGLang"
python3 -c "
try:
    import sglang
    print(f'SGLang version: {sglang.__version__}')
except ImportError:
    print('SGLang: not installed')
except AttributeError:
    print('SGLang: installed (version unavailable)')
" 2>&1

# ── Megatron-LM ────────────────────────────────────────────────────────────
section "Megatron-LM"
python3 -c "
try:
    import megatron
    ver = getattr(megatron, '__version__', 'unknown')
    print(f'Megatron-LM version: {ver}')
except ImportError:
    print('Megatron-LM: not installed')
" 2>&1

# ── Ray ─────────────────────────────────────────────────────────────────────
section "Ray"
python3 -c "
try:
    import ray
    print(f'Ray version: {ray.__version__}')
except ImportError:
    print('Ray: not installed')
" 2>&1
if command -v ray >/dev/null 2>&1; then
    ray status 2>&1 || echo "Ray cluster: not running"
else
    echo "ray CLI: not found"
fi

echo -e "\n${SEP}\n  Verification complete\n${SEP}"
