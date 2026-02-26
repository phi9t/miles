#!/usr/bin/env bash
set -euo pipefail

# Layer pure-Python deps onto an existing MLSys environment using uv.
# Default target is the container-provided sglang env, which already has torch+CUDA.

TARGET_PY="${TARGET_PY:-/opt/mlsys-envs/sglang/bin/python}"

if [[ ! -x "${TARGET_PY}" ]]; then
    echo "Python target not found: ${TARGET_PY}" >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but not found in PATH" >&2
    exit 2
fi

# Do not install flash_attn/flashinfer in this layer.
uv pip install --python "${TARGET_PY}" --no-deps \
    "accelerate>=0.30.0" \
    "megatron-core>=0.8.0" \

uv pip install --python "${TARGET_PY}" \
    "ray>=2.50.0" \
    "sglang-router>=0.2.3" \
    "datasets>=3.0.0" \
    "networkx>=3.0" \
    "pyarrow>=17.0.0" \
    "pandas>=2.0.0" \
    "typer>=0.12.0" \
    "wandb>=0.18.0"

echo "Verifying layered Python deps..."
"${TARGET_PY}" - <<'PY'
import accelerate
import datasets
import megatron
import networkx
import pandas
import pyarrow
import ray
import sglang_router
import torch
import typer
import wandb

print("python", __import__("sys").executable)
print("torch", torch.__version__)
print("ray", ray.__version__)
print("sglang_router", sglang_router.__version__)
print("accelerate", accelerate.__version__)
print("megatron", getattr(megatron, "__version__", "installed"))
print("datasets", datasets.__version__)
print("networkx", networkx.__version__)
print("pyarrow", pyarrow.__version__)
print("pandas", pandas.__version__)
print("typer", typer.__version__)
print("wandb", wandb.__version__)
PY

if [[ "${TARGET_PY}" == */bin/python ]]; then
    ACTIVATE_PATH="${TARGET_PY%/python}/activate"
    if [[ -f "${ACTIVATE_PATH}" ]]; then
        echo "Layer ready on ${TARGET_PY}"
        echo "Activate with: source ${ACTIVATE_PATH}"
        exit 0
    fi
fi

echo "Layer ready on ${TARGET_PY}"
