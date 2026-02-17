#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: docker/verify_runtime_stack.sh --image <image>

Verifies runtime packages required by Miles training are importable in the
active MLSys venv and that Ray CLI is available.
USAGE
}

IMAGE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

[[ -n "${IMAGE}" ]] || { echo "Missing --image" >&2; exit 2; }

docker run --rm --entrypoint /bin/bash "${IMAGE}" -lc '
set -euo pipefail
venv_root="${SYGALDRY_MLSYS_VENV_ROOT:-/opt/mlsys-envs}"
env_name="${SYGALDRY_MLSYS_ENV:-}"
if [[ -z "${env_name}" && -f "${venv_root}/.default-env" ]]; then
  env_name="$(head -n1 "${venv_root}/.default-env" 2>/dev/null || true)"
fi
[[ -n "${env_name}" ]] || { echo "No MLSys env found under ${venv_root}" >&2; exit 1; }
if [[ -f "${venv_root}/${env_name}/bin/activate" ]]; then
  source "${venv_root}/${env_name}/bin/activate"
else
  ap="$(find "${venv_root}/${env_name}" -maxdepth 3 -path "*/bin/activate" -type f 2>/dev/null | head -n1 || true)"
  [[ -n "${ap}" ]] || { echo "Cannot locate activate script for ${env_name}" >&2; exit 1; }
  source "${ap}"
fi
python3 - <<'"'"'PY'"'"'
import ray
import torch
import sglang
import pynvml

print("ray", ray.__version__)
print("torch", torch.__version__)
print("sglang", getattr(sglang, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
print("pynvml", "ok")
PY
ray --version
'
