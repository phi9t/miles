#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: docker/verify_zephyr_core_stack.sh --base <image> --target <image>

Compares immutable Zephyr core stack package versions between base and target:
  torch, jax, jaxlib, triton, llvmlite, llvm-config
USAGE
}

BASE_IMAGE=""
TARGET_IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base)
      BASE_IMAGE="${2:-}"; shift 2 ;;
    --target)
      TARGET_IMAGE="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

[[ -n "${BASE_IMAGE}" ]] || { echo "Missing --base" >&2; exit 2; }
[[ -n "${TARGET_IMAGE}" ]] || { echo "Missing --target" >&2; exit 2; }

collect_versions() {
  local image="$1"
  docker run --rm --entrypoint /bin/bash "${image}" -lc '
venv_root="${SYGALDRY_MLSYS_VENV_ROOT:-/opt/mlsys-envs}"
env_name="${SYGALDRY_MLSYS_ENV:-}"
if [[ -z "${env_name}" && -f "${venv_root}/.default-env" ]]; then
  env_name="$(head -n1 "${venv_root}/.default-env" 2>/dev/null || true)"
fi
if [[ -n "${env_name}" ]]; then
  if [[ -f "${venv_root}/${env_name}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${venv_root}/${env_name}/bin/activate"
  else
    ap="$(find "${venv_root}/${env_name}" -maxdepth 3 -path "*/bin/activate" -type f 2>/dev/null | head -n1 || true)"
    if [[ -n "${ap}" ]]; then
      # shellcheck disable=SC1090
      source "${ap}"
    fi
  fi
fi
python3 - <<'"'"'PY'"'"'
import importlib
import json

mods = ("torch", "jax", "jaxlib", "triton", "llvmlite")
out = {}
for name in mods:
    try:
        mod = importlib.import_module(name)
        out[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        out[name] = f"missing:{exc.__class__.__name__}"

import subprocess
try:
    out["llvm-config"] = subprocess.check_output(
        ["llvm-config", "--version"], text=True
    ).strip()
except Exception as exc:
    out["llvm-config"] = f"missing:{exc.__class__.__name__}"

print(json.dumps(out, sort_keys=True))
PY
'
}

base_json="$(collect_versions "${BASE_IMAGE}")"
target_json="$(collect_versions "${TARGET_IMAGE}")"

echo "Base (${BASE_IMAGE}):   ${base_json}"
echo "Target (${TARGET_IMAGE}): ${target_json}"

if [[ "${base_json}" != "${target_json}" ]]; then
  echo "ERROR: Zephyr core stack drift detected between base and target images." >&2
  exit 1
fi

echo "OK: Zephyr core stack versions are unchanged."
