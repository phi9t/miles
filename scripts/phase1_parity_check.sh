#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_PLANE_DIR="$ROOT_DIR/control-plane"
SPEC_PATH="$CONTROL_PLANE_DIR/examples/job_spec.sample.json"

if [[ ! -f "$SPEC_PATH" ]]; then
  echo "missing spec file: $SPEC_PATH" >&2
  exit 1
fi

if [[ -z "${PORT:-}" ]]; then
  PORT="$(
    python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
  )"
else
  PORT="${PORT}"
fi
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
SNAPSHOT_DIR="$(mktemp -d /tmp/miles-control-snapshots.XXXXXX)"
LOG_FILE="$(mktemp /tmp/miles-control-server.XXXXXX.log)"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$SNAPSHOT_DIR"
  rm -f "$LOG_FILE"
}
trap cleanup EXIT

wait_for_server() {
  local attempts=0
  until curl -fsS "$BASE_URL/v1/workers" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [[ "$attempts" -gt 300 ]]; then
      echo "server did not become ready; log follows:" >&2
      cat "$LOG_FILE" >&2 || true
      return 1
    fi
    sleep 0.1
  done
}

start_server() {
  (
    cd "$CONTROL_PLANE_DIR"
    cargo run -p miles-control-orchestrator -- serve \
      --bind "${HOST}:${PORT}" \
      --snapshot-dir "$SNAPSHOT_DIR"
  ) >"$LOG_FILE" 2>&1 &
  SERVER_PID="$!"
  wait_for_server
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

json_get() {
  local expr="$1"
  python -c "import json,sys; print(${expr})"
}

echo "[phase1] starting API server"
start_server

echo "[phase1] creating job"
create_resp="$(curl -fsS -X POST "$BASE_URL/v1/jobs" -H 'content-type: application/json' --data-binary "@$SPEC_PATH")"
job_id="$(printf '%s' "$create_resp" | json_get 'json.load(sys.stdin)["job_id"]')"

echo "[phase1] starting job: $job_id"
curl -fsS -X POST "$BASE_URL/v1/jobs/$job_id/start" >/dev/null

echo "[phase1] stopping job"
curl -fsS -X POST "$BASE_URL/v1/jobs/$job_id/stop" >/dev/null

state_before_restart="$(curl -fsS "$BASE_URL/v1/jobs/$job_id/state" | json_get 'json.load(sys.stdin)["runtime"]["state"]')"
if [[ "$state_before_restart" != "stopped" ]]; then
  echo "expected stopped before restart, got: $state_before_restart" >&2
  exit 1
fi

echo "[phase1] restarting API server"
stop_server
start_server

state_after_restart="$(curl -fsS "$BASE_URL/v1/jobs/$job_id/state" | json_get 'json.load(sys.stdin)["runtime"]["state"]')"
if [[ "$state_after_restart" != "stopped" ]]; then
  echo "expected stopped after restart, got: $state_after_restart" >&2
  exit 1
fi

echo "[phase1] parity check passed"
