# Rust Control Plane (Phase 1)

This directory contains the Rust control-plane implementation for Miles.

Current status:
- Workspace and crate layout is in place.
- Typed API models (`JobSpec`, `ResourcePlan`, runtime states) are implemented.
- Deterministic resource assignment helper exists in `scheduler`.
- State store supports in-memory and persistent filesystem snapshots.
- Worker client trait + mock implementation exists in `worker_client`.
- Orchestrator binary supports:
  - `validate` for `JobSpec` + placement output
  - `run` no-op lifecycle
  - `serve` control-plane HTTP APIs under `/v1`

Out of scope in this phase:
- Production RPC transport to Python workers.
- Full sync/async parity with existing Python control-plane traces.
- Fault-tolerant retry orchestration and chaos recovery scenarios.

## Commands

Validate spec and placement:

```bash
cd control-plane
cargo run -p miles-control-orchestrator -- validate --spec examples/job_spec.sample.json
```

Run no-op lifecycle:

```bash
cd control-plane
cargo run -p miles-control-orchestrator -- run --spec examples/job_spec.sample.json
```

Serve API with persistent snapshots:

```bash
cd control-plane
cargo run -p miles-control-orchestrator -- serve --bind 127.0.0.1:18080 --snapshot-dir ./snapshots
```

## API Examples

Create a job:

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/jobs \
  -H 'content-type: application/json' \
  --data-binary @examples/job_spec.sample.json
```

Start, pause, resume, stop:

```bash
curl -sS -X POST http://127.0.0.1:18080/v1/jobs/<job_id>/start
curl -sS -X POST http://127.0.0.1:18080/v1/jobs/<job_id>/pause
curl -sS -X POST http://127.0.0.1:18080/v1/jobs/<job_id>/resume
curl -sS -X POST http://127.0.0.1:18080/v1/jobs/<job_id>/stop
```

Query state, events, and metrics:

```bash
curl -sS http://127.0.0.1:18080/v1/jobs/<job_id>/state
curl -sS http://127.0.0.1:18080/v1/jobs/<job_id>/events
curl -sS http://127.0.0.1:18080/v1/jobs/<job_id>/metrics
```

Worker registry endpoints:

```bash
curl -sS http://127.0.0.1:18080/v1/workers
curl -sS -X POST http://127.0.0.1:18080/v1/workers/register -H 'content-type: application/json' -d '{}'
curl -sS -X POST http://127.0.0.1:18080/v1/workers/<worker_id>/heartbeat
curl -sS -X POST http://127.0.0.1:18080/v1/workers/<worker_id>/drain
```

## Snapshot and Recovery

- Snapshot files are stored as JSON at `<snapshot-dir>/<job_id>.json`.
- `put_job` and `update_runtime` persist snapshots automatically.
- On server startup, existing snapshot files are loaded and job state is restored.

## Tests

Run all Rust unit and integration tests:

```bash
cd control-plane
cargo test
```

Run per-crate tests:

```bash
cd control-plane
cargo test -p miles-control-api
cargo test -p miles-control-scheduler
cargo test -p miles-control-state-store
cargo test -p miles-control-worker-client
cargo test -p miles-control-orchestrator
```

Run Phase-1 parity validation script:

```bash
./scripts/phase1_parity_check.sh
```

## Phase-1 Parity Checklist

- [x] Workspace crates: `api`, `orchestrator`, `scheduler`, `state_store`, `worker_client`.
- [x] `JobSpec` schema and validation implemented.
- [x] External job CRUD API endpoints implemented (`/v1/jobs/*`).
- [x] Persistent snapshot-backed state restore implemented.
- [x] Dev acceptance flow validated: create -> start -> stop -> restart -> state recovery.

## Next Phase (Phase 2)

Planned features:
- Replace `MockWorkerClient` with RPC-backed worker adapters.
- Implement adapter contract for `Init`, `Health`, `Generate`, `TrainStep`, `UpdateWeights`.
- Add worker heartbeat/drain handling to orchestrator dispatch logic.

Planned verification and validation:
- Rust transport unit tests for request envelopes and retry classification.
- Cross-language contract tests against Python adapters.
- Controlled-mode integration tests proving orchestrator can drive adapter endpoints.
