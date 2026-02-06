# Miles: Comprehensive Architecture Review

**Date**: 2026-02-05
**Version**: 0.2.1
**Scope**: Full codebase walkthrough, dependency analysis, strengths/weaknesses, improvement proposals

---

## 1. Project Overview

Miles is an enterprise-grade reinforcement learning (RL) framework for large-scale model post-training. It is a fork of [SLIME](https://github.com/THUDM/slime) that integrates three major systems:

| System | Role | Communication |
|--------|------|---------------|
| **SGLang** | High-throughput inference / rollout generation | HTTP API |
| **Megatron-LM** | Distributed training (tensor/pipeline parallelism, MoE) | NCCL / CUDA IPC |
| **Ray** | Distributed orchestration of all components | Ray actors / Object Store |

**Python**: >= 3.10 | **License**: see LICENSE file | **Docker**: `radixark/miles:latest`

---

## 2. Repository Layout

```
miles/
├── train.py                    # Synchronous RL training entry point
├── train_async.py              # Asynchronous variant (pipelined rollout+train)
├── setup.py                    # Package metadata, wheel customization
├── pyproject.toml              # Black/isort/Ruff/pytest configuration
├── requirements.txt            # 17 core dependencies
├── .pre-commit-config.yaml     # Linting hooks
├── build_conda.sh              # Full conda env build from source
│
├── miles/                      # Core framework package
│   ├── ray/                    #   Ray actor orchestration layer
│   ├── backends/               #   Training + inference backends
│   │   ├── megatron_utils/     #     Megatron-LM training backend
│   │   ├── fsdp_utils/         #     PyTorch FSDP training backend
│   │   ├── sglang_utils/       #     SGLang inference engine
│   │   └── training_utils/     #     Shared training utilities (loss, data, logging)
│   ├── rollout/                #   Rollout (inference generation) pipeline
│   │   ├── generate_hub/       #     Pluggable generation strategies
│   │   ├── rm_hub/             #     Pluggable reward model implementations
│   │   ├── filter_hub/         #     Data filtering strategies
│   │   └── inference_rollout/  #     Custom rollout function loading
│   ├── router/                 #   FastAPI-based request routing + sessions
│   │   └── middleware_hub/     #     Pluggable router middleware
│   └── utils/                  #   Arguments, logging, tracking, types
│
├── miles_plugins/              # Plugin extensions
│   ├── mbridge/                #   Model bridges (HF <-> Megatron weight conversion)
│   ├── megatron_bridge/        #   Megatron-specific bridge stubs
│   └── models/                 #   Custom model implementations (Gated Delta Net, etc.)
│
├── tests/
│   ├── fast/                   #   27 fast unit tests (no GPU)
│   └── e2e/                    #   35+ GPU-required E2E tests
│       ├── short/              #     2-4 GPU smoke tests
│       ├── fsdp/               #     FSDP backend tests
│       ├── megatron/           #     Megatron backend tests
│       ├── precision/          #     Cross-framework logprob validation
│       ├── ckpt/               #     Checkpoint save/restore
│       ├── long/               #     Long-running integration tests
│       └── image/              #     Container image tests
│
├── tools/                      # 8 checkpoint conversion scripts + CI generator
├── docker/                     # Dockerfiles (CUDA + ROCm variants) + patches
├── docs/                       # Documentation (EN, quick start, advanced topics)
├── examples/                   # Example training configs
└── scripts/                    # Utility scripts
```

---

## 3. Architecture Walkthrough

### 3.1 Training Loop (Synchronous)

The core loop in `train.py` follows a simple rollout → train → sync cycle:

```
train.py:main()
  │
  ├─ parse_args()                           # Central CLI argument parsing
  ├─ configure_logger() + init_tracking()   # W&B / TensorBoard
  │
  ├─ create_placement_groups(args)          # GPU allocation via Ray
  │   ├─ Create Ray PlacementGroup (PACK strategy)
  │   ├─ Discover physical GPU IDs via ephemeral InfoActors
  │   └─ Return overlapping views: actor_pg, rollout_pg, critic_pg
  │
  ├─ create_rollout_manager(args, pg)       # SGLang inference orchestrator
  │   └─ RolloutManager (Ray actor, 0 GPUs)
  │       ├─ Start SGLang Router (HTTP load balancer)
  │       ├─ Create SGLangEngine actors (one per engine)
  │       ├─ Load DataSource (dataset reader)
  │       └─ Load generate_rollout function (pluggable)
  │
  ├─ create_training_models(args, pg)       # Training actors
  │   └─ RayTrainGroup
  │       └─ N × TrainRayActor (Megatron or FSDP)
  │           ├─ init_process_group("nccl")
  │           ├─ Load model (HF or Megatron checkpoint)
  │           ├─ Wrap with FSDP2 or Megatron parallelism
  │           └─ Create optimizer + LR scheduler
  │
  ├─ actor_model.update_weights()           # Initial weight sync → SGLang
  │
  └─ for rollout_id in range(num_rollouts): # Main RL loop
        │
        ├─ rollout_manager.generate(rollout_id)
        │   ├─ data_source.get_samples(batch_size)
        │   ├─ generate_rollout() → HTTP → SGLang → tokens + logprobs
        │   ├─ async_rm() → reward scoring
        │   ├─ dynamic_filter() → drop zero-variance groups
        │   ├─ convert_samples_to_train_data()
        │   └─ split_train_data_by_dp() → ray.put() per rank
        │
        ├─ actor_model.async_train(rollout_id, data_refs)
        │   └─ N × TrainRayActor.train()
        │       ├─ compute_log_probs("actor", data)
        │       ├─ compute_advantages_and_returns()  # GRPO/PPO
        │       ├─ for microbatch: forward → loss → backward
        │       └─ clip_grad_norm → optimizer.step
        │
        ├─ actor_model.update_weights()     # Push trained weights → SGLang
        │
        └─ periodic: rollout_manager.eval() # Evaluation on test set
```

### 3.2 Async Training Loop (`train_async.py`)

The async variant overlaps rollout generation with training:

```
while not done:
    rollout_future = rollout_manager.generate.remote(id)  # Start next rollout
    ray.get(train_futures)                                 # Wait for current train
    train_data = ray.get(rollout_future)                   # Get rollout results
    train_futures = actor_model.async_train(id, data)      # Start training
    actor_model.update_weights()                           # Sync weights
```

This improves GPU utilization by hiding inference latency behind training compute. Does not support colocated mode (needs separate GPU pools).

### 3.3 Component Communication

```
┌──────────────────────────────────────────────────────────────────┐
│  train.py  (Ray driver — orchestration only)                     │
└──┬────────────────────────┬──────────────────────┬───────────────┘
   │ ray.get()              │ ray.get()            │ ray.get()
   ▼                        ▼                      ▼
┌──────────────┐    ┌──────────────┐      ┌──────────────────┐
│ RolloutManager│    │ RayTrainGroup│      │ SGLangEngine × N │
│ (Ray, 0 GPU) │    │ (Ray actors) │      │ (Ray, frac GPU)  │
│              │    │              │      │                  │
│ - DataSource │    │ - FSDP or    │      │ - HTTP server    │
│ - RM scoring │    │   Megatron   │      │ - /generate API  │
│ - Filtering  │    │ - Optimizer  │      │ - Weight sync    │
└──────┬───────┘    └──────┬───────┘      └────────▲─────────┘
       │                   │                       │
       │  HTTP /generate   │  CUDA IPC / NCCL      │
       └───────────────────┼───────────────────────┘
                           │
                    Weight Updates
                  (zero-copy or broadcast)
```

### 3.4 Weight Synchronization

Two modes depending on GPU topology:

| Mode | Class | Mechanism | When |
|------|-------|-----------|------|
| **Colocated** | `UpdateWeightFromTensor` | CUDA IPC zero-copy memory mapping | Training + inference share GPUs |
| **Separated** | `UpdateWeightFromDistributed` | NCCL broadcast from rank-0 | Training and inference on different GPUs |

Both modes bucket parameters into configurable chunks (default 512MB) for efficient transfer.

---

## 4. Import & Dependency Analysis

### 4.1 External Dependencies

| Package | Import Count | Role | Version Pin |
|---------|-------------|------|-------------|
| **torch** | ~96 (28%) | Core ML framework, distributed training | `>=2.0` (extras only) |
| **megatron** | ~49 (11%) | Large-scale training (TP/PP/MoE) | None |
| **ray** | ~25 (5%) | Distributed orchestration | None |
| **sglang** | ~16 (3%) | Inference engine | None |
| **sglang-router** | — | Load-balanced inference routing | `>=0.2.3` |
| **transformers** | ~13 (3%) | Model loading, tokenizers | None |
| **numpy** | ~7 (2%) | Numerical operations | None |
| **fastapi** | ~7 (2%) | HTTP router for inference | None |
| **wandb** | ~2 (<1%) | Experiment tracking | None |
| **accelerate** | — | Distributed training utilities | None |
| **datasets** | — | HuggingFace dataset loading | None |
| **httpx** | — | HTTP client (HTTP/2 support) | None |
| **omegaconf** | — | Configuration management | None |
| **tensorboard** | — | Alternative experiment tracking | None |

**Not used** despite being common in this space: `deepspeed`, `vllm`.

### 4.2 Internal Dependency Graph

```
train.py / train_async.py
    │
    ├─► miles.utils.arguments      (CLI parsing)
    ├─► miles.utils.logging_utils   (logger setup)
    ├─► miles.utils.tracking_utils  (W&B / TensorBoard)
    │
    ├─► miles.ray.placement_group   (GPU allocation)
    │       ├─► miles.ray.rollout          (RolloutManager)
    │       │       ├─► miles.rollout.*     (generation pipeline)
    │       │       └─► miles.utils.types   (Sample dataclass)
    │       │
    │       └─► miles.ray.actor_group      (RayTrainGroup)
    │               ├─► miles.ray.train_actor        (base TrainRayActor)
    │               ├─► miles.backends.megatron_utils (Megatron backend)
    │               │       └─► miles_plugins.mbridge (weight conversion)
    │               ├─► miles.backends.fsdp_utils     (FSDP backend)
    │               └─► miles.backends.sglang_utils   (SGLang engine)
    │
    └─► miles.router.router         (MilesRouter — FastAPI)
            └─► miles.router.sessions (session management)
```

### 4.3 Lazy Import Patterns

The codebase uses **38 files** with function-level lazy imports. Key patterns:

- **Optional features**: `deep_ep`, `torch_memory_saver` guarded by try/except with feature flags (`_FSDP_AVAILABLE`, `_TORCH_MEMORY_SAVER_AVAILABLE`)
- **Backend isolation**: Megatron imports only in `megatron_utils/`, FSDP imports only in `fsdp_utils/`
- **TYPE_CHECKING guards**: Used in `miles/rollout/base_types.py` and `miles/router/sessions.py` to break circular imports
- **Conditional model loading**: `AutoModelForCausalLM` vs `AutoModelForImageTextToText` loaded at runtime based on model type

### 4.4 Circular Dependency Risks

| Risk | Files | Mitigation |
|------|-------|------------|
| `miles.utils` ↔ `miles.ray` | base_types.py, sessions.py | TYPE_CHECKING guards |
| `sglang_utils` ↔ `megatron_utils`/`fsdp_utils` | Bidirectional imports detected | Lazy imports at function level |

---

## 5. Plugin System

### 5.1 Model Bridge Architecture

The `miles_plugins/mbridge/` package handles weight conversion between HuggingFace and Megatron-Core formats:

```python
from mbridge.core import register_model

@register_model("qwen3_next")
class Qwen3NextBridge(Qwen2MoEBridge):
    # HF ↔ Megatron weight mapping
    # Custom attention layer mappings
    # MoE shared expert configuration
```

**Registered bridges** (4 total):

| Bridge | Base Class | Features |
|--------|-----------|----------|
| `qwen3_next` | Qwen2MoEBridge | Gated Delta Net linear attention, MoE with shared experts |
| `glm4` | LLMBridge | Post-attention/MLP layernorms, QK layernorms |
| `glm4_moe` | Qwen2MoEBridge | MoE + MTP layers, sigmoid router |
| `mimo` | Qwen2Bridge | Multi-Token Prediction on Qwen2 |

**Discovery mechanism**: `AutoBridge.from_hf_pretrained()` uses the registry to select the right bridge based on model config.

### 5.2 Hub Pattern (Pluggable Registries)

Four hubs provide extensibility points:

| Hub | Location | Purpose | Examples |
|-----|----------|---------|----------|
| `generate_hub` | `miles/rollout/generate_hub/` | Generation strategies | single-turn, multi-turn, benchmarkers |
| `rm_hub` | `miles/rollout/rm_hub/` | Reward models | math, f1, gpqa, deepscaler, dapo, remote_rm |
| `filter_hub` | `miles/rollout/filter_hub/` | Data filtering | dynamic sampling, reward variance check |
| `middleware_hub` | `miles/router/middleware_hub/` | Router middleware | RadixTree routing optimization |

All hubs use `load_function()` for dynamic loading, allowing users to specify custom implementations via CLI arguments.

### 5.3 Custom Model Implementations

- **Qwen3NextGatedDeltaNet** (`models/qwen3_next.py`): Linear attention variant using `chunk_gated_delta_rule` from the `fla` library
- **HuggingfaceAttention** (`models/hf_attention.py`): Abstract base for running HF attention within Megatron's tensor/context parallelism

---

## 6. Testing Architecture

### 6.1 Test Coverage Summary

| Category | Files | GPU Required | Trigger |
|----------|-------|-------------|---------|
| Fast unit tests | 27 | No | Every PR |
| E2E short | ~5 | 2-4 GPU | `run-ci-short` label |
| E2E FSDP | ~8 | 2-8 GPU | `run-ci-fsdp` label |
| E2E Megatron | ~8 | 8 GPU | `run-ci-megatron` label |
| E2E precision | ~4 | 4-8 GPU | `run-ci-precision` label |
| E2E checkpoint | ~4 | 8 GPU | `run-ci-ckpt` label |
| E2E long | ~4 | 2 GPU | `run-ci-long` label |
| E2E image | ~2 | varies | `run-ci-image` label |

### 6.2 Testing Patterns

**Fixture architecture**: Shared fixtures in `tests/fast/fixtures/` provide:
- `MockSGLangServer` — HTTP server mimicking SGLang's `/generate` API
- `UvicornThreadServer` — Async FastAPI server in a thread
- `GenerateEnv` / `RolloutEnv` — Complete test environments with args, router, data

**Parameterized testing**: Tests run across multiple rollout variants:
```python
@pytest.mark.parametrize("rollout_env", [
    RolloutEnvConfig(extra_argv=["--old-rollout", "--old-generate"]),
    RolloutEnvConfig(extra_argv=["--new-rollout", "--old-generate"]),
    RolloutEnvConfig(extra_argv=["--new-rollout", "--new-generate"]),
], indirect=True)
```

**E2E execution model**: Tests use `execute_train()` which:
1. Kills stale processes
2. Starts a Ray head node
3. Submits `ray job submit -- python3 train.py <args>`
4. Uses `gpu_lock_exec.py` for exclusive GPU access in CI

### 6.3 Well-Tested vs. Under-Tested

**Strong coverage**:
- Reward models (7 test files, all RM types parametrized)
- Inference rollout integration (8+ files: basic, deterministic, multi-turn, filtering, over-sampling, group RM, agentic tools)
- Router (worker lifecycle, load balancing, health checks, sessions)
- Argument parsing and function loading

**Gaps**:
- Model bridge weight conversion (no direct unit tests for mbridge/)
- Custom model implementations (Gated Delta Net, HF Attention — no unit tests)
- Megatron checkpoint loading (tested only via E2E)
- Data loading edge cases (DataSource variations)
- Negative/error injection tests (limited)

---

## 7. CI/CD Pipeline

```
PR opened
  │
  ├─ [always] pre-commit.yml
  │   └─ Black, isort, Ruff, autoflake, YAML checks
  │
  ├─ [always] pr-test.yml → "fast" job
  │   └─ pytest tests/fast/ (no GPU)
  │
  ├─ [label: run-ci-short] → 2-4 GPU smoke tests
  ├─ [label: run-ci-fsdp]  → FSDP backend tests
  ├─ [label: run-ci-megatron] → Megatron backend tests (8 GPU)
  ├─ [label: run-ci-precision] → Cross-framework validation
  ├─ [label: run-ci-ckpt] → Checkpoint save/restore
  ├─ [label: run-ci-long] → Long-running integration
  └─ [label: run-ci-image] → Container image tests
```

**Notable**: `pr-test.yml` is **auto-generated** from a Jinja2 template (`tools/generate_github_workflows.py`) — the canonical source is `pr-test.yml.j2`.

CI runs in `radixark/miles:latest` container with `--gpus all`, 32GB shared memory, and model/dataset caches mounted from `/mnt/nvme0n1/miles_ci`.

---

## 8. Strengths

### 8.1 Clean Layered Architecture
The separation into `ray/` (orchestration) → `backends/` (training/inference) → `rollout/` (generation) → `utils/` (shared) is well-designed. Each layer has clear responsibilities and the dependency flow is predominantly unidirectional. The `train.py` entry point reads almost like pseudocode because all complexity is properly encapsulated.

### 8.2 Dual Backend Strategy
Supporting both Megatron-LM and PyTorch FSDP through the same `TrainRayActor` interface is a major strength. Users can choose FSDP for simpler setups (< 10B parameters) and Megatron for large-scale MoE/pipeline parallelism — without changing any other part of the system. The backends share the same `update_weights()` interface for SGLang sync.

### 8.3 Hub Pattern for Extensibility
The four pluggable hubs (`generate_hub`, `rm_hub`, `filter_hub`, `middleware_hub`) allow users to add custom generation strategies, reward models, and filters without modifying framework code. Combined with `load_function()` dynamic loading and CLI-configurable paths, this is production-ready extensibility.

### 8.4 Weight Sync Design
The colocated/separated weight sync strategy is elegant. CUDA IPC zero-copy for same-GPU transfers and NCCL broadcast for cross-GPU transfers, both behind a unified interface. The 512MB bucketing prevents memory spikes during sync.

### 8.5 Memory Management
Sophisticated memory management with offloading modes (`offload_train`, `offload_rollout`, `fsdp_cpu_offload`), `torch_memory_saver` integration, and fractional GPU scheduling via Ray. This allows training and inference to timeshare GPUs effectively in colocated mode.

### 8.6 R3 (Routing Replay) for MoE Stability
The `RoutingReplay` mechanism records expert routing decisions during the forward pass and replays them during backward, preventing routing divergence in MoE training. This solves a real problem in MoE RL training that many frameworks ignore.

### 8.7 Fault Tolerance
The `RolloutHealthMonitor` background thread monitors SGLang engine health and `recover_rollout_engines()` can restart dead engines and re-sync weights. CI even includes crash simulation (`_try_ci_fault_injection`) to test recovery paths.

### 8.8 Comprehensive E2E Testing
The label-triggered GPU test matrix covers FSDP, Megatron, precision validation, checkpoint round-trips, and long-running stability — all in real GPU environments. The `gpu_lock_exec.py` tool ensures exclusive GPU access, preventing flaky tests from resource contention.

### 8.9 Async Training Variant
The `train_async.py` entry point overlaps rollout generation with training for better GPU utilization. Having both synchronous and asynchronous variants gives users flexibility to trade simplicity for throughput.

---

## 9. Weaknesses

### 9.1 Dependency Version Pinning

**Severity: HIGH**

Critical dependencies lack version pins in `requirements.txt`:

| Package | Current Pin | Risk |
|---------|------------|------|
| `torch` | `>=2.0` (extras only, not main) | Breaking changes in torch releases |
| `megatron` | Not pinned at all | Megatron-LM API changes frequently |
| `ray` | Not pinned | Major version breaks (Ray 2.x vs 3.x) |
| `transformers` | Not pinned | Tokenizer API changes |
| `sglang` | Not pinned (only `sglang-router>=0.2.3`) | Server API changes |

The Docker image pins specific commits (e.g., Megatron commit `3714d81d`), but anyone installing via `pip install -e .` gets whatever version is current. This is a reproducibility risk.

### 9.2 Missing Unit Tests for Plugin System

**Severity: MEDIUM**

The 4 model bridges (`qwen3_next`, `glm4`, `glm4_moe`, `mimo`) and custom model implementations (`Qwen3NextGatedDeltaNet`, `HuggingfaceAttention`) have **no dedicated unit tests**. Weight mapping correctness is only validated indirectly through E2E tests, meaning:
- A broken weight mapping could silently produce wrong results
- Testing a fix requires 8-GPU Megatron E2E runs (slow, expensive)

### 9.3 Ruff Line Length Misconfiguration

**Severity: LOW**

```toml
# pyproject.toml
[tool.ruff]
line-length = 320  # TODO: currently some file is too long
```

Ruff's line length is set to 320 while Black enforces 119. This effectively disables Ruff's line-length checking entirely. The TODO has been present for some time.

### 9.4 Monolithic Argument Parsing

**Severity: MEDIUM**

`miles/utils/arguments.py` defines **all** CLI arguments for every feature in one function. This creates:
- A single file that every contributor must modify for new features
- Arguments for Megatron-specific features visible even when using FSDP
- No validation that argument combinations are compatible
- Risk of argument name collisions as the framework grows

### 9.5 HTTP-Based Inference in Colocated Mode

**Severity: LOW (design trade-off)**

Even in colocated mode (training and inference on the same GPUs), rollout goes through HTTP to the SGLang server. This adds serialization/deserialization overhead for every generation request. The trade-off is justified (SGLang's optimized batching), but for small-scale setups it adds unnecessary latency.

### 9.6 No Global conftest.py

**Severity: LOW**

There is no `tests/conftest.py` at the top level. Shared fixtures are in `tests/fast/fixtures/` and loaded via individual conftest files. This means E2E tests cannot easily reuse fast test fixtures, and there's no global test configuration.

### 9.7 Bidirectional Imports Between Backend Modules

**Severity: MEDIUM**

`sglang_utils` has bidirectional import relationships with both `megatron_utils` and `fsdp_utils`. While currently managed through lazy imports, this coupling means:
- Changes to the SGLang engine interface can break both training backends
- The backends are not truly independent; they share implicit contracts
- Refactoring one backend requires checking the other

### 9.8 Limited Error Handling Documentation

**Severity: LOW**

The framework handles many error scenarios (engine crashes, GPU OOM, weight sync failures) but the error recovery behavior is not documented. Users encountering failures in production have to read the source code to understand recovery semantics.

### 9.9 No Feature-Based Dependency Groups

**Severity: MEDIUM**

A user who only needs FSDP training must still install all Megatron-related dependencies (which require compiling from source). There's only one optional extra (`[fsdp]` for torch), but no `[megatron]`, `[dev]`, or `[test]` groups.

---

## 10. Proposed Improvement Plans

### Plan A: Dependency Management Hardening (Priority: HIGH)

**Goal**: Reproducible installs without Docker

1. **Pin critical dependencies** in `requirements.txt`:
   ```
   torch>=2.4,<2.7
   ray>=2.40,<3.0
   transformers>=4.45,<5.0
   ```
2. **Add `requirements-lock.txt`** generated from the Docker image for exact reproducibility
3. **Create feature-based extras** in `setup.py`:
   ```python
   extras_require={
       "fsdp": ["torch>=2.4"],
       "megatron": ["megatron-core>=0.10"],
       "dev": ["pytest", "pre-commit", "black", "ruff"],
       "test": ["pytest", "pytest-asyncio"],
   }
   ```
4. **Add CI job** that tests `pip install -e .` on a clean environment (no Docker) to catch missing deps

### Plan B: Plugin System Testing (Priority: HIGH)

**Goal**: Catch weight mapping bugs without E2E GPU runs

1. **Add unit tests for each model bridge**:
   - Test weight name mapping (HF key → Megatron key) without loading actual weights
   - Test config translation (HF config → Megatron config)
   - Test round-trip: `hf_to_megatron(megatron_to_hf(weights)) == weights` with small random tensors
2. **Add unit tests for custom models**:
   - Test `Qwen3NextGatedDeltaNet` forward pass with small random inputs
   - Test `HuggingfaceAttention` abstract interface compliance
3. **Target**: Run in `tests/fast/` without GPUs (use CPU tensors)

### Plan C: Argument System Refactoring (Priority: MEDIUM)

**Goal**: Modular, validated argument parsing

1. **Split `arguments.py` into argument groups**:
   ```
   miles/utils/arguments/
   ├── __init__.py         # parse_args() combines groups
   ├── cluster.py          # GPU allocation args
   ├── training.py         # Backend, optimizer, loss args
   ├── rollout.py          # Generation, RM, filter args
   ├── megatron.py         # Megatron-specific args
   └── fsdp.py             # FSDP-specific args
   ```
2. **Add argument validation**:
   - `--colocate` incompatible with `train_async.py`
   - `--megatron-model-type` required when `--train-backend megatron`
   - `--rollout-num-gpus-per-engine` must divide total rollout GPUs
3. **Generate argument documentation** from argparse definitions

### Plan D: Backend Interface Formalization (Priority: MEDIUM)

**Goal**: Decouple backends from each other

1. **Define explicit `TrainingBackend` protocol**:
   ```python
   class TrainingBackend(Protocol):
       def init(self, args, rank, world_size) -> None: ...
       def train(self, data_refs, rollout_id) -> dict: ...
       def update_weights(self) -> None: ...
       def save_model(self, path) -> None: ...
       def get_state_dict_for_sync(self) -> Iterator[tuple[str, Tensor]]: ...
   ```
2. **Extract weight sync into a standalone module** (currently embedded in both backends)
3. **Remove bidirectional imports** between `sglang_utils` and training backends
4. **Add backend selection validation** at import time (fail fast if megatron not installed)

### Plan E: Observability & Error Handling (Priority: MEDIUM)

**Goal**: Production-ready error reporting

1. **Add structured error types**:
   ```python
   class WeightSyncError(MilesError): ...
   class EngineHealthError(MilesError): ...
   class RolloutTimeoutError(MilesError): ...
   ```
2. **Document recovery behavior** for each error type
3. **Add metrics collection** for weight sync latency, rollout throughput, and engine health
4. **Add `--dry-run` mode** that validates arguments and connectivity without starting training

### Plan F: Code Quality Quick Wins (Priority: LOW)

1. **Fix Ruff line-length**: Set to 119 to match Black, fix offending files
2. **Add global `tests/conftest.py`** with shared markers and skip conditions
3. **Add `py.typed` marker** for type checking support
4. **Document the hub pattern** with a guide for writing custom reward models / generators

---

## 11. Summary

Miles is a well-architected distributed RL framework with clean separation of concerns, production-grade features (fault tolerance, memory management, MoE routing replay), and a thoughtful extensibility model. Its main risks are around dependency management (reproducibility outside Docker) and test coverage gaps in the plugin system. The proposed improvements are incremental — they harden what's already a solid foundation rather than requiring architectural changes.
