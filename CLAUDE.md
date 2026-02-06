# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Miles is an enterprise-grade reinforcement learning (RL) framework for large-scale model post-training. It is a fork of [SLIME](https://github.com/THUDM/slime) that integrates **SGLang** (high-throughput inference/rollout), **Megatron-LM** (distributed training), and **Ray** (distributed orchestration). Version 0.2.1, Python >= 3.10.

## Common Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .
# Or use Docker: docker pull radixark/miles:latest
```

### Running Tests
```bash
# Fast unit tests (no GPUs needed)
pytest tests/fast/

# Run a single test file
pytest tests/fast/utils/test_arguments.py

# Run by marker
pytest -m unit
pytest -m integration

# E2E tests (require GPUs, organized by backend/category)
python tests/e2e/short/test_qwen2.5_0.5B_gsm8k_short.py
```

### Linting & Formatting
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
black --line-length 119 <file>
isort --profile black <file>
ruff check --fix <file>
autoflake --remove-all-unused-imports --in-place <file>
```

### Training
```bash
# Synchronous training (main entry point)
python train.py --advantage-estimator grpo --model-name qwen3-30b-a3b --hf-checkpoint /path/to/model

# Asynchronous training
python train_async.py ...
```

## Code Style

- **Black** formatter with line length **119**
- **isort** with black profile, line length 119
- **Ruff** for linting (E, F, B, UP rules; E402 and E501 ignored)
- First-party packages: `miles`, `miles_plugins`
- Known third-party: `megatron`, `wandb`, `ray`, `transformers`

## Architecture

### Entry Points
- `train.py` — Synchronous RL training loop: allocates GPUs via Ray placement groups, creates a RolloutManager (SGLang inference), creates training actors (Megatron/FSDP), then runs rollout-train-save cycles.
- `train_async.py` — Asynchronous variant of the training loop.

### Core Package (`miles/`)

**`miles/ray/`** — Ray-based distributed orchestration layer
- `placement_group.py` — GPU allocation and placement group creation
- `rollout.py` — `RolloutManager` Ray actor: manages SGLang inference engines, reward scoring, filtering, and evaluation
- `actor_group.py` / `train_actor.py` — Training actor group management wrapping backend-specific trainers
- `ray_actor.py` — Base Ray actor abstraction

**`miles/backends/`** — Training backend implementations (two backends, chosen by config)
- `megatron_utils/` — Megatron-LM backend for large-scale distributed training (tensor/pipeline parallelism, MoE, INT4 QAT)
- `fsdp_utils/` — PyTorch FSDP backend for simpler distributed training
- `sglang_utils/` — SGLang inference engine integration (weight sync, FP8)
- `training_utils/` — Shared training utilities (gradient computation, checkpointing)

**`miles/rollout/`** — Rollout (inference) generation pipeline
- `sglang_rollout.py` — Main SGLang-backed rollout implementation
- `sft_rollout.py` — Supervised fine-tuning rollout variant
- `sleep_rollout.py` — Mock rollout for testing
- `data_source.py` — Data source abstraction for datasets
- `generate_hub/` — Pluggable generation strategies
- `rm_hub/` — Pluggable reward model implementations
- `filter_hub/` — Data filtering strategies
- `inference_rollout/` — Custom rollout function loading

**`miles/router/`** — Request routing
- `router.py` — Main router implementation
- `sessions.py` — Session management
- `middleware_hub/` — Pluggable router middleware

**`miles/utils/`** — Shared utilities
- `arguments.py` — Central CLI argument parsing (all training arguments defined here)
- `logging_utils.py` / `tracking_utils.py` — Logging and W&B experiment tracking
- `routing_replay.py` — R3 (Rollout Routing Replay) for MoE training stability

### Plugin System (`miles_plugins/`)
- `mbridge/` — Model bridge integrations
- `megatron_bridge/` — Megatron-specific model bridges
- `models/` — Custom model implementations

### Test Structure (`tests/`)
- `tests/fast/` — Fast unit tests (no GPU): utils, rollout components, router
- `tests/e2e/` — End-to-end GPU tests organized by category:
  - `short/` — Quick 2-4 GPU smoke tests (FSDP)
  - `fsdp/` — FSDP backend tests
  - `megatron/` — Megatron-LM backend tests
  - `precision/` — Cross-framework log probability validation
  - `ckpt/` — Checkpoint save/restore tests
  - `long/` — Long-running integration tests
  - `image/` — Container image tests

### Key Architectural Patterns
- **Ray actors** orchestrate all distributed communication. The RolloutManager and training actors are Ray actors that communicate via `ray.get()` on remote method calls.
- **Weight sync** between training (Megatron/FSDP) and inference (SGLang) uses CUDA IPC zero-copy mapping for efficiency.
- **Hub pattern** for extensibility: `generate_hub`, `rm_hub`, `filter_hub`, and `middleware_hub` are all pluggable registries.
- **Two training backends** coexist: Megatron for large-scale (MoE, pipeline parallelism) and FSDP for simpler setups. The backend is selected via CLI arguments.

## CI/CD

- **Pre-commit CI** runs on all PRs: black, isort, ruff, autoflake, YAML checks
- **PR tests** are triggered by GitHub labels: `run-unit-test`, `run-ci-short`, `run-ci-fsdp`, `run-ci-megatron`, `run-ci-precision`, `run-ci-ckpt`, `run-ci-long`, `run-ci-image`
- E2E CI runs on self-hosted GPU runners using `radixark/miles:latest` container
- Workflow definitions in `.github/workflows/`; `pr-test.yml` is generated from a Jinja2 template via `tools/generate_github_workflows.py`

## Pytest Markers

`unit`, `integration`, `system`, `acceptance`, `docs`, `skipduringci`, `pleasefixme`
