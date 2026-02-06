# Architecture Walkthrough: `test_qwen3_0.6B_fsdp_colocated_2xGPU.py`

This document traces the full training and inference architecture of Miles through the lens of the simplest end-to-end example: a 2-GPU colocated FSDP run with Qwen3-0.6B on GSM8K.

## Phase 0: Launch scaffolding

The test script itself never touches Miles internals directly. It:

1. **`prepare()`** — Downloads `Qwen3-0.6B` from HuggingFace and the `gsm8k` dataset to `/root/models/` and `/root/datasets/`.

2. **`execute()`** — Assembles a massive CLI argument string, then calls `U.execute_train()` (`miles/utils/external_utils/command_utils.py:93`). This function:
   - Kills any stale `sglang`, `ray`, `miles` processes
   - Starts a **Ray head node** with `ray start --head --num-gpus 2`
   - Submits a **Ray job**: `ray job submit -- python3 train.py <all_args>`

So the test script is purely a **launch harness**. The real system starts at `train.py`.

## Phase 1: GPU allocation (`train.py` → `placement_group.py`)

```
train.py:main → parse_args() → train(args)
```

`train()` calls `create_placement_groups(args)` (`miles/ray/placement_group.py:79`). Because `--colocate` is set, the placement logic at line 92-94 allocates **only `actor_num_nodes * actor_num_gpus_per_node = 2`** GPUs total. Both rollout (inference) and training share the same GPUs — that's what "colocated" means.

The function creates a Ray `PlacementGroup` with `strategy="PACK"` (keep everything on the same node), then uses ephemeral `InfoActor` Ray actors to discover the physical GPU IDs and sort them deterministically. The result is three overlapping views into the same placement group:
- `pgs["actor"]` — bundle indices [0, 1] (training)
- `pgs["rollout"]` — bundle indices [0, 1] (inference)
- `pgs["critic"]` — None (no critic in this example)

**Key insight**: In colocated mode, training and inference timeshare the same GPUs. They alternate — inference fills VRAM during rollout, then releases it; training takes over. This is why `--update-weight-buffer-size 536870912` matters — it controls how much memory the weight sync uses.

## Phase 2: Rollout Manager creation (`placement_group.py:169` → `miles/ray/rollout.py`)

```python
rollout_manager = RolloutManager.options(num_cpus=1, num_gpus=0).remote(args, pg)
```

The `RolloutManager` is a **Ray actor that uses 0 GPUs itself** — it's an orchestrator. During `__init__` (`rollout.py:50`):

1. **Starts the SGLang Router** (`_start_router`, line 652) — a separate process that load-balances HTTP requests across SGLang inference engines. In this 2-GPU example with `--rollout-num-gpus-per-engine 2`, there's just one engine.

2. **Loads the data source** — by default `miles.rollout.data_source.DefaultDataSource`, which reads the gsm8k parquet file.

3. **Loads pluggable rollout functions** — `self.generate_rollout` and `self.eval_generate_rollout` are loaded dynamically. The default is `miles.rollout.sglang_rollout.generate_rollout`.

4. **Creates SGLang Engine actors** (`init_rollout_engines`, line 473) — For each engine, it creates a Ray actor wrapping `SGLangEngine` (`miles/backends/sglang_utils/sglang_engine.py:109`). Each engine actor:
   - Gets 0.2 fractional GPU (to coexist with training actors on the same GPU)
   - Spawns an SGLang HTTP server process via `launch_server_process` (a full inference server)
   - Registers itself with the SGLang router via HTTP POST to `/add_worker`

After this phase, you have a fully running **SGLang inference server cluster** behind a load-balancing router, ready to accept `/generate` requests.

## Phase 3: Training Actor creation (`placement_group.py:132`)

```python
actor_model = allocate_train_group(args, num_nodes=1, num_gpus_per_node=2, pg=pgs["actor"])
```

`RayTrainGroup.__init__` (`actor_group.py:29`) chooses the backend based on `--train-backend fsdp`:

```python
# actor_group.py:85
from miles.backends.fsdp_utils import FSDPTrainRayActor
actor_impl = FSDPTrainRayActor
```

It creates **2 Ray actors** (one per GPU), each getting `num_gpus_per_actor=0.4` fractional GPU. Each actor is an `FSDPTrainRayActor` instance. Rank 0 discovers its IP and picks a master port; all actors get the same master address.

Then `async_init` is called (`actor_group.py:108`), which on each actor:
1. Sets `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `LOCAL_RANK` env vars
2. Calls `dist.init_process_group(backend="nccl")` — standard PyTorch distributed
3. Loads `Qwen3-0.6B` via `AutoModelForCausalLM.from_pretrained()` (rank 0 loads to CPU, other ranks use meta tensors)
4. Wraps the model with **FSDP v2** (`apply_fsdp2`, `actor.py:656`) — shards parameters across the 2 GPUs
5. Creates the AdamW optimizer and LR scheduler
6. Creates the **weight updater** — since `--colocate`, this is `UpdateWeightFromTensor` which uses **CUDA IPC zero-copy** to push weights directly to the SGLang engine on the same GPU

## Phase 4: Initial weight sync

```python
# train.py:27
actor_model.update_weights()
```

Before any training, the FSDP actor model pushes its initial weights to the SGLang inference engines. This happens in `FSDPTrainRayActor.update_weights()` (`actor.py:541`):

1. Fetches the rollout engines and lock from the RolloutManager
2. Calls `self.weight_updater.update_weights()` (`update_weight_utils.py:46`)
3. Iterates through `model.state_dict()`, gathering DTensor shards into full tensors
4. Groups tensors into 512MB buckets (`--update-weight-buffer-size`)
5. Sends each bucket to the SGLang engines via CUDA IPC (same-GPU direct memory access)

**This is the critical bridge** — it ensures the SGLang inference engine has the exact same weights as the FSDP training model.

## Phase 5: The main training loop (`train.py:66`)

```python
for rollout_id in range(0, 60):  # --num-rollout 60
```

Each iteration of this loop is one **RL step** with three phases:

### 5a. Rollout (Inference) — `rollout_manager.generate.remote(rollout_id)`

The `RolloutManager.generate()` method (`rollout.py:140`):

1. Calls `_get_rollout_data()` which invokes the loaded rollout function — default is `miles.rollout.sglang_rollout.generate_rollout`
2. **`generate_rollout`** (`sglang_rollout.py`):
   - Pulls `--rollout-batch-size 32` prompts from the gsm8k dataset
   - For each prompt, generates `--n-samples-per-prompt 8` responses (256 total)
   - Uses `--over-sampling-batch-size 64` for over-sampling with dynamic filtering
   - Sends requests as async HTTP POSTs to the SGLang router at `http://{router_ip}:{router_port}/generate`
   - The router forwards to the SGLang engine, which runs the actual GPU inference
   - After generation, applies the **reward model** (`--rm-type math`) to score each response
   - Applies the **dynamic sampling filter** (`check_reward_nonzero_std`) — drops prompt groups where all responses got the same reward (zero variance = no learning signal)
3. Back in `generate()`:
   - Trims samples to fit `--global-batch-size 256`
   - Calls `_convert_samples_to_train_data()` — converts `Sample` objects into a training dict: `{tokens, response_lengths, rewards, loss_masks, ...}`
   - Normalizes rewards using GRPO group normalization (subtract group mean, divide by group std)
   - Calls `_split_train_data_by_dp()` — splits the training data across data-parallel ranks (2 in this case), puts each shard into Ray object store via `ray.put()`

### 5b. Training — `actor_model.async_train(rollout_id, rollout_data_ref)`

This calls `.train.remote()` on each of the 2 FSDP actors. In `FSDPTrainRayActor.train()` (`actor.py:386`):

1. **Fetch rollout data** from Ray object store via `get_rollout_data()` — each actor gets its DP shard
2. **Compute actor log probs** (`_compute_log_prob("actor", ...)`) — forward pass through the model to get per-token log probabilities for the generated responses
3. **Compute advantages** (`compute_advantages_and_returns()`) — GRPO advantage estimation using the normalized rewards and log probs
4. **Actor training loop** (`actor.py:432`):
   - For each microbatch:
     - Forward pass → logits
     - `loss_function()` computes the **PPO/GRPO clipped surrogate loss** with `--eps-clip 0.2` / `--eps-clip-high 0.28`
     - `loss.backward()`
   - Clip gradients (`clip_grad_norm_`)
   - `optimizer.step()` + `lr_scheduler.step()`

### 5c. Weight sync — `actor_model.update_weights()`

After training, the updated FSDP model weights are pushed back to the SGLang inference engines (same mechanism as Phase 4). This ensures the next rollout uses the freshly trained model.

### 5d. Periodic eval

Every 20 rollouts (`--eval-interval 20`), the RolloutManager runs evaluation:
- Generates responses on gsm8k test set with `--eval-top-k 1` (greedy)
- Scores with math reward model
- Logs pass rate; CI checks against `--ci-metric-checker-threshold 0.71`

## Complete data flow diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Ray Head (orchestrator)                   │
│  train.py: for rollout_id in range(60):                     │
└────────┬──────────────────────────────┬─────────────────────┘
         │                              │
    ┌────▼────┐                   ┌─────▼──────┐
    │ Rollout  │   HTTP /generate │  SGLang     │
    │ Manager  │ ──────────────── │  Engine     │
    │ (Ray,    │   via Router     │  (GPU 0+1)  │
    │  0 GPU)  │                  │  inference   │
    │          │ ◄──── samples ── │  server     │
    │          │                  └─────────────┘
    │ reward   │
    │ scoring  │
    │ + filter │
    │          │
    │ convert  │── ray.put() ──► Ray Object Store
    └──────────┘                       │
                                       │ ray.get()
                              ┌────────▼─────────┐
                              │  FSDP Actors      │
                              │  (GPU 0 + GPU 1)  │
                              │                   │
                              │  1. log_probs     │
                              │  2. advantages    │
                              │  3. PPO/GRPO loss │
                              │  4. backward      │
                              │  5. optimizer.step │
                              └────────┬──────────┘
                                       │
                                  update_weights()
                                  (CUDA IPC zero-copy)
                                       │
                              ┌────────▼──────────┐
                              │  SGLang Engine     │
                              │  (weights updated) │
                              │  ready for next    │
                              │  rollout           │
                              └───────────────────┘
```

## Key architectural decisions visible in this example

1. **Colocated mode** (`--colocate`): Training and inference share GPUs. The system carefully manages GPU memory — SGLang occupies VRAM during rollout, then training takes over. The fractional GPU allocations (0.4 for training, 0.2 for SGLang) are Ray scheduling hints, not hard memory limits.

2. **FSDP over Megatron**: For this small 0.6B model, FSDP is simpler — no need for tensor/pipeline parallelism. The test passes `megatron_model_type=None` to skip Megatron checkpoint conversion entirely. The model loads directly from HuggingFace format.

3. **HTTP-based inference**: Even though everything is on the same machine, rollout goes through HTTP to the SGLang server. This decouples inference from training and lets SGLang use its optimized batching/scheduling (continuous batching, RadixAttention, etc.).

4. **Weight sync via CUDA IPC**: In colocated mode, `UpdateWeightFromTensor` uses direct GPU memory mapping rather than serializing weights over the network. Parameters are bucketed into 512MB chunks and transferred in-place.

5. **Ray as the glue**: Every component (RolloutManager, SGLangEngine, FSDPTrainRayActor) is a Ray actor. The training loop in `train.py` is just `ray.get()` calls orchestrating these actors. This makes it trivial to scale to multiple nodes — swap colocated for non-colocated, add more placement group bundles, and Ray handles the rest.
