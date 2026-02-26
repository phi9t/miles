#!/usr/bin/env bash
set -euo pipefail

# OOM-mitigation ladder for Qwen3-0.6B SFT smoke on small GPUs.
# Stop on first successful run.

MODE="${MODE:-direct}"
NUM_GPUS="${NUM_GPUS:-1}"
SMOKE_OPTIMIZER="${SMOKE_OPTIMIZER:-adam}" # adam|sgd
PROMPT_DATA="${PROMPT_DATA:-/root/datasets/gsm8k/train_sft_messages.parquet}"

if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "Missing ${PROMPT_DATA}; generating a tiny SFT parquet from gsm8k train..."
    python - <<'PY'
import pandas as pd
src = "/root/datasets/gsm8k/train.parquet"
out = "/root/datasets/gsm8k/train_sft_messages.parquet"
df = pd.read_parquet(src).head(8).copy()
msgs = []
for _, r in df.iterrows():
    msgs.append([
        {"role": "system", "content": "You are a helpful assistant. Please put the answer within \\boxed{}."},
        {"role": "user", "content": str(r["question"])},
        {"role": "assistant", "content": str(r["answer"])},
    ])
pd.DataFrame({"messages": msgs}).to_parquet(out, index=False)
print("wrote", out, "rows", len(msgs))
PY
fi

run_attempt() {
    local name="$1"
    shift
    echo "================ ${name} ================"
    set +e
    env \
        MODE="${MODE}" \
        NUM_GPUS="${NUM_GPUS}" \
        OPTIMIZER="${SMOKE_OPTIMIZER}" \
        PROMPT_DATA="${PROMPT_DATA}" \
        TORCH_COMPILE_DISABLE=1 \
        TORCHDYNAMO_DISABLE=1 \
        WANDB_MODE=disabled \
        TOKENIZERS_PARALLELISM=false \
        "$@" \
        bash scripts/run-qwen3-0.6B-sft-pytorch-smoke.sh
    local rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
        echo "[PASS] ${name}"
        return 0
    fi
    echo "[FAIL] ${name} rc=${rc}"
    return 1
}

run_attempt "A1-minimal-token" \
    GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 \
    ROLLOUT_MAX_RESPONSE_LEN=8 USE_DYNAMIC_BATCH_SIZE=1 MAX_TOKENS_PER_GPU=256 LOG_PROBS_MAX_TOKENS_PER_GPU=128 \
    GRADIENT_CHECKPOINTING=0 OFFLOAD_TRAIN=0 OFFLOAD_ROLLOUT=0 FSDP_CPU_OFFLOAD=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 && exit 0

run_attempt "A2-plus-grad-ckpt" \
    GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 \
    ROLLOUT_MAX_RESPONSE_LEN=8 USE_DYNAMIC_BATCH_SIZE=1 MAX_TOKENS_PER_GPU=256 LOG_PROBS_MAX_TOKENS_PER_GPU=128 \
    GRADIENT_CHECKPOINTING=1 OFFLOAD_TRAIN=0 OFFLOAD_ROLLOUT=0 FSDP_CPU_OFFLOAD=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 && exit 0

run_attempt "A3-offload-train-rollout" \
    GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 \
    ROLLOUT_MAX_RESPONSE_LEN=8 USE_DYNAMIC_BATCH_SIZE=1 MAX_TOKENS_PER_GPU=192 LOG_PROBS_MAX_TOKENS_PER_GPU=96 \
    GRADIENT_CHECKPOINTING=1 OFFLOAD_TRAIN=1 OFFLOAD_ROLLOUT=1 FSDP_CPU_OFFLOAD=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 && exit 0

run_attempt "A4-fsdp-cpu-offload-mpi" \
    GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 \
    ROLLOUT_MAX_RESPONSE_LEN=8 USE_DYNAMIC_BATCH_SIZE=1 MAX_TOKENS_PER_GPU=192 LOG_PROBS_MAX_TOKENS_PER_GPU=96 \
    GRADIENT_CHECKPOINTING=1 OFFLOAD_TRAIN=0 OFFLOAD_ROLLOUT=0 FSDP_CPU_OFFLOAD=1 FSDP_CPU_BACKEND=mpi \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 && exit 0

if [[ "${SMOKE_OPTIMIZER}" != "sgd" ]]; then
    run_attempt "A5-sgd-fallback" \
        GLOBAL_BATCH_SIZE=1 MICRO_BATCH_SIZE=1 ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 \
        ROLLOUT_MAX_RESPONSE_LEN=8 USE_DYNAMIC_BATCH_SIZE=1 MAX_TOKENS_PER_GPU=192 LOG_PROBS_MAX_TOKENS_PER_GPU=96 \
        GRADIENT_CHECKPOINTING=1 OFFLOAD_TRAIN=1 OFFLOAD_ROLLOUT=1 FSDP_CPU_OFFLOAD=0 \
        OPTIMIZER=sgd \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 && exit 0
fi

echo "All OOM mitigation attempts failed."
exit 1
