#!/usr/bin/env bash
set -euo pipefail

# Pure-PyTorch smoke run for GPUs that cannot support flash_attn/flashinfer.
# This uses FSDP + SFT rollout in debug-train-only mode, so no SGLang rollout engines are started.

MODEL_DIR="${MODEL_DIR:-/root/models/Qwen3-0.6B}"
PROMPT_DATA="${PROMPT_DATA:-/root/datasets/gsm8k/train.parquet}"
NUM_GPUS="${NUM_GPUS:-1}"
MODE="${MODE:-ray}" # ray|direct
OPTIMIZER="${OPTIMIZER:-adam}" # adam|sgd

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-32}"

USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-0}" # 0|1
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-}"
LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-}"

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}" # 0|1
OFFLOAD_TRAIN="${OFFLOAD_TRAIN:-0}" # 0|1
OFFLOAD_ROLLOUT="${OFFLOAD_ROLLOUT:-0}" # 0|1
FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-0}" # 0|1
FSDP_CPU_BACKEND="${FSDP_CPU_BACKEND:-mpi}" # gloo|mpi|empty

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Missing model checkpoint at ${MODEL_DIR}" >&2
    exit 2
fi

if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "Missing prompt data at ${PROMPT_DATA}" >&2
    exit 2
fi

python - <<'PY'
import torch
print(f"torch.cuda.is_available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
PY

COMMON_ARGS=(
    --train-backend fsdp
    --distributed-backend nccl
    --fp16
    --attn-implementation eager
    --context-parallel-size 1
    --hf-checkpoint "${MODEL_DIR}"
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --rollout-shuffle
    --rollout-function-path miles.rollout.sft_rollout.generate_rollout
    --debug-train-only
    --loss-type sft_loss
    --disable-compute-advantages-and-returns
    --num-rollout 1
    --optimizer "${OPTIMIZER}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --actor-num-nodes 1
    --actor-num-gpus-per-node "${NUM_GPUS}"
    --rollout-num-gpus "${NUM_GPUS}"
    --rollout-num-gpus-per-engine 1
)

if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
    COMMON_ARGS+=(--gradient-checkpointing)
fi

if [[ "${OFFLOAD_TRAIN}" == "1" ]]; then
    COMMON_ARGS+=(--offload-train)
fi

if [[ "${OFFLOAD_ROLLOUT}" == "1" ]]; then
    COMMON_ARGS+=(--offload-rollout)
fi

if [[ "${FSDP_CPU_OFFLOAD}" == "1" ]]; then
    COMMON_ARGS+=(--fsdp-cpu-offload)
    if [[ -n "${FSDP_CPU_BACKEND}" ]]; then
        COMMON_ARGS+=(--fsdp-cpu-backend "${FSDP_CPU_BACKEND}")
    fi
fi

if [[ "${USE_DYNAMIC_BATCH_SIZE}" == "1" ]]; then
    COMMON_ARGS+=(--use-dynamic-batch-size)
    if [[ -n "${MAX_TOKENS_PER_GPU}" ]]; then
        COMMON_ARGS+=(--max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
    fi
    if [[ -n "${LOG_PROBS_MAX_TOKENS_PER_GPU}" ]]; then
        COMMON_ARGS+=(--log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}")
    fi
fi

echo "Running smoke config:"
echo "  OPTIMIZER=${OPTIMIZER} MODE=${MODE} NUM_GPUS=${NUM_GPUS}"
echo "  GBS=${GLOBAL_BATCH_SIZE} MBS=${MICRO_BATCH_SIZE} RBS=${ROLLOUT_BATCH_SIZE} NSPP=${N_SAMPLES_PER_PROMPT}"
echo "  ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN}"
echo "  USE_DYNAMIC_BATCH_SIZE=${USE_DYNAMIC_BATCH_SIZE} MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-unset}"
echo "  LOG_PROBS_MAX_TOKENS_PER_GPU=${LOG_PROBS_MAX_TOKENS_PER_GPU:-unset}"
echo "  GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING} OFFLOAD_TRAIN=${OFFLOAD_TRAIN} OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT}"
echo "  FSDP_CPU_OFFLOAD=${FSDP_CPU_OFFLOAD} FSDP_CPU_BACKEND=${FSDP_CPU_BACKEND:-unset}"

if [[ "${MODE}" == "direct" ]]; then
    python train.py "${COMMON_ARGS[@]}"
    exit 0
fi

if [[ "${MODE}" != "ray" ]]; then
    echo "Unsupported MODE=${MODE}; expected ray or direct" >&2
    exit 2
fi

if ! command -v ray >/dev/null 2>&1; then
    echo "ray command not found in current env. Activate UV layered env first." >&2
    exit 2
fi

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address 127.0.0.1 --num-gpus "${NUM_GPUS}" --disable-usage-stats
ray job submit --address="http://127.0.0.1:8265" -- python train.py "${COMMON_ARGS[@]}"
