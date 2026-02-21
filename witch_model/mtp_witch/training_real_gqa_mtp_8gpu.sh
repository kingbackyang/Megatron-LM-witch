#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/pretrain_witch_real_mtp.py"

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}"
else
    export PYTHONPATH="${REPO_ROOT}"
fi
export WANDB_MODE=offline
export WANDB_PROJECT=witch-gqa-mtp
export WANDB_NAME=run-gqa-mtp-$(date +%m%d-%H%M)
export WANDB_DIR=/data2/wandb_cache

TP_SIZE=4
PP_SIZE=1

MODEL_ARGS="
    --num-layers 28 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --ffn-hidden-size 3072 \
    --seq-length 4096 \
    --max-position-embeddings 40960 \
    --vocab-size 151936 \
    --use-rotary-position-embeddings \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --bf16
"

MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1
"

TRAIN_ARGS="
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters 100000 \
    --eval-interval 10000 \
    --eval-iters 0 \
    --save-interval 10000 \
    --lr 0.0001 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 10 \
    --clip-grad 1.0 \
    --rerun-mode disabled \
    --no-gradient-accumulation-fusion
"

DATA_ARGS="
    --data-path /data2/pretrain_data_megatron_bin \
    --data-cache-path ${REPO_ROOT}/witch_model/cache_indices \
    --num-dataset-builder-threads 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${REPO_ROOT}/witch_model/qwen0_6B_customv2 \
    --split 100,0,0 \
    --seed 1234
"

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000
"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=7200000
export TORCH_DISTRIBUTED_TIMEOUT=7200

echo "Starting Witch MTP run (8 GPUs, TP=4, DP=2) ..."

torchrun ${DISTRIBUTED_ARGS} ${SCRIPT_PATH} \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --save /data2/Megatron-LM/witch_model/checkpoints/witch_gqa_mtp \
    ${MODEL_ARGS} \
    ${MTP_ARGS} \
    ${TRAIN_ARGS} \
    ${DATA_ARGS} \
    --disable-bias-linear
