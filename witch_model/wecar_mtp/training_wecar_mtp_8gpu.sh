#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/pretrain_wecar_mtp.py"

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi

# === Model (Performance Benchmarking 1.7B) ===
MODEL_ARGS="
    --num-layers 24 \
    --hidden-size 2304 \
    --num-attention-heads 24 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --vocab-size 151936 \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --rotary-base 10000 \
    --fp16 \
    --transformer-impl transformer_engine \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 96
"

# === MTP ===
MTP_ARGS="
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1
"

# === Training ===
TRAIN_ARGS="
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --train-iters 1000 \
    --save-interval 1000 \
    --lr 0.0001 \
    --min-lr 0.00001 \
    --lr-decay-style cosine \
    --lr-decay-iters 320 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --eval-iters 1 \
    --eval-interval 1000 \
    --log-interval 100 
"

# === Data / Tokenizer (update paths as needed) ===
DATA_ARGS="
    --data-path /data2/pretrain_data_megatron_bin \
    --data-cache-path ${REPO_ROOT}/witch_model/cache_indices \
    --num-dataset-builder-threads 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${REPO_ROOT}/witch_model/qwen0_6B_customv2 \
    --split 969,30,1 \
    --seed 1234
"

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6001
"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=7200000
export TORCH_DISTRIBUTED_TIMEOUT=7200

echo "Starting WeCar (1.7B) with MTP + GQA ..."

torchrun ${DISTRIBUTED_ARGS} ${SCRIPT_PATH} \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --save /data2/Megatron-LM/witch_model/wecar_mtp/checkpoints \
    ${MODEL_ARGS} \
    ${MTP_ARGS} \
    ${TRAIN_ARGS} \
    ${DATA_ARGS} \
    --disable-bias-linear
