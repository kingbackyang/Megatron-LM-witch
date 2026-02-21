#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/pretrain_witch_1b_mtp.py"

export PYTHONPATH=$PYTHONPATH:/opt/workspace/Megatron-LM
export WANDB_MODE=offline
export WANDB_PROJECT=witch-1b-mtp
export WANDB_NAME=run-witch-1b-mtp-$(date +%m%d-%H%M)
export WANDB_DIR=workspace/wandb_cache

TP_SIZE=1
PP_SIZE=1
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}

MODEL_ARGS="
    --num-layers 16 \
    --hidden-size 2048 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --ffn-hidden-size 6144 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
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
    --train-iters 500000 \
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
    --data-path workspace/dataset/ft_local/pretrain_data_megatron_bin \
    --data-cache-path workspace/Megatron-LM/witch_model/cache_indicesv2 \
    --num-dataset-builder-threads 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model workspace/Megatron-LM/witch_model/qwen0_6B_customv2 \
    --split 100,0,0 \
    --seed 1234
"

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPROC_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=7200000
export TORCH_DISTRIBUTED_TIMEOUT=7200

TOTAL_GPUS=$((NPROC_PER_NODE * NNODES))
DP_SIZE=$((TOTAL_GPUS / (TP_SIZE * PP_SIZE)))
echo "Starting Witch-1B MTP run (${TOTAL_GPUS} GPUs, TP=${TP_SIZE}, DP=${DP_SIZE}) ..."

SAVE_DIR=workspace/Megatron-LM/witch_model/checkpoints/witch_1b_mtp
mkdir -p "${SAVE_DIR}"

torchrun ${DISTRIBUTED_ARGS} ${SCRIPT_PATH} \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --save "${SAVE_DIR}" \
    ${MODEL_ARGS} \
    ${MTP_ARGS} \
    ${TRAIN_ARGS} \
    ${DATA_ARGS} \
    --disable-bias-linear
