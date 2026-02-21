#!/bin/bash

# 多机多卡（12 节点 × 8 卡 = 96 卡）训练脚本
# 使用前请在所有机器上设置相同的文件路径，并填好 MASTER_ADDR / NODE_RANK。

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/pretrain_witch_1b_mtp.py"

export PYTHONPATH=$PYTHONPATH:/opt/workspace/Megatron-LM

# === 分布式环境参数（请按节点修改 MASTER_ADDR / NODE_RANK） ===
MASTER_ADDR=${MASTER_ADDR:-"29.127.69.143"}   # 修改为主节点 IP
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=12
NPROC_PER_NODE=8
# 当前节点在集群中的序号：主节点 0，从节点 1
NODE_RANK=${NODE_RANK:-3}

# === WandB 离线缓存 ===
export WANDB_MODE=offline
export WANDB_PROJECT=witch-1b-mtp
export WANDB_NAME=run-witch-1b-mtp-multinode-$(date +%m%d-%H%M)
export WANDB_DIR=workspace/wandb_cache

# === 模型配置 ===
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

# === 训练配置 ===
TRAIN_ARGS="
    --micro-batch-size 4 \
    --global-batch-size 384 \
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

# === 数据配置 ===

DATA_ARGS="
    --data-path workspace/dataset/ft_local/pretrain_data_megatron_bin \
    --data-cache-path workspace/Megatron-LM/witch_model/cache_indicesv2 \
    --num-dataset-builder-threads 1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model workspace/Megatron-LM/witch_model/qwen0_6B_customv2 \
    --split 100,0,0 \
    --seed 1234
"

# === torchrun 启动参数 ===
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

SAVE_DIR=workspace/Megatron-LM/witch_model/checkpoints/witch_1b_mtp_multinode
mkdir -p "${SAVE_DIR}"

echo "Starting Witch-1B MTP Run (multinode)..."

torchrun ${DISTRIBUTED_ARGS} ${SCRIPT_PATH} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --save "${SAVE_DIR}" \
    ${MODEL_ARGS} \
    ${MTP_ARGS} \
    ${TRAIN_ARGS} \
    ${DATA_ARGS} \
    --disable-bias-linear
