#!/bin/bash

# 你的代码路径
SCRIPT_PATH="pretrain_witch_dummy.py"

# 设置并行度 (单机单卡调试)
TP_SIZE=4
PP_SIZE=1

# 模型参数 (非常小，为了快速跑通验证结构)
# Hidden 128, 2 Layers, 4 Heads
MODEL_ARGS="
    --num-layers 28 \
    --hidden-size 1024 \
    --num-attention-heads 4 \
    --seq-length 4096 \
    --max-position-embeddings 8192 \
    --ffn-hidden-size 512 \
    --vocab-size 1000 \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --swiglu \
    --bf16 
"

# 训练参数
TRAIN_ARGS="
    --micro-batch-size 2 \
    --global-batch-size 4 \
    --train-iters 5000 \
    --lr 0.001 \
    --min-lr 0.0001 \
    --lr-decay-style cosine \
    --lr-warmup-iters 10 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion
"

# 我们没有真实数据路径，但 Megatron 可能会检查 --data-path 参数是否存在
# 我们在 python 脚本里 Mock 掉了 dataset provider，所以这里填个假的没关系
DATA_ARGS="
    --data-path dummy_data \
    --tokenizer-type NullTokenizer \
    --vocab-size 151936 \
    --split 90,5,5 \
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

echo "Starting Witch Attention Debug Run..."

torchrun $DISTRIBUTED_ARGS $SCRIPT_PATH \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    --disable-bias-linear