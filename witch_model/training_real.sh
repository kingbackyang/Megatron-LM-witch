#!/bin/bash

# 你的新代码路径
SCRIPT_PATH="pretrain_witch_real.py"
export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM
TP_SIZE=4
PP_SIZE=1

MODEL_ARGS="
    --num-layers 28 \
    --hidden-size 1024 \
    --num-attention-heads 4 \
    --seq-length 4096 \
    --max-position-embeddings 8192 \
    --ffn-hidden-size 512 \
    --vocab-size 151936 \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --swiglu \
    --bf16 
"

TRAIN_ARGS="
    --micro-batch-size  4 \
    --global-batch-size 32 \
    --train-iters 1000 \
    --eval-interval 100 \
    --eval-iters 10 \
    --save-interval 50 \
    --lr 0.0001 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 10 \
    --clip-grad 1.0 \
    --rerun-mode disabled \
    --no-gradient-accumulation-fusion
"

# 🟢 真实数据配置
# 注意：data-path 这里填的是文件前缀 (去掉 .bin 和 .idx)
DATA_ARGS="
    --data-path /workspace/Megatron-LM/witch_model/qwen_processed_text_document \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /workspace/Megatron-LM/witch_model/qwen0_6B_customv2 \
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
# export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=7200

echo "Starting Witch Attention Real Data Run..."

torchrun $DISTRIBUTED_ARGS $SCRIPT_PATH \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    --disable-bias-linear