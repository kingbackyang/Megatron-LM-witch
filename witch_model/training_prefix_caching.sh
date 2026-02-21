#!/bin/bash

# 你的 Python 脚本路径 (必须和训练用的脚本一致)
SCRIPT_PATH="pretrain_witch_real.py"

# ==========================================
# 🟢 关键修改 1: 并行度强制设为 1
# ==========================================
TP_SIZE=1
PP_SIZE=1

# ==========================================
# 模型参数 (保持和 training_real.sh 一致)
# 这样能保证参数解析不出错，且 seq-length 等影响索引生成的参数一致
# ==========================================
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

# ==========================================
# 🟢 关键修改 2: 训练参数最小化
# 我们只跑 1 步，Batch Size 设为 1，仅仅为了触发 Data Builder
# ==========================================
TRAIN_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 1000 \
    --lr 0.0001 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --no-gradient-accumulation-fusion
"

export NCCL_TIMEOUT=7200000
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_TIMEOUT=7200
# ==========================================
# 🟢 关键修改 3: 数据参数 (必须和 training_real.sh 一模一样!!!)
# ==========================================
# 指向你的数据根目录 (自动扫描模式)
DATA_ARGS="
    --data-path /workspace/Megatron-LM/witch_model/qwen_processed_text_document \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /workspace/Megatron-LM/witch_model/qwen0_6B_customv2 \
    --split 900,50,50 \
    --seed 1234
"

# ==========================================
# 🟢 关键修改 4: 只启动 1 个进程 (nproc_per_node=1)
# ==========================================
DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000
"

export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=================================================================="
echo "Starting Index Builder (Single Process Mode)..."
echo "This will take a while, but it will NOT timeout."
echo "Wait until you see '> Datasets built successfully via Builder'."
echo "=================================================================="

torchrun $DISTRIBUTED_ARGS $SCRIPT_PATH \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    --disable-bias-linear