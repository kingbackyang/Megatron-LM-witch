#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/smoke_test_mtp.py"

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi

torchrun --nproc_per_node 1 "${SCRIPT_PATH}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 2 \
  --hidden-size 128 \
  --num-attention-heads 4 \
  --kv-channels 32 \
  --ffn-hidden-size 512 \
  --seq-length 32 \
  --max-position-embeddings 32 \
  --vocab-size 1024 \
  --position-embedding-type rope \
  --use-rotary-position-embeddings \
  --rotary-base 10000 \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --swiglu \
  --bf16 \
  --micro-batch-size 2 \
  --global-batch-size 2 \
  --train-iters 2 \
  --eval-iters 0 \
  --eval-interval 0 \
  --save-interval 0 \
  --log-interval 1 \
  --lr 0.0001 \
  --min-lr 0.0001 \
  --lr-decay-style constant \
  --clip-grad 1.0 \
  --num-workers 0 \
  --mtp-num-layers 1 \
  --mtp-loss-scaling-factor 0.1 \
  --tokenizer-type NullTokenizer \
  --no-load-optim \
  --no-load-rng \
  --disable-bias-linear
