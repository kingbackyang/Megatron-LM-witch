#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# Comma-separated list of node IPs (update this for your cluster).
HOST_LIST=${HOST_LIST:-"192.168.1.10,192.168.1.11"}
MASTER_ADDR=${MASTER_ADDR:-"29.127.69.143"}
MASTER_PORT=${MASTER_PORT:-12345}

# Derive NNODES from HOST_LIST.
IFS=',' read -r -a NODES_ARR <<< "${HOST_LIST}"
NNODES=${NNODES:-${#NODES_ARR[@]}}

# NCCL settings (adjust per cluster).
export NCCL_IB_HCA=${NCCL_IB_HCA:-"mlx5_0,mlx5_1"}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}

echo "Launching Witch-1B MTP across ${NNODES} nodes: ${HOST_LIST}"

pdsh -w "ssh:${HOST_LIST}" "
  set -euo pipefail

  NODES_ARR=(${HOST_LIST//,/ })
  LOCAL_IP=\$(hostname -I | awk '{print \$1}')
  NODE_RANK=-1
  for i in \"\${!NODES_ARR[@]}\"; do
    if [[ \"\${NODES_ARR[\$i]}\" == \"\$LOCAL_IP\" ]]; then
      NODE_RANK=\$i
    fi
  done

  if [[ \$NODE_RANK -lt 0 ]]; then
    echo \"[ERROR] Could not determine NODE_RANK for \$LOCAL_IP\" >&2
    exit 1
  fi

  export NNODES=${NNODES}
  export NODE_RANK=\${NODE_RANK}
  export MASTER_ADDR=${MASTER_ADDR}
  export MASTER_PORT=${MASTER_PORT}
  export NPROC_PER_NODE=8

  export NCCL_IB_HCA=${NCCL_IB_HCA}
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE}
  export NCCL_DEBUG=${NCCL_DEBUG}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}

  echo \"Launching on Node \${NODE_RANK} (IP: \${LOCAL_IP})...\"
  bash ${REPO_ROOT}/witch_model/mtp_witch_1b/training_witch_1b_mtp_8gpu.sh
"
