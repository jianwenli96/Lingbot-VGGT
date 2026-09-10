#!/bin/bash

set -x
umask 007

NNODES="$MA_NUM_HOSTS"                # 总机器数 MA_NUM_HOSTS
NODE_RANK="$VC_TASK_INDEX"            # 当前机器序号 VC_TASK_INDEX
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"  # 主节点(Rank 0)的内网IP

NGPU="$MA_NUM_GPUS"
MASTER_PORT=${MASTER_PORT:-"29501"}
LOG_RANK=${LOG_RANK:-"0"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
config_name=${CONFIG_NAME}

## cmd setting
export HF_DATASETS_CACHE="/efs-mi-east4-2/Caches/hf_dataset_cache"
export TOKENIZERS_PARALLELISM=false

# 建议增加 NCCL 调试日志和网卡指定（如果有多网卡环境）
export HCCL_DEBUG=INFO
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export ASCEND_HOME_PATH="/usr/local/Ascend/ascend-toolkit/latest"
export LOG_TIME=$(date +"%Y%m%d_%H%M%S")

# 到当前文件目录的上一层（项目根目录）
cd "$(dirname "$0")/.." || exit

/efs-mi-east4-2/lijianwen/Apps/miniforge3/envs/lingbot-vggt/bin/python -m torch.distributed.run \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${master_port} \
    --nproc_per_node=${num_gpu} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
