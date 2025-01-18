#!/bin/bash
REPO_ROOT=$(dirname $0)
cd ${REPO_ROOT}

source /opt/tiger/llama_env/bin/activate

export HADOOP_ROOT_LOGGER="ERROR,console"
export LIBHDFS_OPTS="-Dhadoop.root.logger=$HADOOP_ROOT_LOGGER"
export LIBHDFS_OPTS="$LIBHDFS_OPTS -Xms512m -Xmx10g "
export KRB5CCNAME="/tmp/krb5cc"
export TF_CPP_MIN_LOG_LEVEL="2"
export MKL_THREADING_LAYER="GNU"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="mlx5_2:1"
export NCCL_SOCKET_IFNAME="eth0"
export ARNOLD_FRAMEWORK="pytorch"
export NCCL_DEBUG=ERROR

export NPROC_PER_NODE=$(python -c "import torch;print(torch.cuda.device_count())")

#目前只支持单卡推理
python src/inference.py $@

