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

master_address=$METIS_WORKER_0_HOST
num_processes_per_worker=$NPROC_PER_NODE
num_workers=$ARNOLD_WORKER_NUM
worker_rank=$ARNOLD_ID
master_port=$(python -m src.tool get_rand_port $ARNOLD_TRIAL_ID)

accelerate_yaml=$(python -m src.tool split_accelerate_yaml "$@" $master_address $num_processes_per_worker $num_workers $worker_rank $master_port)
model_yaml=$(python -m src.tool split_model_yaml "$@")
benchmark_yaml=$(python -m src.tool split_benchmark_yaml "$@")

target_script="src/train_bash.py"

accelerate launch --config_file $accelerate_yaml \
    $target_script $model_yaml

CUDA_VISIBLE_DEVICES=0 python src/evaluate.py $benchmark_yaml

