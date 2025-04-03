#!/bin/bash
unset http_proxy
unset https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export WANDB_MODE="online"
export WANDB_API_KEY="9144b562879460494cad9b7abe439e779cfa8af7"
export TOKENIZERS_PARALLELISM=true

export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=162
export NCCL_IB_TIMEOUT=25
export NCCL_PXN_DISABLE=0
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_IB_RETRY_CNT=32

if [ -z "$MASTER_ADDR" ]; then
  export MASTER_ADDR="localhost"
fi

if [ -z "$MASTER_PORT" ]; then
  export MASTER_PORT=29501
fi

if [ -z "$NODE_RANK" ]; then
  export NODE_RANK=0
fi

if [ -z "$NNODES" ]; then
  export NNODES=1
fi

torchrun \
  --nproc_per_node 8 \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --node_rank $NODE_RANK \
  --nnodes $NNODES \
  train_step_distill.py \
  --config scripts/step_distill/14B_32_16_bf16.yaml