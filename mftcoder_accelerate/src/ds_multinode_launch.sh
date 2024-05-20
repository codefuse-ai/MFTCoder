#!/bin/sh
# Author: Chaoyu Chen
# Last Modified: 2024/5/20
# Description: # Launch script on Multiple Nodes

# Run this script on all Nodes.

# You need to export your number of nodes and number of GPUs per node first.
N_NODE=4
N_GPU_PER_NODE=8

# You need to export $RANK, $MASTER_ADDR, $MASTER_PORT automatically for each Node.

# config path
CONFIG="configs/xxx_train_config.json"

# envs used inside training
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=False

TODAY=$(date +%Y-%m%d-%H%M)

# accelerate launch --config_file accelerate_ds_config.yaml \
accelerate launch \
    --num_machines $N_NODE \
    --num_processes $(($N_NODE*$N_GPU_PER_NODE)) \
    --use_deepspeed \
    --deepspeed_multinode_launcher 'standard' \
    --zero_stage 2 \
    --offload_optimizer_device 'cpu' \
    --offload_param_device 'none' \
    --gradient_accumulation_steps 1 \
    --gradient_clipping 1.0 \
    --zero3_init_flag false \
    --zero3_save_16bit_model false \
    --main_training_function 'main' \
    --mixed_precision 'bf16' \
    --dynamo_backend 'no' \
    --same_network \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend 'static' \
    pefts/mft_accelerate.py --train_config "$CONFIG" --distributed_type "deepspeed"