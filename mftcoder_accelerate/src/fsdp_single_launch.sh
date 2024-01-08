#!/bin/sh
# Author: Chaoyu Chen
# Last Modified: 2024/12/11
# Description: An alternative(command line) way to launch FSDP training

# Launch script on single node
N_GPU_PER_NODE=8

# envs used inside training
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=False

TODAY=$(date +%Y-%m%d-%H%M)

# accelerate launch --config_file accelerate_fsdp_config.yaml \
accelerate launch \
    --use_fsdp \
    --num_machines=1 \
    --num_processes=$N_GPU_PER_NODE \
    --fsdp_sharding_strategy=1 \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
    --fsdp_state_dict_type=FULL_STATE_DICT \
    --fsdp_backward_prefetch_policy=BACKWARD_PRE \
    --fsdp_transformer_layer_cls_to_wrap=LlamaDecoderLayer \
    --fsdp_offload_params=false \
    --main_training_function=main \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --same_network \
    --machine_rank=0 \
    --rdzv_backend=static \
    pefts/mft_accelerate.py --train_config configs/"xxx_train_config.json" \
        --distributed_type "fsdp" \
        > MFTCoder-training-"$TODAY".log 2>&1 &

