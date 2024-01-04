# Launch script on single node
N_GPU_PER_NODE=8

# envs used inside training
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=False
MYHOME=path/to/your/log
TODAY=$(date +%Y-%m%d-%H%M)

# accelerate launch --config_file accelerate_ds_config.yaml \
accelerate launch \
    --num_machines 1 \
    --num_processes $(($N_GPU_PER_NODE)) \
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
    --machine_rank 0 \
    --rdzv_backend 'static' \
    pefts/mft_accelerate.py --train_config configs/"xxx_train_config.json" \
        > $MYHOME/logs/MFTCoder-training-$TODAY.log 2>&1 &
