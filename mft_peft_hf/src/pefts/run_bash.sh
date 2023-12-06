
# Specify the number of GPU cards per node. Please set according to your specific situation!
N_GPU_PER_NODE=1
# Specify the number of nodes (machines) can be used. Please set according to your specific situation!
N_NODE=1

# pip install deepspeed==0.9.3 &&
#pip install transformers==4.32.0 &&


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
  mft_accelerate.py --train_config configs/qwen_train_config_1_8B.json
