#######################################################################################################################
MODEL_PATH="path/to/hf/model"
DATA_FILE="path/to/input/file/jsonl"
BATCH_SIZE=2

N_NODE=1
N_GPU_PER_NODE=2
#######################################################################################################################
# you could add some specific commnads here if you need.
pip install transformers==4.44.2
pip install -U accelerate


export OMP_NUM_THREADS=16


# One Node inference
# accelerate launch \
#     --num_machines 1 \
#     --num_processes 2 \
#     --machine_rank 0 \
#     --dynamo_backend 'no' \
#     --same_network \
#     --rdzv_backend 'static' \
#     infer_accelerate.py \
#         --model_path $MODEL_PATH \
#         --data_file $DATA_FILE \
#         --batch_size $BATCH_SIZE \
#         --output_dir "." \
#         > "path/to/local/log" 2>&1 &

# You need to export $MACHINE_RANK, $MASTER_ADDR, $MASTER_PORT automatically for each Node.

echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"


# launch by multi-node, multi-gpu
accelerate launch \
    --num_machines $N_NODE \
    --num_processes $(($N_NODE*$N_GPU_PER_NODE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --dynamo_backend 'no' \
    --same_network \
    --rdzv_backend 'static' \
    infer_accelerate.py \
        --model_path $MODEL_PATH \
        --data_file $DATA_FILE \
        --batch_size $BATCH_SIZE \
        --output_dir "path/to/output/directory"