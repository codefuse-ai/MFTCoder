pip install transformers==4.32.0

N_NODE=1

N_GPU_PER_NODE=1


tasks=(humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-go humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp)


model=/path/to/local/model/checkpoint
model_name=Qwen-7b-chat
generation_base_dir=/path/to/hold/generated/results


if [ ! -d $generation_base_dir ]; then
    mkdir $generation_base_dir
fi


batch_size=1
n_samples=1
eos_token="<|im_end|>"


# SFT Format
user=user
assistant=assistant
system=system
end_tag="<|im_end|>"
start_tag="<|im_start|>"

# If you need to set system prompt, set it here, otherwise you can set it as empty string
system="$start_tag"$system$'\n'"$end_tag"$'\n'

for task in "${tasks[@]}"; do

    if [ "$task" == "mbpp" ]; then
        prefix="$system""$start_tag"${user}$'\n'
        suffix="$end_tag"$'\n'"$start_tag"${assistant}
    else
        prefix=""
        suffix=""
    fi

    generations_path=$generation_base_dir/generations_$model_name/generations_$task\_$model_name.json
    if [ ! -d $generation_base_dir/generations_$model_name ]; then
        mkdir $generation_base_dir/generations_$model_name
    fi

    echo "start to launch ...."
    accelerate launch \
            --num_machines $N_NODE \
            --num_processes $(($N_NODE*$N_GPU_PER_NODE)) \
            main.py \
                --model $model \
                --task $task \
                --prompt instruct \
                --n_samples $n_samples \
                --batch_size $batch_size \
                --max_length_generation 2000 \
                --do_sample False \
                --temperature 0.2 \
                --precision bf16 \
                --eos "$eos_token" \
                --seed 999999999 \
                --add_special_tokens True \
                --trust_remote_code \
                --generation_only \
                --save_generations_path $generations_path \
                --prefix "$prefix" \
                --suffix "$suffix"
    
    echo "Task $task done"
done