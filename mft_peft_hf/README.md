# MFTCoder: Fine-Tuning & Inference & Evaluation & Submission
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/codefuse-ai)
<a href="https://github.com/codefuse-ai/MFTCoder/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
</a>

# SFT Fine-Tuning

## Training Data Format
The training data is in a uniformed JSONL format, in which each line of data has the following JSON format. The "chat_rounds" field is required, and other fields can be added or removed based on specific needs. In "chat_rounds", the element with "system" role is optional.

```json
{
    "id":0,
    "data_name":"code-helper",
    "chat_rounds":[
        {
            "role": "system",
            "content": "You are a expert in coding and help answer code questions",
            "chat_round_id": 0
        },
        {
            "role": "human",
            "content": "Write a python function of quick sort", 
            "chat_round_id": 1
        },
        {
            "role": "bot",
            "content": "Below is the function of quick sort: ...", 
            "chat_round_id": 1
        },
        {
            "role": "human",
            "content": "Explain the code", 
            "chat_round_id": 2
        },
        {
            "role": "bot",
            "content": "OK, this code ...", 
            "chat_round_id": 2
        }
    ]
}
```

An example is [CodeFuse-CodeExercise-Python-27k-dataset](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary).

## Model Training

Currently, the "MFTCoder/mft_peft_hf" codebase supports [QLoRA](https://arxiv.org/pdf/2305.14314.pdf) instruction fine-tuning, and [LoRA](https://arxiv.org/pdf/2106.09685.pdf) instruction fine-tuning. According to the QLoRA paper, this method enables fine-tuning of a 33B model on a single V100 GPU while achieving performance close to that of full-parameter fine-tuning.
In theory, this project can be used to train any publicly available model in the HuggingFace Format.

You can find the implementations in the ```mft_peft_hf/src``` directory. The entry directory for fine-tuning training is ```mft_peft_hf/src/pefts```, and the entry file for training is ```mft_peft_hf/src/pefts/mft_accelerate.py```. 
Configurations are stored in the ```mft_peft_hf/src/pefts/configs``` directory for easy management and modification.

More details you can find in this paper:
```
@article{liu2023mftcoder,
  title={MFTCoder: Boosting Code LLMs with Multitask Fine-Tuning},
  author={Liu, Bingchang and Chen, Chaoyu and Liao, Cong and Gong, Zi and Wang, Huan and Lei, Zhichao and Liang, Ming and Chen, Dajun and Shen, Min and Zhou, Hailian and others},
  journal={arXiv preprint arXiv:2311.02303},
  year={2023}
}
```

### Configuration
An example configuration file for fine-tuning Qwen-1.8B model is [src/pefts/configs/qwen_train_config_1_8B.json](src/pefts/configs/qwen_train_config_1_8B.json).

The parameters in ```configs/*_train_config``` configuration files are explained as follows. **You can modify these parameters according to your needs**.

- **load_raw_dataset**: Must be true at present. Only JSONL format is supported.

- **data_paths**: Input data paths in a String of list format, e.g., "[path1,path2,path3]". Each path represents a task directory and each task directory contains one or more JSONL data files. You can provide one or more task directory.

- **output_dir**: Training output directory to store checkpoints, Lora adapter, etc.

- **tb_dir**: TensorBoard directory to store logs, metrics, etc.

- **model_type**: Type of the model to train, e.g., "llama | starcoder | chatglm2 | qwen | gpt_neox". To fine-tune Qwen-1.8B, it must be "qwen".

- **peft_type**: either "lora" or "qlora". You can make a choice as your needs.

- **lora_rank**: Rank value for Lora.

- **lora_alpha**: Alpha value for Lora.

- **lora_dropout**: Dropout rate for Lora.

- **quantization**: Whether to use quantization."4bit" or "8bit", or null. For QLoRA, it is recommended to use 4-bit quantization.

- **pretrained_model_path**: Local/Shared disk path or model name on HuggingFace for the pre-trained model. *In Qwen AI competition, it should be the local path of your downloaded Qwen-1.8B model.*

- **weighted_loss_mode**: Loss weighting method for multitask training. "case3" is recommended at present.

- **padding_mode**: The way tokenized data is set. "padding" means padding for each sample to seq_length, "pack" means putting samples into seq_length as many as possible. If you have large amounts of training samples, you may use "pack" to achieve faster training speed.

- **num_train_epochs**: Number of training epochs. 

- **per_device_train_batch_size**: Batch size per GPU for training.

- **per_device_eval_batch_size**: Batch size per GPU for evaluation.

- **gradient_accumulation_steps**: Number of gradient accumulation steps. Global batch size is calculated as num_gpus * per_device_train_batch_size * gradient_accumulation_steps.

- **learning_rate**: Initial Learning rate. For full-parameter fine-tuning, it is recommended to use a smaller value such as 1e-5 or 5e-6. For QLoRA, a larger learning rate is generally used, such as 1e-4 or 2e-4.

- **min_lr**: Minimum learning rate. Usually set to one-tenth of the learning rate.

- **seq_length**: Maximum input sequence length during training. 

- **log_interval**: Log training loss every ```log_interval``` steps.

- **checkpointing_steps**: Save a checkpoint every ```checkpointing_steps``` steps.

- **evaluation_steps**: Evaluate on the validation set every ```evaluation_steps``` steps.

- **early_stopping**: Enable early stopping or not.

- **early_stopping_stall_num**: Number of evaluation points without improvement which triggers early stopping.

- **lr_scheduler_type**: Type of learning rate scheduler. "cosine" is a good choice already. 

- **num_warmup_steps**: Number of warm-up steps to gradually increase the learning rate.

- **seed**: Random seed for reproducibility.




### Run

To run LoRA/QLoRA fine-tuing, you can execute the [src/pefts/run_bash.sh](src/pefts/run_bash.sh) script:

```shell
N_GPU_PER_NODE=8
N_NODE=1

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
```

You need to adjust some parametes as your needs, e.g. the configuration path ```--train_config```, the number of nodes ```N_NODE```, gpus of each node ```N_GPU_PER_NODE```. You can also execute the following command to run:
```bash
cd mft_peft_hf/src/pefts

accelerate launch --config_file accelerate_ds_config.yaml mft_accelerate.py --train_config configs/starcoder_train_config.json
```


# Inference Generation

We build our inference framework based on [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). We recommend that you go to [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) to learn some necessary information. We made some modifications to adapt to Qwen AI (Code) competition, including inference format, evaluation datasets localization et.al.

In Qwen AI (Code) competition, we select **HumanEvalPack** and **MBPP** as our evaluation tasks. Two subtasks of HumanEvalPack are selected, including "humanevalsynthesize" and "humanevalfixtests"  and each of them contains 6 languages. In summary, the evaluation tasks are as follows and there are 2,468 questions to be evluated in total. The list of evaluation tasks:

```
humanevalsynthesize-python
humanevalsynthesize-java
humanevalsynthesize-js
humanevalsynthesize-cpp
humanevalsynthesize-go
humanevalsynthesize-rust
humanevalfixtests-python
humanevalfixtests-java
humanevalfixtests-js
humanevalfixtests-cpp
humanevalfixtests-go
humanevalfixtests-rust
mbpp
```


## Inference/Tokenization Format

We take Qwen's ChatML format as our tokenization format:

```
<|im_start|>system
{Here is your system prompt}<|im_end|>
<|im_start|>user
{Here is your 1st-round user prompt}<|im_end|>
<|im_start|>assistant
{Here is the model's inference result of 1st-round}<|im_end|>
<|im_start|>user
{Here is your 2nd-round user prompt}<|im_end|>
<|im_start|>assistant
{Here is the model's inference result of 2nd-round}<|im_end|>
...
```

The eod token is ```<|im_end|>```. During training, only the content of "assistant" and the "<|im_end|>" token following it are token into loss computation. The details you can find in [src/data/tokenization/preprocess_data.py](src/data/tokenization/preprocess_data.py). Based on this, when you try to generate inference results of evaluated tasks, the reference format is:

```
<|im_start|>user
{Prompt of one evaluatation question}<|im_end|>
<|im_start|>assistant
```

Also, you can add a system prompt if you need as follows:

```
<|im_start|>system
{This is your system prompt}<|im_end|>
<|im_start|>user
{Prompt of one evaluatation question}<|im_end|>
<|im_start|>assistant
```

## Inference Script

We provide a shell script to infer the evaluating tasks, i.e. [src/evaluation/launch_generate_codeqwen.sh](src/evaluation/launch_generate_codeqwen.sh):

```shell
N_NODE=1
N_GPU_PER_NODE=1


tasks=(humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-go humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp)


model=/path/to/local/model/checkpoint
model_name={your-model-name}
generation_base_dir=/path/to/hold/generated/results


if [ ! -d $generation_base_dir ]; then
    mkdir $generation_base_dir
fi


batch_size=1
n_samples=1
# For qwen base model, eos is '<|endoftext|>'; for fine-tuned qwen model, eos is '<|im_end|>'
eos_token="<|im_end|>"


# SFT Format
user=user
assistant=assistant
system=system
end_tag="<|im_end|>"
start_tag="<|im_start|>"

# If you need to set system prompt, set it here, otherwise you can set it as empty string. Decide whether to add system prompt by yourself.
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
```

To run this script, you must provide the values of local model path ```model```, model name ```model_name```, generated results' base directory ```generation_base_dir```.

You can modify other parameters as your needs. For example, if you want to set your own system prompt, you have to change current ```system``` variable's value; if you want to infer humanevalpack-tasks in the fine-tuned format requiring your model must be able to do completion tasks in fine-tuned format, you need to adjust ```prefix``` and ```suffix``` variables when performing humanevalpack tasks.

Besides, current script is not task-parallel, you can change it with Slurm as your needs.

After inference with this script, you will get a folder which is named with **generations_{your-model-name}**. In this folder, there're **13** json files which are named in the schema **generations_{task-name}_{your-model-name}.json**. Remeber to replace "**{your-model-name}**" with your own model name, e.g. "*generations_qwen_1_8B_codefuse*".

```
generations_{your-model-name}:
\
  - generations_humanevalsynthesize-python_{your-model-name}.json
  - generations_humanevalsynthesize-java_{your-model-name}.json
  - generations_humanevalsynthesize-js_{your-model-name}.json
  - generations_humanevalsynthesize-cpp_{your-model-name}.json
  - generations_humanevalsynthesize-go_{your-model-name}.json
  - generations_humanevalsynthesize-rust_{your-model-name}.json
  - generations_humanevalfixtests-python_{your-model-name}.json
  - generations_humanevalfixtests-java_{your-model-name}.json
  - generations_humanevalfixtests-js_{your-model-name}.json
  - generations_humanevalfixtests-cpp_{your-model-name}.json
  - generations_humanevalfixtests-go_{your-model-name}.json
  - generations_humanevalfixtests-rust_{your-model-name}.json
  - generations_mbpp_{your-model-name}.json
```
**You must not change these names, otherwise your submission will be 0 score. Besides, we require you must generate reference results in  greedy decoding mode, i.e. 
```doSample=Fase, num_beams=1, num_return_sequences=1```**

# Evaluation (Optional)

Because the generated code cannot guarantee security, we run the generated code in a separate container. We have provided a Docker image and a shell script to run the generated code to get pass@1 evaluation scores.
**In Qwen AI (Code) competition, we take "Greedy Decoding Mode & PASS@1" as mearsure metric.** 


## Docker image

You can pull our built image with the following commands:

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/bingchang/code-qwen-competition:latest
docker tag registry.cn-hangzhou.aliyuncs.com/bingchang/code-qwen-competition:latest code-qwen-competition:latest 
```

Also, you can build by yourself with our provide [Dockerfile](src/evaluation/Dockerfile).

## Running Scripts

We provide a shell script to perform evaluation with our provided image in [src/evaluation//launch_evaluate.sh](src/evaluation//launch_evaluate.sh):

```shell

# please replace this with your own model name which is taken during generation with launch_generate_codeqwen.sh
model={your-model-name}
org=test


tasks=(humanevalsynthesize-python humanevalsynthesize-java humanevalsynthesize-js humanevalsynthesize-cpp humanevalsynthesize-go humanevalsynthesize-rust humanevalfixtests-python humanevalfixtests-java humanevalfixtests-js humanevalfixtests-cpp humanevalfixtests-go humanevalfixtests-rust mbpp)

# if you provide absolute paths remove the $(pwd) from the command below
generations_path=generations_$model
metrics_path=metrics_$model

if [ -d $metrics_path ]; then
    echo "Folder exists. Deleting folder: $metrics_path"
    rm -rf $metrics_path
fi
mkdir $metrics_path

batch_size=1
n_samples=1
eos_token="\"<|im_end|>\""


for task in "${tasks[@]}"; do
    echo "Task: $task"

    gen_suffix=generations_$task\_$model.json
    metric_suffix=metrics_$task\_$model.json
    echo "Evaluation of $model on $task benchmark, data in $generations_path/$gen_suffix"

    sudo docker run -v $(pwd)/$generations_path/$gen_suffix:/app/$gen_suffix:ro  -v $(pwd)/$metrics_path:/app/$metrics_path -it code-qwen-competition bash -c "python3 main.py \
        --model $org/$model \
        --tasks $task \
        --load_generations_path /app/$gen_suffix \
        --metric_output_path /app/$metrics_path/$metric_suffix \
        --allow_code_execution  \
        --trust_remote_code \
        --use_auth_token \
        --temperature 0.2 \
        --max_length_generation 1024 \
        --do_sample False \
        --precision bf16 \
        --eos "$eos_token" \
        --seed 999999999 \
        --batch_size $batch_size \
        --n_samples $n_samples | tee -a logs_$model.txt"
    echo "Task $task done, metric saved at $metrics_path/$metric_suffix"
done
```

To run this script, you must provide ```{your-model-name}``` and place the generations folder (named with **generations_{your-model-name}**) in the sample folder with this script. If not the same, you need to adjust the part ```${pwd}/generations_path/``` as your path. 

This script is not task-parallel, you can change it as your needs.

After evaluation, you will get a metric folder named with **metrices_{your-model-name}** and there are 13 metric json files in it. Each json file holds the evaluation score of a task.

# Submission

When you get the model's genererated results, i.e. the folder named with "generations_{your-model-name}", you need to compress it into a zip file and upload the zip file to TianChi platform of Aliyun [https://tianchi.aliyun.com/competition/entrance/532169](https://tianchi.aliyun.com/competition/entrance/532169).

Your submission must satisfy these requirements:

1. The generation result folder must be compressed into a zip file
2. The compressed result of the zip file must be a folder named with "generations_{your-model-name}"
3. There're must **13** json files corresponding to 13 tasks in the folder
4. Each json file must be named with this schema "generations_{task-name}_{your-model-name}.json". ({task-name} needs to be replaced with an evaluation task name and {your-model-name} needs to be replaced with your model name).

After submitting you generated results, pass@1 score of each task will be evaluated in TianChi platform and the average score of 13 tasks will be taken as your score of this submission. 