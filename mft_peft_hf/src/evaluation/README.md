<h1 align="center">Code Generation LM Evaluation Harness</h1>

**This is forked from [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). Appreciate their outstanding work. You can find more information in the original project.**



# Inference Generation

We build our inference framework based on [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). We recommend that you go to [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) to learn some necessary information. We made some modifications to adapt to Qwen AI (Code) competition, including inference format, evaluation datasets localization et.al.

In the Qwen AI (Code) competition, we have chosen **HumanEvalPack** and **MBPP** as our evaluation tasks. Within the HumanEvalPack, we have selected two subtasks: "humanevalsynthesize" and "humanevalfixtests," each comprising 6 languages. In total, there are 2,468 questions to be evaluated. The complete list of evaluation tasks is as follows:

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

The eod token is ```<|im_end|>```. During training, only the content of the "assistant" and the token "<|im_end|>" following it are taken into consideration for loss computation. You can find further details in [src/data/tokenization/preprocess_data.py](src/data/tokenization/preprocess_data.py). Based on this information, when attempting to generate inference results for evaluated tasks, please adhere to the reference format as follows:

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

We have provided a shell script for inferring the evaluating tasks, i.e. [src/evaluation/launch_generate_codeqwen.sh](src/evaluation/launch_generate_codeqwen.sh):

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

You have the flexibility to modify other parameters according to your specific requirements. For instance, if you wish to customize the system prompt, you can change the value of the current ```system``` variable. Similarly, if you intend to infer humanevalpack-tasks in the fine-tuned format, which necessitates your model's ability to complete tasks in the fine-tuned format, you will need to adjust the ```prefix``` and ```suffix``` variables when executing humanevalpack tasks.

Besides, current script is not task-parallel, you can change it with Slurm as your needs.

Upon running the script for inference, you will obtain a folder named **generations_{your-model-name}**. Within this folder, you will find *13* JSON files named according to the schema **generations_{task-name}_{your-model-name}.json**. Please remember to replace "{your-model-name}" with your specific model name, such as "generations_qwen_1_8B_codefuse".


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

# Evaluation

Due to security concerns, the generated code is executed within a separate container. To facilitate this process and obtain the PASS@1 evaluation scores, we have provided a Docker image and a shell script specifically for running the generated code.
**In the Qwen AI (Code) competition, we adopt "Greedy Decoding Mode & PASS@1" as the metric for evaluation.**


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

In order to execute this script, you must provide the ```{your-model-name}``` parameter and ensure that the generations folder (named *generations_{your-model-name}*) is placed in the same directory as this script. If the paths differ, you will need to adjust the ```${pwd}/generations_path/``` section accordingly.
*Please note that this script does not support task parallelism, but you can modify it according to your specific requirements.*

After the evaluation process, you will find a metric folder named **metrices_{your-model-name}**, which contains 13 JSON files. Each JSON file corresponds to an evaluation task and holds the evaluation score for that particular task.