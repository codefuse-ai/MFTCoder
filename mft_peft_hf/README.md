# MFTCoder: Fine-Tuning & Inference & Evaluation & Submission
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/codefuse-ai)
<a href="https://github.com/codefuse-ai/MFTCoder/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
</a>

# SFT Fine-Tuning / SFT微调

To fine-tune the Qwen-1.8B model, you need to start by preparing the training dataset(s) and then proceed with the fine-tuning training using the dataset(s). Subsequently, we will outline the requirements for the training data format, provide instructions on building the configuration file, and guide you on initiating the training process.

微调Qwen-1.8B模型，首先需要准备训练数据集以便随后将其用于微调。接下来，我们将说明训练数据格式要求、如何配置模型配置文件，以及如何启动微调训练过程。

## Training Data Format / 训练数据格式
The training data is in a uniformed JSONL format, in which each line of data has the following JSON format. The "chat_rounds" field is required, and other fields can be added or removed based on specific needs. In "chat_rounds", the element with "system" role is optional.

训练数据要求是统一的JSONL格式，即文件（扩展名是.jsonl）中每行是一项如下JSON格式的数据。格式中“chat_rounds”项是必须的，其他项可依需求增删。在"chat_rounds"项中，带有"system"角色的项是可选的。

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

一个可参考的示例数据集是[CodeFuse-CodeExercise-Python-27k-dataset](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary).

## Model Training / 模型训练

Currently, the "MFTCoder/mft_peft_hf" codebase supports [QLoRA](https://arxiv.org/pdf/2305.14314.pdf) instruction fine-tuning, and [LoRA](https://arxiv.org/pdf/2106.09685.pdf) instruction fine-tuning. According to the QLoRA paper, this method enables fine-tuning of a 33B model on a single V100 GPU while achieving performance close to that of full-parameter fine-tuning.
In theory, this project can be used to train any publicly available model in the HuggingFace Format.

当前，"MFTCoder/mft_peft_hf" 代码支持[QLoRA](https://arxiv.org/pdf/2305.14314.pdf)和[LoRA](https://arxiv.org/pdf/2106.09685.pdf)指令微调. 按照QLoRA论文这种方法可在一张V100显卡上微调具有33B参数的模型，并取得与全量微调想接近的结果。 理论上，该项目可适配于任何可公开获得的HuggingFace格式的模型。

You can find the implementations in the ```mft_peft_hf/src``` directory. The entry directory for fine-tuning training is ```mft_peft_hf/src/pefts```, and the entry file for training is ```mft_peft_hf/src/pefts/mft_accelerate.py```. 
Configurations are stored in the ```mft_peft_hf/src/pefts/configs``` directory for easy management and modification.

可在```mft_peft_hf/src```目录下找到具体实现，其中，微调训练入口是```mft_peft_hf/src/pefts```，具体入口文件是```mft_peft_hf/src/pefts/mft_accelerate.py```，而具体的配置文件统一保存在```mft_peft_hf/src/pefts/configs```目录下以便于管理。

More details you can find in this paper:

更多细节可在下面这篇文章中发现：


```
@article{liu2023mftcoder,
  title={MFTCoder: Boosting Code LLMs with Multitask Fine-Tuning},
  author={Liu, Bingchang and Chen, Chaoyu and Liao, Cong and Gong, Zi and Wang, Huan and Lei, Zhichao and Liang, Ming and Chen, Dajun and Shen, Min and Zhou, Hailian and others},
  journal={arXiv preprint arXiv:2311.02303},
  year={2023}
}
```

### Configuration / 配置
An example configuration file for fine-tuning Qwen-1.8B model is [src/pefts/configs/qwen_train_config_1_8B.json](src/pefts/configs/qwen_train_config_1_8B.json).

一个具体的用于微调Qwen-1.8B模型的配置文件是 [src/pefts/configs/qwen_train_config_1_8B.json](src/pefts/configs/qwen_train_config_1_8B.json)。

The parameters in ```configs/*_train_config``` configuration files are explained as follows. **You can modify these parameters according to your needs**.

位于```configs/*_train_config```下的配置文件中的各项参数解释如下。可按自己的需求自行调整配置参数值。

- **load_raw_dataset**: Must be true at present. Only JSONL format is supported. *必须是True，并且只支持JSONL格式。*

- **data_paths**: Input data paths in a String of list format, e.g., "[path1,path2,path3]". Each path represents a task directory and each task directory contains one or more JSONL data files. You can provide one or more task directory. *值必须是字符串格式的路径列表，例如["路径1","路径2","路径3"]。每个路径代表一个下游任务，且其中可包含一个或多个JSONL文件。路径数量可以是一个或多个。*

- **output_dir**: Training output directory to store checkpoints, Lora adapter, etc. *用于存储训练产生的检查点的输出目录路径*

- **tb_dir**: TensorBoard directory to store logs, metrics, etc. *用于存储训练产生的TensorBoard日子数据的目录。*

- **model_type**: Type of the model to train, e.g., "llama | starcoder | chatglm2 | qwen | gpt_neox". To fine-tune Qwen-1.8B, it must be "qwen". *要微调的底座模型类型，可选值如"llama | starcoder | chatglm2 | qwen | gpt_neox"。如果要微调Qwen-1.8B模型，则需设为"qwen"。*

- **peft_type**: either "lora" or "qlora". You can make a choice as your needs. *PEFT类型，可选值包括"lora"和"qlora"，可自行选择。*

- **lora_rank**: Rank value for Lora. *LoRA/QloRA模式中的Rank值。*

- **lora_alpha**: Alpha value for Lora. *LoRA/QLoRA模型中的Alpha值*

- **lora_dropout**: Dropout rate for Lora. *LoRA/QLoRA中的dropout率*

- **quantization**: Whether to use quantization."4bit" or "8bit", or null. For QLoRA, it is recommended to use 4-bit quantization. *可设置为"4bit"、"8bit"或者null，null意味着不量化。如果是使用QLoRA模型，推荐使用"4bit"量化。*

- **pretrained_model_path**: Local/Shared disk path or model name on HuggingFace for the pre-trained model. *In Qwen AI competition, it should be the local path of your downloaded Qwen-1.8B model.* *要微调的底座模型的本地或远程路径。在Qwen比赛中，应该填写Qwen-1.8B模型的路径。*

- **weighted_loss_mode**: Loss weighting method for multitask training. "case3" is recommended at present. *多任务训练模式中的loss计算方法。当前推荐使用"case3"类型。*

- **padding_mode**: The way tokenized data is set. "padding" means padding for each sample to seq_length, "pack" means putting samples into seq_length as many as possible. If you have large amounts of training samples, you may use "pack" to achieve faster training speed. *Tokenization方式，可选值有“padding”和"pack"。"padding"表示batch中每个项目会被对齐到seq-length长度；"pack"表示将尽可能多的样本填充到一个seq-length大小的新样本中。如果你有很大量的训练数据，可以使用"pack"方式获得更快的训练速度。*

- **num_train_epochs**: Number of training epochs. *计划训练的Epoch数量*

- **per_device_train_batch_size**: Batch size per GPU for training. *训练时单卡上的batch大小*

- **per_device_eval_batch_size**: Batch size per GPU for evaluation. *验证时单卡上的batch大小*

- **gradient_accumulation_steps**: Number of gradient accumulation steps. Global batch size is calculated as num_gpus * per_device_train_batch_size * gradient_accumulation_steps. *梯度累积步数。全局batch大小等于num_gpus、per_device_train_batch_size和gradient_accumulation_steps的乘积。*

- **learning_rate**: Initial Learning rate. For full-parameter fine-tuning, it is recommended to use a smaller value such as 1e-5 or 5e-6. For QLoRA, a larger learning rate is generally used, such as 1e-4 or 2e-4. *初始学习率。对于全量微调，推荐使用小一些的值，例如1e-5或1e-6。如果是QLoRA微调，推荐使用更大一些的学习率，例如1e-4或2e-4。*

- **min_lr**: Minimum learning rate. Usually set to one-tenth of the learning rate. *最小学习率，通常设置比初始学习率小一个数量级。*

- **seq_length**: Maximum input sequence length during training. *训练过程最大输入序列长度。*

- **log_interval**: Log training loss every ```log_interval``` steps. *设置每隔多少步打印一次日志*

- **checkpointing_steps**: Save a checkpoint every ```checkpointing_steps``` steps. *设置每隔多少步保存一次检查点*

- **evaluation_steps**: Evaluate on the validation set every ```evaluation_steps``` steps. *设置每隔多少步进行一次验证*

- **early_stopping**: Enable early stopping or not. *设置是否启用早停策略*

- **early_stopping_stall_num**: Number of evaluation points without improvement which triggers early stopping. *开启早停策略时，设置连续多少个验证点loss不下降后停止训练。*

- **lr_scheduler_type**: Type of learning rate scheduler. "cosine" is a good choice already.  *学习率调整策略，目前"cosine"是不错的选择。*

- **num_warmup_steps**: Number of warm-up steps to gradually increase the learning rate. *设置逐步增加到初始学习率所需的步数。*

- **seed**: Random seed for reproducibility. *设置随机化种子，用于复现使用。*




### Run

To run LoRA/QLoRA fine-tuing, you can execute the [src/pefts/run_bash.sh](src/pefts/run_bash.sh) script:

要执行LoRA或QLoRA微调，可执行[src/pefts/run_bash.sh](src/pefts/run_bash.sh)脚本：

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

需按自己的情况调整脚本中的部分参数，例如，配置文件路径（即```--train_config```参数的值）、机器节点数量```N_NODE```、每个节点GPU卡数```N_GPU_PER_NODE```等。也可执行下面的命令启动训练：


```bash
cd mft_peft_hf/src/pefts

accelerate launch --config_file accelerate_ds_config.yaml mft_accelerate.py --train_config configs/starcoder_train_config.json
```


# Inference Generation

We build our inference framework based on [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). We recommend that you go to [bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness) to learn some necessary information. We made some modifications to adapt to Qwen AI (Code) competition, including inference format, evaluation datasets localization et.al.

所用推理框架是基于[bigcode-project/bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)项目构建的，因此建议进入该项目主页了解必要的信息。在原项目基础上，我们进行了必要的调整以适配Qwen赛事，例如推理格式、评测集本地化等。

In the Qwen AI (Code) competition, we have chosen **HumanEvalPack** and **MBPP** as our evaluation tasks. Within the HumanEvalPack, we have selected two subtasks: "humanevalsynthesize" and "humanevalfixtests" each comprising 6 languages. In total, there are 2,468 questions to be evaluated. The complete list of evaluation tasks is as follows:

本次Qwen AI（代码）比赛，我们选择使用**HumanEvalPack** and **MBPP**作为评测任务。其中，HumanEvalPack中的两个子类"humanevalsynthesize" and "humanevalfixtests"被具体选择，这两个子类每个都由6种语言的补全任务组成。总体上，共有2468道评测题目。完整的评测任务列表如下所示：


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


## Inference/Tokenization Format / 推理格式

We take Qwen's ChatML format as our tokenization format:

我们使用Qwen所用的ChatML格式作为我们微调训练中的tokenization格式：

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

EOD token是```<|im_end|>```。训练中，只有"assistant"角色的内容和其后紧跟的"<|im_end|>"token被纳入loss计算。可在[src/data/tokenization/preprocess_data.py](src/data/tokenization/preprocess_data.py)发现更多关于数据处理的细节。当你试图为评测任务生成推理结果时，请按照如下的推理格式：

```
<|im_start|>user
{Prompt of one evaluatation question}<|im_end|>
<|im_start|>assistant
```

Also, you can add a system prompt if you need as follows:

另外，你也可按自己的需求增加system提示：

```
<|im_start|>system
{This is your system prompt}<|im_end|>
<|im_start|>user
{Prompt of one evaluatation question}<|im_end|>
<|im_start|>assistant
```

## Inference Script / 推理脚本

We have provided a shell script for inferring the evaluating tasks, i.e. [src/evaluation/launch_generate_codeqwen.sh](src/evaluation/launch_generate_codeqwen.sh):


我们提供了一个shell脚本用于推理评测任务，即[src/evaluation/launch_generate_codeqwen.sh](src/evaluation/launch_generate_codeqwen.sh):

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

欲运行该脚本，必须提供本地模型路径（即```model```的值）、模型名字（即```model_name```的值）、生成结果存放目录（即```generation_base_dir```的值）。

You have the flexibility to modify other parameters according to your specific requirements. For instance, if you wish to customize the system prompt, you can change the value of the current ```system``` variable. Similarly, if you intend to infer humanevalpack-tasks in the fine-tuned format, which necessitates your model's ability to complete tasks in the fine-tuned format, you will need to adjust the ```prefix``` and ```suffix``` variables when executing humanevalpack tasks.

你可以按需灵活调整其他参数。例如，如果要使用定制的system提示，则可调整当前```system```变量的值；如果想用微调格式推理HumanEvalPack各任务的结果，则可调整演算这些任务时```prefix```和```suffix```的值，当前前提是模型能支持以这种格式推理补全任务。

Besides, current script is not task-parallel, you can change it with Slurm as your needs.

除此之外，当前脚本不是各子任务并行的，可按需自行调整为并行版本。

Upon running the script for inference, you will obtain a folder named **generations_{your-model-name}**. Within this folder, you will find *13* JSON files named according to the schema **generations_{task-name}_{your-model-name}.json**. Please remember to replace "{your-model-name}" with your specific model name, such as "generations_qwen_1_8B_codefuse".

推理完成后，你将得到一个文件夹，文件夹命名为**generations_{your-model-name}**，在其中有*13*个JSON文件，每个文件均按**generations_{task-name}_{your-model-name}.json**模式命名。这里，{your-model-name}需替换为你具体的模型名字，例如"generations_qwen_1_8B_codefuse"，{task-name}需替换为具体的评测任务名字，例如“humanevalsyntheize-python”。


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

**绝不要随意更改这些名字，以免成绩被判定为0分。此外，我们要求结果生成必须使用贪心解码模式，即```doSample=Fase, num_beams=1, num_return_sequences=1```**

# Evaluation (Optional) / 测试（可选）

Due to security concerns, the generated code is executed within a separate container. To facilitate this process and obtain the PASS@1 evaluation scores, we have provided a Docker image and a shell script specifically for running the generated code.
**In the Qwen AI (Code) competition, we adopt "Greedy Decoding Mode & PASS@1" as the metric for evaluation.**

出于安全考虑，推理生成的代码将在一个独立的容器中运行。为方便自行评测得出PASS@1分值，我们提供了一个现成可用的Docker镜像和相应的shell处理脚本。
**在Qwen AI比赛中，我们要求使用贪心解码模式并将PASS@1作为评测指标。**


## Docker image / Docker镜像

You can pull our built image with the following commands:

你可通过以下命令拉取镜像：

```bash
docker pull registry.cn-hangzhou.aliyuncs.com/bingchang/code-qwen-competition:latest
docker tag registry.cn-hangzhou.aliyuncs.com/bingchang/code-qwen-competition:latest code-qwen-competition:latest 
```

Also, you can build by yourself with our provide [Dockerfile](src/evaluation/Dockerfile).

你可以使用我们提供的[Dockerfile](src/evaluation/Dockerfile)自己构建镜像。

## Running Scripts / 运行脚本

We provide a shell script to perform evaluation with our provided image in [src/evaluation//launch_evaluate.sh](src/evaluation//launch_evaluate.sh):

我们提供了一个shell脚本[src/evaluation//launch_evaluate.sh](src/evaluation//launch_evaluate.sh)用于完成任务pass@1评测：


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

为执行该脚本，你必须提供```{your-model-name}```参数的值并确保保存生成结果的文件夹（命名模式为*generations_{your-model-name}*）与该脚本放置在同一个目录下，如果不同，则需对应的调整脚本中的```${pwd}/generations_path/```部分。

After the evaluation process, you will find a metric folder named **metrices_{your-model-name}**, which contains 13 JSON files. Each JSON file corresponds to an evaluation task and holds the evaluation score for that particular task.

评测完成后，你将获得一个命名为**metrices_{your-model-name}**的文件夹，它里面包含13个JSON文件，每个JSON文件存储一个评测任务的得分。

# Submission / 提交

Once you have obtained the generated results of your model, which is the folder named "generations_{your-model-name}", you should compress it into a zip file. After compressing the folder, you can proceed to upload the zip file to TianChi platform of Aliyun [https://tianchi.aliyun.com/competition/entrance/532169](https://tianchi.aliyun.com/competition/entrance/532169).


当你获得模型推理结果后（即获得一个名为"generations_{your-model-name}"）的文件夹，你需将该文件夹压缩为一个ZIP文件。随后，你需要将该ZIP文件上传到阿里云天池平台[https://tianchi.aliyun.com/competition/entrance/532169](https://tianchi.aliyun.com/competition/entrance/532169)完成打分。


Your submission must satisfy these requirements:

你的提交必须满足如下要求：

```text
1. The generation result folder must be compressed into a zip file
2. The decompressed result of the zip file must be a folder named with "generations_{your-model-name}"
3. Verify that the folder contains exactly 13 JSON files, each corresponding to one evaluation task.
4. Name each JSON file using the following schema: "generations_{task-name}_{your-model-name}.json". Replace "{task-name}" with the name of the evaluation task and "{your-model-name}" with the name of your model.
```

```text
1. 生成结果文件夹需压缩进一个ZIP文件
2. ZIP文件需能解压出一个命名模式为"generations_{your-model-name}"的文件夹
3. 解压出的文件夹中需刚好包含13个JSON文件，每个JSON文件对应于一个评测任务
3. 每个JSON文件按"generations_{task-name}_{your-model-name}.json"模式命名，其中，"{task-name}"是评测任务的名字，"{your-model-name}"是推理时设置的模型名字
```

Once you have submitted your generated results, the TianChi platform will evaluate the PASS@1 scores. The average score across all 13 tasks will then be calculated and considered as your overall score for the submission.

当你提交生成结果后，天池平台会评测得出PASS@1分值，并将13个任务的平均PASS@1值作为本次提交的得分。

下面是我们未精细微调的一个版本分数，作为本次大赛的 baseline 分数:
```
score:0.0933
humanevalsynthesize-python:0.2195
humanevalfixtests-cpp:0.0183
mbpp:0.2440
humanevalsynthesize-cpp:0.1280
humanevalfixtests-js:0.0122
humanevalfixtests-go:0.0122
humanevalfixtests-python:0.0305
humanevalfixtests-rust:0.0000
humanevalsynthesize-rust:0.0366
humanevalsynthesize-js:0.1341
humanevalsynthesize-go:0.1524
humanevalfixtests-java:0.0000
humanevalsynthesize-java:0.2256
```
