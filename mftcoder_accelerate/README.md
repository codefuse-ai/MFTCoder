# MFTCoder-accelerate: Training Framework with Accelerate and DeepSpeed/FSDP
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/codefuse-ai)
<a href="https://github.com/codefuse-ai/MFTCoder/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
</a>

[[中文]](README_cn.md) [**English**]

## 1. Updates

🔥 MFTCoder-accelerate supports Full-parameters/LoRA using accelerate + FSDP Framework;

🔥 MFTCoder-accelerate supports MFT/SFT on more new mainstream open-source base models: mistral, mixtral-8x7b(Mixture of Experts), deepseek, chatglm3;

🔥 MFTCoder-accelerate supports Self-Paced Loss for Convergence Balance;

🔥 MFTCoder-accelerate supports Full-parameters/QLoRA/LoRA using accelerate + DeepSpeed Framework;

🔥 MFTCoder-accelerate supports Multiple Task Finetuning, which is able to balance diffenrent tasks in data level.

🔥 MFTCoder-accelerate supports finetuning multiple mainstream open-source base models: codellama, llama2, llama, starcoder, codegeex2, chatglm2, qwen.

## 2. Data Format
### 2.1 Training Data Format
The training data is in a uniformed JSONL format, in which each line of data has the following JSON format. The "chat_rounds" field is required, and other fields can be added or removed based on specific needs. 

For the keys of roles in "chat_rounds", you could use "system/human/bot" tuple or "system/user/assistant" tuple.

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

### 2.2 Default Inference Data Format
The default inference data contains strings concatenated by conversation data(system, human and bot contents) in the training data format. 
It is used as the data "seen"(before tokenization) by the model in training process.
It is used as input during the inference process as well.
Here is an example format of the concatenated string:

```python
"""
<s>system
System instruction
<s>human
User 1st round input
<s>bot
Assistant 1st round output{EOS_TOKEN}
<s>human
User 2nd round input
<s>bot
Assistant 2nd round output{EOS_TOKEN}
...
...
...
<s>human
User nth round input
<s>bot
{Assistant output to be genreated}{EOS_TOKEN}
"""
```
When applying inference, you always make your input string end with ```<s>bot\n``` to request the model generating answers.



## 3. Model Training
Currently, the "MFTCoder-accelerate" codebase supports Full-parameters/LoRA/QLoR along with Multi-Task FineTuning(MFT). 
In theory, this project can be used to train any publicly available model in the HuggingFace Format.

Here are some excellent pre-trained models weights available on Huggingface that can be finetuned with this codebase:

🤗 [Latest code pre-trained SOTA, CodeLlama-34b-Python](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) : code-llama-34b, code-llama-34b-python, a new SOTA base model. 

🤗 [Best 10B level pre-trained Code LLM, Starcoder:](https://huggingface.co/bigcode/starcoder) wizardCoder-15B, PanGu-coder2, and other previous SOTA were trained on it.

🤗 [Multilingual powerhouse, Qwen-7b](https://huggingface.co/Qwen/Qwen-7B): Suitable for multilingual tasks, including Chinese tasks, for instruction fine-tuning.

**mftcoder_accelerate directory structure**
```
mftcoder_accelerate
       |
       src
          configs
          |
          data
          |
          model
          |
          *pefts*
          |
          tokenizer
          |
          utils
       |
       evals
```
我们将训练中使用的各种组件抽取出来，以便后续的扩展和优化， 详见```src```目录下的实现。

训练入口文件是```mftcoder_accelerate/src/pefts/mft_accelerate.py```

参数配置存储在```mftcoder_accelerate/src/configs```目录下，方便统一管理和更改。

**_所以，在你开启训练之前，请进入src目录_**
```
cd mftcoder_accelerate/src
```

You can find the implementations in the ```mftcoder_accelerate/src``` directory.
The entry directory for fine-tuning training is ```mftcoder_accelerate/src```, and the entry file for training is ```mftcoder_accelerate/src/pefts/mft_accelerate.py```. 
Configurations are stored in the ```mftcoder_accelerate/src/configs``` directory for easy management and modification.

**_As a result, before you start training, you should first change your dir by_**
```
cd mftcoder_accelerate/src
```

### 3.1 Tokenization
During training, we concatenate multi-turn dialogues into the following format (also known as the inference data format mentioned before) and then tokenize it. 

In default format, ```<s>human\n``` starts the user's input (i.e., prompt),```<s>bot\n``` starts the assistant's output (i.e., response)

```{EOS_TOKEN}``` represents the proper eos_token.
We have different eos_tokens in ```src/pefts/model_mapping.py``` which fits different base models.

Here is a visionable example of the training data after formatting:
```
f"<s>human\n{input1}<s>bot\n{target1}{EOS_TOKEN}\n<s>human\n{input2}<s>bot\ntarget2{EOS_TOKEN}\n"
```
During the calculation of loss, we use a ```loss mask``` to ensure that the loss from the input part does not contribute to parameter updates. Only the loss from the ```target{EOS_TOKEN}``` part is used for updating parameters.
This approach takes full advantage of the benefits of model parallelism, making training more efficient. It also leverages the characteristic of decoder-only models with left-to-right attention. 
By including all target parts from multiple turns in a single training iteration, the training process becomes more efficient.


### 3.2 LoRA/QLoRA

#### Intro
You can refer to the Lora paper for details about LoRA：[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)

You can refer to the Qlora paper for details about QLoRA：[QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)

QLoRA (Quantized LoRA) is a method that combines 4-bit nf4 quantization and additional adapters to achieve a balance between reducing GPU memory consumption and approaching the performance of full-parameter fine-tuning.

According to the QLoRA paper, this method enables fine-tuning of a 33B model on a single V100 GPU while achieving performance close to that of full-parameter fine-tuning.

To perform LoRA/QLoRA fine-tuning, you can execute the following command:

#### Launch via Deepspeed
DeepSpeed config in accelerate_ds_config.yaml.
```bash
accelerate launch --config_file accelerate_ds_config.yaml pefts/mft_accelerate.py --train_config configs/xxx_train_config.json --distributed_type "DeepSpeed" 
```
or
DeepSpeed config in command line arguments
```bash
sh ds_single_launch.sh
```

#### Launch via FSDP
FSDP config in accelerate_fsdp_config.yaml.
```bash
accelerate launch --config_file accelerate_fsdp_config.yaml pefts/mft_accelerate.py --train_config configs/xxx_train_config.json --distributed_type "FSDP"
```
or
FSDP config in command line arguments
```bash
sh ds_single_launch.sh
```

#### Traing Arguments
All arguments allowed in ***_train_config.josn are defined in ```arguments.py```.

Frequently used arguments are provided in ```configs/***_train_config``` and explained as follows. You can modify these parameters according to your needs:

- **load_raw_dataset**:  Need to be true at present. Only JSONL format is supported.

- **data_paths**: Input data paths in a String of list format, e.g., "[path1,path2,path3]". Each path represents a task directory and each task directory contains one or more JSONL data files.

- **output_dir**: Training output directory to store checkpoints, Lora adapter, etc.

- **tb_dir**: TensorBoard directory to store logs, metrics, etc.

- **model_type**: Type of the model to train, e.g., "mixtral | llama | starcoder | chatglm2 | qwen | gpt_neox".

- **attn_implementation**: "flash_attention_2" or "eager" or "sdpa", worked when model is supported by transformers officially

- **peft_type**: null or  "lora" or "qlora". null for full-params training

- **lora_rank**: Rank value for Lora.

- **lora_alpha**: Alpha value for Lora.

- **lora_dropout**: Dropout rate for Lora.

- **target_modules**: List of target modules in lora, we have default values if None

- **quantization**: "4bit" for QLoRA/ null for LoRA and Full-params training.

- **pretrained_model_path**: Local/Shared disk path or model name on HuggingFace for the pre-trained model.

- **weighted_loss_mode**: Loss weighting method for multitask training. "case3" is recommended at present, "self-paced" is supported but need tuning of hyper-parameters.

- **padding_mode**: The way tokenized data is set. "padding" means padding for each sample to seq_length, "pack" means putting samples into seq_length as many as possible.

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

- **saving_limit**: ckpt saving limit num, must be set in Full-parameter training.

- **role_markers**: {"system": "\<s\>system\n", "user": "\<s\>human\n", "assistant": "\<s\>bot\n} as default(null). You could set your preferred role_markers as the templates startting "system", "user" and "assistant". e.g. {"system": "### System:\n", "user": "### Instruction:\n", "assistant": "### Response:\n"}


## 4. Model Usage

### 4.1 Merge Adaptor weights
Using LoRA or QLoRA for training, this project only saves the weights and configuration files of the adapters. 
To merge the adapter weights with the base model: 
```
python pefts/merge_base_and_lora_to_hf.py \
    --base_model_or_path model_path \
    --adaptor_path lora_adapter_path \
    --model_type model_type \
    --merged_output_path output_path
```

### 4.2 Inference demo
Here is the script for inference on models trained by MFTCoder since v0.3.0, which is compatible with most HuggingFace models:
```python
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
model_name_or_path = "codefuse-ai/CodeFuse-Deepseek-33B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="left")
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

HUMAN_ROLE_START_TAG = "<s>human\n"
BOT_ROLE_START_TAG = "<s>bot\n"
texts = ["write a python function of quick sort."]
texts = [f"{HUMAN_ROLE_START_TAG}{text}{BOT_ROLE_START_TAG}" for text in texts]

inputs = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        top_p=0.95,
        temperature=0.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(gen_text)
```


Indeed, the parameters top_p, temperature, repetition_penalty, do_sample, etc., have a significant impact on the model's generation output. 
You can modify these parameters based on your specific use case.

In code generation scenarios, if you are using the sampling mode (do_sample=True), the following parameter settings can yield good results for the Pass@1 metric:

top_p: Set a higher value, such as 0.95, to retain highly probable generated words. This helps ensure more accurate and fluent generation results.

temperature: Set a lower value, such as 0.1, to reduce randomness. Lower temperature values make the generation output more deterministic.

These parameter combinations can control the diversity of the generated outputs while maintaining naturalness. Additionally, you can adjust other related parameters, such as repetition_penalty, to reduce repetition in the generated results.

If you choose the non-sampling mode (do_sample=False), you can consider the following parameter settings:

beam_num: Set a smaller value such as 1 or 3. ```beam_num=1``` represents greedy decoding, which selects the most probable single generated word. ```beam_num=3``` represents beam search mode, which considers multiple potential generation paths and chooses the best path among them.

## 5. FAQ
#### Q1：What should I do when cuda OOM happens？
If OOM happened，you can reduce parameters such as per_device_train_batch_size and seq_length. Since you are dealing with large models (6B, 13B, 34B, 70B, etc.), you are already using gradient checkpointing technology by default, which significantly reduces GPU memory consumption. 
However, this may slightly slow down the training speed.

#### Q2：install packages
Please refer to init_env.sh and requirements.txt
We highly recommend you install Flash Attention 2 (flash_attn>=2.1.0, 2.3.6 used by us) first to get memory-efficient and fast training.

#### Q3：How should I specify the GPUs for training？
You can specify the visiable GPUs as below:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_ds_config.yaml pefts/mft_accelerate.py --train_config configs/xxx_train_config.json
```

#### Q4：Whats is a recommended Distributed Training?
For LoRA/QLoRA, we recommend DeepSpeed(ZeRO2) as the underlying framework, because it is easy and stable to use, moreover it is more compatable for different settings.
And FSDP does not support Quantization(integer type in training).

For Full-parameter finetuning, FSDP is usually faster, and may help you with very large models by sharding parameters and gradients.