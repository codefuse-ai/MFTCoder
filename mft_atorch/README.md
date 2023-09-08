# MFTCoder Training: Atorch Framework
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/codefuse-ai)
<a href="https://github.com/codefuse-ai/MFTCoder/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
</a>

[[中文]](README_cn.md) [**English**]

## 1. Updates

🔥 MFTCoder supports training and fine-tuning of the GPTNeoX model under the Atorch framework.

🔥 MFTCoder supports both fully supervised fine-tuning and self-supervised pretraining.

🔥 MFTCoder supports LoRA using Atorch Framework.

## 2. Data Format
### 2.1 Training Data Format
The training data is in a uniformed JSONL format, in which each line of data has the following JSON format. The "chat_rounds" field is required, and other fields can be added or removed based on specific needs. 

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

### 2.2 Inference Data Format
The inference string is a string concatenated by conversation data(system, human and bot contents) in the training data format. 
It is used as the data "seen"(before tokenization) by the model in training process.
It is used as input during the inference process as well.
Here is an example format of the concatenated string:

```python
"""
<|role_start|>system<|role_end|>System instruction
<|role_start|>human<|role_end|>Human 1st round input
<|role_start|>bot<|role_end|>Bot 1st round output</s>
<|role_start|>human<|role_end|>Human 2nd round input
<|role_start|>bot<|role_end|>Bot 2nd round output</s>
...
...
...
<|role_start|>human<|role_end|>Human nth round input
<|role_start|>bot<|role_end|>{Bot output to be genreated}</s>
"""
```
When applying inference, you always make your input string end with "<|role_start|>bot<|role_end|>" to ask the model generating answers.

## 3. Model Training
Currently, the "MFTCoder/mft_atorch" code repository supports full-parameter SST, full-parameter instruction fine-tuning, and LoRA instruction fine-tuning.
Currently, only training of the GPTNeoX model is supported. In theory, the pretrained weights of the GPTNeoX model available on HuggingFace can be used for training with this project.

We have extracted various components used in training to facilitate future extension and optimization. Please refer to the implementation in the main directory for more details. The entry directory for fine-tuning training is ```train/```, and the entry file for training is ```train/run_train.py```. The parameter configurations are stored in the launch scripts such as ```train/run_gpt_*.sh```, making it easier to manage and modify them uniformly.

### 3.1 Tokenization
During training, we concatenate multi-turn dialogues into the following format (also known as the inference string format mentioned earlier) and then tokenize it. In this format, <|role_start|>human<|role_end|> represents the human input prompt, <|role_start|>bot<|role_end|> represents the bot output prompt, and </s> represents the eos_token.
You can modify and replace the eos_token based on different models' requirements.

Here is an example of the concatenated format with prompts:
```
"<|role_start|>human<|role_end|>input1</s>target1</s>input2</s>target2</s>...
```
During the calculation of loss, we use a ```loss mask``` to ensure that the loss from the input part does not contribute to parameter updates. Only the loss from the ```target</s>``` part is used for updating parameters.
This approach takes full advantage of the benefits of model parallelism, making training more efficient. It also leverages the characteristic of decoder-only models with left-to-right attention. 
By including all target parts from multiple turns in a single training iteration, the training process becomes more efficient.

### 3.2 Fully Self-Supervised Pretraining
To perform fully SST, you can execute the following command:
```bash
sh run_gpt_mpt.sh 10 1 8 5
```
Please note that the four parameters after the launch script have the following meanings:
- The first parameter is the per GPU batch size.
- The second parameter is the number of tensor parallelism (currently only supports 1).
- The third parameter is the number of data parallelism, which should match the number of GPUs used.
- The fourth parameter is the number of training epochs. 

For other training modes, the same four parameters need to be configured in the launch script.

### 3.3 Fully Supervised Fine-Tuning
To perform fully SFT, you can execute the following command:
```bash
sh run_gpt_mft.sh 10 1 8 5
```

### 3.4 LoRA Supervised Fine-Tuning
To perform LoRA SFT, you can execute the following command:
```bash
sh run_gpt_mft_peft.sh 10 1 8 5
```

### 3.5 Parameter Explanations
The main parameter explanations for the ```train/run_gpt_*.sh``` are as follows. You can modify these parameters according to your needs:

- **tokenize_mode**: Choose "sst" or "sft" based on the actual situation.

- **train_mode**: Choose "sst" or "sft" based on the actual situation.

- **load_raw_dataset**: When True, it reads the jsonl input and performs tokenization in real-time before starting the training process. When False, it reads the ".bin" formatted input, including the ".bin" file for loss_mask, and starts training immediately after reading.

- **data_paths**: "[path1,path2,path3]" Input data addresses, a string enclosed in [], with different paths separated by commas (,). Each path is a directory where the last level of the directory name is considered as the task name. Each task directory contains 1 to multiple jsonl data files.

- **output_dir**: Training output directory to store checkpoints, lora_adaptor checkpoints, etc.

- **tensorboard_dir**: Can be temporarily ignored, as the actual tensorboard is stored in the runs directory under output_dir.

- **model_type**: Currently only supports gpt_neox.

- **peft_type**: Currently only supports lora.

- **pretrained_model_path**: Local directory of the pre-trained model. Not required for SST.

- **config_path**: Model config address required for SST, not required for SFT.

- **total_train_batch_size**: The total batch size for training across all GPUs, calculated automatically based on per gpu batch size entered in the script.

- **per_device_valid_batch_size**: The batch size for evaluation on each GPU, calculated automatically based on per gpu batch size entered in the script.

- **gradient_accumulation_steps**: Number of gradient accumulation steps. Global batch size = num_gpus * per_device_train_batch_size * gradient_accumulation_steps.

- **checkpoint_activations**: Enable if running out of GPU memory. Trades time for space by not caching activation states, resulting in two forward passes to save memory.

- **learning_rate**: Learning rate. When fine-tuning the entire model, it is recommended to use a smaller value, such as 1e-5 or 5e-6. For lora, a larger learning rate is generally used, such as 1e-4 or 2e-4.

- **min_lr**: Minimum learning rate, usually one-tenth of the learning_rate.

- **seq_length**: Maximum length during training. Set according to your device, longer lengths require more memory.

- **log_interval**: Frequency of logging training loss.

- **checkpointing_steps**: Frequency of saving a model checkpoint.

- **evalation_steps**: Frequency of evaluating on the validation set.

- **early_stopping_patience**: Number of consecutive eval points without further convergence to stop training.

- **lr_scheduler_type**: Learning rate changing strategy.

- **num_warmup_steps**: Number of warm-up steps for the learning rate to increase to the specified value.

- **seed**: Random seed used for reproducibility of experimental results.

- **train_iters**: Can be temporarily set to a small value, such as 10, which does not affect the actual number of training steps but can affect the number of samples in the train dataset when reading ".bin" data.

- **valid_iters**: Can be temporarily set to a small value, such as 10, which does not affect the actual number of training steps but can affect the number of samples in the valid dataset when reading ".bin" data.

- **evaluation_strategy**: Evaluation strategy during training. "steps" means to evaluate every "valid_interval" steps, "epoch" means to evaluate every epoch. Both can be enabled simultaneously.

- **save_strategy**: Strategy for saving model weights during training. "steps" means to save every "checkpointing_steps" steps.
- **extra_save_by_epoch**: Whether to save an epoch-level checkpoint every epoch.

- **save_total_limit**: Maximum number of model checkpoints to keep. Generally set to 2, retaining the checkpoint with the lowest valid loss and the latest checkpoint. Note that epoch-level checkpoints will always be retained and are not subject to this limit.

- **weighted_loss_mode**: Loss weighting method for multi-task training.

## 4. Model Usage

### 4.1 Merge Adaptor weights
Using LoRA or QLoRA for training, this project only saves the weights and configuration files of the adapters. 
To merge the adapter weights with the base model, see ```src/pefts/merge_base_and_lora_to_hf.py```

### 4.2 Inference demo
Here is the script for inference on our trained models, which is compatible with most Hugging Face models:
```python
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<unk>")
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True)

HUMAN_ROLE_START_TAG = "<|role_start|>human<|role_end|>"
BOT_ROLE_START_TAG = "<|role_start|>bot<|role_end|>"
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
### Q1：What should I do when cuda OOM happens？
If OOM (Out of Memory) occurs, you can mitigate it by reducing parameters such as per GPU batch size (the first argument when starting the training script) and seq_length. You can also set gradient_checkpointing=true, which significantly reduces memory usage but may slow down the training speed.