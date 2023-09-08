# MFTCoder训练: Huggingface accelerate + DeepSpeed框架篇
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/codefuse-ai)
<a href="https://github.com/codefuse-ai/MFTCoder/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
</a>

[**中文**] [[English]](README.md)

## 1. 更新

🔥 MFTCoder在Huggingface accelerate + DeepSpeed框架下支持QLoRA/LoRA微调； 

🔥 MFTCoder在训练中支持了多任务微调， 可以同时平衡多个任务的训练，训练的模型支持多任务推理； 

🔥 MFTCoder在训练中支持多种模型基座： codellama, llama2, llama, starcoder, codegeex2, chatglm2, qwen等

## 2. 数据格式
### 2.1 训练数据格式
训练数据为jsonl格式，每一行的数据格式如下，其中chat_rounds字段是必需的，可以根据实际需求添加或删除其他字段。
可以参考项目中的xxx.jsonl文件。
```json
{
    "id":0,
    "data_name":"code-helper",
    "chat_rounds":[
        {
            "role": "system",
            "content": "你是一个智能代码助手，可以回复用户与代码相关的问题",
            "chat_round_id": 0
        },
        {
            "role": "human",
            "content": "写一个快速排序", 
            "chat_round_id": 1
        },
        {
            "role": "bot",
            "content": "以下是一个快速排序算法xxxxxx", 
            "chat_round_id": 1
        },
        {
            "role": "human",
            "content": "解释一下这段代码", 
            "chat_round_id": 2
        },
        {
            "role": "bot",
            "content": "好的，这段代码xxx", 
            "chat_round_id": 2
        }
    ]
}
```

### 2.2 推理数据格式
推理数据格式为模型在训练数据格式下拼接的字符串形式，它也是推理时输入prompt拼接的方式：
```python
"""
<|role_start|>system<|role_end|>这是System指令
<|role_start|>human<|role_end|>这是第1轮用户输入的问题
<|role_start|>bot<|role_end|>这是第1轮模型生成的内容</s>
<|role_start|>human<|role_end|>这是第2轮用户输入的问题
<|role_start|>bot<|role_end|>这是第2轮模型生成的内容</s>
...
...
...
<|role_start|>human<|role_end|>这是第n轮用户输入的问题
<|role_start|>bot<|role_end|>{模型现在要生成的内容}</s>
"""
```


## 3. 模型训练
目前支持全量参数指令微调、QLoRA指令微调，LoRA指令微调。
一些优秀的代码预训练模型权重，理论上，HuggingFace上开源的模型，均可使用本项目进行训练：

🤗 [最新代码预训练SOTA，CodeLlama](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) ：code-llama-34b， code-llama-34b-python, 新的SOTA基座。

🤗 [10B级别最佳代码预训练模型Starcoder](https://huggingface.co/bigcode/starcoder) wizardCoder-15B, PanGu-coder2等前SOTA的基座模型。

🤗 [多语言能手Qwen-7b](https://huggingface.co/Qwen/Qwen-7B) ：适用于多语言任务，也适用中文任务。进行指令微调时。

我们将训练中使用的各种组件抽取出来，以便后续的扩展和优化，详见src目录下的实现。微调训练的入口目录是```src/pefts```, 训练入口文件是```src/pefts/mft_accelerate.py```, 参数配置存储在```src/pefts/configs```目录下，方便统一管理和更改。

### 3.1 数据tokenization
训练时，我们将多轮对话拼接成如下格式（也是上文中的推理string格式），然后进行tokenize。其中<|role_start|>human<|role_end|>表示human输入提示符，<|role_start|>bot<|role_end|>表示bot输出提示符，`````</s>````` 表示eos_token。
其中eos_token可以根据不同模型修改替换。
```
"<|role_start|>human<|role_end|>input1</s>target1</s>input2</s>target2</s>...
```
在计算loss时，我们通过loss mask的方式，input部分的loss不参与参数更新，只有“target</s>”部分的loss参与参数更新。
这种方式充分利用了模型并行计算的优势，训练更加高效，同时也充分利用了decoder-only模型从左到右attention的特性，一次性将多轮对话中的每个target部分都参与了训练，训练更充分高效。

### 3.2 LoRA/QLoRA微调
关于LoRA的详细介绍可参考论文：[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
关于QLoRA的详细介绍可参考论文：[QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)

QLoRA通过4-bit的nf4量化，且加入更多adapter，在大幅减少显存消耗的同时，尽可能逼近全量参数微调的效果。
QLoRA论文指出，该方法可以在一张V100上对33B的模型进行微调，并且性能逼近全量参数微调。

执行如下命令即可进行Lora/QLora微调：
```bash
accelerate launch --config_file accelerate_ds_config.yaml mft_accelerate.py --train_config configs/starcoder_train_config.json
```

```configs/*_train_config```中的主要参数说明如下，以下参数可以根据需求进行修改，其他参数建议不做修改：
- load_raw_dataset : 需要保持true，后续会支持其它模式数据，当前仅支持jsonl输入
- data_paths: "[path1,path2,path3]" 输入数据地址，字符串，开头结尾用[]，中间用```,```间隔不同path，每个path是一个目录，目录的最后一级名字作为任务名称，下面包含1到多个jsonl数据
- output_dir：训练输出目录，存储checkpoint、lora_adaptor checkpoint等
- tb_dir: 存储tensorboard等
- model_type
- peft_type: lora或者qlora
- lora_rank: lora rank
- lora_alpha: lora alpha
- lora_dropout: lora dropout
- quantization: 是否量化，"4bit", "8bit" 或者null， qlora推荐4bit量化
- pretrained_model_path：预训练模型的本地目录，或者在huggingface上的模型名称。
- **weighted_loss_mode**: 多任务loss加权模式， "case3"是当前推荐。
- **padding_mode**: 数据的样本组织方式， "padding"是将每个原始样本填充到seq_length, "pack"是将尽量多的样本打包到每个seq_length的序列中。
- num_train_epochs：训练的轮次。如果数据量足够大，一般建议只训1-2个epoch。
- per_device_train_batch_size：每张显卡train的batch size。
- per_device_eval_batch_size：每张显卡eval的batch size。
- gradient_accumulation_steps：梯度累计步数。global batch=num_gpus * per_device_train_batch_size * gradient_accumulation_steps。
- learning_rate：学习率。全量参数微调的时候，建议小一些，1e-5或5e-6。qlora中的学习率设置更大一些，一般为1e-4、2e-4。
- min_lr: 最低学习率， 一般是learning_rate的十分之一
- seq_length：训练时的最大长度。按照自己的设备进行设置，越长需要占用越多显存。
- log_interval：每隔多少步统计一次train loss。
- checkpointing_steps：每隔多少步保存一个模型。
- evalation_steps：每隔多少步在验证集上evaluate一次。
- early_stopping ： 是否执行early_stop
- early_stopping_stall_num： 多少个eval point不继续收敛，则停止训练
- lr_scheduler_type：学习率变化策略。
- warmup_steps：warm up步数。学习率经过多少步，增长到指定的数值。
- seed：随机种子，用于复现实验结果。

## 4. 模型使用

### 4.1 权重合并
如果使用LoRA或者QLoRA进行训练，本项目仅保存adapter的权重和配置文件，需要将adapter权重与base model进行合并。脚本见```src/pefts/merge_base_and_lora_to_hf.py```

### 4.2 模型推理
我们提供了单轮对话和多轮对话的如下脚本，该脚本可同时兼容大部分huggingface格式的模型。
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


生成脚本中的top_p、temperature、repetition_penalty、do_sample等参数对模型的生成效果影响较大，可按照自己的使用场景进行调试修改。
实践中，在代码生成场景中，如果采样模式，do_sample=True, top_p=0.95, temperature=0.1是pass@1指标的不错选择；
如果非采样模式， do_sample=False, beam_num=1或者3是不错的选择，其中beam_num=1即为greedy decoding。

## 5. FAQ
#### 问题1：OOM如何解决？
如果发生OOM，可以缩小per_device_train_batch_size、seq_length等参数来缓解。由于面对的模型普遍较大（6b， 13b， 34b， 70b等）我们已经默认使用gradient_checkpointing技术，可以大幅降低显存占用，但训练速度会稍慢一些。

#### 问题2：安装包错误
参考init_env.sh和requirements.txt

#### 问题3：如何指定使用某些卡训练？
通过如下方式，即可指定使用0和1号卡进行训练:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_ds_config.yaml mft_accelerate.py --train_config configs/starcoder_train_config.json
```





