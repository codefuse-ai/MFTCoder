# MFTCoder: Accelerate + DeepSpeed/FSDP 框架篇
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/codefuse-ai)
<a href="https://github.com/codefuse-ai/MFTCoder/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
</a>

[**中文**] [[English]](README.md)

## 1. 更新
🔥 MFTCoder-accelerate 增加了xxpo模块，支持dpo训练。

🔥 MFTCoder-accelerate 增加了mpt模块，借助offline_tokenization模块，支持全量参数加训。

🔥 MFTCoder-accelerate 增加了CoBa Loss的最新实现（原selfpaced Loss）, 让收敛均衡更进一步。

🔥 MFTCoder-accelerate 最新支持的训练模式包括: QLoRA/LoRA + DeepSpeed ZeRO2， QLoRA + DeepSpeed ZeRO3, 全量 + DeepSpeed ZeRO3, QLoRA + FSDP, 全量 + FSDP。

🔥 MFTCoder-accelerate 新增支持QLoRA + DeepSpeed ZeRO3， 支持QLoRA + FSDP, 可以训练更大的模型。

🔥 MFTCoder-accelerate 新增支持accelerate + FSDP框架， 支持全量微调和LoRA。

🔥 MFTCoder-accelerate 支持最新更多主流开源模型: mistral, mixtral-8x7b(Mixture of Experts), deepseek, chatglm3。

🔥 MFTCoder-accelerate 新增self-paced Loss, 用于收敛均衡。

🔥 MFTCoder-accelerate 支持使用accelerate + DeepSpeed框架下支持 全量参数/QLoRA/LoRA微调。

🔥 MFTCoder-accelerate 在训练中支持了多任务微调MFT， 可以同时平衡多个任务的训练，训练的模型支持多任务推理。

🔥 MFTCoder-accelerate 在训练中支持多种模型基座： codellama, llama2, llama, starcoder, codegeex2, chatglm2, qwen等。

## 2. 数据格式
### 2.1 MFT训练数据格式
训练数据为jsonl格式，每一行的数据格式如下，其中chat_rounds字段是必需的，可以根据实际需求添加或删除其他字段。
可以参考项目中的xxx.jsonl文件。
```json
{
    "id":0,
    "data_name":"code-helper",
    "chat_rounds":[
        {
            "role": "system",
            "content": "你是一个智能代码助手，可以回复用户与代码相关的问题"
        },
        {
            "role": "human",
            "content": "写一个快速排序"
        },
        {
            "role": "bot",
            "content": "以下是一个快速排序算法xxxxxx"
        },
        {
            "role": "human",
            "content": "解释一下这段代码"
        },
        {
            "role": "bot",
            "content": "好的，这段代码xxx"
        }
    ]
}
```

### 2.2 推理数据格式
推理数据格式为模型在训练数据格式下拼接的字符串形式，它也是推理时输入prompt拼接的方式：
```
"""
<s>system
这是System指令
<s>human
这是第1轮用户输入的问题
<s>bot
这是第1轮模型生成的内容{EOS_TOKEN}
<s>human
这是第2轮用户输入的问题
<s>bot
这是第2轮模型生成的内容{EOS_TOKEN}
...
...
...
<s>human
这是第n轮用户输入的问题
<s>bot
{模型现在要生成的内容}{EOS_TOKEN}
"""
```

### 2.3 DPO训练数据格式
训练数据为jsonl格式，每一行的数据格式如下，其中chosen字段和rejected字段分别代表偏好对齐中的```chosen```和```rejected```，其内部依然是MFT的chatml格式，并且只有最后一轮对话的bot content不同。
```json
{
    "chosen":[
        {
            "role": "system",
            "content": "你是一个智能代码助手，可以回复用户与代码相关的问题"
        },
        {
            "role": "human",
            "content": "写一个快速排序"
        },
        {
            "role": "bot",
            "content": "以下是一个快速排序算法xxxxxx"
        },
        {
            "role": "human",
            "content": "解释一下这段代码"
        },
        {
            "role": "bot",
            "content": "好的，这段代码xxx"
        }
    ],
    "rejected":[
        {
            "role": "system",
            "content": "你是一个智能代码助手，可以回复用户与代码相关的问题"
        },
        {
            "role": "human",
            "content": "写一个快速排序"
        },
        {
            "role": "bot",
            "content": "以下是一个快速排序算法xxxxxx"
        },
        {
            "role": "human",
            "content": "解释一下这段代码"
        },
        {
            "role": "bot",
            "content": "对不起，我不会"
        }
    ]
}
```


## 3. 模型训练
目前支持全量参数(Full-parameters)指令微调、QLoRA指令微调，LoRA指令微调。
一些优秀的代码预训练模型权重，理论上，HuggingFace上开源的模型，均可使用本项目进行训练：

🤗 [最新代码预训练SOTA，CodeLlama](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) ：code-llama-34b， code-llama-34b-python, 新的SOTA基座。

🤗 [10B级别最佳代码预训练模型Starcoder](https://huggingface.co/bigcode/starcoder) wizardCoder-15B, PanGu-coder2等前SOTA的基座模型。

🤗 [多语言能手Qwen-7b](https://huggingface.co/Qwen/Qwen-7B) ：适用于多语言任务，也适用中文任务。进行指令微调时。

**mftcoder_accelerate文件结构**
```
mftcoder_accelerate
       |
       data
       |
       inference
       |
       model
       |
       tokenization
       |
       tokenizer
       |
       training
          |
          configs
          |
          *pefts*
          |
          *xxpo*
          |
          *mpt*
          |
          train_utils

```
我们将训练中使用的各种组件抽取出来，以便后续的扩展和优化， 详见```mftcoder_accelerate```目录下的实现。

MFT训练入口文件是```mftcoder_accelerate/training/mft_accelerate.py```

DPO/ORPO训练入口文件是```mftcoder_accelerate/training/xxpo_accelerate.py```

MPT(全量加训)训练入口文件是```mftcoder_accelerate/training/mpt_accelerate.py```. MPT加训需要提前做好数据的tokenziation，通过```mftcoder_accelerate/run_offline_tokenization.sh```，你可以将数据通过cpu进行离线的tokenization。这和MFT/DPO中使用的在线tokenziation不同。

参数配置存储在```mftcoder_accelerate/training/configs```目录下，方便统一管理和更改。

**你开启训练之前**
```
cd mftcoder_accelerate
```



### 3.1 数据tokenization
MFT/DPO训练时，我们将多轮对话拼接成如下格式（也是上文中的推理数据格式），然后进行tokenize。
其中，默认情况下：

```<s>human\n```作为human/user的起始符，```<s>bot\n```作为bot/assistant的起始符，```{EOS_TOKEN}``` 表示eos_token。
其中eos_token可以根据不同模型修改替换。不同角色的起始符可以配置，用来实现不同的对话/问答模版。
```
"<s>human\n{input1}<s>bot\n{target1}{EOS_TOKEN}<s>human\n{input2}<s>bot\n{target2}{EOS_TOKEN}\n"
```
在计算loss时，我们通过loss mask的方式，input部分的loss不参与参数更新，只有“target{EOS_TOKEN}”部分的loss参与参数更新。
这种方式充分利用了模型并行计算的优势，训练更加高效，同时也充分利用了decoder-only模型从左到右attention的特性，一次性将多轮对话中的每个target部分都参与了训练，训练更充分高效。

### 3.2 LoRA/QLoRA微调

#### LoRA/QLoRA微调简介
关于LoRA的详细介绍可参考论文：[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)

关于QLoRA的详细介绍可参考论文：[QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)

QLoRA通过4-bit的nf4量化，且加入更多adapter，在大幅减少显存消耗的同时，尽可能逼近全量参数微调的效果。
QLoRA论文指出，该方法可以在一张V100上对33B的模型进行微调，并且性能逼近全量参数微调。

执行如下命令即可进行 Lora/QLora/全量 微调：
#### Deepspeed 单机启动
DeepSpeed配置在accelerate_ds_config.yaml中。
```bash
accelerate launch --config_file accelerate_ds_config.yaml training/mft_accelerate.py --train_config training/configs/xxx_train_config.json --distributed_type "DeepSpeed" 
```
或者

DeepSpeed Zero2 配置在脚本中通过命令行输入。
```bash
sh ds_single_launch.sh
```

DeepSpeed Zero3 配置在脚本中通过命令行输入
```bash
sh ds_zero3_single_launch.sh
```

#### FSDP 单机启动
FSDP配置在accelerate_fsdp_config.yaml中。
```bash
accelerate launch --config_file accelerate_fsdp_config.yaml training/mft_accelerate.py --train_config training/configs/xxx_train_config.json --distributed_type "FSDP"
```
或者

FSDP配置在脚本中通过命令行输入。
```bash
sh fsdp_single_launch.sh
```

#### 多机启动
多机启动请参考如下deepspeed多机启动脚本
```bash
sh ds_multinode_launch.sh
```

#### 训练参数
_**训练需要的参数配置在```training/configs/*_train_config```中，主要参数说明如下：**_

- **load_raw_dataset**: 需要保持true，后续会支持其它模式数据，当前仅支持jsonl输入
- **data_paths**: "[path1,path2,path3]" 输入数据地址，字符串，开头结尾用[]，中间用```,```间隔不同path，每个path是一个目录，目录的最后一级名字作为任务名称，下面包含1到多个jsonl数据
- **output_dir**：训练输出目录，存储checkpoint(全量训练时)、lora_adaptor（Lora或者Qlora时）等
- **tb_dir**: 存储tensorboard等
- **model_type**: "mixtral|mistral|deepseek|llama|starcoder|chatglm2|qwen|gpt_neox"
- **attn_implementation**: "flash_attention_2" 或者 "eager"
- **peft_type**: lora或者qlora或者null(全量微调)
- **lora_rank**: lora rank
- **lora_alpha**: lora alpha
- **lora_dropout**: lora dropout
- **target_modules**: List[str], lora目标模块，如果null，会使用默认，参考model_mapping.py
- **quantization**: 是否量化，"4bit", "8bit" 或者null， qlora推荐4bit量化
- **pretrained_model_path**：预训练模型的本地目录，或者在huggingface上的模型名称。
- **weighted_loss_mode**: 多任务loss加权模式， "case3"是当前推荐。
- **padding_mode**: 数据的样本组织方式， "padding"是将每个原始样本填充到seq_length, "pack"是将尽量多的样本打包到每个seq_length的序列中。
- **num_train_epochs**：训练的轮次。如果数据量足够大，一般建议只训1-2个epoch。
- **per_device_train_batch_size**：每张显卡train的batch size。
- **per_device_eval_batch_size**：每张显卡eval的batch size。
- **gradient_accumulation_steps**：梯度累计步数。global batch=num_gpus * per_device_train_batch_size * gradient_accumulation_steps。
- **learning_rate**：学习率。全量参数微调的时候，建议小一些，1e-5或5e-6。qlora中的学习率设置更大一些，一般为1e-4、2e-4。
- **min_lr**: 最低学习率， 一般是learning_rate的十分之一
- **seq_length**：训练时的最大长度。按照自己的设备进行设置，越长需要占用越多显存。
- **log_interval**：每隔多少步统计一次train loss。
- **checkpointing_steps**：每隔多少步保存一个模型。
- **evaluation_steps**：每隔多少步在验证集上evaluate一次。
- **early_stopping** ： 是否执行early_stop
- **early_stopping_stall_num**： 多少个eval point不继续收敛，则停止训练
- **lr_scheduler_type**：学习率变化策略。常用"cosine"
- **warmup_steps**：warm up步数。学习率经过多少步，增长到指定的数值。
- **seed**：随机种子，用于复现实验结果。
- **saving_limit**：整数，ckpt存储数量上限， 全量训练必须设置。默认null即不限制数量。
- **role_markers**: null，即使用{"system": "\<s\>system\n", "user": "\<s\>human\n", "assistant": "\<s\>bot\n"}。 你可以自定义 "system", "user" and "assistant"的模板， 用于定制自己的问答或者对话模板，比如 {"system": "### System:\n", "user": "### Instruction:\n", "assistant": "### Response:\n"}

#### CoBa相关参数配置
- **coba_warmup_steps**: CoBa的warm-up步数。在warm-up期间，各任务权重相等，warm-up之后，开始动态调整权重。一般建议设置为与valid batch总数量相近即可。
- **coba_history_length**: CoBa维护的valid loss的历史窗口长度，用于拟合当前步收敛斜率。一般建议设置为2倍**coba_warmup_steps**至5倍**coba_warmup_steps**之间。一般该值越大，权重的变化幅度就会越小。
- **coba_tau**: 发散因子（DF）的温度系数。一般设置为5即可。
- **coba_update_interval**: CoBa更新权重的频率。一般设置为1，即每一步都对权重做更新。
- **coba_sample_valid_num**: CoBa每一步要取的valid batch数。理论上当该值等于valid batch总数量时，拟合出的收敛斜率最逼近真实情况，但考虑到计算需求，建议设置为1。

#### DPO 相关参数配置
- **xxpo**: 偏好对齐方法, "dpo" 或者 "orpo"。
- **beta**: DPO beta, beta 越小，允许对齐后的dpo模型与ref模型的距离越远。
- **rpo_alpha**: 加到dop损失的```chosen``` NLL损失的系数，0的话就是原始DPO。

## 4. 模型使用

### 4.1 权重合并
如果使用LoRA或者QLoRA进行训练，本项目仅保存adapter的权重和配置文件，需要将adapter权重与base model进行合并。
可以使用如下merge_base_and_lora_to_hf.py脚本。
```
python pefts/merge_base_and_lora_to_hf.py \
    --base_model_or_path model_path \
    --adaptor_path lora_adapter_path \
    --model_type model_type \
    --merged_output_path output_path
```

### 4.2 模型推理
我们提供了单轮对话和多轮对话的如下脚本，该脚本可同时兼容大部分huggingface格式的模型。
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


生成脚本中的top_p、temperature、repetition_penalty、do_sample等参数对模型的生成效果影响较大，可按照自己的使用场景进行调试修改。
实践中，在代码生成场景中，如果采样模式，do_sample=True, top_p=0.95, temperature=0.1是pass@1指标的不错选择；
如果非采样模式， do_sample=False, beam_num=1或者3是不错的选择，其中beam_num=1即为greedy decoding。

## 5. FAQ
#### 问题1：OOM如何解决？
如果发生OOM，可以缩小per_device_train_batch_size、seq_length等参数来缓解。由于面对的模型普遍较大（6b， 13b， 34b， 70b等）我们已经默认使用gradient_checkpointing技术，可以大幅降低显存占用，但训练速度会稍慢一些。
如果是模型太大，可以使用QLoRA + DeepSpeed ZeRO3(配置 zero stage = 3)，这个方案可以在卡数足够的情况下，微调更大的模型。
#### 问题2：安装包错误
参考init_env.sh和requirements.txt

#### 问题3：如何指定使用某些卡训练？
通过如下方式，即可指定使用0和1号卡进行训练:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file pefts/accelerate_ds_config.yaml pefts/mft_accelerate.py --train_config configs/xxx_train_config.json --distributed_type "deepspeed"
```

#### 问题4：关于Flash Attention, 该如何配置训练？
首先，我们强烈建议您安装Flash Attention 2(FA2)，（>=2.1.0, 2.3.6功能更齐全）。

训练参数中"attn_implementation" 设置成 "eager" 可以用naive attention，也就是未经加速的attention。

训练参数中"attn_implementation" 设置成 "flash_attention_2" 可以用FA2，速度快，省显存。

如果你可以自行安装环境并使用torch>=2.1.1，可以尝试设置参数"attn_implementation"为 "sdpa"。这样会尝试使用transformers兼容的torch.nn.functional.scaled_dot_product_attention。支持的模型还不全面。

#### 问题5：推荐的分布式框架是怎样的？
对于LoRA, 我们推荐使用DeepSpeed Zero2作为底层分布式框架，它具有易用性和兼容性好的特点，并且速度很快, 模型加载模式上类似DDP。
对于QLoRA, DeepSpeed Zero2 适合中小模型, DeepSpeed Zero3 适合很大的模型。

对于全量微调，可以使用DeepSpeed Zero3, 或者FSDP。二者都是Fully Sharding模式，即模型加载平分在每张卡。

#### 问题6：当前支持的模型中，有什么区别
国产大模型比如chatglm2， chatglm3， baichuan2， qwen， aquila2等，使用的是和模型共同发布的modeling_xxx.py. 
其它被transformers官方支持的大模型，比如llama, qwen2, starcoder2, mistral等，全面切换到官方的modeling支持训练，之前的自定义modeling会被deprecated。





