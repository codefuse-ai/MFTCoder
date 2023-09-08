# CodeFuse-MFTCoder: Multitask Fine-Tuned Code LLMs

<p align="center">
  <img src="./assets/codefuse_logo_blue.png" width="100%" />
</p>


<div align="center">

<p>
    <a href="https://github.com/codefuse-ai/MFTCoder">
        <img alt="stars" src="https://img.shields.io/github/stars/codefuse-ai/mftcoder?style=social" />
    </a>
    <a href="https://github.com/codefuse-ai/MFTCoder">
        <img alt="forks" src="https://img.shields.io/github/forks/codefuse-ai/mftcoder?style=social" />
    </a>
    <a href="https://github.com/codefuse-ai/MFTCoder/LICENCE">
      <img alt="License: MIT" src="https://badgen.net/badge/license/apache2.0/blue" />
    </a>
     <a href="https://github.com/codefuse-ai/MFTCoder/releases">
      <img alt="Release Notes" src="https://img.shields.io/github/release/codefuse-ai/MFTCoder" />
    </a>
    <a href="https://github.com/codefuse-ai/MFTCoder/issues">
      <img alt="Open Issues" src="https://img.shields.io/github/issues-raw/codefuse-ai/MFTCoder" />
    </a>
</p>


[[中文]](README_cn.md) [**English**]

</div>



## Contents
- [News](#News)
- [Articles](#Articles)
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Training](#Training)
- [Models](#Models)
- [Datasets](#Datasets)


## News
🔥🔥🔥 [2023/09/07]We released **CodeFuse-CodeLlama-34B**, which achieves the **74.4 pass@1** (greedy decoding) and surpasses GPT4 (2023/03/15), ChatGPT-3.5, and Claude2 on the [HumanEval Benchmarks](https://github.com/openai/human-eval).

🔥 [2023/08/26]We released MFTCoder which supports finetuning CodeLlama, Llama, Llama2, StarCoder, Chatglm2, CodeGeex2, Qwen, and GPT-NEOX models with LoRA/QLoRA.

| Model                              | HumanEval(pass@1) | 
|:-----------------------------------|:-----------------:|
| CodeLlama-34b                      |       48.8%       |
| CodeLlama-34b-Python               |       53.7%       |
| WizardCoder-Python-34B-V1.0        |       73.2%       |
| **CodeFuse-CodeLlama-34B**         |     **74.4%**     |

## Articles


## Introduction
**CodeFuse-MFTCoder** is an open-source project of CodeFuse for multitasking Code-LLMs(large language model for code tasks), which includes models, datasets, training codebases and inference guides.
In MFTCoder, we released two codebases for finetuning Large Language Models: 
- ```mft_peft_hf``` is based on huggingface accelerate and deepspeed framework.
- ```mft_atorch``` is based on [ATorch frameworks](https://github.com/intelligent-machine-learning/dlrover), which is a fast distributed training framework of LLM.
The project aims to share and collaborate on advancements in large language models specifically in the domain of code.

### Frameworks
![img.png](./assets/img.png)

### Highlights
1. [x] **Multi-task**: It is able to train a model on multiple tasks, ensuring a balance between them and even generalizing to new, unseen tasks.
2. [x] **Multi-model**: It supports various state-of-the-art open-source models, including gpt-neox, llama, llama-2, baichuan, Qwen, chatglm2, and more.
3. [x] **Multi-framework**: It provides support for both HuggingFace Accelerate(deepspeed used) and [ATorch frameworks](https://github.com/intelligent-machine-learning/dlrover).
4. [x] **Efficient fine-tuning**: It supports LoRA and QLoRA, enabling the fine-tuning of large models with minimal resources. The training speed is capable of meeting the demands of almost all fine-tuning scenarios.

The main content of this project includes:
- Support for both SFT (Supervised FineTuning) and MFT (Multi-task FineTuning). The current MFTCoder has achieved data balance among multiple tasks, and future releases will realize balance of difficulty and convergence in traning process.
- Support for QLoRA instruction fine-tuning, as well as LoRA fine-tuning.
- Support for most mainstream open-source large models, specifically for potential Code-LLMs, such as Code-LLaMA, Starcoder, Codegeex2, Qwen, GPT-Neox and more.
- Support for weight merging between LoRA adaptor and base models, making inference more convenient.
- Release of 2 high-quality code-related instruction fine-tuning datasets: CodeFuse13B-evol-instruction-4K, CodeFuse-CodeExercise-Python-27k.
- Release of 2 model weights in [CodeFuse series model weights](https://huggingface.co/codefuse-ai).


## Requirements
Firstly, you need to make sure you have installed CUDA(>=11.4, we have used 11.7) related, and torch(2.0.1) successfully.

Then we provide init_env.sh to install required packages:
```bash
sh init_env.sh
```
If you need flash attention, please install via reference https://github.com/Dao-AILab/flash-attention


## Training
🚀 [Huggingface accelerate + deepspeed Codebase for MFT(Multi-task Finetuning)](./mft_peft_hf/README.md)

🚀 [Atorch Codebase for MFT(Multi-task Finetuning)](./mft_atorch/README.md)


## Models

We are releasing the 2 fowllowed CodeLLMs trianed by MFTCoder on Hugging Face.

| Model                                                                | Base Model         | Num of examples trained | Batch Size | Seq Length | Licence |
|----------------------------------------------------------------------|--------------------|-------------------------|------------|------------|-----|
| [🔥🔥🔥 CodeFuse-CodeLlama-34B](https://huggingface.co/codefuse-ai/) | CodeLlama-34b-Python | 600k                    | 80         | 4096       |     | 
| [🔥 CodeFuse-13B](https://huggingface.co/codefuse-ai/)               | CodeFuse-13B       | 66k                     | 64         | 4096       |  |



## Datasets
We are releasing the 2 fowllowed code-related instruction datasets on Hugging Face.

| Dataset                                                                | Introduction           | Licence |
|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| [⭐ Evol-instruction-66k](https://huggingface.co/datasets/)    | Based on open-evol-instruction-80k, filter out low-quality, repeated, and similar instructions to HumanEval, thus get high-quality code instruction dataset. |     | 
| [⭐ CodeExercise-Python-27k](https://huggingface.co/datasets/) | python code exercise instruction dataset generated by chatgpt                                                                                                |  |

