# coding=utf-8
# @author Chaoyu Chen
# @date 2024/11/12
# @module ppl.py
"""PPL demo"""
from typing import Iterable, Dict, List
import gzip
import json
import os
import argparse
import time
from tqdm import tqdm
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel
from infer_base import HFInferenceBase

MFTCoder_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = false %}"
    "{% endif %}"
    "{% for message in loop_messages %}"  # Loop over all non-system messages
    "{% if (message['role'] == 'user' or message['role'] == 'human') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
    "{% set content = '<s>system\n' + system_message + '\n' %}"
    "{% else %}"
    "{% set content = '' %}"
    "{% endif %}"
    "{% if message['role'] == 'user' or message['role'] == 'human' %}"
    "{{ content + '<s>human\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' or message['role'] == 'bot' %}"
    "{{ '<s>bot\n' + message['content'] + '\n' +  eos_token + '\n'}}"
    "{% else %}"
    "{{ raise_exception('Only user/human and assistant/bot roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<s>bot\n' }}"
    "{% endif %}"
)


class HFPerplexityPairInfer(HFInferenceBase):
    def __init__(self):
        super().__init__()

    def load_model_tokenizer(self, args):
        """
        load ppl model and tokenizer using self._load_model_tokenizer
        """
        self._load_model_tokenizer(AutoModelForCausalLM, args.model_path)

    def handler(self, dataloader, args):
        for batch in dataloader:
            for key in ["chosen", "rejected"]:
                # apply chat template on chatml messages
                input_text = [self.tokenizer.apply_chat_template(sample[key], tokenize=False) for sample in batch]

                # tokenization
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to("cuda")
                # print(inputs)

                # ppl computing
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = inputs["input_ids"]
                print(input_ids.shape)
                # forward outputs
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    )
                    logits = outputs.logits

                # loss fuction
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction="none")
                # shift logits and labels to make them aligned
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # get mean loss of all tokens of each sample
                losses = loss.view(len(input_text), -1).mean(dim=1)
                # save loss and ppl
                for i, individual_loss in enumerate(losses):
                    batch[i][f"{key}_loss"] = individual_loss.item()
                    batch[i][f"{key}_ppl"] = torch.exp(individual_loss).item()
            yield batch

    def prepare_args(self, args):
        name = args.data_file.split("/")[-1].replace(".jsonl", "") + "-PPL"
        args.output_path = os.path.join(args.output_dir, f"{name}.jsonl")


def get_args():
    parser = argparse.ArgumentParser(description="PPL args.")
    parser.add_argument(
        "--model_path",
        type=str,
        help="huggingface model path",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        help="data file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    return args


def main(args):
    st = time.time()
    runner = HFPerplexityPairInfer()
    runner.run(args)

    print("{} finish in: {:.4f} Minutes {}".format("+++" * 10, (time.time() - st) / 60, "+++" * 10))


if __name__ == "__main__":
    print("-----------" * 10)
    print("-----------" * 10)
    print("-----------" * 10)
    main(get_args())
    print("-----------" * 10)
    print("-----------" * 10)
    print("-----------" * 10)
