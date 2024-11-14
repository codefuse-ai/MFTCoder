# @author Chaoyu Chen
# @date 2024/11/13
# @module gen.py
"""Generation Demo"""
from typing import Iterable, Dict, List
import gzip
import json
import os
import argparse
import time
import sys
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


class HFGenerationInfer(HFInferenceBase):
    def __init__(self):
        super().__init__()

    def load_model_tokenizer(self, args):
        """
        load generation model and tokenizer using self._load_model_tokenizer
        """
        self._load_model_tokenizer(AutoModelForCausalLM, args.model_path, peft_path=args.peft_path)

    def handler(self, dataloader, args):
        for batch in dataloader:
            prompts = [
                self.tokenizer.apply_chat_template([{"role": "user", "content": sample["prompt"]}], tokenize=False)
                for sample in batch
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

            # print(inputs)
            print("=================Prompts and Generations===================")

            outputs = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            gen_text = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            num_return_sequences = len(gen_text) // len(prompts)
            for i in range(len(prompts)):
                print("=========" * 10)
                print(f"Prompt:\n{prompts[i]}")
                batch[i]["generations"] = gen_text[
                    i * num_return_sequences : i * num_return_sequences + num_return_sequences
                ]
                for j in range(num_return_sequences):
                    print(f"Generation {j+1}/{num_return_sequences}:\n{gen_text[i * num_return_sequences + j]}")
                    # print(f"Outputs ids:\n{outputs[i]}")
                    sys.stdout.flush()
            yield batch

    def prepare_args(self, args):
        name = args.data_file.split("/")[-1].replace(".jsonl", "") + "-GEN"
        args.output_path = os.path.join(args.output_dir, f"{name}.jsonl")


def get_args():
    parser = argparse.ArgumentParser(description="Generation args.")
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
    parser.add_argument("--num_return_sequences", type=int, default=20, help="pass1:20,pass10:20,pass100:100")
    parser.add_argument("--num_beams", type=int, default=1, help="beam1, beam3, beam5, beam7")
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--peft_path", type=str, default="", help="peft path：None")
    parser.add_argument("--eos_token", type=str, default=None, help="eos token")
    args = parser.parse_args()
    return args


def main(args):
    st = time.time()
    runner = HFGenerationInfer()
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
