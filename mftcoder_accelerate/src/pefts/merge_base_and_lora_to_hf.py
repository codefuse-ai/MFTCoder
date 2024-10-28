"""
# @author Chaoyu Chen
# @date 2023/10/19

Merge base and lora adaptor
"""

import os
import sys
import time
import shutil
import argparse
from typing import List
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from peft import PeftModel

# insert src as import path
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_dir)
print("In merge_base_and_lora_to_hf.py, sys path:", sys.path)

from tokenizer import init_tokenizer


def copy_tokenizer_files(mode_path: str, files_list: List[str], save_path: str):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for filename in files_list:

        src_file = os.path.join(mode_path, filename)

        if os.path.exists(src_file):
            dest_file = os.path.join(save_path, filename)

            shutil.copy(src_file, dest_file)
            print(f"Copied {filename} to {save_path}")
        else:
            print(f"File {filename} does not exist in {mode_path}")


if __name__ == "__main__":

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_or_path", type=str, default=None)
    parser.add_argument("--adaptor_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--merged_output_path", type=str, default=None)
    args = parser.parse_args()

    model_path = args.base_model_or_path
    lora_adapter = args.adaptor_path
    model_type = args.model_type
    save_path = args.merged_output_path

    t0 = time.time()

    tokenizer = init_tokenizer(args.base_model_or_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float32,
        return_dict=True,
        device_map="auto",
    )
    print("--------------------------------------Base Model--------------------------------------------")
    print(base_model)
    print("--------------------------------------------------------------------------------------------")

    print("-----------------------------------Base Model Config----------------------------------------")
    print(base_model.config)
    print("--------------------------------------------------------------------------------------------")

    # merge, save model and tokenizer
    model_to_merge = PeftModel.from_pretrained(base_model, lora_adapter)
    merged_model = model_to_merge.merge_and_unload()
    # merged_model.to(torch.bfloat16)

    print("---------------------------------Merged Model Config----------------------------------------")
    print(merged_model.config)
    print("--------------------------------------------------------------------------------------------")
    merged_model.save_pretrained(save_path)

    print("-------------------------------------Tokenizer----------------------------------------------")
    print(tokenizer)
    print("--------------------------------------------------------------------------------------------")
    if model_type.lower() == "deepseek":
        copy_tokenizer_files(
            model_path,
            ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
            save_path,
        )
    else:
        tokenizer.save_pretrained(save_path)

    print(f"Merge finised: {save_path} saved, Cost {time.time() - t0:.2f}s")
