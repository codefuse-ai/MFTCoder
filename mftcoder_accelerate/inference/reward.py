# @author Chaoyu Chen
# @date 2024/11/12
# @module reward.py
"""Reward Demo"""

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
    AutoModelForSequenceClassification,
)

from infer_base import HFInferenceBase


class HFRewardPairInfer(HFInferenceBase):
    def __init__(self):
        super().__init__()

    def load_model_tokenizer(self, args):
        """
        load ppl model and tokenizer using self._load_model_tokenizer
        """
        self._load_model_tokenizer(AutoModelForCausalLM, args.model_path)

    def handler(self, dataloader, args):
        correct_num = 0
        total_num = 0
        for batch in dataloader:
            try:
                for key in ["chosen", "rejected"]:
                    # apply chat template on chosen and rejected chatml messages
                    input_text = [
                        self.tokenizer.apply_chat_template(sample[key], tokenize=False, add_generation_prompt=False)
                        for sample in batch
                    ]

                    # tokenization
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=4096,
                        add_special_tokens=False,
                    ).to("cuda")

                    # prepare input_ids and attention_mask
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]

                    # generate score
                    response_token_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                    # print(response_token_ids["scores"], response_token_ids["scores"][0].shape)
                    rewards = response_token_ids["scores"][0][:, 0].reshape(-1)
                    for i, reward in enumerate(rewards):
                        batch[i][f"{key}_reward"] = reward.item()
                        # print(reward.item())
                for sample in batch:
                    total_num += 1
                    # print(sample["chosen_reward"], sample["rejected_reward"])
                    if sample["chosen_reward"] > sample["rejected_reward"]:
                        correct_num += 1
                print(f"correct {correct_num} of total {total_num}")
                torch.cuda.empty_cache()
                yield batch
            except Exception as e:
                print(f"[ERROR] {e}")
                continue
            # break
        print(f"correct {correct_num} of total {total_num}")

    def prepare_args(self, args):
        name = args.data_file.split("/")[-1].replace(".jsonl", "") + "-REWARD"
        args.output_path = os.path.join(args.output_dir, f"{name}.jsonl")


def get_args():
    parser = argparse.ArgumentParser(description="REWARD args.")
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

    return parser.parse_args()


def main(args):
    st = time.time()
    runner = HFRewardPairInfer()
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
