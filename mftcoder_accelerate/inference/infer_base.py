# @author Chaoyu Chen
# @date 2024/11/12
# @module infer_base.py
"""HF InferenceBase"""

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
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

from utils import (
    print_args,
    get_line_count,
    stream_jsonl,
    batch_stream_jsonl,
    flatten_batch_stream,
    write_jsonl,
)


class HFInferenceBase:
    def __init__(self):
        # load model and tokenizer
        self.tokenizer = None
        self.model = None

    def _load_model_tokenizer(
        self,
        model_cls,
        model_path,
        torch_dtype=torch.bfloat16,
        peft_path=None,
        quantization=None,
        eos_token=None,
        **kwargs,
    ):
        """
        load model and tokenizer by transfromers
        """
        # pass if ckpt already loaded
        if self.model and self.tokenizer:
            print("CKPT: {} already loaded".format(model_path))
            return

        # load tokenizer first
        print("LOAD CKPT and Tokenizer: {}".format(model_path))
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"

        config, unused_kwargs = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, return_unused_kwargs=True
        )
        print("unused_kwargs:", unused_kwargs)
        print("config input:\n", config)

        # eos token parsing
        if eos_token:
            eos_token = eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            print(f"eos_token {eos_token} from user input")
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
            print(f"Initial eos_token_id {tokenizer.eos_token_id} from tokenizer")
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)
        elif hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
            print(f"Initial eos_token {tokenizer.eos_token} from tokenizer")
            eos_token = tokenizer.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        elif hasattr(config, "eos_token_id") and config.eos_token_id:
            print(f"Initial eos_token_id {config.eos_token_id} from config.json")
            eos_token_id = config.eos_token_id
            eos_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)
        elif hasattr(config, "eos_token") and config.eos_token:
            print(f"Initial eos_token {config.eos_token} from config.json")
            eos_token = config.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)
        else:
            raise ValueError("No available eos_token or eos_token_id.")

        try:
            tokenizer.eos_token = eos_token
            tokenizer.eos_token_id = eos_token_id
            # set pad_token to be same as eos_token, it is ok because is will be masked out.
            tokenizer.pad_token = eos_token
            tokenizer.pad_token_id = eos_token_id
        except:
            print(f"[WARNING]Cannot set tokenizer.eos_token")

        print(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
        print(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")
        print(type(tokenizer))

        base_model = model_cls.from_pretrained(
            model_path,
            config=config,
            load_in_8bit=(quantization == "8bit"),
            load_in_4bit=(quantization == "4bit"),
            device_map="auto",
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs,
        )

        if peft_path:
            print("Loading PEFT MODEL...")
            model = PeftModel.from_pretrained(base_model, peft_path)
        else:
            print("Loading Original MODEL...")
            model = base_model

        model.eval()

        print("===============================MODEL Configs=============================")
        print(model.config)
        print("=========================================================================")
        print("==============================MODEL Archetecture=========================")
        print(model)
        print("=========================================================================")

        self.model = model
        self.tokenizer = tokenizer

    def load_model_tokenizer(self, args):
        raise NotImplementedError

    def handler(self, dataloader, args):
        """consume batch dataloader, yield result batch"""
        raise NotImplementedError

    @staticmethod
    def check_args(args):
        if not os.path.exists(args.model_path):
            raise FileNotFoundError("Model path {} not exists!".format(args.model_path))

        if not os.path.exists(args.data_file):
            raise FileNotFoundError("Data file {} not exists!".format(args.data_file))

        os.makedirs(args.output_dir, exist_ok=True)

    def prepare_args(self, args):
        raise NotImplementedError("You need assign args.output_path for result writing")

    def run(self, args):
        self.check_args(args)
        self.prepare_args(args)
        print_args(args)

        # load model if not yet loaded
        self.load_model_tokenizer(args)

        # get dataloader
        total_num = get_line_count(args.data_file)
        stream = stream_jsonl(args.data_file)
        dataloader = batch_stream_jsonl(stream, args.batch_size)

        # handel batch dataloader
        batch_res_stream = self.handler(dataloader, args)
        res_stream = flatten_batch_stream(batch_res_stream)

        # write results into output_path streamingly
        write_jsonl(args.output_path, res_stream, total=total_num)
