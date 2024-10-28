"""
# @author Chaoyu Chen
# @date 2023/6/19
"""

import numpy as np
from typing import List, Union
from utils.common_utils import print_rank_0
from transformers import AutoTokenizer, AutoConfig
from tokenizer.chat_template import MFTCoder_template


def init_tokenizer(path):
    """
    Init a Huggingface tokenizer, parsing eos_token from the tokenizer_config then config.
    Set pad_token same as eos_token for easy life.
    :param path: model path or tokenizer path
    :return: Tokenizer (TokenizerFast is preferred)
    """
    # tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False, legacy=False)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config, unused_kwargs = AutoConfig.from_pretrained(path, trust_remote_code=True, return_unused_kwargs=True)

    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
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
        raise ValueError(
            "No available eos_token or eos_token_id, please provide eos_token by params or eos_token_id by config.json"
        )
    try:
        tokenizer.eos_token = eos_token
        tokenizer.eos_token_id = eos_token_id
        # set pad_token to be same as eos_token, it is ok because is will be masked out.
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
    except:
        print(f"[WARNING]Cannot set tokenizer.eos_token")

    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False
    tokenizer.chat_template = MFTCoder_template
    print_rank_0(f"Tokenizer: {type(tokenizer)}")
    print_rank_0(f"Length of tokenizer: {len(tokenizer)}")
    print_rank_0(f"build_tokenizer pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")
    print_rank_0(f"build_tokenizer pad_token : {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")

    return tokenizer


def build_tokenizer(args):
    """Initialize tokenizer."""
    print_rank_0(f"> building {args.tokenizer_type} tokenizer ...")
    # Select and instantiate the tokenizer.
    if args.tokenizer_type.lower() == "AutoTokenizer".lower():
        assert args.pretrained_model_path is not None
        tokenizer = init_tokenizer(args.pretrained_model_path)
    else:
        raise NotImplementedError(f"{args.tokenizer_type} tokenizer is not implemented.")

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size thus it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.model_parallel_size
    while (after % multiple) != 0:
        after += 1
    print_rank_0(
        " > padded vocab (size: {}) with {} dummy tokens "
        "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after)
    )

    return after
