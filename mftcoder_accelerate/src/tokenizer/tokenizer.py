"""
# @author Chaoyu Chen
# @date 2023/6/19
"""


import numpy as np
from typing import List, Union
from utils.common_utils import print_rank_0
from transformers import AutoTokenizer
from tokenizer.chat_template import MFTCoder_template


def build_tokenizer(args):
    """Initialize tokenizer."""
    print_rank_0(f"> building {args.tokenizer_type} tokenizer ...")
    # Select and instantiate the tokenizer.
    if args.tokenizer_type.lower() == "AutoTokenizer".lower():
        assert args.pretrained_model_path is not None
        # tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, trust_remote_code=True, use_fast=False, legacy=False)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, trust_remote_code=True)
        tokenizer.eod_id = tokenizer.convert_tokens_to_ids(args.eos_token)
        tokenizer.pad_id = tokenizer.convert_tokens_to_ids(args.pad_token)
        try:
            tokenizer.eos_token = args.eos_token
            tokenizer.pad_token = args.pad_token
        except:
            print(f"[WARNING]Cannot set tokenizer.eos_token")
        print_rank_0(f"Tokenizer: {type(tokenizer)}")
        print_rank_0(f"Length of tokenizer: {len(tokenizer)}")
        print_rank_0(f"build_tokenizer PAD id: {tokenizer.pad_id}, EOD id: {tokenizer.eod_id}")
        print_rank_0(f"build_tokenizer PAD token : {args.pad_token}, EOD token: {args.eos_token}")
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
