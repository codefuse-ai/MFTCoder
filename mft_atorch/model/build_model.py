import os
import torch
import sys
sys.path.append("..")
from utils.common_utils import get_model_params_num
from transformers import (  # noqa: E402
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast
)
from .gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from .gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from .gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
)
from utils.common_utils import print_rank_0, is_old_version
from tokenizer import build_tokenizer
from tokenizer.tokenizer import HFTokenizer

import peft
from peft.tuners.lora import LoraLayer
from model.peft.utils import prepare_model_for_kbit_training
from peft import (  # noqa
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model
)
import model.peft.modeling_peft # noqa
from model.peft.tuner import AdaLoraConfig

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
try:
    import bitsandbytes as bnb # noqa
except ImportError:
    bnb = None
from packaging import version


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def setup_model(args, logger, use_cache=False):
    # Load pretrained model and tokenizer

    if args.pretrained_model_path:
        if args.model_type == 'gpt_neox':
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.pretrained_model_path)
            tokenizer.eod_token = "<|endoftext|>"
            tokenizer.pad_token = "<|pad|>"
            tokenizer.sop_token = "<|endoftext|>"
            tokenizer.eop_token = "<|endoftext|>"
            tokenizer.eod_id = tokenizer.convert_tokens_to_ids(tokenizer.eod_token)
            tokenizer.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
            print_rank_0(f'tokenizer {tokenizer.eod_token} id: {tokenizer.eod_id}')
            print_rank_0(f'tokenizer {tokenizer.pad_token} id: {tokenizer.pad_id}')
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_path."
        )
    
    if args.model_type == 'gpt_neox':
        auto_config = GPTNeoXConfig
        auto_model_class = GPTNeoXForCausalLM
    else:
        auto_config = AutoConfig
        auto_model_class = AutoModelForCausalLM

    # with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
    if args.pretrained_model_path:
        logger.info("Training model from checkpoint")
        config = auto_config.from_pretrained(args.pretrained_model_path)
        if args.peft_type != "qlora":
            model = auto_model_class.from_pretrained(args.pretrained_model_path, trust_remote_code=True).cuda()
        # TODO: qlora
    else:
        logger.info("Training model from scratch")
        if args.model_type == 'gpt_neox':
            config = GPTNeoXConfig.from_json_file(args.config_path + '/config.json')
            model = GPTNeoXForCausalLM._from_config(config)
        else:
            config = AutoConfig.from_json_file(args.config_path + '/config.json')
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    print_rank_0('embedding size: ' + str(embedding_size))
    print_rank_0('vocab size: ' + str(tokenizer.vocab_size))
    if tokenizer.vocab_size > embedding_size:
        model.resize_token_embeddings(tokenizer.vocab_size)
    print_rank_0('resize embedding size: ' + str(model.get_input_embeddings().weight.shape[0]))
    
    print_rank_0(config)
    num_params = get_model_params_num(model)
    print_rank_0("num_params of this model:", num_params)
    args.total_model_param = num_params
    args.hidden_size = config.hidden_size
    args.num_hidden_layers = config.num_hidden_layers
    args.vocab_size = tokenizer.vocab_size
    print_rank_0(f'hidden size: {args.hidden_size}')
    print_rank_0(f'num hidden layers: {args.num_hidden_layers}')
    print_rank_0(f'vocab size: {args.vocab_size}')

    if args.peft_type:
        if args.peft_type in ['lora', 'qlora']:
            target_modules = None
            # TODO: qlora
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            print_rank_0(f'target modules: {target_modules}')
            peft_config = LoraConfig(
                task_type=TaskType.ANT_CAUSAL_LM,
                inference_mode=False,
                r=96,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=target_modules,
            )
        logger.info(
            f"Load Peft {args.peft_type} model ......")
        if args.checkpoint_activations and args.peft_type in ["lora", "qlora"]:
            # Make Lora and gradient checkpointing compatible
            # https://github.com/huggingface/peft/issues/137
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        logger.info(
            f"Reduce trainalbe params:\n")
        model.print_trainable_parameters()

    return model, config, tokenizer
