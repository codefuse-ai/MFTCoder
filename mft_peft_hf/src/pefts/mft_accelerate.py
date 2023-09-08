"""
# @author Chaoyu Chen
# @date 2023/6/19
# @module mft_accelerate.py

Hugging face accelerate + deepspeed zero stage2 + DP
QLoRA + MFT entry
"""

import gc
import os
import sys
import argparse
import math
import logging
import json
import time
import transformers
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from datasets import Dataset
import datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModel,
)
from accelerate import Accelerator
from accelerate.logging import get_logger

sys.path.append("..")
from data.gpt2_multi_task_dataset import load_dataset_from_jsonl
from utils.common_utils import generate_task_id, TASK2ID, ID2TASK
from train_utils import accelerate_train
from model_mapping import MODEL_TYPES, QLORA_TARGETING_MODULES, MODEL_SPECIAL_TOKENS
logger = get_logger(__name__)


def get_task_mask(args, task_id):
    task_num = len(TASK2ID)
    task_mask = torch.zeros(task_id.shape[0], task_num)
    task_mask[torch.arange(task_id.size(0)).unsqueeze(1), task_id] = 1

    return task_mask


def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    attention_mask = torch.ones((batch_size, seq_length), device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data).clone()

    return attention_mask, position_ids


@dataclass
class DataCollatorForMFTDataset(object):
    args: None

    # tokenizer: None

    def __call__(self, instances):
        input_ids, loss_mask, weights, task_id = tuple(
            [instance[key] if key in instance else None for instance in instances] for key in
            ("input_ids", "loss_mask", "weight", "task_id"))

        result_batch = {}
        '''
        outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                # labels=(batch['labels'], batch['loss_mask'], batch['task_mask']),
                # labels=(batch['labels'], batch['loss_mask']),
                position_ids=batch['position_ids'],
            )
        '''
        
        # if loss_mask is not None:
        loss_mask = torch.tensor(np.array(loss_mask))
        if self.args.use_dynamic_padding:
            last_one_pos = (loss_mask == 1).long().cumsum(dim=1).argmax(dim=1)
            # get last non-padding position
            max_pos = last_one_pos.max().item() + 1
        else:
            max_pos = loss_mask.shape[-1]
        result_batch['loss_mask'] = loss_mask.float()[:, 1:max_pos].contiguous()

        input_ids = torch.tensor(np.array(input_ids)).long()
        # print(f"shape of input_ids: {input_ids.shape}")
        result_batch['input_ids'] = input_ids[:, :max_pos-1].contiguous()
        result_batch['labels'] = input_ids[:, 1:max_pos].contiguous()

        # Get the masks and position ids.
        result_batch['attention_mask'], result_batch['position_ids'] = get_ltor_masks_and_position_ids(
            data=result_batch['input_ids'])

        if task_id is not None:
            task_id = torch.tensor(np.array(task_id))
            result_batch['task_mask'] = get_task_mask(self.args, task_id)  # bsz * task_num
            result_batch['task_id'] = task_id

        return result_batch


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if not is_gptq_quantized:
        # cast all non INT8 parameters to fp32
        for param in model.parameters():
            if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                param.data = param.data.to(torch.float32)

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


def pprint_args(args, accelerator):
    message = '\n'.join([f'{k:<30}:   {v}' for k, v in vars(args).items()])
    accelerator.print('====' * 30)
    accelerator.print(message)
    accelerator.print('====' * 30)
    accelerator.print("GPU: {}".format(torch.cuda.current_device()))


def get_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default='./train_config.json')

    parser.add_argument("--data_paths", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--tb_dir", type=str, default='')
    parser.add_argument("--pretrained_model_path", type=str, default='')

    return parser.parse_args()


def main():
    t0 = time.time()
    parser = get_configs()
    train_config_file = parser.train_config
    # get configs
    with open(train_config_file, 'r') as f:
        train_config = json.load(f)

    args = argparse.Namespace(**train_config)
    
    # get eos token和 pad token
    args.eos_token = MODEL_SPECIAL_TOKENS[args.model_type]['eos_token']
    args.pad_token = MODEL_SPECIAL_TOKENS[args.model_type]['pad_token']

    # refactor args
    if parser.data_paths:
        args.data_paths = parser.data_paths
    if parser.output_dir:
        args.output_dir = parser.output_dir
    if parser.tb_dir:
        args.tb_dir = parser.tb_dir
    if parser.pretrained_model_path:
        args.pretrained_model_path = parser.pretrained_model_path
        args.vocab_file = parser.pretrained_model_path

    if args.peft_type == 'qlora' and args.quantization != '4bit' and args.quantization != '8bit':
        print(f"[WARNING]peft_type is qlora but quantization is not 4bit or 8bit, setting it to 4bit")
        args.quantization = '4bit'
    
    # define accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    pprint_args(args, accelerator)

    # logger
    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    
    if args.seed is not None:
        set_seed(args.seed)

    # get world_size and global_rank
    # args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    # global_rank = int(os.environ.get('RANK', 0))
    args.world_size = accelerator.num_processes
    global_rank = accelerator.process_index
    print(f'world_size: {args.world_size}, global_rank: {global_rank}, local_rank: {accelerator.local_process_index}')
    
    # TASK2ID, ID2TASK
    generate_task_id(args.data_paths)
    # # multi task blendable dataset(sharded)
    train_dataset, valid_dataset = load_dataset_from_jsonl(args, shard_data=True, world_size=args.world_size,
                                                           global_rank=global_rank, local_rank=accelerator.local_process_index)
    t1 = time.time()
    logger.info(f"dataset loading time: {t1 - t0:.4f}")

    # memory
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f"{free_in_GB - 2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    accelerator.print("max memory: ", max_memory, n_gpus)

    # peft config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=QLORA_TARGETING_MODULES[args.model_type],
       
    )

    # creating base model
    ModelClass = MODEL_TYPES[args.model_type]
    model = ModelClass.from_pretrained(
        args.pretrained_model_path,
        # max_memory=max_memory,
        # trust_remote_code=True,
        load_in_8bit=(args.quantization=='8bit'),
        load_in_4bit=(args.quantization=='4bit'),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=args.low_cpu_mem_usage,  # not for zero3
        use_safetensors=False,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=(args.quantization=='4bit'),
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if args.quantization=='4bit' else None,
    )

    accelerator.print("load in 8bit: ", args.quantization=='8bit')
    accelerator.print("load in 4bit: ", args.quantization=='4bit')
    if args.peft_type == 'lora':
        # for name, param in model.named_parameters():
        #     # cast layer norm in fp32 for stability
        #     if param.ndim == 1 and "layer_norm" in name:
        #         param.data = param.data.to(torch.float32)
        #     if "lm_head" in name:
        #         param.data = param.data.to(torch.float32)
        model.gradient_checkpointing_enable()

    elif args.peft_type == 'qlora':
        # prepare base model for 8bit or 4bit model(cast non-8bit or non-4bit layers to fp32)
        model = prepare_model_for_kbit_training(model)
        logging.info(f"device map: {model.hf_device_map}")

    # Potentially load in the lora from a previous save
    if not args.resume_from_checkpoint:
        model = get_peft_model(model, peft_config)
    else:

        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        # accelerator.load_state(args.resume_from_checkpoint)
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)

    t2 = time.time()
    if accelerator.is_main_process:
        logging.info(f"model loading time: {t2 - t1:.4f}")
    model.print_trainable_parameters()
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    model.config.use_logn_attn = False # special for qwen model
    accelerator.print(model.config)

    # dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=DataCollatorForMFTDataset(args),
        batch_size=args.per_device_train_batch_size, pin_memory=True, drop_last=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=DataCollatorForMFTDataset(args), batch_size=args.per_device_eval_batch_size,
        pin_memory=True, drop_last=True
    )

    from deepspeed.ops.adam import FusedAdam as Adam

    adam_optimizer = Adam
    optimizer = adam_optimizer(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, train_dataloader, valid_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, valid_dataloader, optimizer, lr_scheduler
    )
    print(model.device)
    accelerator.print(model)
    # accelerator.print(model.config)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # zero 3 flag
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
    accelerator.print(f"is_ds_zero_3: {is_ds_zero_3}")

    # Train!
    accelerate_train(accelerator, model, train_dataloader, valid_dataloader, optimizer, lr_scheduler, num_update_steps_per_epoch, len(train_dataset), args)


if __name__ == "__main__":
    main()
