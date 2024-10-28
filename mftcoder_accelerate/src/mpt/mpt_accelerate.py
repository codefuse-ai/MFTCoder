"""
# @author Chaoyu Chen
# @date 2024/6/1
# @module mpt_accelerate.py

Accelerate + DeepSpeed + Full-parameter + Multi-task + Pre-training/Continue Training/Finetuning

Entry
"""

import os
import sys
import argparse
import math
import logging
import json
import time
from tqdm.auto import tqdm
import transformers
import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from datasets import Dataset
import datasets
from torch.utils.data import DataLoader
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
    get_scheduler,
)

from accelerate import Accelerator, DistributedType, FullyShardedDataParallelPlugin, DataLoaderConfiguration
from accelerate.logging import get_logger
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from transformers.optimization import Adafactor

# insert src as import path
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_dir)

from tokenizer import build_tokenizer
from data.multi_task_dataset import load_dataset_from_jsonl, compile_helper
from data.data_utils import load_dataset_from_bin
from utils.common_utils import print_rank_0, generate_task_id, TASK2ID, ID2TASK
from mpt.mpt_trainer import MptTrainer
from mpt.mpt_arguments import MptTrainArgs
from utils.model_mapping import MODEL_TYPES, SUPPORT_IN_TRANSFORMERS


logger = get_logger(__name__)


def get_task_mask(args, task_id):
    task_num = len(TASK2ID)
    task_mask = torch.zeros(task_id.shape[0], task_num)
    task_mask[torch.arange(task_id.size(0)).unsqueeze(1), task_id] = 1

    return task_mask


def get_attention_mask_and_position_ids(data):
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

    def __call__(self, instances):
        (input_ids, loss_mask, weights, task_id) = tuple(
            [instance.get(key, None) for instance in instances]
            for key in ("input_ids", "loss_mask", "weight", "task_id")
        )

        result_batch = {}
        """
        outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    # labels=(batch['labels'], batch['loss_mask'], batch['task_mask']),
                    # labels=(batch['labels'], batch['loss_mask']),
                    position_ids=batch['position_ids'])
        """

        # if loss_mask is not None:
        loss_mask = torch.tensor(np.array(loss_mask)).long()
        last_one_pos = (loss_mask == 1).long().cumsum(dim=1).argmax(dim=1)
        if self.args.use_dynamic_padding:
            # get last non-padding position
            max_pos = last_one_pos.max().item() + 1
        else:
            max_pos = loss_mask.shape[-1]

        if self.args.tokenize_mode == "sst" and self.args.padding_mode == "pack":
            # 兼容sst + pack tokenization, 最后一位是脏数据，需要去掉
            result_batch["loss_mask"] = loss_mask.float()[:, 1 : max_pos - 1].contiguous()
            input_ids = torch.tensor(np.array(input_ids)).long()
            result_batch["input_ids"] = input_ids[:, : max_pos - 2].contiguous()
            result_batch["labels"] = input_ids[:, 1 : max_pos - 1].contiguous()
        else:
            result_batch["loss_mask"] = loss_mask.float()[:, 1:max_pos].contiguous()
            input_ids = torch.tensor(np.array(input_ids)).long()
            # print(f"shape of input_ids: {input_ids.shape}")
            result_batch["input_ids"] = input_ids[:, : max_pos - 1].contiguous()
            result_batch["labels"] = input_ids[:, 1:max_pos].contiguous()

        # Get the masks and position ids.

        # if you want to be compatible with non-gpt models, something you can do here
        if self.args.model_type in ["antglm"]:
            (result_batch["attention_mask"], result_batch["position_ids"]) = get_attention_mask_and_position_ids(
                data=result_batch["input_ids"]
            )
        elif self.args.model_type in ["mixtral", "mtx-qwen2", "qwen2_moe"]:
            batch_size, seq_length = result_batch["input_ids"].shape
            # bsz * seq_length
            range_tensor = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1)
            # attention_mask for padding tokens
            attention_mask = (range_tensor <= last_one_pos.reshape(batch_size, 1)).long()
            result_batch["attention_mask"], result_batch["position_ids"] = attention_mask, None
        else:
            # For decoder-only models, transformers will create them.
            result_batch["attention_mask"], result_batch["position_ids"] = None, None

        if task_id is not None:
            task_id = torch.tensor(np.array(task_id))
            result_batch["task_mask"] = get_task_mask(self.args, task_id)  # bsz * task_num
            result_batch["task_id"] = task_id

        return result_batch


def pprint_args(args, accelerator):
    # 计算所有键的最大字符串长度
    max_key_length = max(len(str(key)) for key in vars(args).keys())

    message = ""
    message += "====" * 60 + "\n"
    message += "\n".join([f"{k:<{max_key_length}} : {v}" for k, v in vars(args).items()]) + "\n"
    message += "====" * 60 + "\n"
    accelerator.print(message)
    accelerator.print("GPU: {}".format(torch.cuda.current_device()))


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)

    parser.add_argument("--data_paths", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tb_dir", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--distributed_type", type=str, default="deepspeed")

    parsed = parser.parse_args()
    # get json configs
    with open(parsed.train_config, "r") as f:
        train_config = json.load(f)

    # parse args from cofig.json
    # args = argparse.Namespace(**train_config)
    args = MptTrainArgs(**train_config)

    # override args by cli arguments
    if parsed.data_paths:
        args.data_paths = parsed.data_paths
    if parsed.output_dir:
        args.output_dir = parsed.output_dir
    if parsed.tb_dir:
        args.tb_dir = parsed.tb_dir
    if parsed.pretrained_model_path:
        args.pretrained_model_path = parsed.pretrained_model_path
        args.vocab_file = parsed.pretrained_model_path
    if parsed.micro_batch_size:
        args.per_device_train_batch_size = parsed.micro_batch_size
        args.per_device_eval_batch_size = parsed.micro_batch_size
    if parsed.model_type:
        args.model_type = parsed.model_type

    args.distributed_type = parsed.distributed_type

    # refactor args

    args.vocab_file = args.pretrained_model_path

    args.data_weights = "[" + ",".join(["1."] * len(args.data_paths[1:-1].split(","))) + "]"

    # generate TASK2ID, ID2TASK
    generate_task_id(args.data_paths)

    if args.weighted_loss_mode == "selfpaced":
        args.task_weights = [1.0] * len(ID2TASK)
    elif args.task_weights is not None:
        args.task_weights = [float(wt) for wt in args.task_weights[1:-1].split(",")]
        assert len(args.task_weights) == len(ID2TASK), f"length of task_weights must equal to length of data_paths"
    else:
        args.task_weights = [1.0] * len(ID2TASK)

    return args


def main():
    t0 = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "false"
    # get input args, set TASK2ID, ID2TASK, refactor args
    args = prepare_args()

    # fix randomness
    if args.seed is not None:
        set_seed(args.seed)

    # define accelerator
    init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.init_timeout_seconds))
    
    if args.distributed_type and args.distributed_type.lower() == "fsdp":
        fsdp_plugin = FullyShardedDataParallelPlugin(
            # state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            # optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            limit_all_gathers=True,
            sync_module_states=True,
            use_orig_params=True,
            cpu_offload=False,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fsdp_plugin=fsdp_plugin,
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
            kwargs_handlers=[init_process_kwargs],
        )
    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
            kwargs_handlers=[init_process_kwargs],
        )

    # print key infos
    accelerator.print("In mft_accelerate.py, sys path:", sys.path)
    accelerator.print(f"transformers.__version__: {transformers.__version__}")

    # get world_size
    args.world_size = accelerator.num_processes

    # backup args
    pprint_args(args, accelerator)
    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(args.dict(), f, indent=2)

    # deal with autoresume, args.resume_from_checkpoint prior to auto_resume from latest
    latest = None
    if os.path.exists(os.path.join(args.output_dir, "latest")):
        with open(os.path.join(args.output_dir, "latest"), "r") as fl:
            latest = json.load(fl)
        accelerator.print(f"[INFO] Existing latest: {latest}")

    if args.auto_resume and args.resume_from_checkpoint is None and latest:
        args.resume_from_checkpoint = latest["latest_ckpt"]

    # logger
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s][%(name)s]%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
        # compile Cpp helper
        compile_helper()
        time.sleep(10)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # get global_rank and local rank for current process
    global_rank = accelerator.process_index
    local_rank = accelerator.local_process_index
    print(f"world_size: {args.world_size}, global_rank: {global_rank}, local_rank: {local_rank}")

    # TASK2ID, ID2TASK
    # generate_task_id(args.data_paths)

    # multi task blendable dataset(sharded)
    if args.load_raw_dataset:
        print_rank_0("> load raw jsonl dataset")
        train_dataset, valid_dataset = load_dataset_from_jsonl(
            args=args, shard_data=True, world_size=args.world_size, global_rank=global_rank, local_rank=local_rank
        )
    else:
        print_rank_0("> load tokenized bin dataset, refer to gpt_neox indexed dataset")
        train_dataset, valid_dataset, _ = load_dataset_from_bin(args=args)

    t1 = time.time()
    logger.info(f"dataset loading time: {t1 - t0:.4f}")

    # cuda memory
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB - 2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    accelerator.print("max memory: ", max_memory, n_gpus)

    # # 是否要加入新的special tokens
    # num_added_toks = tokenizer.tokenizer.add_special_tokens(["<role_start>", "<role_end>"])
    # accelerator.print("We have added", num_added_toks, "tokens")
    # accelerator.print(f"role marker tokens {tokenizer.convert_tokens_to_ids('<role_start>')} {tokenizer.convert_tokens_to_ids('<role_end>')}, resized tokenizer_size: {len(tokenizer)}")

    # creating model
    ModelClass = MODEL_TYPES[args.model_type]
    if args.model_type in SUPPORT_IN_TRANSFORMERS:
        accelerator.print(f"[INFO] Model Type {args.model_type} is supported by Transformers")
        model = ModelClass.from_pretrained(
            args.pretrained_model_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
        )
    else:
        accelerator.print(f"[INFO] Model Type {args.model_type} is supported in our local model dir for remote code")  
        model = ModelClass.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=torch.bfloat16,
        )

    # build a tokenizer for possible resizing or saving
    tokenizer = build_tokenizer(args)
    # Note: resize_token_embeddings expects to receive the full size of the new vocabulary,
    # i.e. the length of the tokenizer.
    # 如果新增special tokens, 需要resize input embedding 和output embedding
    # model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)

    model.gradient_checkpointing_enable()

    if args.saving_limit is None or not isinstance(args.saving_limit, int) or args.saving_limit < 1:
        # saving_limit is set automatically if needed
        args.saving_limit = 2
        accelerator.print(
            "[WARNING]saving_limit must be a integer greater than 1 in Full-Parameters Training, we set it to 2"
        )

    t2 = time.time()
    if accelerator.is_main_process:
        logging.info(f"model loading time: {t2 - t1:.4f}")

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    if hasattr(model.config, "use_logn_attn"):
        model.config.use_logn_attn = False  # special for qwen model
    # load balance for moe training
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    model_config = model.config
    accelerator.print(model.config)

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForMFTDataset(args),
        batch_size=args.per_device_train_batch_size,
        pin_memory=True,
        drop_last=True,
    )
    if valid_dataset:
        valid_dataloader = DataLoader(
            valid_dataset,
            collate_fn=DataCollatorForMFTDataset(args),
            batch_size=args.per_device_eval_batch_size,
            pin_memory=True,
            drop_last=True,
        )
    else:
        valid_dataloader = None

    # optimizer
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.print("DISTRIBUTED TRAINING USING DEEPSPEED")
        # from deepspeed.ops.adam import FusedAdam as Adam
        # adam_optimizer = Adam
        adam_optimizer = torch.optim.AdamW
    elif accelerator.distributed_type == DistributedType.FSDP:
        accelerator.print("DISTRIBUTED TRAINING USING FSDP")
        model = accelerator.prepare(model)
        adam_optimizer = torch.optim.AdamW
    else:
        raise ValueError("Only support DeepSpeed and FSDP")

    optimizer = adam_optimizer(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
    )
    # for group in optimizer.param_groups:
    #     group.setdefault("initial_lr", group["lr"])

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if isinstance(args.num_warmup_steps, float) and args.num_warmup_steps < 1.0:
        args.num_warmup_steps = int(args.max_train_steps * args.num_warmup_steps) // accelerator.num_processes
    accelerator.print(f"num_warmup_steps: {args.num_warmup_steps}")
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        # scheduler_specific_kwargs={"last_epoch": scheduler_last_ep}
    )
    # prepare all
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        if valid_dataloader:
            (model, train_dataloader, valid_dataloader, optimizer, lr_scheduler) = accelerator.prepare(
                model, train_dataloader, valid_dataloader, optimizer, lr_scheduler
            )
        else:
            (model, train_dataloader, optimizer, lr_scheduler) = accelerator.prepare(
                model, train_dataloader, optimizer, lr_scheduler
            )

    # prepare all except model, which is prepared before
    elif accelerator.distributed_type == DistributedType.FSDP:
        if valid_dataloader:
            (optimizer, train_dataloader, valid_dataloader, lr_scheduler) = accelerator.prepare(
                optimizer, train_dataloader, valid_dataloader, lr_scheduler
            )
        else:
            (optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
                optimizer, train_dataloader, lr_scheduler
            )
    print(model.device)
    accelerator.print(model)
    # accelerator.print(model.config)

    # Recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterward we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # zero 3 flag
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3
        accelerator.print(f"DEEPSPEED plugin: {accelerator.state.deepspeed_plugin}")
    elif getattr(accelerator.state, "fsdp_plugin", None):
        accelerator.print(f"FSDP plugin: {accelerator.state.fsdp_plugin}")

    trainer = MptTrainer(
        accelerator=accelerator,
        model=model,
        model_config=model_config,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        total_train_dataset_size=len(train_dataset),
        args=args,
    )
    trainer.accelerate_train()


if __name__ == "__main__":
    main()
