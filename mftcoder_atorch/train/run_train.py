import sys
sys.path.append("..")
import json
import logging
import math
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class cb:
    def __init__(self, path):
        self.path = path
    def __call__(self, s):
        with open(f"{self.path}/fsdp_mapping.html", "w") as f:
            f.write(s)

# handle multi-processing writing
os.environ["HF_MODULES_CACHE"] = os.path.join("/root/.cache/huggingface/modules", os.getenv("RANK", ""))
import random  # noqa: E402
import datasets  # noqa: E402
import transformers  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402
from transformers import (  # noqa: E402
    default_data_collator,
    # get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version  # noqa: E402
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload  # noqa: E402

from transformers import AutoTokenizer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from utils.common_utils import (
    is_local_main_process, generate_task_id, print_rank_0, is_old_version,
    atorch_init_distributed, atorch_reset_distributed, TASK2ID, ID2TASK,
    get_rank, get_world_size
)
from utils.auto_accelerate_utils import DataCollatorForMFTDataset, loss_func_mft
from arguments.get_arguments import parse_args
from model.build_model import setup_model
from data.gpt2_multi_task_dataset import load_dataset_from_jsonl
from train.trainer.atorch_trainer import AtorchTrainer
from pathlib import Path


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    if is_local_main_process():
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    generate_task_id(args.data_paths, args.train_mode)  # generate TASK2ID, ID2TASK mapping
    print(TASK2ID)
    print(ID2TASK)

    model, model_config, tokenizer = setup_model(args, logger, use_cache=False)
    print(f'args.total_model_param: {args.total_model_param}')

    train_dataset, dataloader_args = None, None
    train_dataloader, valid_dataloader, test_dataloader = None, None, None

    args.world_size = get_world_size()
    global_rank = get_rank()
    print(f'world_size: {args.world_size}, global_rank: {global_rank}')
    args.per_device_train_batch_size = args.total_train_batch_size // args.world_size
    if args.load_raw_dataset:
        print_rank_0('load raw dataset')
        if args.model_type in ['gpt_neox']:
            train_dataset, valid_dataset = load_dataset_from_jsonl(args, tokenizer, shard_data=True, world_size=args.world_size, global_rank=global_rank)

        if train_dataset is not None:
            args.do_train = True
        if valid_dataset is not None:
            args.do_valid = True
    else:
        print_rank_0('please set load_raw_dataset to True and rerun')

    if args.resume_from_checkpoint == 'true':
        logger.info(f'Resume from {args.output_dir}')
        resume_from_checkpoint = True
    else:
        logger.info(f'Train from scratch')
        resume_from_checkpoint = False
    if args.model_type in ['gpt_neox']:
        gpt_data = True
    else:
        gpt_data = False
    data_collator = DataCollatorForMFTDataset(args.model_type, args.weighted_loss_mode, args.use_dynamic_padding)
    my_loss_function = loss_func_mft
    trainer = AtorchTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        tokenizer=tokenizer,
        # files_to_save=files_to_save,
        args_to_save={
            # 'max_length': args.max_length,
            'max_length': args.seq_length,
            'peft_type': args.peft_type,
            'gpt_model': gpt_data
        },
        data_collator=data_collator,
        my_loss_func=my_loss_function,
        custom_lr_scheduler_type=args.custom_lr_scheduler_type,
        rank=global_rank
    )
    if args.do_train:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    atorch_init_distributed("nccl")
    main()
    atorch_reset_distributed()
