"""
# @author qumu
# @date 2023/12/11
# @module mft_accelerate.py

Accelerate + DeepSpeed/FSDP + QLoRA/LoRA/Full + DPO/RPO/ORPO

Entry
"""

import os
import sys
import argparse
import math
import logging
import json
import time
from datetime import timedelta
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Dict, Optional, Union, List

import datasets
from datasets import Dataset, load_dataset, concatenate_datasets

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from accelerate import Accelerator, DistributedType, FullyShardedDataParallelPlugin, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs

# insert src as import path
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
sys.path.insert(0, parent_dir)

from tokenizer import build_tokenizer

from utils.common_utils import print_rank_0, generate_task_id, TASK2ID, ID2TASK
from utils.model_mapping import MODEL_TYPES, SUPPORT_IN_TRANSFORMERS

logger = get_logger(__name__)


from trl import (
    DPOConfig,
    DPOTrainer,
    ORPOConfig,
    ORPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from xxpo.xxpo_arguments import XXPOTrainArgs
from xxpo.custom_callbacks import CustomProgressCallback
from xxpo.custom_callbacks import LogCallback


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
    args = XXPOTrainArgs(**train_config)

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

    if args.peft_type == "qlora":
        print_rank_0(f"[INFO] args.peft_type is set 'qlora', setting quantization to '4bit'")
        args.quantization = "4bit"
    else:
        args.quantization = None

    args.vocab_file = args.pretrained_model_path

    return args


def get_model(args, accelerator):
    ModelClass = MODEL_TYPES[args.model_type]
    if args.model_type in SUPPORT_IN_TRANSFORMERS:
        accelerator.print(f"[INFO] Model Type {args.model_type} is supported by Transformers")
        model = ModelClass.from_pretrained(
            args.pretrained_model_path,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
            # device_map=get_kbit_device_map() if args.quantization == "4bit" else None,
            quantization_config=(
                BitsAndBytesConfig(
                    load_in_4bit=(args.quantization == "4bit"),
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_quant_storage=torch.bfloat16,
                )
                if args.quantization == "4bit"
                else None
            ),
        )
    else:
        accelerator.print(f"[INFO] Model Type {args.model_type} is supported in our local model dir for remote code")
        model = ModelClass.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=(
                BitsAndBytesConfig(
                    load_in_4bit=(args.quantization == "4bit"),
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_quant_storage=torch.bfloat16,
                )
                if args.quantization == "4bit"
                else None
            ),
        )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    return model


def chatml_to_dpo_format(
    data_file: str,
    tokenizer,
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=16,
) -> Dataset:
    """Load the standard-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'chosen': List[dict], chatml
        'rejected': List[dict], chatml
    }
    """

    dataset = load_dataset(
        "json",
        split="train",
        data_files=data_file,
        cache_dir=cache_dir,
        verification_mode="no_checks",
    )
    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    def process(samples):
        samples["prompt"] = [
            tokenizer.apply_chat_template(chosen[:-1], tokenize=False, add_generation_prompt=True)
            for chosen in samples["chosen"]
        ]
        samples["chosen"] = [chosen[-1]["content"] + tokenizer.eos_token for chosen in samples["chosen"]]
        samples["rejected"] = [rejected[-1]["content"] + tokenizer.eos_token for rejected in samples["rejected"]]
        return samples

    return dataset.map(
        process,
        batched=True,
        num_proc=num_proc,
        # remove_columns=original_columns,
    )


def main():
    t0 = time.time()
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
    accelerator.print("In dpo_accelerate.py, sys path:", sys.path)
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
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # get global_rank and local rank for current process
    global_rank = accelerator.process_index
    local_rank = accelerator.local_process_index
    print(f"world_size: {args.world_size}, global_rank: {global_rank}, local_rank: {local_rank}")

    # 1. dataset

    # build tokenizer
    tokenizer = build_tokenizer(args)
    # tokenizer.chat_template = MFTCoder_template

    # Load the dpo dataset
    all_datasets = []
    # print(args.data_paths, type(args.data_paths))
    if isinstance(args.data_paths, str):
        args.data_paths = list(args.data_paths[1:-1].split(","))
        # print(f"DATA_PATHS: {args.data_paths}")
    for data_file in args.data_paths:
        ds = chatml_to_dpo_format(data_file=data_file, tokenizer=tokenizer, sanity_check=args.sanity_check)
        all_datasets.append(ds)

    all_dataset = concatenate_datasets(all_datasets)
    # all_dataset = all_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= args.max_length
    # )
    accelerator.print(f"Length of all_dataset: {len(all_dataset)}")

    # split train/eval dataset
    splits = [float(s) for s in args.data_split.split(",")][:2]
    print(f"data splits: {splits}")

    all_dataset = all_dataset.train_test_split(test_size=splits[1] / sum(splits), shuffle=True, seed=args.seed)
    all_dataset.flatten_indices()

    train_dataset, eval_dataset = all_dataset["train"], all_dataset["test"]
    accelerator.print(f"Length of train_dataset: {len(train_dataset)}\nLength of eval_dataset: {len(eval_dataset)}")
    print(eval_dataset[0])
    t1 = time.time()
    logger.info(f"dataset loading time: {t1 - t0:.4f}")

    # cuda memory
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB - 2}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    accelerator.print("max memory: ", max_memory, n_gpus)

    # target_modules, default all-linear for all linear layers
    if args.target_modules:
        target_modules = args.target_modules
    else:
        target_modules = "all-linear"

    # peft config
    if args.peft_type:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="lora_only",
        )
    else:
        peft_config = None

    # creating base model
    model = get_model(args, accelerator)
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    accelerator.print("Model load_in_4bit: ", args.quantization == "4bit")

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    if hasattr(model.config, "use_logn_attn"):
        model.config.use_logn_attn = False  # special for qwen model
    # load balance for moe training
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    model_config = model.config
    accelerator.print(model.config)

    t2 = time.time()
    if accelerator.is_main_process:
        logging.info(f"model loading time: {t2 - t1:.4f}")

    # 4. initialize training arguments:
    if args.xxpo == "dpo":
        ConfigClass = DPOConfig
    elif args.xxpo == "orpo":
        ConfigClass = ORPOConfig
    logging.info(f"{args.xxpo} Used.")

    training_args = ConfigClass(
        beta=args.beta,
        rpo_alpha=args.rpo_alpha,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to="tensorboard",
        logging_dir=args.tb_dir,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="",
        gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reentrant),
        seed=args.seed,
        dataset_num_proc=args.dataset_num_proc,
        disable_tqdm=args.disable_tqdm,
        save_only_model=args.save_only_model,
        save_total_limit=args.saving_limit,
    )

    # 5. initialize the DPO trainer
    if not args.peft_type and args.xxpo == "dpo":
        model_ref = get_model(args, accelerator)
        model_ref.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    else:
        model_ref = None

    if args.xxpo == "dpo":
        xxpo_trainer = DPOTrainer(
            model,
            ref_model=model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
    elif args.xxpo == "orpo":
        xxpo_trainer = ORPOTrainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    # callbacks
    if args.disable_tqdm:
        xxpo_trainer.remove_callback(PrinterCallback)
        xxpo_trainer.add_callback(LogCallback)
    else:
        xxpo_trainer.remove_callback(ProgressCallback)
        xxpo_trainer.add_callback(CustomProgressCallback)

    # 6. train
    xxpo_trainer.train()

    # 7. save
    output_dir = os.path.join(args.output_dir, "epoch_final")
    xxpo_trainer.save_model(output_dir)
    # dpo_trainer.model.save_pretrained(output_dir)
    logger.info(f"Training Finished!")


if __name__ == "__main__":
    main()
