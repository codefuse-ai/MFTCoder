#!/usr/bin/env python
# coding=utf-8

import datetime
import json
import logging
import math
import os
import random
import re
import shutil
import time
import warnings
from functools import partial
from pathlib import Path
import gc

import numpy as np

import atorch
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import get_scheduler as get_scheduler_trans
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME
)
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
)
from transformers.utils import WEIGHTS_NAME
from torch.nn import CrossEntropyLoss
from utils.common_utils import print_rank_0, get_tflops_megatron, get_computation_speed, TASK2ID, ID2TASK, EarlyStopping
from utils.auto_accelerate_utils import get_ltor_masks_and_position_ids

from atorch.auto import auto_accelerate
from atorch.utils.version import torch_version
from model.gpt_neox.modeling_gpt_neox import GPTNeoXLayer, GPTNeoXAttention, GPTNeoXMLP
from model.peft.modeling_peft import PeftModel

HYPER_PARAMETER_NAME = 'hyper_parameters.json'
ATORCH_CHECKPOINT_NAME = 'atorch_checkpoint.bin'
EPOCH_CHECKPOINT_NAME = 'epoch'

logger = logging.getLogger(__name__)


def is_local_main_process():
    return atorch.local_rank() == 0


def is_global_main_process():
    return atorch.rank() == 0


def has_inf_or_nan(x):
    try:
        # if x is half, the .float() incurs an additional deep copy, but it's necessary if
        # Pytorch's .sum() creates a one-element tensor of the same type as x
        # (which is true for some recent version of pytorch).
        cpu_sum = float(x.float().sum())
        # More efficient version that can be used if .sum() returns a Python scalar
        # cpu_sum = float(x.sum())
    except RuntimeError as instance:
        # We want to check if inst is actually an overflow exception.
        # RuntimeError could come from a different error.
        # If so, we still want the exception to propagate.
        if "value cannot be converted" not in instance.args[0]:
            raise
        return True
    else:
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        return False


def count_model_params(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return all_params, trainable_params


class AtorchArguments:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def get_linear_schedule_with_log_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        inverse_log_warm_up = 1.0 / math.log(num_warmup_steps)
        if current_step == 0:
            return 0.0
        if current_step < num_warmup_steps:
            return inverse_log_warm_up * math.log(current_step)
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    scheduler_map = {
        'log_warmup_linear_decay': get_linear_schedule_with_log_warmup}
    try:
        lr_scheduler = get_scheduler_trans(
            name, optimizer, num_warmup_steps, num_training_steps)
        return lr_scheduler
    except Exception:
        schedule_func = scheduler_map[name]
        return schedule_func(optimizer, num_warmup_steps, num_training_steps)


class AtorchTrainer:
    def __init__(self,
                 model,
                 args,
                 train_dataset,
                 valid_dataset,
                 tokenizer=None,
                 callbacks=None,
                 no_save_atorch_checkpoint=None,
                 save_pytorch_model_bin_checkpoint=True,
                 train_peft=False,
                 rank=0,
                 max_shard_size='10GB',
                 files_to_save=None,
                 args_to_save=None,
                 data_collator=None,
                 my_loss_func=None,
                 **kwargs,
                 ):
        self.args = args
        self.TASK2ID = TASK2ID
        self.ID2TASK = ID2TASK
        print('in atorch trainer')
        print(TASK2ID)
        print(ID2TASK)
        self.model = model
        self.no_save_atorch_checkpoint = no_save_atorch_checkpoint
        self.save_pytorch_model_bin_checkpoint = save_pytorch_model_bin_checkpoint
        self.train_peft = train_peft
        self.rank = rank
        self.kwargs = kwargs
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.tokenizer = tokenizer
        self.max_shard_size = max_shard_size
        self.files_to_save = files_to_save
        self.args_to_save = args_to_save
        self.best_metric = None
        self.best_model_checkpoint = None
        self.no_save_base_model = True

        self.device = f"cuda:{atorch.local_rank()}"

        self.total_train_batch_size = self.args.per_device_train_batch_size * \
            self.args.gradient_accumulation_steps * \
            atorch.world_size()

        self.data_collator = data_collator
        self.my_loss_func = my_loss_func
        if self.args.early_stopping_patience > 0:
            print(f'early_stopping_patience: {self.args.early_stopping_patience}')
            patience = self.args.early_stopping_patience
            self.early_stopping = EarlyStopping(patience, verbose=True)
        
        self.train_dataloader_args = {
            "shuffle": True,
            "batch_size": self.total_train_batch_size,
            "pin_memory": True,
            "collate_fn": data_collator,
            "drop_last": True,
            # "num_workers": args.num_workers,
            # "persistent_workers": args.num_workers > 0,
        }

        self.valid_dataloader = DataLoader(
            valid_dataset,
            sampler=DistributedSampler(valid_dataset, shuffle=True),
            batch_size=args.per_device_valid_batch_size,
            pin_memory=True,
            collate_fn=data_collator
        )
        self.valid_dataloader_length = len(self.valid_dataloader)

        if self.args.resume_from_checkpoint == 'true':
            self.resume_checkpoint_dir = self.get_last_checkpoint(
                self.args.output_dir)

        self.atorch_args = AtorchArguments(
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_eps=args.adam_epsilon,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2)

        self.atorch_init()

        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        print(f'number of update steps per epoch: {self.num_update_steps_per_epoch}')
        if self.args.max_steps == -1:
            self.args.max_steps = int(
                self.args.num_train_epochs * self.num_update_steps_per_epoch)
        else:
            self.args.num_train_epochs = math.ceil(
                self.args.max_steps / self.num_update_steps_per_epoch)

        # self.args.warmup_steps = self.args.get_warmup_steps(
        #     self.args.max_steps)  # 找不到get_warmup_steps
        custom_lr_scheduler_type = self.kwargs.get(
            'custom_lr_scheduler_type', None)
        self.lr_scheduler = get_scheduler(
            name=custom_lr_scheduler_type if custom_lr_scheduler_type else self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_steps,
        )
        print_rank_0(f'lr_scheduler{self.lr_scheduler}')
        if self.args.resume_from_checkpoint == 'true':
            with warnings.catch_warnings(record=True):
                self.lr_scheduler.load_state_dict(torch.load(
                    os.path.join(self.resume_checkpoint_dir, SCHEDULER_NAME)))
            self._load_rng_state(self.resume_checkpoint_dir)
        torch.distributed.barrier()
        now_datetime = datetime.datetime.now()
        timestr = datetime.datetime.strftime(now_datetime, '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(self.args.output_dir, 'runs', timestr)
        self.summary_writer = None
        if torch.distributed.get_rank() == 0:
            self.summary_writer = SummaryWriter(log_dir=self.log_dir)
    
    def get_last_checkpoint(self, folder):
        _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)")
        content = sorted(os.listdir(folder))
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
        ]
        if len(checkpoints) == 0:
            return
        return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

    def _load_rng_state(self, resume_checkpoint_dir):
        # Load RNG states from `checkpoint`
        if resume_checkpoint_dir is None:
            return

        if self.args.world_size > 1:
            rng_file = os.path.join(
                resume_checkpoint_dir, f"rng_state_{self.rank}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {self.rnak}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(resume_checkpoint_dir, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.local_rank != -1:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state_all(
                        checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def load_atorch_model_state(self, model_state_dict, **kwargs):
        print('resume atorch model state')
        if self.is_rank0():
            self.model.load_state_dict(model_state_dict)
        # 在 rank 0 加载完毕后，再通过sync_module_states分发参数
        torch.distributed.barrier()
        # self.model = FSDP(self.model, sync_module_states=True, **kwargs)

    def load_atorch_optim_state(self, optim_state_dict):

        print('resume optimizer state')
        optim_state_dict = FSDP.scatter_full_optim_state_dict(
            optim_state_dict, self.model)  # may be removed after PyTorch 2.2
        
        def move_optim_state_to_cpu(optim_state_dict):
            for k in optim_state_dict:
                if isinstance(optim_state_dict[k], torch.Tensor):
                    optim_state_dict[k] = optim_state_dict[k].cpu()
                elif isinstance(optim_state_dict[k], dict):
                    move_optim_state_to_cpu(optim_state_dict[k])
        
        move_optim_state_to_cpu(optim_state_dict)

        self.optimizer.load_state_dict(optim_state_dict)


    def atorch_init(self):
        assert torch_version() >= (2, 0, 0), "use pt2.0 for use orig param if fsdp"
        
        if self.args.model_type == 'gpt_neox':
            # wrap_class = (GPTNeoXAttention, GPTNeoXMLP)
            wrap_class = (GPTNeoXLayer,)
        
        parallel_mode = []
        if self.args.dp:
            # p_mode = ([("data", torch.distributed.get_world_size())], None)
            parallel_mode.append(("data", self.args.dp))
        if self.args.tp:
            parallel_mode.append(("tensor_parallel", self.args.tp))
        strategy = [
            # ("parallel_mode", p_mode),
            ("parallel_mode", (parallel_mode, None)),
            "module_replace",
            # ("fsdp", fsdp_config),
            # ("amp_native", {"dtype": torch.bfloat16}) if self.args.bf16 else "amp_native",
            # ("checkpoint", wrap_class),
        ]
        if self.args.peft_type is None or self.args.peft_type == 'lora':
            # cpu_offload = False if self.args.peft_type is None else True
            cpu_offload = False if self.args.total_model_param < 1e9 else True
            fsdp_config = {
                "atorch_wrap_cls": wrap_class,
                "sync_module_states": True,
                "use_orig_params": True,
                "limit_all_gathers": True,
                # "cpu_offload": True,
            }
            print(fsdp_config)
            fsdp_opt = ("fsdp", fsdp_config)
            strategy.append(fsdp_opt)
            self.args.atorch_opt = "fsdp"
        else:
            # TODO: qlora
            self.args.atorch_opt = "ddp"

        if self.args.bf16 or self.args.fp16:
            if self.args.bf16:
                amp_config = {"dtype": torch.bfloat16, "skip_if_nonfinite": True}
                # TODO: qlora
            elif self.args.fp16:
                amp_config = {"dtype": torch.float16}
            strategy.append(("amp_native", amp_config))
        
        if self.args.checkpoint_activations:
            strategy.append(("checkpoint", wrap_class))
        print(f"Manually loaded auto acc strategy: {strategy}")
        
        def prepare_input(batch, device):
            batch = {k: v.to(device=device, non_blocking=True) if v is not None else None
                     for k, v in batch.items()}
            return batch

        def optim_param_func(model, args):
            no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            return optimizer_grouped_parameters

        # load fsdp checkpoint参数
        if self.args.resume_from_checkpoint == 'true':
            logger.info(f'Resume training from {self.resume_checkpoint_dir}')
            if self.is_rank0():
                sd = torch.load(os.path.join(
                    self.resume_checkpoint_dir, ATORCH_CHECKPOINT_NAME), map_location='cpu')
                model_state_dict, optim_state_dict = sd['model_state_dict'], sd['optimizer_state_dict']
            else:
                model_state_dict, optim_state_dict = None, None
            torch.distributed.barrier()  # other rank waiting
            ##########
            self.load_atorch_model_state(model_state_dict)
            ##########

        if self.is_rank0():
            print(f'GPU mem before fsdp:')
            print(torch.cuda.memory_summary(device=self.device, abbreviated=False))
        optim_func = torch.optim.AdamW
        print(f'optimizer before fsdp: {optim_func}')

        ddp_find_unused_parameters = None
        if self.args.atorch_opt == "ddp" and not (self.args.peft_type in ["lora", "qlora"] and self.args.checkpoint_activations):
            ddp_find_unused_parameters = True

        status, result, best_strategy = auto_accelerate(
            self.model,
            optim_func,
            self.train_dataset,
            dataloader_args=self.train_dataloader_args,
            loss_func=self.my_loss_func,
            prepare_input=prepare_input,
            optim_args={
                "lr": self.atorch_args.lr,
                "weight_decay": self.atorch_args.weight_decay,
                "eps": self.atorch_args.adam_eps,
                "betas": (self.atorch_args.adam_beta1, self.atorch_args.adam_beta2),
            },
            optim_param_func=partial(
                optim_param_func, args=self.atorch_args),
            load_strategy=strategy,
            ignore_dryrun_on_load_strategy=True,
            find_unused_parameters=ddp_find_unused_parameters,
        )
        assert (
            status
        ), f"auto_accelerate failed. status: {status}, result: {result}, best_strategy: {best_strategy}"
        print(f"Best strategy is: {best_strategy}")

        self.model = result.model
        self.optimizer = result.optim
        print(f'optimizer after fsdp: {self.optimizer}')
        self.loss_func = result.loss_func
        self.train_dataloader = result.dataloader
        self.prepare_input = result.prepare_input

        if self.args.resume_from_checkpoint == 'true':
            self.load_atorch_optim_state(optim_state_dict)
        print(f"atorch use optimizer: {self.optimizer}")
        if self.is_rank0():
            print(f'GPU mem after fsdp:')
            print(torch.cuda.memory_summary(device=self.device, abbreviated=False))

    def evaluate(self):
        logger.info(f"Start evaluation")
        if self.is_rank0():
            print(torch.cuda.memory_summary(device=self.device, abbreviated=False))
        print(f'valid dataset length is: {len(self.valid_dataset)}')
        print(f'valid dataloader length is: {len(self.valid_dataloader)}')
        print(f'per device batch size: {self.args.per_device_valid_batch_size}')
        progress_bar = tqdm(range(len(self.valid_dataloader)), 
                            disable=not is_local_main_process(), 
                            smoothing=0)
        self.model.eval()
        losses = []
        accumulated_task_loss_np = np.zeros(len(self.ID2TASK))
        accumulated_task_num_np = np.zeros(len(self.ID2TASK))
        accumulated_step = 0
        for step, batch in enumerate(self.valid_dataloader):
            # if step >= self.args.valid_iters:
            if step >= self.args.valid_iters and (self.args.total_model_param >= 1e9 or self.args.train_mode == 'sst'):
                break
            with torch.no_grad():
                # batch = {k: v.to(self.device) for k, v in batch.items()}
                # batch = self.prepare_input(batch, self.device)
                # outputs = self.model(**batch)
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    position_ids=batch['position_ids'].to(self.device)
                )
                # loss = outputs["loss"]
                loss, task_loss, task_num = self.loss_func(outputs, batch, self.args.weighted_loss_mode)

                repeated_loss = loss.repeat(
                    self.args.per_device_valid_batch_size)
                if repeated_loss.ndim == 0:
                    repeated_loss = repeated_loss.clone()[None]
                output_tensors = [repeated_loss.clone()
                                  for _ in range(atorch.world_size())]
                torch.distributed.all_gather(output_tensors, repeated_loss)
                for tensor in output_tensors:
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        accumulated_step -= 1
                        continue
                losses.append(torch.cat(output_tensors, dim=0).cpu())
                task_loss = task_loss.cpu().numpy()
                task_num = task_num.cpu().numpy()
                accumulated_task_loss_np += task_loss
                accumulated_task_num_np += task_num
            
            accumulated_step += 1
            progress_bar.update(1)

        losses = torch.cat(losses)
        losses = losses[: len(self.valid_dataset)]
        mean_loss = torch.mean(losses).item()
        accumulated_task_loss = torch.tensor(accumulated_task_loss_np).to(self.device)
        accumulated_task_num = torch.tensor(accumulated_task_num_np).to(self.device)
        torch.distributed.all_reduce(accumulated_task_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(accumulated_task_num, op=torch.distributed.ReduceOp.SUM)
        accumulated_task_loss /= torch.distributed.get_world_size()
        valid_task_loss = accumulated_task_loss / (accumulated_step - 1)
        logs = {'valid_loss': mean_loss}
        per_task_valid_loss = {self.ID2TASK[i]+'_loss': valid_task_loss[i].item() for i in range(len(self.ID2TASK))}
        logs.update(per_task_valid_loss)
        if is_global_main_process():
            logger.info('log point')
            for i in range(len(self.ID2TASK)):
                if accumulated_task_num[i] != 0:
                    logger.info(f"{self.ID2TASK[i]}_loss: {valid_task_loss[i]}, sample nums: {accumulated_task_num[i]}")
            self.log(logs, step=self.global_steps, phase='Evaluation')
        metrics = {'valid_loss': mean_loss, 'valid_task_loss': valid_task_loss}
        logger.info(f"Finish evaluation")
        if self.is_rank0():
            print(torch.cuda.memory_summary(device=self.device, abbreviated=False))

        return metrics

    def log(self, logs, step, phase='Train'):
        if not self.summary_writer:
            return
        logger.info(json.dumps(logs))
        for key, value in logs.items():
            self.summary_writer.add_scalar(f'{phase}/{key}', value, step)

    def _sorted_checkpoints(
        self,
        output_dir=None,
        checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
        checkpoint_name_pattern='([0-9]+)',
        use_mtime=False
    ):
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(
            f"{checkpoint_prefix}-*") if os.path.isdir(x)]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append(
                    (os.path.getmtime(path), path))
            else:
                regex_match = re.search(
                    f".*{checkpoint_prefix}-({checkpoint_name_pattern})", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.best_model_checkpoint)))
            # for i in range(best_model_index, len(checkpoints_sorted) - 2):
            for i in range(best_model_index, len(checkpoints_sorted) - 1):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        print_rank_0(f'checkpoints sorted list: {checkpoints_sorted}')
        return checkpoints_sorted

    def _rotate_checkpoints(
            self,
            use_mtime=False,
            output_dir=None,
            prefix=PREFIX_CHECKPOINT_DIR,
            checkpoint_name_pattern='.*') -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=use_mtime,
            output_dir=output_dir,
            checkpoint_prefix=prefix,
            checkpoint_name_pattern=checkpoint_name_pattern)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit

        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    def _clean_atorch_checkpoints(self, output_dir=None, prefix=PREFIX_CHECKPOINT_DIR):
        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            output_dir=output_dir,
            checkpoint_prefix=prefix,
            checkpoint_name_pattern='([0-9]+)')

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.

        for checkpoint in checkpoints_sorted[:-1]:
            logger.info(
                f"Deleting older atorch checkpoint [{checkpoint}] due to self.args.save_total_limit")
            try:
                os.remove(os.path.join(checkpoint, ATORCH_CHECKPOINT_NAME))
            except Exception:
                continue

    def _save_peft_model(self, output_dir, state_dict=None):
        logger.info(f"Start saving peft model to {output_dir}")
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = unwrap_model(self.model)
        if isinstance(model, PeftModel):
            if state_dict is None:
                state_dict = model.state_dict()
            model.save_pretrained(
                output_dir, state_dict=state_dict, is_main_process=self.is_rank0())
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            if self.is_rank0():
                torch.save(state_dict, os.path.join(
                    output_dir, "pytorch_model.bin"))
        logger.info(f"Saving peft model done.")
    
    def _save_model(self, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                print_rank_0('save in not pretrained model~~~~~~~')
                if state_dict is None:
                    state_dict = self.model.state_dict()
                    state_dict = {key: value.bfloat16() if self.args.bf16 else value.half() for key, value in state_dict.items()}
                model = unwrap_model(self.model)
                model.save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    max_shard_size=self.max_shard_size,
                    is_main_process=self.is_rank0())
                # unwrap_model(self.model).save_pretrained(
                #     output_dir, state_dict=state_dict, max_shard_size=self.max_shard_size)
            elif isinstance(unwrap_model(self.model), PeftModel):
                if state_dict is None:
                    state_dict = unwrap_model(self.model).base_model.model.state_dict()
                    # state_dict = {key: value.bfloat16() if self.args.bf16 else value.half() for key, value in state_dict.items()}
                # Filter the peft params ...
                param_keys = list(state_dict.keys())
                base_model_state_dict = {}
                for key in param_keys:
                    if LORA_KEY in key:
                        # state_dict.pop(key)
                        continue
                    elif PEFT_PARAM_PREFIX in key:
                        # value = state_dict.pop(key)
                        value = state_dict[key]
                        new_key = key.replace(PEFT_PARAM_PREFIX, "")
                        base_model_state_dict[new_key] = value
                    else:
                        base_model_state_dict[key] = value
                if self.is_rank0():
                    torch.save(base_model_state_dict,
                               os.path.join(output_dir, WEIGHTS_NAME))
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                    state_dict = {key: value.bfloat16() if self.args.bf16 else value.half() for key, value in state_dict.items()}
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            print(f'save in pretrained model!!!!!!')
            if state_dict is None:
                state_dict = self.model.state_dict()
                state_dict = {key: value.bfloat16() if self.args.bf16 else value.half() for key, value in state_dict.items()}
            self.model.save_pretrained(
                    output_dir, 
                    state_dict=state_dict, 
                    max_shard_size=self.max_shard_size)
        if self.tokenizer is not None:    
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def is_rank0(self):
        return self.rank == 0

    def _save_atorch_checkpoint(self, output_dir):

        # StateDictType.FULL_STATE_DICT得到完整的模型状态。
        # FullStateDictConfig指定保存到CPU，仅rank0保存
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = self.model.state_dict()
            optim_state_dict = FSDP.full_optim_state_dict(self.model, self.optimizer)  # may be removed after PyTorch 2.2

        if self.is_rank0():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optim_state_dict,
                    "global_steps": self.global_steps,
                },
                os.path.join(output_dir, ATORCH_CHECKPOINT_NAME),
            )
        torch.distributed.barrier()  # other rank waiting


    def save(self, suffix=None, metrics=None):
        logger.info('Save start')
        if self.is_rank0():
            print(torch.cuda.memory_summary(device=self.device, abbreviated=False))
        if not self.save_pytorch_model_bin_checkpoint:
            return
        if suffix is None:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_steps}"
        else:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{suffix}"

        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        # self._save_model(output_dir)
        # 获取要存的state_dict, 每个rank都要调用
        if isinstance(self.model, FSDP):
            save_policy = FullStateDictConfig(offload_to_cpu=atorch.world_size() > 1, rank0_only=atorch.world_size() > 1)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state_dict = self.model.state_dict()
                optim_state_dict = FSDP.full_optim_state_dict(self.model, self.optimizer)  # may be removed after PyTorch 2.2
        else:
            model_state_dict = unwrap_model(self.model).state_dict()
            optim_state_dict = self.optimizer.state_dict()
        if not self.no_save_atorch_checkpoint:
            if self.args.peft_type is None or not self.no_save_base_model: 
                if self.is_rank0():
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optim_state_dict,
                            "global_steps": self.global_steps,
                        },
                        os.path.join(output_dir, ATORCH_CHECKPOINT_NAME),
                    )
                torch.distributed.barrier()  # other rank waiting
        if self.args.peft_type is not None:
            print(f'no_save_base_model: {self.no_save_base_model}')
            if not self.no_save_base_model:
                self._save_model(output_dir=output_dir)
            self._save_peft_model(output_dir=output_dir)
        else:
            self._save_model(output_dir=output_dir)
        
            # if not self.no_save_atorch_checkpoint:
            #     self._save_atorch_checkpoint(output_dir)
            # else:
            #     torch.save(self.optimizer.state_dict(),
            #             os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            
            # Save RNG state in non-distributed training
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "cpu": torch.random.get_rng_state(),
            }
            if torch.cuda.is_available():
                if self.args.local_rank == -1:
                    # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                    rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
                else:
                    rng_states["cuda"] = torch.cuda.random.get_rng_state()

            os.makedirs(output_dir, exist_ok=True)

            if torch.distributed.get_world_size() <= 1:
                torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
            else:
                # torch.save(rng_states, os.path.join(
                #     output_dir, f"rng_state_{self.args.process_index}.pth"))  # process_index = rank
                torch.save(rng_states, os.path.join(
                    output_dir, f"rng_state_{self.rank}.pth"))

            if self.args_to_save:
                json.dump(self.args_to_save, open(os.path.join(output_dir,
                        HYPER_PARAMETER_NAME), 'w'), ensure_ascii=False, indent=2)
            # save state
            state = {'global_steps': self.global_steps}
            json.dump(state, open(os.path.join(
                output_dir, TRAINER_STATE_NAME), 'w'), ensure_ascii=False, indent=2)
            
            # if self.files_to_save:
            #     for name in self.files_to_save:
            #         if not os.path.exists(name):
            #             continue
            #         try:
            #             if os.path.isfile(name):
            #                 shutil.copy(name, output_dir)
            #             elif os.path.isdir(name):
            #                 shutil.copytree(name, os.path.join(
            #                     output_dir, os.path.basename(name)))
            #         except Exception:
            #             continue
        
        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("valid_"):
                metric_to_check = f"valid_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better == 'true' else np.less
            if (
                self.best_metric is None
                or self.best_model_checkpoint is None
                or operator(metric_value, self.best_metric)
            ):
                self.best_metric = metric_value
                self.best_model_checkpoint = output_dir
                print_rank_0(f'current best model checkpoint is: {self.best_model_checkpoint}, valid_loss: {self.best_metric}')
        
        if self.is_rank0():
            if self.args.extra_save_by_epoch:
                print('extra_save_by_epoch')
                # 如果是每个epoch extra save的，那么每个epoch的checkpoint不会删除，不受save_total_limit的影响，
                # 而对按step存的，则会只保留save_total_limit个
                self._rotate_checkpoints(
                    output_dir=run_dir, prefix=PREFIX_CHECKPOINT_DIR, checkpoint_name_pattern='([0-9]+)$')
            else:
                self._rotate_checkpoints(
                    output_dir=run_dir, prefix=PREFIX_CHECKPOINT_DIR)
            # 只保留最新一个checkpoint的atorch checkpoint
            self._clean_atorch_checkpoints(
                output_dir=run_dir, prefix=PREFIX_CHECKPOINT_DIR)

            print(torch.cuda.memory_summary(device=self.device, abbreviated=False))
        torch.distributed.barrier()
        logger.info('Save finished')

    
    def train(self, **kwargs):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_train_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_steps}")

        progress_bar = tqdm(range(self.args.max_steps), 
                            disable=not is_local_main_process(), 
                            smoothing=0)
        training_time = 0

        self.global_steps = 0
        start_epoch = 0
        steps_trained_in_current_epoch = 0
        exit_flag = False
        if self.args.resume_from_checkpoint == 'true':
            state = json.load(
                open(os.path.join(self.resume_checkpoint_dir, TRAINER_STATE_NAME), 'r'))
            self.global_steps = state.get('global_steps', 0)
            # progress_bar.update(self.global_steps)
            progress_bar = tqdm(range(self.args.max_steps),
                                disable=not is_local_main_process(),
                                initial=self.global_steps, 
                                smoothing=0)
            start_epoch = self.global_steps // self.num_update_steps_per_epoch
            steps_trained_in_current_epoch = self.global_steps % self.num_update_steps_per_epoch
            steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            print(f'Start training at step {self.global_steps}')
        self.last_step_logged = self.global_steps
        self.skipped_steps = 0
        self.accumulated_loss = 0
        self.accumulated_task_loss = np.zeros(len(self.ID2TASK))
        self.accumulated_task_num = np.zeros(len(self.ID2TASK))

        self.train_task_loss_prev = None
        self.valid_task_loss_prev = None  # L_valid at step t-1
        self.ema_valid_task_loss_prev = None  # L_valid_ema at step t-2
        self.ratio_valid_task_loss_prev = torch.zeros(len(self.ID2TASK)).to(self.device)  # ema ratio at step t-1

        for epoch in range(start_epoch, int(self.args.num_train_epochs)):
            self.train_dataloader.set_epoch(epoch)
            self.model.train()
            start_time = time.time()

            valid_iterator = iter(self.valid_dataloader)
            for step, batch in enumerate(self.train_dataloader):
                if step == 0:
                    print_rank_0(f"step 1 batch shape: {batch['input_ids'].shape},\n"
                                 f"last 10 tokens: {batch['input_ids'][:, -10:]}")
                                # f"last 10 loss mask: {batch['loss_mask'][:, -10:]}"
                    print_rank_0(f"first 1000 tokens")
                    for pt in range(10):
                        print_rank_0(f"{batch['input_ids'][:, 10 * pt:10 * pt + 10]}")
                        # print_rank_0(f"{batch['loss_mask'][:, 10 * pt:10 * pt + 10]}")
                # self.global_steps += 1
                skipped = False
                self.model.train()
                step_start = time.time()
                if steps_trained_in_current_epoch and step < steps_trained_in_current_epoch:
                    continue
                steps_trained_in_current_epoch = 0  # 恢复到上一次的steps in current epoch后,需要置零,否则后面的每个epoch都会跳过前面的steps
                # batch = self.prepare_input(batch, self.device)
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    position_ids=batch['position_ids'].to(self.device),
                    # labels=batch['labels'],
                )
                
                loss, task_loss, task_num = self.loss_func(outputs, batch, self.args.weighted_loss_mode)
                # print(f'rank: {self.rank}, loss: {loss}, task loss: {task_loss}')

                loss = loss / self.args.gradient_accumulation_steps
                loss_tensor = torch.zeros(
                    [1], device=loss.device, dtype=loss.dtype)
                loss_tensor[0] = loss.item()
                torch.distributed.all_reduce(loss_tensor)
                torch.distributed.all_reduce(task_loss, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(task_num, op=torch.distributed.ReduceOp.SUM)
                reduce_loss = loss_tensor.sum() / torch.distributed.get_world_size()
                if has_inf_or_nan(reduce_loss):
                    print_rank_0(f'There have nan loss.')
                    self.skipped_steps += 1
                    skipped = True
                else:
                    self.accumulated_loss += reduce_loss.item()
                    mean_task_loss = task_loss / torch.distributed.get_world_size()
                    self.accumulated_task_loss += mean_task_loss.cpu().numpy()
                    self.accumulated_task_num += task_num.cpu().numpy()
                    loss.backward()
                self.global_steps += 1
                if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        # 如果是fp16，需要unscale。如果是bf16，self.optimizer里没有unscale这个方法
                        try:
                            self.optimizer.unscale_()
                        except Exception:
                            pass
                        if isinstance(self.model, FSDP):
                            self.model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    overflow = hasattr(self.optimizer, "step_was_skipped") and self.optimizer.step_was_skipped
                    if skipped != overflow:
                        print(f'skipped != overflow!!!!!!!!!!!!!!!!')
                    if not overflow:
                        self.lr_scheduler.step()
                    # if not skipped:
                    #     self.optimizer.step()
                    #     self.lr_scheduler.step()

                    self.optimizer.zero_grad()

                    step_time = time.time() - step_start
                    step_tflops = get_tflops_megatron(self.args.total_model_param, self.args.hidden_size, self.args.num_hidden_layers, 
                                                      self.args.per_device_train_batch_size, self.args.seq_length, step_time)
                    step_speed = get_computation_speed(self.args.per_device_train_batch_size, self.args.seq_length, step_time)
                    progress_bar.update(1)

                    if self.global_steps % self.args.log_interval == 0:
                        print_rank_0(f'max memory allocated: {torch.cuda.max_memory_allocated()}')
                        # print_rank_0(f'max memory allocated: {torch.cuda.max_memory_reserved()}')
                        if (self.global_steps - self.last_step_logged - self.skipped_steps) == 0:
                            self.accumulated_loss = 0
                            self.skipped_steps = 0
                            self.accumulated_task_loss = torch.zeros(len(self.ID2TASK)).to(self.device)
                            self.accumulated_task_num = torch.zeros(len(self.ID2TASK)).to(self.device)
                            self.last_step_logged = self.global_steps
                            self.epoch = self.global_steps / self.num_update_steps_per_epoch
                            print_rank_0(f'this log interval is skipped!')
                            continue

                        # train_loss = round(
                        #     self.accumulated_loss / (self.global_steps - self.last_step_logged), 4)
                        train_loss = round(
                            self.accumulated_loss / (self.global_steps - self.last_step_logged - self.skipped_steps), 4)

                        # train_task_loss = self.accumulated_task_loss / (self.global_steps - self.last_step_logged)
                        train_task_loss = self.accumulated_task_loss / (self.global_steps - self.last_step_logged - self.skipped_steps)
                        
                        if is_global_main_process():
                            logger.info('log point')
                            logger.info(f'skipped steps: {self.skipped_steps}')
                            for i in range(len(self.ID2TASK)):
                                if self.accumulated_task_num[i] != 0:
                                    logger.info(f"{self.ID2TASK[i]}_loss: {train_task_loss[i]}, sample nums: {self.accumulated_task_num[i]}")
                        self.accumulated_loss = 0
                        self.skipped_steps = 0
                        self.accumulated_task_loss = np.zeros(len(self.ID2TASK))
                        self.accumulated_task_num = np.zeros(len(self.ID2TASK))
                        self.last_step_logged = self.global_steps
                        self.epoch = self.global_steps / self.num_update_steps_per_epoch
                        learning_rate = self.lr_scheduler.get_last_lr()[0]
                        if torch.is_tensor(learning_rate):
                            learning_rate = learning_rate.item()
                        logs = {'train_loss': train_loss,
                                'epoch': self.epoch, 
                                'learning_rate': learning_rate,
                                }
                        per_task_train_loss = {self.ID2TASK[i]+'_loss': train_task_loss[i].item() for i in range(len(self.ID2TASK))}
                        logs.update(per_task_train_loss)
                        if is_global_main_process():
                            compute_mode = 'labels = -100' if batch['loss_mask'] is None else 'loss mask'
                            logger.info(f'weighted loss mode: {self.args.weighted_loss_mode}, compute mode: {compute_mode}')
                            self.log(logs, step=self.global_steps,
                                     phase='train')
                            logger.info(f"tflops: {step_tflops} | token speed: {step_speed:.2f} tokens/gpu/s | sample speed: {step_speed / self.args.seq_length:.2f}")
                    
                        if 'steps' in self.args.evaluation_strategy.split(',') and self.global_steps % self.args.valid_interval == 0:
                            exit_flag = False
                            del loss, outputs
                            metrics = self.evaluate()
                            print_rank_0(f'Global steps: {self.global_steps} evaluate metrics: {metrics}')
                            if self.args.early_stopping_patience > 0:
                                self.early_stopping(metrics['valid_loss'], self.model)
                                # 若满足 early stopping 要求
                                if self.early_stopping.early_stop:
                                    exit_flag = True
                                    print("Early stopping")
                                    # 结束模型训练
                                    break
                        if 'steps' in self.args.save_strategy.split(',') and self.global_steps % self.args.checkpointing_steps == 0:
                            self.save(metrics=metrics)

                        if self.global_steps >= self.args.max_steps:
                            break

            if exit_flag:
                print("Early stopping")
                break
            logger.info(f"Training of epoch {epoch + 1} finished")

            training_time += time.time() - start_time
            if 'epoch' in self.args.evaluation_strategy.split(','):
                metrics = self.evaluate()
                if self.args.early_stopping_patience > 0:
                    self.early_stopping(metrics['valid_loss'], self.model)
                    # 若满足 early stopping 要求
                    if self.early_stopping.early_stop:
                        exit_flag = True
                        print("Early stopping")
                        # 结束模型训练
                        break
            if 'epoch' in self.args.save_strategy.split(',') or self.args.extra_save_by_epoch:
                if epoch + 1 < (int(self.args.num_train_epochs) // 3) and self.args.total_model_param < 1e9:
                    continue
                print_rank_0(f'Global steps: {self.global_steps} | Epoch {epoch + 1} checkpoint metrics: {metrics}')
                self.save(
                    suffix=f'{self.global_steps}-{EPOCH_CHECKPOINT_NAME}-{epoch + 1}')  # 不用保留epoch级别的atorch_checkpoint.bin
