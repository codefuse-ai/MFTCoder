"""
# @author Chaoyu Chen
# @date 2024/4/12
# @module trainer.py

Accelerate + DeepSpeed/FSDP 
QLoRA/LoRA/Full + SFT/MFT/MPT

Trainer
"""

import gc
import os
import sys
import threading
import argparse
import math
import logging
import json
import time
import transformers
import numpy as np
import psutil
import shutil
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple, Union
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from accelerate import Accelerator
from transformers import set_seed

# sys.path.append("..")
from utils.common_utils import generate_task_id, TASK2ID, ID2TASK
from utils.loss_utils import loss_func_mft, SelfpacedStatus, load_balancing_loss_func

logger = get_logger(__name__)


def copy_tokenizer_files(mode_path: str, files_list: List[str], save_path: str):
    # create path if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # copy each file in files_list to save_path
    for filename in files_list:
        src_file = os.path.join(mode_path, filename)

        # copy only if src exists
        if os.path.exists(src_file):
            dest_file = os.path.join(save_path, filename)

            # copy
            shutil.copy(src_file, dest_file)
            print(f"Copied {filename} to {save_path}")
        else:
            print(f"File {filename} does not exist in {mode_path}")


def check_existing_ckpts(output_dir):
    prefix = "step_"

    if not os.path.exists(output_dir):
        return []
    # list all files and dirs
    contents = os.listdir(output_dir)

    # find dirs starts with "step_"
    matching_folders = [
        folder for folder in contents if os.path.isdir(os.path.join(output_dir, folder)) and folder.startswith(prefix)
    ]

    return matching_folders


def extract_epochs_and_steps(path, num_update_steps_per_epoch, gradient_accumulation_steps):
    """
    extract starting_epoch, completed_steps, resume_step of train_dataloader for resumed training
    """
    # Extract `epoch_{i}` or `step_{i}`
    training_difference = os.path.splitext(path)[0]

    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", ""))
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch
        logger.info(f"Resume from exact Epoch {starting_epoch}: completed_steps {completed_steps}")
    else:
        # need to multiply `gradient_accumulation_steps` to reflect real steps
        completed_steps = int(training_difference.replace("step_", ""))
        starting_epoch = completed_steps // num_update_steps_per_epoch
        resume_step = (completed_steps % num_update_steps_per_epoch) * gradient_accumulation_steps
        logger.info(f"Resume from Epoch {starting_epoch} + step {resume_step}: completed_steps {completed_steps}")

    return starting_epoch, completed_steps, resume_step


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f"{key}", value, completed_steps)


def delete_ckpts_over_limits(output_dir, saving_limit, best_step):
    """delete ckpts more than saving_limits except for the best_step ckpt"""
    existing_ckpts = check_existing_ckpts(output_dir)
    logger.info(f"Existing step ckpts folders: {existing_ckpts}, best step ckpt: step_{best_step}")
    # sorted only step num ascendingly
    ckpt_steps = sorted([int(ckpt.replace("step_", "")) for ckpt in existing_ckpts])
    # delete the oldest steps except for the best step at present
    if len(ckpt_steps) > saving_limit:
        deletable_steps = [ckpt_step for ckpt_step in ckpt_steps if ckpt_step != best_step]
        # print(deletable_steps[:len(ckpt_steps) - saving_limit])
        for del_step in deletable_steps[: len(ckpt_steps) - saving_limit]:
            shutil.rmtree(os.path.join(output_dir, f"step_{del_step}"))
            logger.info(f"Removed ckpt step_{del_step}")


class MftTrainer:
    """
    Multitask FineTuing Trainer, supporting MFT/SFT/ContinueTrain with Lora/Qlora/Full-parameters.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        model,
        model_config,
        train_dataloader,
        valid_dataloader,
        optimizer,
        lr_scheduler,
        tokenizer,
        num_update_steps_per_epoch,
        total_train_dataset_size,
        args,
    ):
        self.accelerator = accelerator
        self.model = model
        # hf model config
        self.model_config = model_config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.num_update_steps_per_epoch = num_update_steps_per_epoch
        self.total_train_dataset_size = total_train_dataset_size
        # training arguments
        self.args = args
        # tensorboard writer
        self.summary_writer = SummaryWriter(log_dir=args.tb_dir)
        self.default_writer = SummaryWriter(log_dir="/home/admin/logs/tfevent")

    def print(self, msg: str):
        """
        accelerator print, default on main process
        Args:
            msg:

        Returns:

        """
        self.accelerator.print(msg)

    def touch(self, batch, num_tokens=10):
        """touch first and last tokens and labels for debugging usage"""
        self.print(
            f"step 1 batch shape: {batch['input_ids'].shape},\n"
            f"last {num_tokens} labels: {batch['labels'][:, -num_tokens:]}"
            f"last {num_tokens} loss mask: {batch['loss_mask'][:, -num_tokens:]}"
        )
        self.print(f"first {num_tokens} input_ids and loss_mask")
        for pt in range(1):
            self.print(f"{batch['input_ids'][:, num_tokens * pt: num_tokens * pt + num_tokens]}")
            self.print(f"{batch['loss_mask'][:, num_tokens * pt: num_tokens * pt + num_tokens]}")

    @staticmethod
    def format_tensor(tensor, n):
        return list(map(lambda x: round(x, n), tensor.tolist()))

    def accelerate_saving_checkpoint(self, output_dir: str, completed_steps: int):
        """
        Saving lora adaptor or full checkpoint using accelerator
        Args:
            output_dir: exact dir for saving ckpt
            completed_steps:

        Returns:

        """
        self.accelerator.wait_for_everyone()

        logger.info(f"[CHECKPOINT] Saving checkpoint", main_process_only=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        # self.print(f"unwrapped model type {type(unwrapped_model)}")
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.accelerator.get_state_dict(self.model),
        )
        self.accelerator.wait_for_everyone()
        # for full-parameter training, save whole ckpt and tokenizer together because it does not need a merge.
        if not self.args.peft_type and self.accelerator.is_main_process:
            if self.args.model_type.lower() == "deepseek":
                copy_tokenizer_files(
                    self.args.pretrained_model_path, ["tokenizer.json", "tokenizer_config.json"], output_dir
                )
            else:
                self.tokenizer.save_pretrained(output_dir)
        
            sf = os.path.join(output_dir, "model.safetensors")
            index_file = os.path.join(output_dir, "model.safetensors.index.json")
            if os.path.isfile(sf) and os.path.isfile(index_file):
                self.print(f"Remove bug dummy ckpt {sf}")
                os.remove(sf)

        if self.accelerator.is_main_process:
            latest = {
                "latest_ckpt": output_dir,
                "lr": self.optimizer.param_groups[0]["lr"],
                # 1 step back because ckping is after schuduler.step()
                # "scheduler_last_ep": self.lr_scheduler.state_dict().get("last_epoch", 0) - 1,
            }
            with open(os.path.join(self.args.output_dir, "latest"), "w") as f:
                json.dump(latest, f, indent=2)

            logger.info(
                f"[CHECKPOINT][complete_steps={completed_steps}], checkpoint {output_dir} saved, latest: {latest}",
                main_process_only=True,
            )
        self.accelerator.wait_for_everyone()

    def accelerate_monitor(
        self,
        reduce_loss,
        reduce_task_loss,
        reduce_task_exist,
        completed_steps,
        selfpaced_status=None,
    ):
        """
        gather reduce_loss and reduce_task_loss from all N devices.
        train logging and tensorboarding.
        """
        # gather reduce_loss and reduce_task_loss from all N devices
        reduce_losses = self.accelerator.gather(reduce_loss).detach().float()
        reduce_task_losses = self.accelerator.gather(reduce_task_loss).reshape(-1, len(ID2TASK))
        reduce_task_exists = self.accelerator.gather(reduce_task_exist).reshape(-1, len(ID2TASK))

        # get train loss and per-task train loss
        train_loss = torch.mean(reduce_losses) / (self.args.log_interval * self.args.gradient_accumulation_steps)
        # train_task_loss = torch.mean(reduce_task_losses, dim=0) / (self.args.log_interval * self.args.gradient_accumulation_steps)
        train_task_loss = torch.sum(reduce_task_losses, dim=0) / torch.sum(reduce_task_exists, dim=0)

        # logging and writing tensorboard
        logger.info(
            f"[TRAIN][complete_steps={completed_steps}][train_loss={train_loss:.6f}]"
            f"[train_task_loss={self.format_tensor(train_task_loss, 4)}]"
            f"[gather shape={list(reduce_losses.shape)}]"
            f"[lr={self.lr_scheduler.get_lr()[0]:.4e}, {self.optimizer.param_groups[0]['lr']:.4e}]",
            main_process_only=True,
        )
        if selfpaced_status is not None:
            if completed_steps > selfpaced_status.selfpaced_history_length:
                selfpaced_status.log_per_task_weight = selfpaced_status.log_per_task_weight / torch.sum(
                    selfpaced_status.log_per_task_weight
                )
            else:
                selfpaced_status.log_per_task_weight = torch.ones(len(ID2TASK)) / len(ID2TASK)
            logger.info(
                f"[TRAIN][per_task_train_weight={selfpaced_status.log_per_task_weight}]", main_process_only=True
            )
        train_log_dict = {"Loss/train": train_loss}
        for i in range(len(ID2TASK)):
            train_log_dict[f"{ID2TASK[i]}_loss/train"] = train_task_loss[i]
            if selfpaced_status is not None:
                train_log_dict[f"{ID2TASK[i]}_selfpaced_weight/train"] = selfpaced_status.log_per_task_weight[i].item()

        if self.accelerator.is_main_process:
            write_tensorboard(self.summary_writer, train_log_dict, completed_steps)
            write_tensorboard(self.default_writer, train_log_dict, completed_steps)

        if selfpaced_status is not None:
            selfpaced_status.log_per_task_weight = torch.zeros(len(ID2TASK))

    def accelerate_evaluate(
        self,
        completed_steps,
        step,
        min_eval_loss,
        stall_num,
        best_step,
    ):
        """
        evaluate the model at current completed_steps on valid_dataloader and gather eval_loss on all devices.
        eval logging and tensorboarding.
        """
        losses = []
        accumulated_task_loss = torch.zeros(len(ID2TASK)).to(self.model.device)
        accumulated_task_exist = torch.zeros(len(ID2TASK)).to(self.model.device)
        for valid_step, valid_batch in enumerate(self.valid_dataloader):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=valid_batch["input_ids"],
                    attention_mask=valid_batch["attention_mask"],
                    position_ids=valid_batch["position_ids"],
                    return_dict=True,
                )

                loss, task_loss, _ = loss_func_mft(
                    outputs=outputs,
                    labels=valid_batch["labels"],
                    task_mask=valid_batch["task_mask"],
                    task_id=valid_batch["task_id"],
                    weighted_loss_mode=self.args.weighted_loss_mode,
                    loss_mask=valid_batch["loss_mask"],
                    task_weights=self.args.task_weights,
                )

                losses.append(self.accelerator.gather(loss.repeat(self.args.per_device_eval_batch_size)))
                accumulated_task_loss += task_loss.detach().float()
                accumulated_task_exist += (task_loss != 0.0).detach().float()

        self.accelerator.wait_for_everyone()
        valid_batch_num = len(losses)
        gathered_size = losses[0].shape
        losses = torch.cat(losses)
        # task_losses = torch.cat(task_losses).reshape(-1, len(ID2TASK))
        task_losses = self.accelerator.gather(accumulated_task_loss).reshape(-1, len(ID2TASK))
        task_exists = self.accelerator.gather(accumulated_task_exist).reshape(-1, len(ID2TASK))

        try:
            eval_loss = torch.mean(losses)
            # eval_task_loss = torch.mean(task_losses, dim=0) / valid_batch_num
            eval_task_loss = torch.sum(task_losses, dim=0) / torch.sum(task_exists, dim=0)
            if eval_loss <= min_eval_loss:
                min_eval_loss = eval_loss
                stall_num = 0
                best_step = completed_steps
            else:
                stall_num += 1
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(
            f"[EVAL][completed_steps={completed_steps}]"
            f"[eval_loss={eval_loss:.6f}][eval_task_loss={self.format_tensor(eval_task_loss, 4)}]"
            f"[perplexity={perplexity:.4f}][valid_batch_num={valid_batch_num}]"
            f"[gather_size={list(gathered_size)}]",
            main_process_only=True,
        )
        eval_log_dict = {
            "Loss/valid": eval_loss,
            "Perplexity/valid": perplexity,
            "Epochs": round(completed_steps / self.num_update_steps_per_epoch, 2),
        }
        for i in range(len(ID2TASK)):
            eval_log_dict[f"{ID2TASK[i]}_loss/valid"] = eval_task_loss[i]

        if self.accelerator.is_main_process:
            write_tensorboard(self.summary_writer, eval_log_dict, completed_steps)
            write_tensorboard(self.default_writer, eval_log_dict, completed_steps)

        return eval_loss, eval_task_loss, min_eval_loss, stall_num, best_step

    def accelerate_train(self):
        # Train!
        if self.args.seed is not None:
            set_seed(self.args.seed)

        global_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        logger.info("************************************** Running training ****************************************")
        logger.info(f"  Num examples = {self.total_train_dataset_size}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total global train batch size (w. parallel, distributed & accumulation) = {global_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization(update/completed) steps = {self.args.max_train_steps}")
        logger.info(f"  Complete/optimize steps per Epoch = {self.args.max_train_steps // self.args.num_train_epochs}")
        logger.info("************************************************************************************************")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)

        # set starting_epoch, completed_steps and resume_step of train_dataloader
        completed_steps = 0
        starting_epoch = 0
        resume_step = None

        if self.args.resume_from_checkpoint:
            path = os.path.basename(self.args.resume_from_checkpoint)
            starting_epoch, completed_steps, resume_step = extract_epochs_and_steps(
                path, self.num_update_steps_per_epoch, self.args.gradient_accumulation_steps
            )

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        # monitor minimum eval_loss, stalling num, and best_step
        min_eval_loss = float("inf")
        stall_num = 0
        best_step = None

        # monitor train loss
        reduce_loss = torch.tensor(0.0).to(self.model.device)
        reduce_aux_loss = torch.tensor(0.0).to(self.model.device)
        reduce_task_loss = torch.zeros(len(ID2TASK)).to(self.model.device)
        reduce_task_exist = torch.zeros(len(ID2TASK)).to(self.model.device)
        per_task_weight = self.args.task_weights

        if self.args.weighted_loss_mode == "selfpaced":
            selfpaced_status = SelfpacedStatus(
                self.args.selfpaced_scale_factor,
                self.args.selfpaced_interval,
                self.args.selfpaced_history_length,
                self.args.selfpaced_sample_valid_num,
                self.valid_dataloader,
            )
            selfpaced_status.sample_valid_batch(self.model, completed_steps)
            selfpaced_status.valid_iterator = iter(selfpaced_status.valid_dataloader)
        else:
            selfpaced_status = None

        # Training Loop!
        for epoch in range(starting_epoch, self.args.num_train_epochs):
            # set_epoch 
            self.train_dataloader.set_epoch(epoch)
            
            # if we early stop by some ckpts not converging
            if self.args.early_stopping and stall_num == self.args.early_stopping_stall_num:
                break

            if self.args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
            else:
                active_dataloader = self.train_dataloader
            tail_num = len(active_dataloader) - len(active_dataloader) % self.args.gradient_accumulation_steps
            print(f"length of dataloader: {len(active_dataloader)}")

            self.model.train()
            # Inner Loop!
            for step, batch in enumerate(active_dataloader):
                if step == tail_num:
                    break
                with self.accelerator.accumulate(self.model):
                    if step == 0:
                        self.touch(batch, num_tokens=10)
                    # forward
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"],
                        return_dict=True,
                    )

                    if (
                        self.args.weighted_loss_mode == "selfpaced"
                        and step % self.args.gradient_accumulation_steps == 0
                        and completed_steps % self.args.selfpaced_interval == 0
                        and completed_steps >= self.args.selfpaced_history_length
                    ):
                        per_task_weight = selfpaced_status.compute_per_task_weight(completed_steps=completed_steps)
                        selfpaced_status.log_per_task_weight += per_task_weight

                    # loss
                    loss, task_loss, _ = loss_func_mft(
                        outputs=outputs,
                        labels=batch["labels"],
                        task_mask=batch["task_mask"],
                        task_id=batch["task_id"],
                        weighted_loss_mode=self.args.weighted_loss_mode,
                        loss_mask=batch["loss_mask"],
                        task_weights=per_task_weight,
                    )

                    # accelerator.print(len(outputs.router_logits), outputs.router_logits[0], outputs.router_logits[-1])
                    # accelerator.print(batch['attention_mask'].shape, batch['attention_mask'])
                    aux_loss = None
                    if hasattr(self.model_config, "output_router_logits") and self.model_config.output_router_logits:
                        if hasattr(self.model_config, "num_local_experts"):
                            num_experts = self.model_config.num_local_experts
                        elif hasattr(self.model_config, "num_experts"):
                            num_experts = self.model_config.num_experts
                        else:
                            raise ValueError("model has no attribute num_local_experts or num_experts")
                        aux_loss = load_balancing_loss_func(
                            outputs.router_logits,
                            num_experts,
                            self.model_config.num_experts_per_tok,
                            batch["attention_mask"],
                        )
                        aux_loss = self.model_config.router_aux_loss_coef * aux_loss.to(loss.device)
                        loss += aux_loss  # make sure to reside in the same device

                    # backward
                    self.accelerator.backward(loss)
                    # print(self.lr_scheduler.state_dict(), self.accelerator.process_index)
                    # update(sync_gradients)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    # support args.min_lr
                    if self.optimizer.param_groups[0]["lr"] <= self.args.min_lr:
                        self.optimizer.param_groups[0]["lr"] = self.args.min_lr

                    # accumulate resuce_loss and reduce_task_loss in a log_interval
                    if not torch.isnan(loss):
                        reduce_loss += loss.detach().float()
                    if aux_loss and not torch.isnan(aux_loss):
                        reduce_aux_loss += aux_loss.detach().float()
                    # self.print("task loss devices: ", reduce_task_loss.device, task_loss.device)
                    reduce_task_loss += task_loss.detach().float()
                    reduce_task_exist += (task_loss != 0).detach().float()

                    # If the accelerator has performed an optimization step behind the scenes, thus a completed_step done.
                    if self.accelerator.sync_gradients:
                        if (
                            self.args.weighted_loss_mode == "selfpaced"
                            and completed_steps % self.args.selfpaced_interval == 0
                            and completed_steps >= 1
                        ):
                            selfpaced_status.sample_valid_batch(self.model, completed_steps)

                        # progress_bar.update(1)
                        completed_steps += 1
                        # monitoring training process and logging and tensorboarding
                        if completed_steps % self.args.log_interval == 0:
                            progress_bar.update(self.args.log_interval)
                            if reduce_aux_loss > 0.0:
                                self.print(f"[INFO] aux_loss: {reduce_aux_loss/self.args.log_interval}")
                            self.accelerate_monitor(
                                reduce_loss,
                                reduce_task_loss,
                                reduce_task_exist,
                                completed_steps,
                                selfpaced_status,
                            )
                            # reset reduce_loss
                            reduce_loss = torch.tensor(0.0).to(self.model.device)
                            reduce_aux_loss = torch.tensor(0.0).to(self.model.device)
                            reduce_task_loss = torch.zeros(len(ID2TASK)).to(self.model.device)
                            reduce_task_exist = torch.zeros(len(ID2TASK)).to(self.model.device)

                        # steps checkpointing
                        if self.args.checkpointing_steps and completed_steps % self.args.checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if self.args.output_dir is not None:
                                output_dir = os.path.join(self.args.output_dir, output_dir)
                            self.accelerate_saving_checkpoint(output_dir, completed_steps)

                        # steps evaluation
                        if completed_steps % self.args.evaluation_steps == 0:
                            self.model.eval()
                            eval_loss, eval_task_loss, min_eval_loss, stall_num, best_step = self.accelerate_evaluate(
                                completed_steps,
                                step,
                                min_eval_loss,
                                stall_num,
                                best_step,
                            )
                            self.model.train()

                            # delete ckpts over args.saving_limit
                            if self.accelerator.is_main_process and self.args.saving_limit:
                                delete_ckpts_over_limits(self.args.output_dir, self.args.saving_limit, best_step)

                            # early stoppin when stalling more than args.early_stopping_stall_num
                            if self.args.early_stopping and stall_num == self.args.early_stopping_stall_num:
                                self.print(f"[WARNING] Early stopping at {completed_steps}")
                                break

                        if completed_steps >= self.args.max_train_steps:
                            break
                        self.accelerator.wait_for_everyone()

            # epoch checkpointing
            if self.args.epoch_checkpointing:
                output_dir = f"epoch_{epoch + 1}"
                if self.args.output_dir is not None:
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                self.accelerate_saving_checkpoint(output_dir, completed_steps)

        self.summary_writer.close()
        self.default_writer.close()

        # final save
        # output_dir = f"final_step_{completed_steps}"
        # if self.args.output_dir is not None:
        #     output_dir = os.path.join(self.args.output_dir, output_dir)
        # self.accelerate_saving_checkpoint(output_dir, completed_steps)
