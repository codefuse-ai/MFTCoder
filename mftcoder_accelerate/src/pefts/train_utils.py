"""
# @author Chaoyu Chen
# @date 2023/10/19
# @module train_utils.py

Accelerate + DeepSpeed zero stage2 + DistributedDataParallel
QLoRA/LoRA/Full + MFT/MPT, resource and parameters efficient training

training functions
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
from tqdm.auto import tqdm

sys.path.append("..")
from utils.common_utils import generate_task_id, TASK2ID, ID2TASK
from utils.auto_accelerate_utils import loss_func_mft, SelfpacedStatus
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger

logger = get_logger(__name__)


def check_existing_ckpts(output_dir):
    prefix = "step_"

    if not os.path.exists(output_dir):
        return []
    # 列出目录中的所有文件和文件夹
    contents = os.listdir(output_dir)

    # 使用列表推导式筛选以"step_"开头的文件夹
    matching_folders = [folder for folder in contents if
                        os.path.isdir(os.path.join(output_dir, folder)) and folder.startswith(prefix)]

    return matching_folders


def extract_epochs_and_steps(path, num_update_steps_per_epoch, gradient_accumulation_steps):
    """
    extract starting_epoch, completed_steps, resume_step of train_dataloader for resumed training
    """
    # Extract `epoch_{i}` or `step_{i}`
    training_difference = os.path.splitext(path)[0]

    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "")) + 1
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch
        print(f"resume from epoch {starting_epoch} and completed_steps {completed_steps}")
    else:
        # need to multiply `gradient_accumulation_steps` to reflect real steps
        completed_steps = int(training_difference.replace("step_", ""))
        starting_epoch = completed_steps // num_update_steps_per_epoch
        resume_step = (completed_steps % num_update_steps_per_epoch) * gradient_accumulation_steps
        print(f"resume from epoch {starting_epoch} resusme step {resume_step} and completed_steps {completed_steps}")

    return starting_epoch, completed_steps, resume_step


def write_tensorboard(summary_writer: SummaryWriter, log_dict: dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f'{key}', value, completed_steps)


def accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir: str, completed_steps: int, args):
    """
    Saving lora adaptor or full checkpoint using accelerator
    """
    accelerator.wait_for_everyone()

    logger.info(
        f"[CHECKPOINT] Saving checkpoint",
        main_process_only=True
    )
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
    )
    # for full-parameter training, save whole ckpt and tokenizer together because it does not need a merge.
    if not args.peft_type and accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

    logger.info(
        f"[CHECKPOINT][complete_steps={completed_steps}], checkpoint {output_dir} saved",
        main_process_only=True
    )
    accelerator.wait_for_everyone()


def accelerate_monitor(accelerator, model, reduce_loss, reduce_task_loss, reduce_task_exist, args, completed_steps,
                       lr_scheduler, optimizer, summary_writer, selfpaced_status=None):
    """
    gather reduce_loss and reduce_task_loss from all N devices.
    train logging and tensorboarding.
    """
    # gather reduce_loss and reduce_task_loss from all N devices
    reduce_losses = accelerator.gather(reduce_loss).detach().float()
    reduce_task_losses = accelerator.gather(reduce_task_loss).reshape(-1, len(ID2TASK))
    reduce_task_exists = accelerator.gather(reduce_task_exist).reshape(-1, len(ID2TASK))
    # get train loss and per-task train loss
    train_loss = torch.mean(reduce_losses) / (args.log_interval * args.gradient_accumulation_steps)
    # train_task_loss = torch.mean(reduce_task_losses, dim=0) / (args.log_interval * args.gradient_accumulation_steps)
    train_task_loss = torch.sum(reduce_task_losses, dim=0) / torch.sum(reduce_task_exists, dim=0)

    # logging and tensorboard
    logger.info(
        f"[TRAIN][complete_steps={completed_steps}][train_loss={train_loss:.6f}][train_task_loss={train_task_loss}]"
        f"[gather shape={reduce_losses.shape}][lr={lr_scheduler.get_lr()[0]:.4e}, {optimizer.param_groups[0]['lr']:.4e}]",
        main_process_only=True)
    if selfpaced_status is not None:
        if completed_steps > selfpaced_status.selfpaced_history_length:
            selfpaced_status.log_per_task_weight = selfpaced_status.log_per_task_weight / torch.sum(selfpaced_status.log_per_task_weight)
        else:
            selfpaced_status.log_per_task_weight = torch.ones(len(ID2TASK)) / len(ID2TASK)
        logger.info(f"[TRAIN][per_task_train_weight={selfpaced_status.log_per_task_weight}]", main_process_only=True)
    train_log_dict = {"training_loss": train_loss}
    for i in range(len(ID2TASK)):
        train_log_dict[f"{ID2TASK[i]}_train_loss"] = train_task_loss[i]
        if selfpaced_status is not None:
            train_log_dict[f"{ID2TASK[i]}_train_selfpaced_weight"] = selfpaced_status.log_per_task_weight[i].item()

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, train_log_dict, completed_steps)
    
    if selfpaced_status is not None:
        selfpaced_status.log_per_task_weight = torch.zeros(len(ID2TASK))


def accelerate_evaluate(accelerator, model, valid_dataloader, args, completed_steps, step, min_eval_loss, stall_num,
                        best_step, summary_writer):
    """
    evaluate the model at current completed_steps on valid_dataloader and gather eval_loss on all devices.
    eval logging and tensorboarding.
    """
    losses = []
    accumulated_task_loss = torch.zeros(len(ID2TASK)).to(model.device)
    accumulated_task_exist = torch.zeros(len(ID2TASK)).to(model.device)
    for valid_step, valid_batch in enumerate(valid_dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=valid_batch['input_ids'],
                attention_mask=valid_batch['attention_mask'],
                position_ids=valid_batch['position_ids'],
                return_dict=True,
            )

            loss, task_loss, _ = loss_func_mft(
                outputs=outputs,
                labels=valid_batch['labels'],
                task_mask=valid_batch['task_mask'],
                task_id=valid_batch['task_id'],
                weighted_loss_mode=args.weighted_loss_mode,
                loss_mask=valid_batch['loss_mask'],
                task_weights=args.task_weights
            )

            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
            accumulated_task_loss += task_loss.detach().float()
            accumulated_task_exist += (task_loss != 0.0).detach().float()

    accelerator.wait_for_everyone()
    valid_batch_num = len(losses)
    gathered_size = losses[0].shape
    losses = torch.cat(losses)
    # task_losses = torch.cat(task_losses).reshape(-1, len(ID2TASK))
    task_losses = accelerator.gather(accumulated_task_loss).reshape(-1, len(ID2TASK))
    task_exists = accelerator.gather(accumulated_task_exist).reshape(-1, len(ID2TASK))

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

    logger.info(f"[EVAL][global_steps={step + 1}][completed_steps={completed_steps}]"
                f"[valid_batch_num={valid_batch_num}], [gather_size={gathered_size}]"
                f"[perplexity={perplexity:.4f}][eval_loss={eval_loss:.6f}]"
                f"[eval_task_loss={eval_task_loss}]",
                main_process_only=True)
    eval_log_dict = {"valid_loss": eval_loss, "perplexity": perplexity}
    for i in range(len(ID2TASK)):
        eval_log_dict[f"{ID2TASK[i]}_valid_loss"] = eval_task_loss[i]

    if accelerator.is_main_process:
        write_tensorboard(summary_writer, eval_log_dict, completed_steps)

    return eval_loss, eval_task_loss, min_eval_loss, stall_num, best_step


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
        for del_step in deletable_steps[:len(ckpt_steps) - saving_limit]:
            shutil.rmtree(os.path.join(output_dir, f"step_{del_step}"))
            logger.info(f"Removed ckpt step_{del_step}")


def touch_print(accelerator, batch, num_tokens=10):
    """touch first and last tokens and labels for debugging usage"""
    accelerator.print(f"step 1 batch shape: {batch['input_ids'].shape},\n"
                      f"last {num_tokens} labels: {batch['labels'][:, -num_tokens:]}"
                      f"last {num_tokens} loss mask: {batch['loss_mask'][:, -num_tokens:]}")
    accelerator.print(f"first {num_tokens} input_ids and loss_mask")
    for pt in range(1):
        accelerator.print(f"{batch['input_ids'][:, num_tokens * pt: num_tokens * pt + num_tokens]}")
        accelerator.print(f"{batch['loss_mask'][:, num_tokens * pt: num_tokens * pt + num_tokens]}")


def accelerate_train(accelerator, model, train_dataloader, valid_dataloader, optimizer, lr_scheduler, tokenizer,
                     num_update_steps_per_epoch, total_train_dataset_size, args):
    # tensorboard writer
    summary_writer = SummaryWriter(log_dir=args.tb_dir)
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("**************************************** Running training ****************************************")
    logger.info(f"  Num examples = {total_train_dataset_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total global train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization(update/completed) steps = {args.max_train_steps}")
    logger.info(f"  Complete/Optimization steps per Epoch = {args.max_train_steps // args.num_train_epochs}")
    logger.info("***************************************************************************************************")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # set starting_epoch, completed_steps and resume_step of train_dataloader
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        starting_epoch, completed_steps, resume_step = extract_epochs_and_steps(
            path, num_update_steps_per_epoch, args.gradient_accumulation_steps
        )
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # monitor minimum eval_loss, stalling num, and best_step
    min_eval_loss = float('inf')
    stall_num = 0
    best_step = None
    
    # monitor train loss
    reduce_loss = 0
    reduce_task_loss = torch.zeros(len(ID2TASK)).to(model.device)
    reduce_task_exist = torch.zeros(len(ID2TASK)).to(model.device)
    per_task_weight = args.task_weights

    if args.weighted_loss_mode == "selfpaced":
        selfpaced_status = SelfpacedStatus(args.selfpaced_scale_factor, args.selfpaced_interval, args.selfpaced_history_length, args.selfpaced_sample_valid_num, valid_dataloader)
        selfpaced_status.sample_valid_batch(model, completed_steps)
        selfpaced_status.valid_iterator = iter(selfpaced_status.valid_dataloader)
    else:
        selfpaced_status = None

    # Training Loop!
    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.early_stopping and stall_num == args.early_stopping_stall_num:
            break

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        tail_num = len(active_dataloader) - len(active_dataloader) % args.gradient_accumulation_steps
        print(f"length of dataloader: {len(active_dataloader)}")

        model.train()
        # Inner Loop!
        for step, batch in enumerate(active_dataloader):
            if step == tail_num:
                break
            with accelerator.accumulate(model):
                if step == 0:
                    touch_print(accelerator, batch, num_tokens=10)
                # forward
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    position_ids=batch['position_ids'],
                    return_dict=True
                )

                if args.weighted_loss_mode == 'selfpaced' and step % args.gradient_accumulation_steps == 0 and completed_steps % args.selfpaced_interval == 0 and completed_steps >= args.selfpaced_history_length:
                    per_task_weight = selfpaced_status.compute_per_task_weight(completed_steps=completed_steps)
                    selfpaced_status.log_per_task_weight += per_task_weight

                # loss
                loss, task_loss, _ = loss_func_mft(
                    outputs=outputs,
                    labels=batch['labels'],
                    task_mask=batch['task_mask'],
                    task_id=batch['task_id'],
                    weighted_loss_mode=args.weighted_loss_mode,
                    loss_mask=batch['loss_mask'],
                    task_weights=per_task_weight
                )

                # backward
                accelerator.backward(loss)

                # update(sync_gradients)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # support args.min_lr
                if optimizer.param_groups[0]['lr'] <= args.min_lr:
                    optimizer.param_groups[0]['lr'] = args.min_lr

                # accumulate resuce_loss and reduce_task_loss in a log_interval
                if not torch.isnan(loss):
                    reduce_loss += loss.detach().float()
                # accelerator.print("task loss devices: ", reduce_task_loss.device, task_loss.device)
                reduce_task_loss += task_loss.detach().float()
                reduce_task_exist += (task_loss != 0).detach().float()

                # If the accelerator has performed an optimization step behind the scenes, thus a completed_step done.
                if accelerator.sync_gradients:
                    if args.weighted_loss_mode == 'selfpaced' and completed_steps % args.selfpaced_interval == 0 and completed_steps >= 1:
                        selfpaced_status.sample_valid_batch(model, completed_steps)

                    # progress_bar.update(1)
                    completed_steps += 1
                    # monitoring training process and logging and tensorboarding
                    if completed_steps % args.log_interval == 0:
                        progress_bar.update(args.log_interval)
                        accelerate_monitor(
                            accelerator, model, reduce_loss, reduce_task_loss, reduce_task_exist, args, completed_steps,
                            lr_scheduler, optimizer, summary_writer, selfpaced_status
                        )
                        # reset reduce_loss
                        reduce_loss = 0
                        reduce_task_loss = torch.zeros(len(ID2TASK)).to(model.device)
                        reduce_task_exist = torch.zeros(len(ID2TASK)).to(model.device)

                    # steps checkpointing
                    if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir, completed_steps, args)

                    # steps evaluation
                    if completed_steps % args.evaluation_steps == 0:
                        model.eval()
                        eval_loss, eval_task_loss, min_eval_loss, stall_num, best_step = accelerate_evaluate(
                            accelerator, model, valid_dataloader, args, completed_steps, step,
                            min_eval_loss, stall_num, best_step, summary_writer
                        )
                        model.train()

                        # delete ckpts over args.saving_limit
                        if accelerator.is_main_process and args.saving_limit:
                            delete_ckpts_over_limits(args.output_dir, args.saving_limit, best_step)

                        # early stoppin when stalling more than args.early_stopping_stall_num
                        if args.early_stopping and stall_num == args.early_stopping_stall_num:
                            accelerator.print(f"[WARNING] Early stopping at {completed_steps}")
                            break

                    if completed_steps >= args.max_train_steps:
                        break
                    accelerator.wait_for_everyone()

        # epoch checkpointing
        if args.epoch_checkpointing:
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir, completed_steps, args)

    summary_writer.close()

    # final save
    output_dir = f"final_step_{completed_steps}"
    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, output_dir)
    accelerate_saving_checkpoint(accelerator, model, tokenizer, output_dir, completed_steps, args)
