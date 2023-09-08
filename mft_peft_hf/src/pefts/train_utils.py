"""
# @author Chaoyu Chen
# @date 2023/7/19
# @module train_utils.py

Hugging face accelerate + deepspeed zero stage2 + DP
QLoRA + MFT Training
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
import torch
from torch import nn
from tqdm.auto import tqdm
sys.path.append("..")
from utils.common_utils import generate_task_id, TASK2ID, ID2TASK
from utils.auto_accelerate_utils import loss_func_mft
from torch.utils.tensorboard import SummaryWriter
from accelerate.logging import get_logger
logger = get_logger(__name__)

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2 ** 20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def accelerate_train(accelerator, model, train_dataloader, valid_dataloader, optimizer, lr_scheduler, num_update_steps_per_epoch, total_train_dataset_size, args):
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
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # 配置starting_epoch and completed_steps 从哪里开始训练
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
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
            resume_step = (completed_steps % num_update_steps_per_epoch) * args.gradient_accumulation_steps
            print(f"resume from epoch {starting_epoch} resusme step {resume_step} and completed_steps {completed_steps}")

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    min_eval_loss = float('inf')
    stall_num = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.early_stopping and stall_num == args.early_stopping_stall_num:
            break
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            reduce_loss = 0
            reduce_task_loss = torch.zeros(len(ID2TASK)).to(model.device)
            reduce_task_exist = torch.zeros(len(ID2TASK)).to(model.device)
            t3 = time.time()
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            tail_num = len(active_dataloader) - len(active_dataloader) % args.gradient_accumulation_steps
            print(f"length of dataloader: {len(active_dataloader)}")
            for step, batch in enumerate(active_dataloader):
                if step == tail_num:
                    break
                with accelerator.accumulate(model):
                    if step == 0:
                        accelerator.print(f"step 1 batch shape: {batch['input_ids'].shape},\n"
                                f"last 10 tokens: {batch['input_ids'][:, -10:]}"
                                f"last 10 loss mask: {batch['loss_mask'][:, -10:]}")
                        accelerator.print(f"first 10 tokens and loss_mask")
                        for pt in range(1):
                            accelerator.print(f"{batch['input_ids'][:, 10 * pt:10 * pt + 10]}")
                            accelerator.print(f"{batch['loss_mask'][:, 10 * pt:10 * pt + 10]}")
                                
                    t4 = time.time()
                    # forward & loss
                    outputs = model(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    position_ids=batch['position_ids'],
                                    return_dict=True,
                                    )
                    # loss
                    loss, task_loss, _ = loss_func_mft(outputs, 
                                                    batch['labels'],
                                                    batch['task_mask'],
                                                    batch['task_id'],
                                                    args.weighted_loss_mode,
                                                    batch['loss_mask']
                                                   )
                    t5 = time.time()

                    # backward
                    if not torch.isnan(loss):
                        total_loss += loss.detach().float()  
                    accelerator.backward(loss)

                    # update(sync_gradients)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # support args.min_lr
                    if optimizer.param_groups[0]['lr'] <= args.min_lr:
                        optimizer.param_groups[0]['lr'] = args.min_lr
                    t6 = time.time()
                    # accumulate resuce_loss
                    if not torch.isnan(loss):
                        reduce_loss += loss.detach().float()
                    # accelerator.print("task loss devices: ", reduce_task_loss.device, task_loss.device)
                    reduce_task_loss += task_loss.detach().float()
                    reduce_task_exist += (task_loss != 0).detach().float()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # progress_bar.update(1)
                        completed_steps += 1

                        # loggging 主进程打印所有卡平均loss
                        if completed_steps % args.log_interval == 0:
                            progress_bar.update(args.log_interval)
                            # gather reduce_loss and reduce_task_loss from all N devices
                            reduce_losses = accelerator.gather(reduce_loss).detach().float()
                            # reduce_task_losses = accelerator.gather_for_metrics(reduce_task_loss).reshape(-1, len(ID2TASK))
                            # reduce_task_exists = accelerator.gather_for_metrics(reduce_task_exist).reshape(-1, len(ID2TASK))
                            reduce_task_losses = accelerator.gather(reduce_task_loss).reshape(-1, len(ID2TASK))
                            reduce_task_exists = accelerator.gather(reduce_task_exist).reshape(-1, len(ID2TASK))
                            # get train loss and per-task train loss
                            train_loss = torch.mean(reduce_losses) / (args.log_interval * args.gradient_accumulation_steps)
                            # train_task_loss = torch.mean(reduce_task_losses, dim=0) / (args.log_interval * args.gradient_accumulation_steps)
                            train_task_loss = torch.sum(reduce_task_losses, dim=0) / torch.sum(reduce_task_exists, dim=0)
                            t7 = time.time()

                            # logging and tensorboard
                            logger.info(
                                f"[TRAIN][complete_steps={completed_steps}][train_loss={train_loss:.6f}][train_task_loss={train_task_loss}]"
                                f"[gather shape={reduce_losses.shape}][lr={lr_scheduler.get_lr()[0]:.4e}, {optimizer.param_groups[0]['lr']:.4e}]",
                                # f"dataloader time: {t4 - t3:.4f}, forward time: {t5 - t4:.4f}, gather time: {t7 - t6:.4f}, backward time: {t6 - t5:.4f}",
                                main_process_only=True)
                            train_log_dict = {"training_loss": train_loss}
                            for i in range(len(ID2TASK)):
                               train_log_dict[f"{ID2TASK[i]}_train_loss"] = train_task_loss[i]
                            # accelerator.log(train_log_dict, step=completed_steps)
                            if accelerator.is_main_process:
                                for key, value in train_log_dict.items():
                                    summary_writer.add_scalar(f'{key}', value, completed_steps)
                            # summary_writer.close()
                            # accelerator.print(optimizer)
                            reduce_loss = 0
                            reduce_task_loss = torch.zeros(len(ID2TASK)).to(model.device)
                            reduce_task_exist = torch.zeros(len(ID2TASK)).to(model.device)
                        # steps checkpointing
                        if args.checkpointing_steps and completed_steps % args.checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            # accelerator.save_state(output_dir)
                            accelerator.wait_for_everyone()
                            logger.info(
                                f"[CHECKPOINT] saving lora checkpoint",
                                main_process_only=True)
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir,
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                                state_dict=accelerator.get_state_dict(model)
                            )
                            accelerator.wait_for_everyone()
                            logger.info(
                                f"[CHECKPOINT][global_steps={step + 1}][complete_steps={completed_steps}], lora checkpoint {output_dir} saved",
                                main_process_only=True)
                        if completed_steps >= args.max_train_steps:
                            break

                        accelerator.wait_for_everyone()

                        # evaluation
                        if completed_steps % args.evalation_steps == 0:
                            model.eval()
                            losses = []
                            accumulated_task_loss = torch.zeros(len(ID2TASK)).to(model.device)
                            accumulated_task_exist = torch.zeros(len(ID2TASK)).to(model.device)
                            for valid_step, valid_batch in enumerate(valid_dataloader):
                                # if valid_step > args.max_valid_steps:
                                #     break
                                with torch.no_grad():
                                    outputs = model(
                                        input_ids=valid_batch['input_ids'],
                                        attention_mask=valid_batch['attention_mask'],
                                        position_ids=valid_batch['position_ids'],
                                        return_dict=True,
                                    )

                                    loss, task_loss, _ = loss_func_mft(outputs, valid_batch['labels'],
                                                                            valid_batch['task_mask'],
                                                                             valid_batch['task_id'],
                                                                             args.weighted_loss_mode,
                                                                             valid_batch['loss_mask'])
                                    # losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
                                    losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
                                    # task_losses.append(accelerator.gather_for_metrics(task_loss))
                                    # [[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4]...]
                                    # [[1, 2, 3, 4], .....]
                                    accumulated_task_loss += task_loss.detach().float()
                                    accumulated_task_exist += (task_loss != 0.0).detach().float()

                            accelerator.wait_for_everyone()
                            # if accelerator.is_main_process:
                            valid_batch_num = len(losses)
                            gathered_size = losses[0].shape
                            losses = torch.cat(losses)
                            # task_losses = torch.cat(task_losses).reshape(-1, len(ID2TASK))
                            # task_losses = accelerator.gather_for_metrics(accumulated_task_loss).reshape(-1, len(ID2TASK))
                            # task_exists = accelerator.gather_for_metrics(accumulated_task_exist).reshape(-1, len(ID2TASK))
                            task_losses = accelerator.gather(accumulated_task_loss).reshape(-1, len(ID2TASK))
                            task_exists = accelerator.gather(accumulated_task_exist).reshape(-1, len(ID2TASK))
                            
                            try:
                                eval_loss = torch.mean(losses)
                                # eval_task_loss = torch.mean(task_losses, dim=0) / valid_batch_num
                                eval_task_loss = torch.sum(task_losses, dim=0) / torch.sum(task_exists, dim=0)
                                if eval_loss <= min_eval_loss:
                                    min_eval_loss = eval_loss
                                    stall_num = 0
                                else:
                                    stall_num += 1
                                perplexity = math.exp(eval_loss)
                            except OverflowError:
                                perplexity = float("inf")

                            logger.info(f"[EVAL][global_steps={step + 1}][completed_steps={completed_steps}]"
                                        f"[valid_batch_num={valid_batch_num}], [gather_size={gathered_size}]"
                                        f"[perplexity={perplexity:.4f}][eval_loss={eval_loss:.6f}][eval_task_loss={eval_task_loss}]",
                                        main_process_only=True)
                            eval_log_dict = {"valid_loss": eval_loss, "perplexity": perplexity} 
                            for i in range(len(ID2TASK)):
                                eval_log_dict[f"{ID2TASK[i]}_valid_loss"] = eval_task_loss[i]
                            # accelerator.log(eval_log_dict, step=completed_steps)
                            if accelerator.is_main_process:
                                for key, value in eval_log_dict.items():
                                    summary_writer.add_scalar(f'{key}', value, completed_steps)

                            model.train()
                            if args.early_stopping and stall_num == args.early_stopping_stall_num:
                                accelerator.print(f"[WARNING] Early stopping at {completed_steps}")
                                break

                    t3 = time.time()

            # epoch checkpointing
            if args.epoch_checkpointing:
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                # accelerator.save_state(output_dir)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model)
                )
                logger.info(f"[CHECKPOINGING], lora checkpoint {output_dir} saved", main_process_only=True)
                accelerator.wait_for_everyone()

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

    # end training if accelerator.init_trackers()
    # accelerator.end_training()
    summary_writer.close()

    # final save
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(args.output_dir, f"final_step_{completed_steps}"),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
    )
    accelerator.wait_for_everyone()
