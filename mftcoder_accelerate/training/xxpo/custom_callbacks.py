"""
Customized Callbacks to use with the Trainer class and customize the training loop.
"""

import copy
import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from transformers.trainer_utils import IntervalStrategy, has_length
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers import TrainerCallback

logger = logging.get_logger(__name__)


class CustomProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and state.global_step % args.logging_steps == 0:
            self.training_bar.update(args.logging_steps)
            self.current_step = state.global_step
        # pass

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # if state.is_world_process_zero and has_length(eval_dataloader):
        #     if self.prediction_bar is None:
        #         self.prediction_bar = tqdm(
        #             total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
        #         )
        #     self.prediction_bar.update(1)
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # avoid modifying the logs object as it is shared between callbacks
            logs = copy.deepcopy(logs)
            # _ = logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in logs:
                logs["epoch"] = round(logs["epoch"], 2)
            # self.training_bar.write(str(logs))
            logger.info(logs)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


class LogCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)