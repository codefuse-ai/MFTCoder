#!/usr/bin/env python
# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../..")
from typing import List, Optional
from dataclasses import dataclass, field
from peft.utils import PeftConfig


@dataclass
class PetuningConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of [`ROEM`], or [`BitFit`].

    Args:
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )