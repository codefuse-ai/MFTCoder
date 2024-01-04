import sys
sys.path.append("..")
sys.path.append("../..")
from peft import get_peft_model, PeftModel
import math
import re
import warnings
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    WEIGHTS_NAME,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from model.peft.tuner import PeftType
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING


from .pe_base_model import PEBaseModel


# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class UniPELTModel(torch.nn.Module):
    r"""
    改编自LoraModel
    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = (
                self.model.config.to_dict()
                if hasattr(self.model.config, "to_dict")
                else self.model.config
            )
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. \
                When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]

        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in lora_config.target_modules
                )
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, GatedLoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if True:  # lazy modification
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = (
                                target.in_features,
                                target.out_features,
                            )
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs[
                                    "fan_in_fan_out"
                                ] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape
                                if hasattr(target.weight, "ds_shape")
                                else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs[
                                    "fan_in_fan_out"
                                ] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = PELTLinear(
                            adapter_name, in_features, out_features, bias=bias, **kwargs
                        )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {
                k: v.value if isinstance(v, Enum) else v
                for k, v in asdict(value).items()
            }
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, GatedLoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, GatedLoraLayer):
                if module.merged:
                    warnings.warn(
                        "Adapter cannot be set when the model is merged. Unmerging the model first."
                    )
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, GatedLoraLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, GatedLoraLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if (
                model_config["model_type"]
                not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            ):
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = (
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                    model_config["model_type"]
                ]
            )
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError(
                "Cannot merge LORA layers when the model is loaded in 8-bit mode"
            )

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, GatedLoraLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(
                    target.in_features, target.out_features, bias=bias
                )
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(
                    parent, target_name, target.modules_to_save[target.active_adapter]
                )

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        if len({self.peft_config[adapter].r for adapter in adapters}) != 1:
            raise ValueError("All adapters must have the same r value")
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self.peft_config[adapter_name].lora_alpha = self.peft_config[adapters[0]].r
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, GatedLoraLayer):
                if adapter_name in target.lora_A:
                    target.lora_A[adapter_name].weight.data = (
                        target.lora_A[adapter_name].weight.data * 0.0
                    )
                    target.lora_B[adapter_name].weight.data = (
                        target.lora_B[adapter_name].weight.data * 0.0
                    )
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_A:
                            continue
                        target.lora_A[adapter_name].weight.data += (
                            target.lora_A[adapter].weight.data
                            * weight
                            * target.scaling[adapter]
                        )
                        target.lora_B[adapter_name].weight.data += (
                            target.lora_B[adapter].weight.data * weight
                        )

                elif adapter_name in target.lora_embedding_A:
                    target.lora_embedding_A[adapter_name].data = (
                        target.lora_embedding_A[adapter_name].data * 0.0
                    )
                    target.lora_embedding_B[adapter_name].data = (
                        target.lora_embedding_B[adapter_name].data * 0.0
                    )
                    for adapter, weight in zip(adapters, weights):
                        if adapter not in target.lora_embedding_A:
                            continue
                        target.lora_embedding_A[adapter_name].data += (
                            target.lora_embedding_A[adapter].data
                            * weight
                            * target.scaling[adapter]
                        )
                        target.lora_embedding_B[adapter_name].data += (
                            target.lora_embedding_B[adapter].data * weight
                        )


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if (
                isinstance(m, GatedLoraLayer)
                and hasattr(m, "bias")
                and m.bias is not None
            ):
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class GatedLoraLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """UniPELT里使用了带gate的Lora，在peft Lora上增加了`self.lora_gate`作为门控

        Args:
            in_features (int): _description_
            out_features (int): _description_
        """
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.lora_gate = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        """在lora dict里添加新的实例。

        Args:
            adapter_name (_type_): _description_
            r (_type_): _description_
            lora_alpha (_type_): _description_
            lora_dropout (_type_): _description_
            init_lora_weights (_type_): _description_
        """
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(self.in_features, r, bias=False)}
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(r, self.out_features, bias=False)}
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
            self.lora_gate.update(
                nn.ModuleDict(
                    {adapter_name: nn.Linear(self.in_features, 1, bias=False)}
                )
            )
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_embedding_A.update(
                nn.ParameterDict(
                    {
                        adapter_name: nn.Parameter(
                            self.weight.new_zeros((r, self.in_features))
                        )
                    }
                )
            )
            self.lora_embedding_B.update(
                nn.ParameterDict(
                    {
                        adapter_name: nn.Parameter(
                            self.weight.new_zeros((self.out_features, r))
                        )
                    }
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_gate[adapter_name].weight)
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


class PELTLinear(nn.Linear, GatedLoraLayer):
    # GatedLora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        GatedLoraLayer.__init__(
            self, in_features=in_features, out_features=out_features
        )
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += (
                transpose(
                    self.lora_B[self.active_adapter].weight
                    @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= (
                transpose(
                    self.lora_B[self.active_adapter].weight
                    @ self.lora_A[self.active_adapter].weight,
                    self.fan_in_fan_out,
                )
                * self.scaling[self.active_adapter]
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            gate = self.lora_gate[self.active_adapter](x)

            result += (
                gate
                * self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](
                        self.lora_dropout[self.active_adapter](x)
                    )
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        result = result.to(previous_dtype)

        return result


@dataclass
class UniPELTConfig(PeftConfig):
    """
    因为是在Lor上增加门控，所以其他结构不变
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) 
        and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"
        },
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.UNIPELT


class PEUniPELTModel(PEBaseModel):
    def __init__(self, model, task_type, r, lora_alpha, lora_dropout, model_name):
        """
        实现了UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuninghttps://arxiv.org/abs/2110.07577


        Args:
            model (_type_): huggingface transformers model
            task_type (_type_): "SEQ_CLS", "SEQ_2_SEQ_LM","CAUSAL_LM","TOKEN_CLS"
            r (_type_): lora rank
            lora_alpha (_type_): lora alpha
            lora_dropout (_type_): The dropout probability for Lora layers.
            model_name (_type_): model_name

        Raises:
            NotImplementedError: _description_
        """
        self.base_model = model
        if task_type not in ["SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"]:
            raise NotImplementedError("this task_type is not supported")
        from solutions.antllm.antllm.models.peft.utils import (
            TRANSFORMERS_MODELS_TO_LORA_LAGE_TARGET_MODULES_MAPPING,
        )

        self.config = UniPELTConfig(
            task_type=task_type,
            target_modules=TRANSFORMERS_MODELS_TO_LORA_LAGE_TARGET_MODULES_MAPPING[
                model_name
            ],
            inference_mode=False,
            lora_alpha=lora_alpha,
            r=r,
            lora_dropout=lora_dropout,
        )

    def get_model(self):
        self.pe_model = get_peft_model(model=self.base_model, peft_config=self.config)
        return self.pe_model

    def get_model_state_dict(self, model, state_dict=None, adapter_name="default"):
        """
        改了PeftModel.save_pretrained，使其支持UniPELT
        Get the state dict of the Peft model.

        Args:
            model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
            the model should be the underlying model/unwrapped model (i.e. model.module).
            state_dict (`dict`, *optional*, defaults to `None`):
                The state dict of the model. If not provided, the state dict of the model
            will be used.
        """
        config = model.peft_config[adapter_name]
        if state_dict is None:
            state_dict = model.state_dict()
            # if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
            # to_return = lora_state_dict(model, bias=model.peft_config.bias)
            # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
            # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {
                k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k
            }
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {
            k: v
            for k, v in to_return.items()
            if (("lora_" in k and adapter_name in k) or ("bias" in k))
        }
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {
                    k.replace(f".{adapter_name}", ""): v
                    for k, v in rank_pattern.items()
                }
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(
                    rank_pattern, to_return, adapter_name
                )
        if model.modules_to_save is not None:
            for key, value in state_dict.items():
                if any(
                    f"{module_name}.modules_to_save.{adapter_name}" in key
                    for module_name in model.modules_to_save
                ):
                    to_return[key.replace("modules_to_save.", "")] = value

        to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
        return to_return

    def save(self, save_directory, **kwargs):
        r"""
        改了PeftModel.save_pretrained，使其支持UniPELT
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)

        for adapter_name, peft_config in self.pe_model.peft_config.items():
            # save only the trainable weights
            output_state_dict = self.get_model_state_dict(
                self.pe_model,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            )
            # logger.info(output_state_dict)
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            )
            os.makedirs(output_dir, exist_ok=True)
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True
            peft_config.save_pretrained(output_dir)
            peft_config.inference_mode = inference_mode

    @classmethod
    def restore(
        cls,
        model=None,
        path=None,
        adapter_name="default",
        is_trainable=False,
        **kwargs,
    ):
        r"""
        改写了PeftModel.from_pretrained，使其支持UniPELT
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
        """
        from peft.mapping import (
            MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
        )

        # load the config
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
            PeftConfig.from_pretrained(
                path, subfolder=kwargs.get("subfolder", None)
            ).peft_type
        ].from_pretrained(path, subfolder=kwargs.get("subfolder", None))

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            peft_model = cls(model, config, adapter_name)
        else:
            # for example model is of  PeftModelForSeq2SeqLM
            peft_model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](
                model, config, adapter_name
            )
        peft_model = load_adapter(peft_model, path, adapter_name, **kwargs)
        return peft_model


def load_adapter(
    peft_model: PeftModel,
    path,
    adapter_name,
    is_trainable=False,
    **kwargs,
):
    """改写了PeftModel.load_adapter，使其支持UniPELT

    Args:
        peft_model (PeftModel): _description_
        path (_type_): _description_
        adapter_name (_type_): _description_
        is_trainable (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if adapter_name not in peft_model.peft_config:
        # load the config
        peft_config = UniPELTConfig.from_pretrained(
            path, subfolder=kwargs.get("subfolder", None)
        )
        peft_config.inference_mode = not is_trainable
        # base model is pretrained model
        peft_model.base_model.add_adapter(adapter_name, peft_config)

        peft_model.set_additional_trainable_modules(peft_config, adapter_name)

    # load weights if any
    path = (
        os.path.join(path, kwargs["subfolder"])
        if kwargs.get("subfolder", None) is not None
        else path
    )

    if os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
    else:
        pass

    adapters_weights = torch.load(
        filename,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # load the weights into the model
    set_model_state_dict(peft_model, adapters_weights, adapter_name=adapter_name)
    if (
        (getattr(peft_model, "hf_device_map", None) is not None)
        and (
            len(set(peft_model.hf_device_map.values()).intersection({"cpu", "disk"}))
            > 0
        )
        and len(peft_model.peft_config) == 1
    ):
        device_map = kwargs.get("device_map", "auto")
        max_memory = kwargs.get("max_memory", None)
        offload_dir = kwargs.get("offload_folder", None)
        offload_index = kwargs.get("offload_index", None)

        dispatch_model_kwargs = {}
        # Safety checker for previous `accelerate` versions
        # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        no_split_module_classes = peft_model._no_split_modules

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                peft_model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                peft_model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
            )
        dispatch_model(
            peft_model,
            device_map=device_map,
            offload_dir=offload_dir,
            **dispatch_model_kwargs,
        )
        hook = AlignDevicesHook(io_same_device=True)
        add_hook_to_module(peft_model.get_base_model(), hook)

    # Set model in evaluation mode to deactivate Dropout modules by default
    peft_model.eval()
    return peft_model


def set_model_state_dict(
    model: PeftModel, peft_model_state_dict, adapter_name="default"
):
    """
    改写了peft.uitls下的set_peft_model_state_dict，使其支持UniPELT
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(
                            module_name, f"{module_name}.modules_to_save.{adapter_name}"
                        )
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.UNIPELT):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(
                        suffix_to_replace, f"{adapter_name}.{suffix_to_replace}"
                    )
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    model.load_state_dict(peft_model_state_dict, strict=False)


def print_lora_parameters(model):
    """debug 用，查看权重是否正确加载

    Args:
        model (_type_): _description_
    """
    for n, p in model.named_parameters():
        if "lora_B" in n:
            print(n)
            print(p)
            # break
        if "lora_A" in n:
            print(n)
            print(p)
            # break
        if "lora_gate" in n:
            print(n)
            print(p)
            break
