import sys
sys.path.append("..")
sys.path.append("../..")
import importlib
import re
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union # noqa

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.utils import transpose
from transformers.pytorch_utils import Conv1D
from model.peft.utils import TRANSFORMERS_MODELS_TO_ROUTELORA_TARGET_MODULES_MAPPING
from model.peft.tuner import PeftType

from peft.tuners.lora import (
    LoraConfig,
    LoraLayer,
    LoraModel
)


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class RouteLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.RouteLora`].

    Args:
        - r (`int`): Lora attention dimension
        - route_size (`int`): The number of router models. 
        - target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        - lora_alpha (`float`): The alpha parameter for Lora scaling.
        - lora_dropout (`float`): The dropout probability for Lora layers.
        - merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        - fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        - enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        - bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        - modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    route_size: int = field(default=1, metadata={"help": "The size of router"})

    def __post_init__(self):
        self.peft_type = PeftType.ROUTELORA


class RouteLoraModel(LoraModel):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):
        super(RouteLoraModel, self).__init__(config, model)

        self.route_size = self.peft_config.route_size
        if self.route_size > 0:
            self.activate_route_lora(self.route_size - 1)

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": self.peft_config.merge_weights or self.peft_config.inference_mode,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        kwargs.update({
                            "enable_lora": self.peft_config.enable_lora,
                            "route_size": self.peft_config.route_size
                        })
                        new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = RouteLinear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    kwargs.update({
                        "enable_lora": self.peft_config.enable_lora,
                        "route_size": self.peft_config.route_size
                    })
                    if isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                    else:
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is not a Conv1D. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = False
                    new_module = MergedRouteLinear(in_features, out_features, bias=bias, **kwargs)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def expand_external_router(self, weight_path: str):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit:
            raise NotImplementedError(
                "The route lora method is not support for int quantization, "
                "we will implement this fuction in the future."
            )

        states_dict = torch.load(weight_path)
        external_key_list = states_dict.keys()
        if "0" in external_key_list[0]:
            raise NotImplementedError("The merge with other router is not support, pls wait.")
        self.peft_config.route_size += 1

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            
            if target_module_found:
                expand_cnt = 0
                for external_key in external_key_list:
                    if external_key.beginwith(key):
                        if "0" in external_key:
                            raise NotImplementedError("The merge with other router is not support, pls wait.")
                        else:
                            _, target, _ = self._get_submodules(key)
                            target.route_size += 1
                            weights = states_dict[external_key]
                            new_linear_moudle = nn.Linear(weights.size(0), weights.size(1))
                            new_linear_moudle.weight.data = weights
                            new_linear_moudle.to(target.weights.device)

                            if "lora_A" in external_key:
                                target.lora_A.append(new_linear_moudle)
                            else:
                                target.lora_B.append(new_linear_moudle)
                            expand_cnt += 1
                
                assert expand_cnt == 2, ValueError("123")

    def activate_route_lora(self, route_id: int):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit:
            raise NotImplementedError(
                "The route lora method is not support for int quantization, "
                "we will implement this fuction in the future."
            )

        if route_id < self.route_size:
            key_list = [key for key, _ in self.model.named_modules()]
            for key in key_list:
                if isinstance(self.peft_config.target_modules, str):
                    target_module_found = re.fullmatch(self.peft_config.target_modules, key)
                else:
                    target_module_found = any(
                        key.endswith(target_key) for target_key in self.peft_config.target_modules)
                if target_module_found:
                    _, target, _ = self._get_submodules(key)
                    target.activate_route_lora(route_id)
        else:
            warnings.warn("The route id need less than the route size,"
                          f"but the route id is {route_id} "
                          f"and the route size is {self.route_size}.")

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ROUTELORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_ROUTELORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config


# 以下代码基于https://github.com/microsoft/LoRA/blob/main/loralib/layers.py改进，
# 用于扩展RouteLora方法并适配AntNLP框架


class RouteLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        route_size: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.route_size = route_size

        # Actual trainable parameters
        self.active_route_id = None
        if r > 0 and route_size > 0:
            self.lora_A = nn.ParameterList()
            self.lora_B = nn.ParameterList()
            for _ in range(route_size):
                self.lora_A.append(nn.Linear(in_features, r, bias=False))
                self.lora_B.append(nn.Linear(r, out_features, bias=False))
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for sub_lora_A, sub_lora_B in zip(self.lora_A, self.lora_B):
                nn.init.kaiming_uniform_(sub_lora_A, a=math.sqrt(5))
                nn.init.zeros_(sub_lora_B.weight)

    def activate_route_lora(self, route_id: int):
        if route_id != self.active_route_id:
            if route_id >= self.route_size:
                warnings.warn(f"The choice route id is great than route size,"
                              f"where the route id is {route_id} and route size is {self.route_size}.")
            elif not self.merged:
                self.active_route_id = route_id
            elif self.merged and self.r > 0 and self.active_route_id is not None:
                self.weight.data -= (
                    transpose(
                        self.lora_B[self.active_route_id].weight @ self.lora_A[self.active_route_id].weight,
                        self.fan_in_fan_out
                    ) * self.scaling
                )
            self.merged = False                

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged and self.active_route_id is not None:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (
                    transpose(
                        self.lora_B[self.active_route_id].weight @ self.lora_A[self.active_route_id].weight,
                        self.fan_in_fan_out
                    ) * self.scaling
                )
            self.merged = True
        elif self.merge_weights and self.merged and self.active_route_id is not None:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (
                    transpose(
                        self.lora_B[self.active_route_id].weight @ self.lora_A[self.active_route_id].weight,
                        self.fan_in_fan_out
                    ) * self.scaling
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged and self.active_route_id is not None:
                self.weight.data -= (
                    transpose(
                        self.lora_B[self.active_route_id].weight @ self.lora_A[self.active_route_id].weight,
                        self.fan_in_fan_out
                    ) * self.scaling
                )
                self.merged = False
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged and self.active_route_id is not None:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B[self.active_route_id](
                    self.lora_A[self.active_route_id](self.lora_dropout(x))
                ) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)


class MergedRouteLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        route_size: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if out_features % len(enable_lora) != 0:
            raise ValueError("The length of enable_lora must divide out_features")
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.route_size = route_size

        # Actual trainable parameters
        self.active_route_id = None
        if r > 0 and route_size > 0 and any(enable_lora):
            self.lora_A = nn.ParameterList() 
            self.lora_B = nn.ParameterList()
            for _ in range(route_size):
                self.lora_A.append(nn.Linear(in_features, r * sum(enable_lora), bias=False))
                self.lora_B.append(nn.Conv1d(
                    r * sum(enable_lora),
                    out_features // len(enable_lora) * sum(enable_lora),
                    kernel_size=1,
                    groups=2,
                    bias=False,
                ))
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            for sub_lora_A, sub_lora_B in zip(self.lora_A, self.lora_B):
                nn.init.kaiming_uniform_(sub_lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(sub_lora_B.weight)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def activate_route_lora(self, route_id: int):
        if route_id != self.active_route_id:
            if route_id >= self.route_size:
                warnings.warn(f"The choice route id is great than route size,"
                              f"where the route id is {route_id} and route size is {self.route_size}.")
            elif not self.merged:
                self.active_route_id = route_id
            elif self.merged and self.r > 0 and self.active_route_id is not None:
                delta_w = F.conv1d(
                    self.lora_A[self.active_route_id].weight.data.unsqueeze(0),
                    self.lora_B[self.active_route_id].weight.data.unsqueeze(-1),
                    groups=sum(self.enable_lora),
                ).squeeze(0)
                self.weight.data -= self.zero_pad(transpose(delta_w * self.scaling, self.fan_in_fan_out))
            self.merged = False   

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and self.active_route_id is not None and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A[self.active_route_id].weight.data.unsqueeze(0),
                    self.lora_B[self.active_route_id].weight.data.unsqueeze(-1),
                    groups=sum(self.enable_lora),
                ).squeeze(0)
                self.weight.data += self.zero_pad(transpose(delta_w * self.scaling, self.fan_in_fan_out))
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and self.active_route_id is not None and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A[self.active_route_id].weight.data.unsqueeze(0),
                    self.lora_B[self.active_route_id].weight.data.unsqueeze(-1),
                    groups=sum(self.enable_lora),
                ).squeeze(0)
                self.weight.data -= self.zero_pad(transpose(delta_w * self.scaling, self.fan_in_fan_out))
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged and any(self.enable_lora) and self.active_route_id is not None:
                delta_w = F.conv1d(
                    self.lora_A[self.active_route_id].weight.data.unsqueeze(0),
                    self.lora_B[self.active_route_id].weight.data.unsqueeze(-1),
                    groups=sum(self.enable_lora),
                ).squeeze(0)
                self.weight.data -= self.zero_pad(transpose(delta_w * self.scaling, self.fan_in_fan_out))
                self.merged = False
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.merged:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0 and self.active_route_id is not None:
                after_A = self.lora_A[self.active_route_id](self.lora_dropout(x))
                after_B = self.lora_B[self.active_route_id](after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            # Actual trainable parameters
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                    result += output
                else:
                    output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                    result += output
            return result

    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
            if out_features % len(enable_lora) != 0:
                raise ValueError("The length of enable_lora must divide out_features")
            self.enable_lora = enable_lora
            # Actual trainable parameters
            if r > 0 and any(enable_lora):
                self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
                self.lora_B = nn.Conv1d(
                    r * sum(enable_lora),
                    out_features // len(enable_lora) * sum(enable_lora),
                    kernel_size=1,
                    groups=2,
                    bias=False,
                )
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False
                # Compute the indices
                self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
                self.lora_ind[enable_lora, :] = True
                self.lora_ind = self.lora_ind.view(-1)
            self.reset_parameters()

        def reset_parameters(self):
            if hasattr(self, "lora_A"):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)

        def zero_pad(self, x):
            result = x.new_zeros((*x.shape[:-1], self.out_features))
            result = result.view(-1, self.out_features)
            result[:, self.lora_ind] = x.reshape(
                -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
            )
            return result.view((*x.shape[:-1], self.out_features))

        def forward(self, x: torch.Tensor):
            result = super().forward(x)
            if self.disable_adapters:
                return result
            elif self.r > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype
                    if x.dtype != torch.float32:
                        x = x.float()
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B).to(expected_dtype) * self.scaling
                    result += output
                else:
                    after_A = self.lora_A(self.lora_dropout(x))
                    after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                    output = self.zero_pad(after_B) * self.scaling
                    result += output
            return result