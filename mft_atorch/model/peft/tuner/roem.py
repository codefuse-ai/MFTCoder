import sys
sys.path.append("..")
sys.path.append("../..")
import torch
import importlib
from enum import Enum
from peft.utils import PeftType
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union

from .pe_base_model import PEBaseModel
from model.peft.utils import (
    PetuningConfig,
    TRANSFORMERS_MODELS_ROME_LAYER_MODULES_MAPPING
)
from model.peft.utils.others import _freeze_model


def is_alps_available():
    return importlib.util.find_spec("alps") is not None


if is_alps_available():
    from alps.util import logger
else:
    import logging
    logger = logging.getLogger(__file__)


class PEROEMModel(PEBaseModel):
    """
    只训练模型中间偏上层mlp：参考 https://arxiv.org/pdf/2202.05262.pdf ； https://arxiv.org/abs/2012.14913
    model: huggingface transformers model
    tokenizer:  huggingface transformers tokenizer
    """

    def __init__(self, model, model_name, task_type=None):
        self.model = model
        self.model_name = model_name

    def get_model(self):
        layer_mapping = TRANSFORMERS_MODELS_ROME_LAYER_MODULES_MAPPING[self.model_name]
        assert len(layer_mapping) == 2
        not_freeze_param_name = []
        for i in range(layer_mapping[0], layer_mapping[1]):
            no_freeze_name = str(i) + ".mlp"
            logger.info(f"Freeze the {no_freeze_name} layer of model")
            not_freeze_param_name.append(no_freeze_name)
        set_parameter_requires_grad(self.model, not_freeze_param_name)
        return self.model

    @classmethod
    def restore(self, model=None, path=None):
        logger.info("roem不需要额外加载参数")
        return model


# 根据名称锁定参数层
def set_parameter_requires_grad(model, freeze_param_name=[]):
    if not isinstance(freeze_param_name, list):
        freeze_param_name = [freeze_param_name]

    for idx, (name, param) in enumerate(model.named_parameters()):
        for p in freeze_param_name:
            if p not in name:
                param.requires_grad = False
        # 打印参数层名
    for idx, (name, param) in enumerate(model.named_parameters()):
        for p in freeze_param_name:
            if p in name:
                print("The name of used parameter used by ROEM is:")
                print(name)
                param.requires_grad = True


@dataclass
class PeftROEMConfig(PetuningConfig):
    """
    This is the configuration class to store the configuration of a [`PeftROEMModel`].

    Args:
        target_layers (`Union[List[int], int]`): The names of the modules to apply Lora to.
    """

    target_layers: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "List of layers of the model to freeze the parameters."
            "For example, [20, 30] or '30' "
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.ROEM


class PeftROEMModel(torch.nn.Module):
    """
    Creates ROEM model for ant peft.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be freeze with some layers.
        config ([`PeftROEMConfig`]): The configuration of the ROEM model.

    Returns:
        `torch.nn.Module`: The ROEM model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be freezed.
        - **peft_config** ([`PeftROEMConfig`]): The configuration of the ROEM model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model

        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if not isinstance(config, PeftROEMConfig):
            raise ValueError(
                f"The PeftROEMModel need PeftROEMConfig, but get {type(config)}."
            )

        model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
        if config is not None:
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config

        if len(self.peft_config) > 1:
            raise ValueError(
                "ROEMModel supports only 1 peft config or name."
                "Because it only freeze the shallow layers without any additional parameters."
            )

        model_name = model_config["model_type"]
        self.model = PEROEMModel(self.model, model_name).get_model()

        if self.peft_config[adapter_name].inference_mode:
            _freeze_model(self.model)

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_layers is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_ROME_LAYER_MODULES_MAPPING:
                raise ValueError("Please specify `target_layers` in `peft_config`")
            peft_config.target_layers = TRANSFORMERS_MODELS_ROME_LAYER_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config