import sys
sys.path.append("..")
sys.path.append("../..")
import torch
import importlib
from enum import Enum
from peft.utils import PeftType
from dataclasses import dataclass, field, asdict
from typing import Optional, List

from .pe_base_model import PEBaseModel
from model.peft.utils import PetuningConfig
from model.peft.utils.others import _freeze_model


def is_alps_available():
    return importlib.util.find_spec("alps") is not None


if is_alps_available():
    from alps.util import logger
else:
    import logging
    logger = logging.getLogger(__file__)


class PEBitfitModel(PEBaseModel):
    """
    只训练模型bias：参考 https://arxiv.org/pdf/2106.10199.pdf
    model: huggingface transformers model
    tokenizer:  huggingface transformers tokenizer
    """

    def __init__(self, model):
        self.model = model

    def get_model(self):
        not_freeze_param_name = ["bias"]
        set_parameter_requires_grad(self.model, not_freeze_param_name)
        return self.model

    @classmethod
    def restore(self, model=None, path=None):
        logger.info("bitfit不需要额外加载参数")
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
                print("trainable parameter name is:")
                print(name)
                param.requires_grad = True


@dataclass
class PeftBitfitConfig(PetuningConfig):
    """
    This is the configuration class to store the configuration of a [`PeftBitfitModel`].

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

    def __post_init__(self):
        self.peft_type = PeftType.BITFIT


class PeftBitfitModel(torch.nn.Module):
    """
    Creates Bitfit model for ant peft.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be freeze with some layers.
        config ([`PeftBitfitConfig`]): The configuration of the Bitfit model.

    Returns:
        `torch.nn.Module`: The Bitfit model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be freezed.
        - **peft_config** ([`PeftBitfitConfig`]): The configuration of the Bitfit model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model

        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if not isinstance(config, PeftBitfitConfig):
            raise ValueError(
                f"The PeftBitfitModel need PeftBitfitConfig, but get {type(config)}."
            )

        if config is not None:
            config = self._prepare_lora_config(config)
            self.peft_config[adapter_name] = config

        if len(self.peft_config) > 1:
            raise ValueError(
                "BitfitModel supports only 1 peft config or name."
                "Because it only freeze the shallow layers without any additional parameters."
            )

        self.model = PEBitfitModel(self.model).get_model()

        if self.peft_config[adapter_name].inference_mode:
            _freeze_model(self.model)

    @staticmethod
    def _prepare_lora_config(peft_config):
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