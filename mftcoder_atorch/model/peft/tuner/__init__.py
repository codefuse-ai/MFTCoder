"""peft tuner methods interface."""

from peft.utils import PeftType
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

from .adalora import AdaLoraConfig, AdaLoraModel
from .routelora import RouteLoraConfig, RouteLoraModel
from .unipelt import UniPELTConfig, UniPELTModel, PEUniPELTModel
from .pe_base_model import PEBaseModel
from .bitfit import PeftBitfitConfig, PEBitfitModel, PeftBitfitModel
from .roem import PeftROEMConfig, PEROEMModel, PeftROEMModel

# Register new ant peft methods
PeftType.ROUTELORA = "ROUTELORA"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ROUTELORA] = RouteLoraModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.ROUTELORA] = RouteLoraConfig

PeftType.UNIPELT = "UNIPELT"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.UNIPELT] = UniPELTModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.UNIPELT] = UniPELTConfig

PeftType.ROEM = "ROEM"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.ROEM] = PeftROEMModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.ROEM] = PeftROEMConfig

PeftType.BITFIT = "BITFIT"
PEFT_TYPE_TO_MODEL_MAPPING[PeftType.BITFIT] = PeftBitfitModel
PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.BITFIT] = PeftBitfitConfig