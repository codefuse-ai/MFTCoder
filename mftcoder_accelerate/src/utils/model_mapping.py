"""
 @author qumu
 transformers==4.40 is stable now
"""

# Models that Transformers support Code and FA2 when flash_attn>=2.1.0
from transformers import (
    GPTNeoXForCausalLM,
    GPTBigCodeForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    MixtralForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
    Qwen2ForCausalLM,
    Qwen2MoeForCausalLM,
    Starcoder2ForCausalLM,
)

# model in local model dir and support transformers FA2
from model.deepseek_v2.modeling_deepseek import DeepseekV2ForCausalLM

# model in local model and self-contained
from model.aquila2.modeling_aquila import AquilaForCausalLM
from model.baichuan2.modeling_baichuan import BaichuanForCausalLM
from model.qwen.modeling_qwen import QWenLMHeadModel
from model.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLMForConditionalGeneration2
from model.chatglm3.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLMForConditionalGeneration3

# from model.phi.modeling_mixformer_sequential import MixFormerSequentialForCausalLM

MODEL_TYPES = {
    "aquila2": AquilaForCausalLM,
    "baichuan": BaichuanForCausalLM,
    "chatglm2": ChatGLMForConditionalGeneration2,
    "chatglm3": ChatGLMForConditionalGeneration3,
    "code_llama": LlamaForCausalLM,
    "deepseek": LlamaForCausalLM,
    "gpt_neox": GPTNeoXForCausalLM,
    "llama": LlamaForCausalLM,
    "mistral": MistralForCausalLM,
    "mixtral": MixtralForCausalLM,
    "phi": PhiForCausalLM,
    "qwen": QWenLMHeadModel,
    "starcoder": GPTBigCodeForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "gemma": GemmaForCausalLM,
    "qwen2_moe": Qwen2MoeForCausalLM,
    "starcoder2": Starcoder2ForCausalLM,
    "deepseek_v2": DeepseekV2ForCausalLM,
}

SUPPORT_IN_TRANSFORMERS = [
    "code_llama",
    "llama",
    "deepseek",
    "mistral",
    "mixtral",
    "gpt_neox",
    "phi",
    "starcoder",
    "qwen2",
    "qwen2_moe",
    "gemma",
    "starcoder2",
    "deepseek_v2",
]
