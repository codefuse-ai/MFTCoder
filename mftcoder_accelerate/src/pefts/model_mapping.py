"""
 # @author Chaoyu Chen
 # @date 2024/5/20

 Manage supported models and their special token used in training.
 Default targeting modules for LoRA/QLora
 4.40 is stable now
"""

# Models that have both cutomized modeling and Transformers modeling

CUSTOMIZE = False
if CUSTOMIZE:
    from model.code_llama.modeling_llama import LlamaForCausalLM
    from model.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
    from model.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM
else:
    from transformers import (
        GPTNeoXForCausalLM,
        GPTBigCodeForCausalLM,
        LlamaForCausalLM,
    )

# Models that Transformers support Code and FA2 when flash_attn>=2.1.0
from transformers import (
    MistralForCausalLM,
    MixtralForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
    Qwen2ForCausalLM,
    Qwen2MoeForCausalLM,
    Starcoder2ForCausalLM,
)
# Models that Code from "remote_code"
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
}

FULL_LORA_TARGETING_MODULES = {
    "aquila": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "baichuan": ["W_pack", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "chatglm2": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "chatglm3": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "code_llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "w1", "w2", "w3"],
    "phi": ["query_key_value", "dense", "fc1", "fc2"],
    "qwen": ["c_proj", "c_attn", "w1", "w2"],
    "starcoder": ["c_proj", "c_attn", "q_attn", "c_fc"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "qwen2_moe": "all-linear",
    "starcoder2": "all-linear",
}


MODEL_SPECIAL_TOKENS = {
    "gpt_neox": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
    },
    "llama": {
        "eos_token": "</s>",
        "pad_token": "<unk>",
    },
    "code_llama": {
        "eos_token": "</s>",
        "pad_token": "<unk>",
    },
    "baichuan": {
        "eos_token": "</s>",
        "pad_token": "<unk>",
    },
    "starcoder": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<fim_pad>",
    },
    "qwen": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|extra_1|>",
    },
    "chatglm2": {
        "eos_token": "</s>",
        "pad_token": "<unk>",
    },
    "chatglm3": {
        "eos_token": "</s>",
        "pad_token": "<unk>",
    },
    "phi": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    },
    "aquila": {
        "eos_token": "</s>",
        "pad_token": "<|endoftext|>",
    },
    "deepseek": {
        "eos_token": "<｜end▁of▁sentence｜>",
        "pad_token": "<｜end▁of▁sentence｜>",
    },
    "mixtral": {
        "eos_token": "</s>",
        "pad_token": "<unk>",
    },
    "mistral": {
        "eos_token": "</s>",
        "pad_token": "<unk>",
    },
    "qwen2": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    },
    "gemma": {
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    },
    "qwen2_moe": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    },
    "starcoder2": {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    },
}
