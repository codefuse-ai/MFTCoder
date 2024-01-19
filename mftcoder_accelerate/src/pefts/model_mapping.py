"""
 # @author Chaoyu Chen
 # @date 2023/12/11

 Manage supported models and their special token used in training.
 Default targeting modules for LoRA/QLora
 4.36 is stable now
"""
# Models that Transformers support FA2
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTNeoXForCausalLM,
    GPTBigCodeForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    MixtralForCausalLM,
    PhiForCausalLM,
)

# Models that Transformers not support FA2, supported by publisher or ourself
from model.aquila2.modeling_aquila import AquilaForCausalLM
from model.baichuan2.modeling_baichuan import BaichuanForCausalLM
from model.qwen.modeling_qwen import QWenLMHeadModel
from model.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLMForConditionalGeneration2
from model.chatglm3.modeling_chatglm import ChatGLMForConditionalGeneration as ChatGLMForConditionalGeneration3

# from model.phi.modeling_mixformer_sequential import MixFormerSequentialForCausalLM

MODEL_TYPES = {
    "aquila2": AquilaForCausalLM,
    "baichuan": BaichuanForCausalLM,
    'chatglm2': ChatGLMForConditionalGeneration2,
    'chatglm3': ChatGLMForConditionalGeneration3,
    "code_llama": LlamaForCausalLM,
    "deepseek": LlamaForCausalLM,
    "gpt_neox": GPTNeoXForCausalLM,
    "llama": LlamaForCausalLM,
    "mistral": MistralForCausalLM,
    "mixtral": MixtralForCausalLM,
    'phi': PhiForCausalLM,
    'qwen': QWenLMHeadModel,
    "starcoder": GPTBigCodeForCausalLM,
}

FULL_LORA_TARGETING_MODULES = {
    "aquila": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "baichuan": ["W_pack", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "chatglm2": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "chatglm3": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "code_llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "gpt_neox": ["query_key_value", 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "phi": ["query_key_value", 'dense', 'fc1', 'fc2'],
    "qwen": ["c_proj", "c_attn", "w1", "w2"],
    "starcoder": ["c_proj", "c_attn", "q_attn", "c_fc"],
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
}
