"""
# @author Chaoyu Chen
# @date 2023/6/19

"""
# from model.llama2.modeling_llama import LlamaForCausalLM
# from model.llama2.configuration_llama import LlamaConfig
from model.code_llama.modeling_llama import LlamaForCausalLM
from model.code_llama.configuration_llama import LlamaConfig
from model.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from model.baichuan.modeling_baichuan import BaichuanForCausalLM
from model.baichuan.configuration_baichuan import BaichuanConfig
from model.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM
from model.qwen.modeling_qwen import QWenLMHeadModel
from model.chatglm2.modeling_chatglm import ChatGLMForConditionalGeneration
from transformers import AutoModelForCausalLM

MODEL_TYPES = {
  "gpt_neox": GPTNeoXForCausalLM,
  "llama": LlamaForCausalLM,
  "baichuan": BaichuanForCausalLM,
  "starcoder": GPTBigCodeForCausalLM,
  'qwen': QWenLMHeadModel,
  'chatglm2': ChatGLMForConditionalGeneration,
}

MODEL_CONFIGS = {
    "gpt_neox": None,
    "llama": LlamaConfig,
    "baichuan": BaichuanConfig,
    "starcoder": None,
    'qwen': None,
    'chatglm2': None,
}

QLORA_TARGETING_MODULES = {
    "gpt_neox": ["query_key_value", 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    "llama": ["q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "baichuan": ["W_pack", "o_proj", "gate_proj", "down_proj", "up_proj"],
    "starcoder": ["c_proj", "c_attn", "q_attn", "c_fc"],
    "qwen": ["c_proj", "c_attn", "w1", "w2"],
    "chatglm2": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}

MODEL_SPECIAL_TOKENS = {
    "gpt_neox": {

        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",

    },
    "llama": {

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
}