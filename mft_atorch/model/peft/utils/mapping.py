#!/usr/bin/env python
# coding=utf-8

import sys
sys.path.append("..")
sys.path.append("../..")
import torch
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
)


# needed for prefix-tuning of bloom model
def bloom_model_postprocess_past_key_value(past_key_values):
    past_key_values = torch.cat(past_key_values)
    (
        total_layers,
        batch_size,
        num_attention_heads,
        num_virtual_tokens,
        head_dim,
    ) = past_key_values.shape
    keys = past_key_values[: total_layers // 2]
    keys = keys.transpose(2, 3).reshape(
        total_layers // 2,
        batch_size * num_attention_heads,
        head_dim,
        num_virtual_tokens,
    )
    values = past_key_values[total_layers // 2 :]
    values = values.reshape(
        total_layers // 2,
        batch_size * num_attention_heads,
        num_virtual_tokens,
        head_dim,
    )

    return tuple(zip(keys, values))


NEW_TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "bloomz": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "glm": ["query_key_value"],
}

NEW_TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "bloomz": ["query_key_value"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gptj": ["q_proj", "v_proj"],
    # "gpt_neox": ["query_key_value"],
    # "gpt_neo": ["q_proj", "v_proj"],
    # "bert": ["query", "value"],
    "roberta": ["query", "key", "value", "dense"],
    # "xlm-roberta": ["query", "value"],
    # "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "chatglm": ["query_key_value"],
    "glm": ["query_key_value"],
    # "deberta": ["in_proj"],
    # "layoutlm": ["query", "value"],
}

TRANSFORMERS_MODELS_TO_LORA_LAGE_TARGET_MODULES_MAPPING = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gpt2": ["c_attn"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "bloomz": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # "gptj": ["q_proj", "v_proj"],
    # "gpt_neox": ["query_key_value"],
    # "gpt_neo": ["q_proj", "v_proj"],
    # "bert": ["query", "value"],
    "roberta": ["query", "key", "value", "dense"],
    # "xlm-roberta": ["query", "value"],
    # "electra": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "glm": ["query_key_value", "dense"]
    # "deberta": ["in_proj"],
    # "layoutlm": ["query", "value"],
}

TRANSFORMERS_MODELS_TO_ROUTELORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "k", "v", "o", "wi", "wo"],
    "mt5": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    "bart": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "roberta": ["query", "key", "value", "dense"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "dense"],
    "chatglm": ["query_key_value"],
    "glm": ["query_key_value"]
}

TRANSFORMERS_MODELS_ROME_LAYER_MODULES_MAPPING = {
    "glm": [0, 22],
    "bloom": [17, 22],
    "bloomz": [17, 22],
}

TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING = {
    "bloom": bloom_model_postprocess_past_key_value,
    "bloomz": bloom_model_postprocess_past_key_value,
}

WEIGHTS_NAME = "adapter_model.bin"
CONFIG_NAME = "adapter_config.json"


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.update(
    NEW_TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
)
TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING.update(
    NEW_TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
)
