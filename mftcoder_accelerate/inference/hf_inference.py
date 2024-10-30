# -*- coding: utf-8 -*-
# @author Chaoyu Chen
# @date 2024/1/4
# @module hf_inference.py
"""
# @author qumu
# @date 2023/9/19
# @module hf_inference.py
"""
import os
import sys
import torch
import textwrap
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel


def load_model_tokenizer(
    path,
    model_type=None,
    peft_path=None,
    torch_dtype=torch.bfloat16,
    quantization=None,
    eos_token=None,
    pad_token=None,
    batch_size=1,
):
    """
    load model and tokenizer by transfromers
    """

    # load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    config, unused_kwargs = AutoConfig.from_pretrained(path, trust_remote_code=True, return_unused_kwargs=True)
    print("unused_kwargs:", unused_kwargs)
    print("config input:\n", config)

    # eos token parsing
    if eos_token:
        eos_token = eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        print(f"eos_token {eos_token} from user input")
    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
        print(f"Initial eos_token_id {tokenizer.eos_token_id} from tokenizer")
        eos_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)
    elif hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
        print(f"Initial eos_token {tokenizer.eos_token} from tokenizer")
        eos_token = tokenizer.eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    elif hasattr(config, "eos_token_id") and config.eos_token_id:
        print(f"Initial eos_token_id {config.eos_token_id} from config.json")
        eos_token_id = config.eos_token_id
        eos_token = tokenizer.convert_ids_to_tokens(config.eos_token_id)
    elif hasattr(config, "eos_token") and config.eos_token:
        print(f"Initial eos_token {config.eos_token} from config.json")
        eos_token = config.eos_token
        eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)
    else:
        raise ValueError(
            "No available eos_token or eos_token_id, please provide eos_token by params or eos_token_id by config.json"
        )

    try:
        tokenizer.eos_token = eos_token
        tokenizer.eos_token_id = eos_token_id
        # set pad_token to be same as eos_token, it is ok because is will be masked out.
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
    except:
        print(f"[WARNING]Cannot set tokenizer.eos_token")

    print(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    print(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")
    print(type(tokenizer))

    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        load_in_8bit=(quantization == "8bit"),
        load_in_4bit=(quantization == "4bit"),
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if peft_path:
        print("Loading PEFT MODEL...")
        model = PeftModel.from_pretrained(base_model, peft_path)
    else:
        print("Loading Original MODEL...")
        model = base_model

    model.eval()

    print("=======================================MODEL Configs=====================================")
    print(model.config)
    print("=========================================================================================")
    print("=======================================MODEL Archetecture================================")
    print(model)
    print("=========================================================================================")

    return model, tokenizer


def hf_inference(model, tokenizer, text_list, args=None, max_new_tokens=512, do_sample=True, **kwargs):
    """
    transformers models inference by huggingface
    """
    # text_list = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False) for text in text_list]
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    # inputs["attention_mask"][0][:100] = 0
    # print(inputs)
    print("================================Prompts and Generations=============================")

    outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )

    gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    for i in range(len(text_list)):
        print("=========" * 10)
        print(f"Prompt:\n{text_list[i]}")
        gen_text[i] = gen_text[i].replace(tokenizer.pad_token, "")
        print(f"Generation:\n{gen_text[i]}")
        # print(f"Outputs ids:\n{outputs[i]}")
        sys.stdout.flush()

    return gen_text


if __name__ == "__main__":
    # Default template used in MFTCoder training
    HUMAN_ROLE_START_TAG = "<s>human\n"
    BOT_ROLE_START_TAG = "<s>bot\n"

    instruction = "Write quick sort function in python."

    prompts = [f"{HUMAN_ROLE_START_TAG}{instruction}\n{BOT_ROLE_START_TAG}"]

    # if you use base + adaptor for inference, provide peft_path or left it None for normal inference
    base_model = "path/to/basemodel"
    peft_path = None
    model, tokenizer = load_model_tokenizer(
        base_model, model_type="", peft_path=peft_path, eos_token="</s>", pad_token="<unk>"
    )

    # hf_inference(model, tokenizer, prompts, do_sample=False, num_beams=1, num_return_sequences=1)
    hf_inference(model, tokenizer, prompts, do_sample=True, temperature=0.8)
