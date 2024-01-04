# -*- coding: utf-8 -*-
# @author qumu
# @date 2023/12/25
# @module chat_template
# store possible chat_template for tokenizers to prepare input string
# -------------------------------------------------- Import ------------------------------------------------------------
from transformers import (
    AutoTokenizer
)

# ----------------------------------------------- func and class -------------------------------------------------------
instruction_template = (
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token}}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)

MFTCoder_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = false %}"
    "{% endif %}"
    "{% for message in loop_messages %}"  # Loop over all non-system messages
    "{% if (message['role'] == 'user' or message['role'] == 'human') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
    "{% set content = '<s>system\n' + system_message + '\n' %}"
    "{% else %}"
    "{% set content = '' %}"
    "{% endif %}"
    "{% if message['role'] == 'user' or message['role'] == 'human' %}"
    "{{ content + '<s>user\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' or message['role'] == 'bot' %}"
    "{{ '<s>assistant\n' + message['content'] + '\n' +  eos_token + '\n'}}"
    "{% else %}"
    "{{ raise_exception('Only user/human and assistant/bot roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<s>assistant\n' }}"
    "{% endif %}"
)

if __name__ == '__main__':
    pass
