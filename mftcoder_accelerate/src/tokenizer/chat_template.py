# -*- coding: utf-8 -*-
# @author Chaoyu Chen
# @date 2023/12/25

# store possible chat_template for tokenizers to prepare input string
# -------------------------------------------------- Import ------------------------------------------------------------
"""
Usage:
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
messages = [
    {"role": "system", "content": "Be smart"},
    {"role": "human", "content": "Hello, how are you?"},
    {"role": "bot", "content": "I'm doing great. How can I help you today?"},
    {"role": "human", "content": "I'd like to show off how chat templating works!"},
]
prompts = tokenizer.apply_chat_template(message, chat_template=MFTCoder_template, tokenize=False, add_generation_prompt=True)
"""

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
    "{{ content + '<s>human\n' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' or message['role'] == 'bot' %}"
    "{{ '<s>bot\n' + message['content'] + '\n' +  eos_token + '\n'}}"
    "{% else %}"
    "{{ raise_exception('Only user/human and assistant/bot roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<s>bot\n' }}"
    "{% endif %}"
)

if __name__ == "__main__":
    pass
