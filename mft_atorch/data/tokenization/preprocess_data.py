"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys
import numpy as np
import random
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# )

# 将父目录的父目录加入path 
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
# print(grandparent_dir)

import data.tokenization.lm_dataformat as lmd

import time
import tqdm
import torch
import ftfy
import glob

from tokenizer import build_tokenizer
from threading import Semaphore




table = {ord(f):ord(t) for f,t in zip(
     u'，。！？：【】（）％＃＠＆１２３４５６７８９０',
     u',.!?:[]()%#@&1234567890')}


def punctuation_format(text: str):
    # Replace non-breaking space with space
    # text = text.strip() + '\n'
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # change chinese punctuation to english ones
    text = text.translate(table)
    return text

def is_prompt_answer_format(data):

    if "prompt" in data and "answer" in data:
        return True
    else:
        return False


def is_chatml_format(data):
    if "chat_rounds" in data and len(data["chat_rounds"]) > 0:
        return True
    else:
        return False


def is_text_format(data):
    if "text" in data:
        return True
    else:
        return False

class Encoder(object):
    def __init__(self, args, tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer

    def initializer(self):
        # Use Encoder class as a container for global data
        if self.tokenizer is None:
            self.tokenizer = build_tokenizer(self.args)
        else:
            self.tokenizer = self.tokenizer

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(self.tokenizer.eod_id)
            ids[key] = doc_ids
        return ids, len(text)


class UniformEncoder(Encoder):
    def __init__(self, args, mode='sft', tokenizer=None):
        super().__init__(args, tokenizer=tokenizer)
        self.mode = mode
        # 实际计算时会Shift一位，因此这里seq_length + 1
        if args.load_raw_dataset:
            self.seq_length = args.seq_length + 1
            self.stride = args.seq_length
        else:
            self.seq_length = args.seq_length
            
        self.remain_input_ids = []
        self.remain_loss_mask = []

    def encode(self, data):
        
        encode_res = {
            "input_ids":[],
            "loss_mask":[]
        }

        if is_prompt_answer_format(data):
            data_type = 'prompt_answer'
        elif is_chatml_format(data):
            data_type = 'chatML'
        elif is_text_format(data):
            data_type = 'text'
        else:
            raise ValueError("data format not supported, please use prompt/answer, or chatML or pretrain text")

        for token_res in self._tokenize_fields(data, data_type=data_type):
            for k, v in token_res.items():
                encode_res[k].append(v)
        
        length = 0
        if data_type == 'prompt_answer':
            length = len(data['prompt']) + len(data['answer'])
        elif data_type == 'chatML':
            for chat in data['chat_rounds']:
                length += len(chat['content'])
        elif data_type == 'text':
            length += len(data['text'])
        
        return encode_res, length


    def _tokenize_fields(self, data, data_type):

        CHAT_COL = 'chat_rounds'
        ROLE_COL = 'role'
        CONTENT_COL = 'content'

        PROMPT_COL = 'prompt'
        ANSWER_COL = 'answer'
        SYSTEM_COL = 'system'

        TEXT_COL = 'text'

        if self.mode == 'sft':
            HUMAN = 'human'
            BOT = 'bot'
            SYSTEM = 'system'
            ROLE_START_MARKER = '<|role_start|>'
            ROLE_END_MARKER = '<|role_end|>'
        elif self.mode == 'pretrain' or data_type == 'text':
            HUMAN = ''
            BOT = ''
            SYSTEM = ''
            ROLE_START_MARKER = ''
            ROLE_END_MARKER = ''
        else:
            raise ValueError(f"tokenize_mode does not support {self.mode}, please use sft or pretrain")


        human_marker_ids = self.tokenizer.encode(f"{ROLE_START_MARKER}{HUMAN}{ROLE_END_MARKER}", add_special_tokens=False)
        bot_marker_ids = self.tokenizer.encode(f"{ROLE_START_MARKER}{BOT}{ROLE_END_MARKER}", add_special_tokens=False)
        system_marker_ids = self.tokenizer.encode(f"{ROLE_START_MARKER}{SYSTEM}{ROLE_END_MARKER}", add_special_tokens=False)
        sft_end_marker_ids = [self.tokenizer.eod_id]

        # 处理逻辑：
        # 统一处理SST,单轮、多轮sft的需求

        input_ids = []
        loss_mask = []

        if data_type == "prompt_answer":
            system = data.get(SYSTEM_COL, '')
            prompt = data[PROMPT_COL]
            answer = data[ANSWER_COL]
            system = punctuation_format(system)
            prompt = punctuation_format(prompt)
            answer = punctuation_format(answer)
            system_ids = system_marker_ids + self.tokenizer.encode(system, add_special_tokens=False) if system else []
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False) + sft_end_marker_ids
            input_ids += system_ids + human_marker_ids + prompt_ids + bot_marker_ids + answer_ids
            loss_mask += [0] * len(system_ids) + [0] * len(human_marker_ids) + [0] * len(prompt_ids) + \
                         [0] * len(bot_marker_ids) + [1] * len(answer_ids)
        elif data_type == 'chatML':
            chat = data[CHAT_COL]
            for r in chat:
                role = r[ROLE_COL]
                content = r[CONTENT_COL]
                content = punctuation_format(content)
                # if not content.endswith('\n'):  # chatML格式
                #     content = content + '\n'
                if role == HUMAN:
                    role_marker_ids = human_marker_ids
                    content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                elif role == BOT:
                    # 每一个bot输出结尾的eod,计算loss, 学会在哪里停， human和system的eod不需要计算loss
                    role_marker_ids = bot_marker_ids
                    content_ids = self.tokenizer.encode(content, add_special_tokens=False) + sft_end_marker_ids
                elif role == SYSTEM:
                    role_marker_ids = system_marker_ids
                    content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                else:
                    raise ValueError(f"Role {role} not supported.")

                input_ids += role_marker_ids + content_ids
                masklet = [1] if role == BOT else [0]
                loss_mask += [0] * len(role_marker_ids) + masklet * len(content_ids)
        elif data_type == "text":
            text = data[TEXT_COL]
            text = punctuation_format(text)
            text_ids = self.tokenizer.encode(text, add_special_tokens=False) + sft_end_marker_ids
            input_ids += text_ids
            loss_mask += [1] * len(text_ids)
        else:
            raise ValueError(
                f"data_type does not support {self.args.data_type}, please use chatML or prompt_answer or text(for pretrain)")
            
        # print(self.mode)
        if self.mode == 'pretrain':
            # change loss mask to all 1s
            input_ids = input_ids
            loss_mask = [1] * len(loss_mask)
        elif self.mode == 'sft':
            # do nothing
            input_ids = input_ids
            loss_mask = loss_mask

        assert len(input_ids) == len(loss_mask)
        if self.args.padding_mode == 'padding':
            if len(input_ids) <= self.seq_length:
                yield self.padding(input_ids, loss_mask)

            # 如果超长，直接抛弃 or 使用seq_length窗口滑动采样
            else:
                # cursor = 0
                # while cursor < len(input_ids):
                #     end_idx = min(cursor + self.seq_length, len(input_ids))
                #     yield self.padding(input_ids[cursor: end_idx], loss_mask[cursor: end_idx])
                #     cursor = end_idx
                yield {}
        elif self.args.padding_mode == 'concat':
            input_ids = self.remain_input_ids + input_ids
            loss_mask = self.remain_loss_mask + loss_mask
            if len(input_ids) < self.seq_length:
                self.remain_input_ids = input_ids
                self.remain_loss_mask = loss_mask
                assert len(self.remain_input_ids) == len(self.remain_loss_mask)
                yield {}
            else:
                cursor = 0
                while cursor + self.seq_length <= len(input_ids):
                    yield {
                        "input_ids": input_ids[cursor: cursor + self.seq_length],
                        "loss_mask": loss_mask[cursor: cursor + self.seq_length]
                    }
                    cursor = cursor + self.stride
                self.remain_input_ids = input_ids[cursor:]
                self.remain_loss_mask = loss_mask[cursor:]
                assert len(self.remain_input_ids) == len(self.remain_loss_mask)
                yield {}
        elif self.args.padding_mode == 'pack':
            if len(input_ids) > self.seq_length:
                yield {}
            elif len(self.remain_input_ids) + len(input_ids) > self.seq_length:
                input_ids, self.remain_input_ids = self.remain_input_ids, input_ids
                loss_mask, self.remain_loss_mask = self.remain_loss_mask, loss_mask
                assert len(input_ids) == len(loss_mask)
                yield self.padding(input_ids, loss_mask)
            else:
                self.remain_input_ids = self.remain_input_ids + input_ids
                self.remain_loss_mask = self.remain_loss_mask + loss_mask
                assert len(self.remain_input_ids) == len(self.remain_loss_mask)
                yield {}


    def padding(self, input_ids, loss_mask):
        pad_id = self.tokenizer.pad_id
        assert len(input_ids) <= self.seq_length, f"padding sequence: {len(input_ids)} > {self.seq_length}"
        input_ids += [pad_id] * (self.seq_length - len(input_ids))
        loss_mask += [0] * (self.seq_length - len(loss_mask))
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask
        } 

# 输入为args.input, 使用","进行分割，每一个itme可以是jsonl或文件夹
def find_jsonl_fnames(inputs):
    fnames = []
    for p in inputs.split(","):
        if not os.path.isdir(p):
            if p.endswith(".jsonl"):
                print(f"loading from {p}")
                fnames.append(p)
        else:
            p_list = glob.glob(p + "/*")
            for p_ in p_list:
                if p_.endswith(".jsonl"):
                    print(f"loading from {p_}")
                    fnames.append(p_)
    return fnames

def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data(key=['task', 'src_language', 'src_code', 'tgt_language', 'tgt_code', 'sql', 'prompt', 'answer', 'bad_answer'])):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)