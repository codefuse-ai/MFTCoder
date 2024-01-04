"""
# @author Chaoyu Chen
# @date 2023/9/13
Preprocessing data and tokenization.
"""

import os
import sys
import ftfy
import glob
print("In preprocess_data.py, sys path:", sys.path)

from tokenizer import build_tokenizer

CHAT_COL = 'chat_rounds'
ROLE_COL = 'role'
CONTENT_COL = 'content'

SYSTEM_COL = 'system'
PROMPT_COL = 'prompt'
ANSWER_COL = 'answer'

TEXT_COL = 'text'

table = {ord(f): ord(t) for f, t in zip(
    u'，。！？：【】（）％＃＠＆１２３４５６７８９０',
    u',.!?:[]()%#@&1234567890')}


def content_format(content: str):
    # Replace non-breaking space with space
    content = content.replace('\u202f', ' ').replace('\xa0', ' ')

    # change chinese punctuation to english ones
    # text = text.translate(table)

    content += '\n'

    return content


def is_text_format(data):
    if "text" in data:
        return True
    else:
        return False


def is_chatml_format(data):
    if "chat_rounds" in data and len(data["chat_rounds"]) > 0:
        return True
    else:
        return False


def is_prompt_answer_format(data):
    if "prompt" in data and "answer" in data:
        return True
    else:
        return False


def is_prompt_response_format(data):
    if "prompt" in data and "response" in data:
        return True
    else:
        return False


def is_input_output_format(data):
    if "input" in data and "output" in data:
        return True
    else:
        return False


def is_instruction_output_format(data):
    if "instruction" in data and "output" in data:
        return True
    else:
        return False


def is_instruction_response_format(data):
    if "instruction" in data and "response" in data:
        return True
    else:
        return False


def is_question_response_format(data):
    if "question" in data and "response" in data:
        return True
    else:
        return False

def is_question_answer_format(data):
    if "question" in data and "answer" in data:
        return True
    else:
        return False


class Encoder(object):
    tokenizer = None

    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        # self.tokenizer = build_tokenizer(self.args)
    
    def pure_encode(self, content):
        return Encoder.tokenizer.encode(content, add_special_tokens=False)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = self.pure_encode(text)
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod_id)
            ids[key] = doc_ids
        return ids, len(text)


class UniformEncoder(Encoder):
    

    def __init__(self, args, mode='sft'):
        super().__init__(args)
        self.mode = mode
        # seq_length + 1 for shifting
        if args.load_raw_dataset:
            self.seq_length = args.seq_length + 1
            self.stride = args.seq_length
        else:
            self.seq_length = args.seq_length

        self.remain_input_ids = []
        self.remain_loss_mask = []

    def encode(self, data, verbose=False):
        self.verbose = verbose
        encode_res = {
            "input_ids": [],
            "loss_mask": []
        }

        if is_prompt_answer_format(data):
            data_type = 'prompt_answer'
        elif is_prompt_response_format(data):
            data_type = 'prompt_response'
        elif is_input_output_format(data):
            data_type = 'input_output'
        elif is_instruction_output_format(data):
            data_type = 'instruction_output'
        elif is_instruction_response_format(data):
            data_type = 'instruction_response'
        elif is_question_response_format(data):
            data_type = 'question_response'
        elif is_question_answer_format(data):
            data_type = 'question_answer'
        elif is_chatml_format(data):
            data_type = 'chatML'
        elif is_text_format(data):
            data_type = 'text'
        else:
            raise ValueError(
                f"data_type does not support"
                f"please use chatML or prompt/answer, prompt/response, question/response, "
                f"instruction/output, input/output, instruction/output or text(only for pretrain)")
        
        length = 0
        if data_type == 'chatML':
            for chat in data['chat_rounds']:
                length += len(chat['content'])
        elif data_type == 'text':
            length += len(data['text'])
        else:
            # update key 
            global PROMPT_COL, ANSWER_COL
            PROMPT_COL, ANSWER_COL = tuple(data_type.split('_'))
            length = len(data[PROMPT_COL]) + len(data[ANSWER_COL])
        
        for token_res in self._tokenize_fields(data, data_type=data_type):
            for k, v in token_res.items():
                encode_res[k].append(v)

        return encode_res, length

    def _tokenize_fields(self, data, data_type):
        if self.mode == 'sft':
            if self.args.role_markers:
                system_marker = self.args.role_markers["system"]
                user_marker = self.args.role_markers["user"]
                assistant_marker = self.args.role_markers["assistant"]
            else:
                system_marker = '<s>system\n'
                user_marker = '<s>user\n'
                assistant_marker = '<s>assistant\n'
        elif self.mode == 'pretrain':
            system_marker = ''
            user_marker = ''
            assistant_marker = ''
        else:
            raise ValueError(f"tokenize_mode does not support {self.mode}, please use sft or pretrain")

        sft_end_marker_ids = [Encoder.tokenizer.eod_id]
        # uniform SST,SFT,MFT
        input_ids = []
        loss_mask = []

        if data_type == 'chatML':
            chat = data[CHAT_COL]
            if chat[0][ROLE_COL] == 'system':
                sys_content_ids = self.pure_encode(system_marker + content_format(chat[0][CONTENT_COL]))
                chat = chat[1:]
                input_ids += sys_content_ids
                loss_mask += [1] * len(sys_content_ids)

            for i, r in enumerate(chat):
                role = r[ROLE_COL]
                content = r[CONTENT_COL]
                content = content_format(content)
                if (role == 'human' or role == 'user') != (i % 2 == 0):
                    raise ValueError("Conversation roles must alternate user/assistant/user/assistant/... or human/bot/human/bot/...')")
                
                # compute loss only for assistant's content and eos token afterward
                if role == 'human' or role == 'user':
                    content_ids = self.pure_encode(user_marker + content + assistant_marker)
                    input_ids += content_ids
                    loss_mask += [0] * len(content_ids)
                elif role == 'bot' or role == 'assistant':
                    content_ids = self.pure_encode(content) + sft_end_marker_ids
                    input_ids += content_ids
                    loss_mask += [1] * len(content_ids)
                    extra_ids = self.pure_encode("\n")
                    input_ids += extra_ids
                    loss_mask += [0] * len(extra_ids)
                else:
                    raise ValueError(f"Role {role} not supported.")

        elif data_type == "text":
            text = data[TEXT_COL]
            text = content_format(text)
            text_ids = self.pure_encode(text) + sft_end_marker_ids
            input_ids += text_ids
            loss_mask += [1] * len(text_ids)
        else:
            system = data.get(SYSTEM_COL, '')
            prompt = data[PROMPT_COL]
            answer = data[ANSWER_COL]

            system = content_format(system_marker + system) if system else ""
            prompt = content_format(prompt)
            answer = content_format(answer)

            prompt_ids = self.pure_encode(system + user_marker + prompt + assistant_marker)
            answer_ids = self.pure_encode(answer) + sft_end_marker_ids

            input_ids += prompt_ids + answer_ids
            loss_mask += [0] * len(prompt_ids) + [1] * len(answer_ids)

            
        # print(self.mode)
        if self.mode == 'pretrain':
            # change loss mask to all 1s
            input_ids = input_ids
            loss_mask = [1] * len(loss_mask)
        elif self.mode == 'sft':
            # do nothing
            input_ids = input_ids
            loss_mask = loss_mask
        
        if self.verbose:
            print(f"original data:\n{data}")
            print(f"decoding back:\n{Encoder.tokenizer.decode(input_ids)}")

        assert len(input_ids) == len(loss_mask)
        if self.args.padding_mode == 'padding':
            if len(input_ids) <= self.seq_length:
                yield self.padding(input_ids, loss_mask)

            # drop if too long
            else:
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
        pad_id = Encoder.tokenizer.pad_id
        assert len(input_ids) <= self.seq_length, f"padding sequence: {len(input_ids)} > {self.seq_length}"
        input_ids += [pad_id] * (self.seq_length - len(input_ids))
        loss_mask += [0] * (self.seq_length - len(loss_mask))
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask
        }


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
