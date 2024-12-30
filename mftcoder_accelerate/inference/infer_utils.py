# @author Chaoyu Chen
# @date 2024/11/12
"""Some inference utils"""
import torch
import json
import os
import gzip
import re
import ast
from tqdm import tqdm
from typing import Iterable, Dict, List


def print_args(args):
    message = "\n".join([f"{k:<20}:   {v}" for k, v in vars(args).items()])
    print("====" * 30)
    print(message)
    print("====" * 30)
    print("GPU: {}".format(torch.cuda.current_device()))


def get_line_count(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def batch_stream_jsonl(stream: Iterable[Dict], batch_size) -> Iterable[List]:
    batch = list()
    for item in stream:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def flatten_batch_stream(batch_stream):
    for batch in batch_stream:
        for item in batch:
            yield item


def write_jsonl(filename: str, data: Iterable[Dict], total, append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in tqdm(data, total=total):
                fp.write((json.dumps(x) + "\n").encode("utf-8"))


def is_compilable(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def is_tests(code):
    return code.strip().startswith("assert")


def extract_python_code_block(response):
    # pattern = r"^```[Pp]ython\s*\n(.*?)(?=^```)"
    pattern = r"^```\s*(?:[Pp]ython)?\s*\n(.*?)(?=^```)"
    result = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
    return "\n".join([x for x in result if is_compilable(x) and not is_tests(x)])


def extract_python_test_block(response):
    # pattern = r"^```[Pp]ython\s*\n(.*?)(?=^```)"
    pattern = r"^```\s*(?:[Pp]ython)?\s*\n(.*?)(?=^```)"
    result = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
    return "\n".join([x for x in result if is_compilable(x) and is_tests(x)])


def extract_code_with_lang(text):
    """
    使用正则表达式从文本中提取任意语言的代码块。

    参数:
    - text: 包含Markdown代码块的字符串。

    返回:
    - 一个包含所有匹配的代码块及其语言的列表，每一个元组都是 (language, code_block)。
    """
    # 编译正则表达式模式，匹配任意语言的代码块
    pattern = re.compile(r"```([\w+#-]*)\s*\n(.*?)```", re.DOTALL | re.MULTILINE)

    # 使用findall方法找出所有匹配的代码块及其语言标识
    # 返回的是一个元组列表，每个元组包含 (language, code_block)
    return pattern.findall(text)