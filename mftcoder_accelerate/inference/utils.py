# @author Chaoyu Chen
# @date 2024/11/12
# @module utils.py

import torch
import json
import os
import gzip
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
