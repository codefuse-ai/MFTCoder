# -*- coding: utf-8 -*-

import argparse
import multiprocessing
import os
import sys
import random
import time
import tqdm
import glob
import json
import numpy as np


# 将父目录的父目录加入path 
current_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_path))
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from tokenizer import init_tokenizer
from pack_encoder import PackSSTBinEncoder, load_tokenizer
from data import indexed_dataset

from threading import Semaphore
from colorama import Fore
import lm_fmt as lmd


def yield_from_files(files: list, semaphore):
    """
    Iterator over input documents 

    :param fnames: list of filenames
    """
    def yielder(fname, semaphore):
        with open(fname, 'r') as f:
            for line in f:
                semaphore.acquire()
                yield json.loads(line)

    for fname in files:
        semaphore.acquire()
        yield from yielder(fname, semaphore)

def yield_from_files2(fnames: list, semaphore, sample_percent):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """
    def yielder(fname, semaphore):
        try:
            sample_interval = int(1/sample_percent)
            for f in filter(lambda x: x, lmd.Reader(fname).stream_data(key=None)):
                rand_value = random.randint(1, sample_interval*100)
                if rand_value % sample_interval != 0:
                    continue
                semaphore.acquire()
            
                #rand_value = random.randint(1, sample_interval*100)
                #if rand_value % sample_interval != 0:
                #    yield None

                yield f
        except Exception as e:
            print('####Exception:', e.args)
            yield None

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def print_example_doc(input_ids, tokenizer):
    print(Fore.YELLOW + f'INPUT IDS len: {len(input_ids)}')
    print(Fore.BLUE + f'INPUT IDS:\n {input_ids}\n\n')

    print(Fore.RED + f'DETOKENIZED INPUT:\n{tokenizer.decode(input_ids)}')


def core_process(encoded_docs, semaphore, seq_length, tokenizer, encoder, builder, output_idx_file):
    """
    core of Data Pack SFT processing
    """
    input_ids_key = 'input_ids'

    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    sentence_droped = 0
    loss_token_cnt = 0

    print("PRINT BEFORE STREAM PROCESS DATA")

    print_example_count = 0  
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence,
        # For sft, each document has only one sample
        input_ids_sentence = doc[input_ids_key][0]
        if len(input_ids_sentence) < 1:
            sentence_droped += 1
            continue

        builder.add_item(np.array(input_ids_sentence, dtype=builder.dtype))
        builder.end_document()
        #builder.finalize_without_close(output_idx_file)
        #builder.add_item_and_end_document_and_finalize(np.array(input_ids_sentence, dtype=builder.dtype), output_idx_file)

        # print the first packed sample as example
        if print_example_count < 1:
            print_example_doc(input_ids_sentence, tokenizer)
            print_example_count += 1

        # log progress
        if i % 100 == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i} documents ({i / elapsed} docs/s, {mbs} MB/s)."
            )
            if i != 0:
                pbar.update(100)

    # 尾部处理
    builder.finalize(output_idx_file)

    print(Fore.RED + "\ndroped docs: {}".format(sentence_droped))


def process_dataset(dataset_path, output_path, model_path, parallel_num, seq_length, dataset_name, sample_percent):
    """
    Re-organize samples in the given data path into a Data Pack file.
    """

    # get all jsonl files and corresponding reading handler
    files = glob.glob(os.path.join(dataset_path, '**/*.jsonl'), recursive=True)

    # build a semaphore object to stop `yield_from_files` from getting ahead 
    # of encoder.encode and hence building up memory
    semaphore = Semaphore(1000 + parallel_num)

    # build sample iterator
    sample_iterator = yield_from_files2(files, semaphore, sample_percent)  

    # load tokenizer
    # tokenizer = load_tokenizer(model_path, tokenizer_type)
    tokenizer = init_tokenizer(model_path)
    print('TOKEN of id=2:', tokenizer.convert_ids_to_tokens(2))
    print('ID of </s>:', tokenizer.convert_tokens_to_ids('</s>'))
    print('TOKEN of id=0:', tokenizer.convert_ids_to_tokens(0))
    print('ID of </unk>:', tokenizer.convert_tokens_to_ids('</unk>'))

    # init encoder
    encoder = PackSSTBinEncoder(seq_length, model_path)

    # create writer builder
    key = "input_ids"
    output_prefix = os.path.join(output_path, dataset_name)
    output_bin_file = "{}_{}.bin".format(
        output_prefix, key
    )
    output_idx_file = "{}_{}.idx".format(
        output_prefix, key
    )
    builder = indexed_dataset.make_builder(
        output_bin_file,
        impl="mmap",
        vocab_size=tokenizer.vocab_size,
    )

    if parallel_num > 1:
        pool = multiprocessing.Pool(parallel_num, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, sample_iterator, chunksize=32)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in sample_iterator)

    if dataset_name is None:
        dataset_path = dataset_path[:-1] if dataset_path.endswith(os.path.sep) else dataset_path
        dataset_name = dataset_path.split(os.path.sep)[-1]

    core_process(encoded_docs, semaphore, seq_length, tokenizer, encoder, builder, output_idx_file)


def main(data_path, output_path, model_path, parallel_num, seq_length, dataset_name, sample_percent):
    """
    Entry
    """

    process_dataset(data_path, output_path, model_path, parallel_num, seq_length, dataset_name, sample_percent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a packed jsonl file in the Data Pack SFT way.")
    parser.add_argument('--model-path', type=str, help='Path of a pretrained model which contains tokenizer-related files.')
    parser.add_argument('--parallel', type=int, default=1, help='The num of parallel processing.')
    parser.add_argument('--output-path', type=str, help='Path to store the genered result file.')
    parser.add_argument('--data-path', type=str, default=None, help='Path of files to be processed')
    parser.add_argument('--seq-length', type=int, default=4096, help='The max input length (i.e. the max number of tokens in a sample)')
    # parser.add_argument('--eod-token-id', type=int, default=2, help='EOD token id')
    # parser.add_argument('--pad-token-id', type=int, default=0, help='PAD token id')
    # parser.add_argument('--tokenizer-type', type=str, choices=["LLAMATokenizer", None], default=None, help="What type of tokenizer to use. Default is None.")
    parser.add_argument('--dataset-name', type=str, default=None, help='The generated result dataset name. The folder name will be token by default.')
    parser.add_argument('--sample-percent', type=float, default=1.0, help='Sample percentage')

    args = parser.parse_args()
    print('ARGS\n', '\n'.join([str(key) + ':' + str(value) for key,value in vars(args).items()]))

    random.seed(9999)

    main(args.data_path, args.output_path, args.model_path, args.parallel, args.seq_length, args.dataset_name, args.sample_percent)
