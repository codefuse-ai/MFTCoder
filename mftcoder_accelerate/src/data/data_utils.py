# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import math
import torch
import numpy as np
from typing import List, Tuple
from functools import partial

from utils.common_utils import print_rank_0, TASK2ID, ID2TASK
from data.indexed_dataset import make_dataset as make_indexed_dataset
from data.blendable_dataset import BlendableDataset
from data.gpt2_dataset import GPT2Dataset, GPT2PromptDataset


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    print_rank_0(
        " > finished creating indexed dataset in {:4f} "
        "seconds".format(time.time() - start_time)
    )
    print_rank_0("    number of documents: {}".format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def build_train_valid_test_datasets(
    data_prefix,
    use_shared_fs,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    seq_length,
    seed,
    skip_warmup,
    build_index_mappings=True,
    shuffle_before_split=False,
    weighted_loss_mode=None,
    ds_weights=[1., 1., 1.],
    train_mode='sft',
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    assert os.path.exists(data_prefix + "_input_ids.bin"), f"Input tokens datafile not found: {data_prefix}_input_ids.bin"

    # Indexed dataset.
    input_ids_indexed_dataset = get_indexed_dataset_(data_prefix + "_input_ids", data_impl, skip_warmup)
    if train_mode == 'sft':
        loss_mask_indexed_dataset = get_indexed_dataset_(data_prefix + "_loss_mask", data_impl, skip_warmup)
    else:
        print(f'pretrain mode, loss mask is ones')
        loss_mask_indexed_dataset = None

    total_num_of_documents = input_ids_indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    # shuffle index before build_dataset
    shuffle_doc_index = []
    if shuffle_before_split:
        total_num_docs = splits[-1] - splits[0]
        shuffle_doc_index = np.arange(start=0, stop=total_num_docs, step=1, dtype=np.uint32)
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(shuffle_doc_index)

    def build_dataset(index, name, ds_weight=1.0):
        dataset = None
        if splits[index + 1] > splits[index]:
            if shuffle_before_split:
                documents = shuffle_doc_index[splits[index]:splits[index + 1]]
            else:
                documents = np.arange(
                    start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32
                )

            dataset = GPT2PromptDataset(
                name,
                data_prefix,
                documents,
                input_ids_indexed_dataset,
                loss_mask_indexed_dataset,
                train_valid_test_num_samples[index],
                seq_length,
                seed,
                build_index_mappings=build_index_mappings,
                use_shared_fs=use_shared_fs,
                weighted_loss_mode=weighted_loss_mode,
                ds_weight=ds_weight,
                train_mode=train_mode,
            )
        return dataset

    train_dataset = build_dataset(0, "train", ds_weights[0])
    valid_dataset = build_dataset(1, "valid", ds_weights[1])
    test_dataset = build_dataset(2, "test", ds_weights[2])

    return train_dataset, valid_dataset, test_dataset, total_num_of_documents


def build_multiple_train_valid_test_datasets(args, train_valid_test_num_samples, use_shared_fs=True, data_impl="mmap", mmap_warmup=False):
    """Build multiple train, valid, and test datasets."""
    data_prefixes = list(args.data_paths[1:-1].split(','))

    data_weights = list(map(float, args.data_weights[1:-1].split(',')))
    print("data weights: ")
    print(data_weights)
    use_shared_fs = use_shared_fs
    data_impl = data_impl
    splits_string = args.data_split
    seq_length = args.seq_length
    # seq_length = args.block_size
    seed = args.seed
    skip_warmup = (not mmap_warmup)
    weight_by_num_documents = args.weight_by_num_documents
    shuffle_before_split = args.shuffle_before_split
    weighted_loss_mode = args.weighted_loss_mode

    weights, weighted_train_valid_test_num_samples = get_datasets_normalized_weights_and_num_samples(
        data_weights, train_valid_test_num_samples
    )

    train_weights, valid_weights, test_weights = weights, weights, weights

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(data_prefixes)):
        train_ds, valid_ds, test_ds, _ = build_train_valid_test_datasets(
            data_prefixes[i],
            use_shared_fs,
            data_impl,
            splits_string,
            weighted_train_valid_test_num_samples[i],
            seq_length,
            seed,
            skip_warmup,
            build_index_mappings=not weight_by_num_documents,
            shuffle_before_split=shuffle_before_split,
            weighted_loss_mode=weighted_loss_mode,
            train_mode=args.tokenize_mode,
        )
        if train_ds is not None:
            train_datasets.append(train_ds)
        if valid_ds is not None:
            valid_datasets.append(valid_ds)
        if test_ds is not None:
            test_datasets.append(test_ds)

    factor = 1
    if weight_by_num_documents:
        # gets the number of documents in each data path
        get_num_docs_list = lambda datasets: [
            dataset.input_ids_indexed_dataset.sizes.shape[0] for dataset in datasets
        ]
        train_num_docs, valid_num_docs, test_num_docs = (
            get_num_docs_list(train_datasets),
            get_num_docs_list(valid_datasets),
            get_num_docs_list(test_datasets),
        )

        # builds weights according to the number of docs
        fn = partial(weights_by_num_docs_sft)
        train_weights, valid_weights, test_weights = (
            fn(train_num_docs),
            fn(valid_num_docs),
            fn(test_num_docs),
        )
        assert sum(train_weights) != 0.0, "found train weights to be 0.0"
        assert sum(valid_weights) != 0.0, "found valid weights to be 0.0"
        
        train_weights, train_num_samples = get_normalized_weights_and_num_samples(
            train_weights, train_valid_test_num_samples[0]
        )
        print_rank_0(f"> train sample weights: {train_weights}")
        valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
            valid_weights, train_valid_test_num_samples[1]
        )
        print_rank_0(f"> valid sample weights: {valid_weights}")
        if sum(test_weights) == 0.0:
            test_weights = weights  # use original weights
        test_weights, test_num_samples = get_normalized_weights_and_num_samples(
            test_weights, train_valid_test_num_samples[2]
        )

        # weighted loss
        num_tokens = []
        ds_fn = partial(ds_weights_by_num_docs_sft)
        train_ds_weights, valid_ds_weights, test_ds_weights = (
            ds_fn(train_num_docs),
            ds_fn(valid_num_docs),
            ds_fn(test_num_docs),
        )

        assert sum(train_ds_weights) != 0.0, "found train loss weights to be 0.0"
        assert sum(valid_ds_weights) != 0.0, "found valid loss weights to be 0.0"

        if sum(test_ds_weights) == 0.0:
            test_ds_weights = weights  # use original weights
        print_rank_0(f"> train loss weights: {train_ds_weights}")
        print_rank_0(f"> valid loss weights: {valid_ds_weights}")

        train_datasets = []
        valid_datasets = []
        test_datasets = []
        total_sample_cnt = []
        for i in range(len(data_prefixes)):
            train_ds, valid_ds, test_ds, total_num_of_documents = build_train_valid_test_datasets(
                data_prefixes[i],
                use_shared_fs,
                data_impl,
                splits_string,
                [train_num_samples[i], valid_num_samples[i], test_num_samples[i]],
                seq_length,
                seed,
                skip_warmup,
                build_index_mappings=True,
                shuffle_before_split=shuffle_before_split,
                weighted_loss_mode=weighted_loss_mode,
                ds_weights=[train_ds_weights[i], valid_ds_weights[i], test_ds_weights[i]],
                train_mode=args.tokenize_mode,
            )
            total_sample_cnt.append(total_num_of_documents)
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        # calcualte common factor based on token cnt and total sample cnt
        if num_tokens:
            factor = sum(num_tokens) / (sum(total_sample_cnt) * args.seq_length)
            factor /= sum([1.0 / w for w in train_ds_weights]) / len(train_ds_weights)
            
    print_rank_0(f"> common denomination factor for CE loss: {factor}")

    # Blend.
    blending_train_dataset = None
    if train_datasets:
        i = 0
        for ds in train_datasets:
            ds.update_ds_weight(ds.ds_weight / factor)
            print(f'loss weight of dataset {i} after update: {ds.ds_weight}')
            i += 1
        blending_train_dataset = BlendableDataset(train_datasets, train_weights)
    blending_valid_dataset = None
    if valid_datasets:
        for ds in valid_datasets:
            ds.update_ds_weight(ds.ds_weight / factor)
        blending_valid_dataset = BlendableDataset(valid_datasets, valid_weights)
    blending_test_dataset = None
    if test_datasets:
        for ds in test_datasets:
            ds.update_ds_weight(ds.ds_weight / factor)
        blending_test_dataset = BlendableDataset(test_datasets, test_weights)

    return blending_train_dataset, blending_valid_dataset, blending_test_dataset


def get_train_valid_test_split_(splits_string, size):
    """Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] + int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def get_normalized_weights_and_num_samples(
    weights: List[float], num_samples: int
) -> Tuple[List[float], List[int]]:
    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    weighted_num_samples = []
    for weight in weights:
        weighted_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
    return weights, weighted_num_samples


def get_datasets_normalized_weights_and_num_samples(
    weights: List[float], num_samples: List[int]
) -> Tuple[List[float], List[List[int]]]:
    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    weighted_num_samples = []
    for weight in weights:
        weighted_num_samples.append(
            [
                int(math.ceil(val * weight * 1.005))
                for val in num_samples
            ]
        )
    return weights, weighted_num_samples


def ds_weights_by_num_docs_sft(l, alpha=0.3):
    # ignore alpha
    weights = [1 / i for i in l]
    weights_sum = sum(weights)
    weights = [weight / weights_sum for weight in weights]
    return weights


def weights_by_num_docs_sft(l, alpha=0.3):
    # ignore alpha
    total_n_docs = sum(l)
    unbiased_sample_probs = [i / total_n_docs for i in l]

    return unbiased_sample_probs


def weights_by_num_docs(l: list, alpha=0.3):
    """
    Builds weights from a multinomial distribution over groups of data according to the number of
    samples in each group.

    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups

    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.

    See https://arxiv.org/abs/1911.02116 for more details
    """
    if len(l) == 1:
        return [1.0]

    total_n_docs = sum(l)
    unbiased_sample_probs = [i / total_n_docs for i in l]

    probs = [i**alpha for i in unbiased_sample_probs]

    # normalize
    total = sum(probs)
    probs = [i / total for i in probs]

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = [1 - p for p in unbiased_sample_probs]
    weights = [p * p2 for p, p2 in zip(probs, unbiased_sample_probs_inverse)]

    # normalize
    total = sum(weights)
    weights = [i / total for i in weights]

    return weights


def load_dataset_from_bin(args):
    """XXX"""

    print_rank_0("> building train, validation, and test datasets ...")

    # Number of train/valid/test samples.
    train_iters = 2
    valid_iters = 2
    test_iters = 2
    train_val_test_num_samples = [
        train_iters * 10,
        valid_iters * 10,
        test_iters * 10,
    ]

    # multiple data paths for SFT task
    train_ds, valid_ds, test_ds = build_multiple_train_valid_test_datasets(
        args=args,
        train_valid_test_num_samples=train_val_test_num_samples,
    )

    return train_ds, valid_ds, test_ds
