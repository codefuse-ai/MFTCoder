"""
# @author Chaoyu Chen
# @date 2023/8/18

"""
import os
import json
import math
import time
import numpy as np
import torch
from functools import partial
from data.tokenization.preprocess_data import UniformEncoder
from utils.common_utils import TASK2ID, ID2TASK


class GPT2FromRawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        input_dataset,
        seq_length,
        weighted_loss_mode=None,
        ds_weight=1.0,
    ):

        self.name = name
        self.input_dataset = input_dataset
        self.num_samples = len(self.input_dataset['input_ids'])
        self.seq_length = seq_length

        self.weighted_loss_mode = weighted_loss_mode
        self.ds_weight = ds_weight
        self.task_name = data_prefix.split('/')[-1]
        self.task_id = TASK2ID[self.task_name]

        # Checks

    def update_ds_weight(self, weight):
        self.ds_weight = weight

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            # Get the shuffled index.
            idx = idx % self.num_samples
            idx_data = {key: self.input_dataset[key][idx]
                        for key in self.input_dataset}

            if self.weighted_loss_mode:
                idx_data["weight"] = np.array([self.ds_weight], dtype=np.float32)
                idx_data["task_id"] = np.array([self.task_id], dtype=np.int)
                return idx_data
            else:
                idx_data["task_id"] = np.array([self.task_id], dtype=np.int)
                return idx_data
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING in GPT2FromRawDataset: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


def ds_weights_by_num_docs_sft(l, alpha=0.3):
    # ignore alpha
    weights = [1 / i for i in l]
    weights_sum = sum(weights)
    weights = [weight / weights_sum for weight in weights]
    return weights


class GPT2BlendableDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights, global_num_samples, local_num_samples):
        self.datasets = datasets
        num_datasets = len(datasets)
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

        assert local_num_samples == self.size
        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights

        # recompute weights
        weights = self.calc_weights()

        # Build indices.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        self.global_num_samples = global_num_samples
        self.local_num_samples = local_num_samples

        from data import helpers

        helpers.build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,
            weights,
            num_datasets,
            self.size,
            torch.distributed.get_rank() == 0,
        )

        print(
            "> RANK {} elapsed time for building blendable dataset indices: "
            "{:.2f} (sec)".format(
                torch.distributed.get_rank(), time.time() - start_time
            )
        )

    def calc_weights(self):
        dataset_sample_cnt = [len(ds) for ds in self.datasets]
        total_cnt = sum(dataset_sample_cnt)
        weights = np.array([(cnt + 0.0) / total_cnt for cnt in dataset_sample_cnt], dtype=np.float64)
        return weights

    def __len__(self):
        return self.global_num_samples

    def __getitem__(self, idx):
        try:
            idx = idx % self.local_num_samples
            dataset_idx = self.dataset_index[idx]
            sample_idx = self.dataset_sample_index[idx]
            return self.datasets[dataset_idx][sample_idx]
        except IndexError:
            # new_idx = idx % len(self)
            new_idx = idx % self.local_num_samples
            print(self.local_num_samples)
            print(
                f"WARNING in GPT2MultiTaskDataset: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


def shuffle_arrays(arrays, set_seed=-1):
        """Shuffles arrays in-place, in the same order, along axis=0

        Parameters:
        -----------
        arrays : List of NumPy arrays.
        set_seed : Seed value if int >= 0, else seed is random.
        """
        assert all(len(arr) == len(arrays[0]) for arr in arrays)
        seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed

        for arr in arrays:
            rstate = np.random.RandomState(seed)
            rstate.shuffle(arr)


def load_dataset_from_jsonl(args, shard_data=False, world_size=1, global_rank=0, local_rank=0):

    # tokenization编码器
    encoder = UniformEncoder(args, args.tokenize_mode)
    encoder.initializer()

    data_prefixes = list(args.data_paths[1:-1].split(','))
    
    # data_weights = list(map(float, args.data_weights[1:-1].split(',')))
    # print("data weights: ")
    # print(data_weights)
    splits = []
    splits_string = args.data_split
    if splits_string.find(",") != -1:
        splits = [float(s) for s in splits_string.split(",")]
    elif splits_string.find("/") != -1:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    print(f'data splits: {splits}')

    all_train_datasets = []
    all_valid_datasets = []
    all_train_datasets_length = []
    all_valid_datasets_length = []
    # 每个数据集的有效token数
    num_tokens = []
    effective_token_rate = []
    total_sample_cnt = []

    local_train_num = 0
    local_valid_num = 0

    # 不同数据集在不同文件夹下
    for dataset_index in range(len(data_prefixes)):
        files = os.listdir(data_prefixes[dataset_index])
        cur_dataset_input_ids = []
        cur_dataset_loss_mask = []
        # support multiple jsonl files under task dir
        for file in files:
            file_name = data_prefixes[dataset_index] + '/' + file
            if os.path.isdir(file_name):
                continue
            fin = open(file_name, 'r')
            print(f'[Global Rank {global_rank}] open file {file_name}')

            if args.padding_mode == 'padding' or args.padding_mode == 'pack':
                for i, line in enumerate(fin):
                    # pre-sharding
                    if shard_data and i % world_size != global_rank:
                        continue
                    data = json.loads(line.rstrip('\n\r'))
                    features, length = encoder.encode(data)
                    # may have more samples
                    for idx in range(len(features['input_ids'])):
                        cur_dataset_input_ids.append(features['input_ids'][idx])
                        cur_dataset_loss_mask.append(features['loss_mask'][idx])
                        
                fin.close()
            else:
                i = 0
                for line in fin:
                    data = json.loads(line.rstrip('\n\r'))
                    features, length = encoder.encode(data)
                    # 一个document可能编码不出sample,可能编码出多个sample
                    for idx in range(len(features['input_ids'])):
                        # post-sharding
                        if shard_data and i % world_size != global_rank:
                            i += 1
                            continue
                        i += 1
                        cur_dataset_input_ids.append(features['input_ids'][idx])
                        cur_dataset_loss_mask.append(features['loss_mask'][idx])
                fin.close()
        
        cur_dataset_input_ids = np.array(cur_dataset_input_ids, dtype=np.float32)
        cur_dataset_loss_mask = np.array(cur_dataset_loss_mask, dtype=np.float32)
        cur_dataset_num_tokens = np.sum(cur_dataset_loss_mask, dtype=np.int32)
        cur_dataset_sample_num = len(cur_dataset_input_ids)
        num_tokens.append(cur_dataset_num_tokens)
        total_sample_cnt.append(cur_dataset_sample_num)
        effective_token_rate.append(cur_dataset_num_tokens / (cur_dataset_sample_num * args.seq_length))
        
        # shuffle before split
        shuffle_arrays([cur_dataset_input_ids, cur_dataset_loss_mask], args.seed)
        train_ratio = splits[0] / 100.0
        train_num = int(math.ceil(train_ratio * cur_dataset_sample_num))
        # split train/valid
        cur_train_input_ids, cur_valid_input_ids = cur_dataset_input_ids[: train_num], cur_dataset_input_ids[train_num: ]
        cur_train_loss_mask, cur_valid_loss_mask = cur_dataset_loss_mask[: train_num], cur_dataset_loss_mask[train_num: ]
        local_train_num += train_num
        local_valid_num += (cur_dataset_sample_num - train_num)

        cur_train_dataset = {'input_ids': cur_train_input_ids,
                             'loss_mask': cur_train_loss_mask
                        }
        cur_valid_dataset = {'input_ids': cur_valid_input_ids,
                             'loss_mask': cur_valid_loss_mask
                        }
        print(f"[Global Rank {global_rank}]shape of cur train dataset: {cur_train_dataset['input_ids'].shape}")
        print(f"[Global Rank {global_rank}]shape of cur valid dataset: {cur_valid_dataset['input_ids'].shape}")

        cur_train_ds = GPT2FromRawDataset(
            'train',
            data_prefixes[dataset_index],
            cur_train_dataset,
            args.seq_length,
            weighted_loss_mode=args.weighted_loss_mode,
            ds_weight=splits[0]
        )
        cur_valid_ds = GPT2FromRawDataset(
            'valid',
            data_prefixes[dataset_index],
            cur_valid_dataset,
            args.seq_length,
            weighted_loss_mode=args.weighted_loss_mode,
            ds_weight=splits[1]
        )
        
        all_train_datasets.append(cur_train_ds)
        all_valid_datasets.append(cur_valid_ds)
        all_train_datasets_length.append(len(cur_train_ds))
        all_valid_datasets_length.append(len(cur_valid_ds))
    
    print(f'[Global Rank {global_rank}]num tokens: {num_tokens}')
    print(f'[Global Rank {global_rank}]effective token rate: {effective_token_rate}')

    num_tokens = []
    ds_fn = partial(ds_weights_by_num_docs_sft)
    train_loss_weights, valid_loss_weights = (
        ds_fn(all_train_datasets_length),
        ds_fn(all_valid_datasets_length),
    )
    
    print(f"> train loss weights in rank {global_rank}: {train_loss_weights}")
    print(f"> valid loss weights in rank {global_rank}: {valid_loss_weights}")

    factor = 1
    # calcualte common factor based on token cnt and total sample cnt
    if num_tokens:
        factor = sum(num_tokens) / (sum(total_sample_cnt) * args.seq_length)
        factor /= sum([1.0 / w for w in train_loss_weights]) / len(train_loss_weights)
    print(f"> common denomination factor for CE loss in rank {global_rank}: {factor}")
    
    train_sample_weights = [x / sum(all_train_datasets_length) for x in all_train_datasets_length]
    valid_sample_weights = [x / sum(all_valid_datasets_length) for x in all_valid_datasets_length]
    print(f"> train sample weights in rank {global_rank}: {train_sample_weights}")
    print(f"> valid sample weights in rank {global_rank}: {valid_sample_weights}")

    # recompute global_train_num and global_valid_num
    
    torch.distributed.barrier()
    device = f"cuda:{local_rank}"
    
    global_train_num_samples_tensor = torch.tensor(local_train_num, dtype=torch.int32)
    global_train_num_samples_tensor = global_train_num_samples_tensor.to(device)
    torch.distributed.all_reduce(global_train_num_samples_tensor, op=torch.distributed.ReduceOp.SUM)
    global_train_num = global_train_num_samples_tensor.item()
    
    global_valid_num_samples_tensor = torch.tensor(local_valid_num, dtype=torch.int32)
    global_valid_num_samples_tensor = global_valid_num_samples_tensor.to(device)
    torch.distributed.all_reduce(global_valid_num_samples_tensor, op=torch.distributed.ReduceOp.SUM)
    global_valid_num = global_valid_num_samples_tensor.item()
    print(f"> global train num in rank {global_rank}: {global_train_num}")
    print(f"> global valid num in rank {global_rank}: {global_valid_num}")
    
    torch.distributed.barrier()

    for i in range(len(all_train_datasets)):
        print(f'loss weight of train dataset {i} before update in rank {global_rank}: {all_train_datasets[i].ds_weight}')
    blending_train_dataset = None
    if all_train_datasets:
        args.do_train = True
        for i in range(len(all_train_datasets)):
            all_train_datasets[i].update_ds_weight(train_loss_weights[i] / factor)
            print(f'loss weight of train dataset {i} after update in rank {global_rank}: {all_train_datasets[i].ds_weight}')
        blending_train_dataset = GPT2BlendableDataset(all_train_datasets, train_sample_weights, global_train_num, local_train_num)
    
    for i in range(len(all_train_datasets)):
        print(f'loss weight of valid dataset {i} before update in rank {global_rank}: {all_train_datasets[i].ds_weight}')
    blending_valid_dataset = None
    if all_valid_datasets:
        args.do_valid = True
        for i in range(len(all_valid_datasets)):
            all_valid_datasets[i].update_ds_weight(valid_loss_weights[i] / factor)
            print(f'loss weight of valid dataset {i} after update in rank {global_rank}: {all_train_datasets[i].ds_weight}')
        blending_valid_dataset = GPT2BlendableDataset(all_valid_datasets, valid_sample_weights, global_valid_num, local_valid_num)
    
    return blending_train_dataset, blending_valid_dataset
