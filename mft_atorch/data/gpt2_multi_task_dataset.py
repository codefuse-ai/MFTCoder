import os
import json
import math
import time
import numpy as np
import torch
from functools import partial
from data.tokenization.preprocess_data import UniformEncoder
from utils.common_utils import get_local_rank, print_rank_0, TASK2ID, ID2TASK


class GPT2FromRawDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        input_dataset,
        # loss_mask_dataset,
        # num_samples,
        seq_length,
        weighted_loss_mode=None,
        ds_weight=1.0,
    ):

        self.name = name
        self.input_dataset = input_dataset
        self.num_samples = len(self.input_dataset['input_ids'])
        # self.loss_mask_dataset = loss_mask_dataset
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


class GPT2MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 args,
                 dataset_type,
                 data_paths,
                 tokenizer,
                 max_input_length=550,
                 max_output_length=550,
                 max_length=1024,
                 gpt_data=False,
                 world_size=1,
                 global_rank=0,
                 left_truncate=False,
                 shard_data=False,
                 **kwargs):
        super().__init__()
        self.args = args
        self.dataset_type = dataset_type
        self.mode = args.tokenize_mode
        self.seq_length = args.seq_length
        self.max_seq_length = args.seq_length + 1
        # self.max_seq_length = args.seq_length
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_length = max_length
        self.left_truncate = left_truncate
        self.kwargs = kwargs
        self.shard_data = shard_data
        self.world_size = world_size
        self.global_rank = global_rank
        self.datasets = None
        self.weights = None
        self.table = {ord(f):ord(t) for f,t in zip(
                u'，。！？：【】（）％＃＠＆１２３４５６７８９０',
                u',.!?:[]()%#@&1234567890')}
        self.BAD_MARKERS = [
                'An unhelpful answer:\n',
                'The following is a worse answer.\n',
                'The following is a less accurate answer.\n',
                'The following is a less correct answer.\n',
                # 'Generate a worse answer.\n',
                # 'Generate a less accurate answer.\n',
                # 'Generate a less correct answer.\n',
                '一个没有帮助的回答:\n',
                '下面是一个更差的回答.\n',
                '下面是一个不太准确的回答.\n',
                '下面是一个不太正确的回答.\n',
                # '请生成一个更差的回答.\n',
                # '请生成一个不太准确的回答.\n',
                # '请生成一个不太正确的回答.\n',
            ]
        self.GOOD_MARKERS = [
                'A helpful answer:\n',
                'The following is a better answer.\n',
                'The following is a more accurate answer.\n',
                'The following is a more correct answer.\n',
                # 'Generate a better answer.\n',
                # 'Generate a more accurate answer.\n',
                # 'Generate a more correct answer.\n',
                '一个有帮助的回答:\n',
                '下面是一个更好的回答.\n',
                '下面是一个更准确的回答.\n',
                '下面是一个更正确的回答.\n',
                # '请生成一个更好的回答.\n',
                # '请生成一个更准确的回答.\n',
                # '请生成一个更正确的回答.\n',
            ]
        self._load_dataset_from_jsonl()

        # self.datasets = None
        num_datasets = len(self.datasets)
        weights = self.weights
        assert num_datasets == len(weights)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

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
        print(f'size of {self.dataset_type} is {self.size} in rank {self.global_rank}')

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
    
    def encode(self, data):
        encode_res = {
            "input_ids":[],
            "loss_mask":[],
            # "labels": [],
        }
        
        for token_res in self._tokenize_fields(data):
            for k, v in token_res.items():
                encode_res[k].append(v)
        length = 0
        for chat in data['chat_rounds']:
            length += len(chat['content'])
        return encode_res, length

    def punctuation_format(self, text):
        # Replace non-breaking space with space
        # text = text.strip() + '\n'
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # change chinese punctuation to english ones
        text = text.translate(self.table)
        return text

    # def _tokenize_fields(self, prompt, content, bad_content, good_marker, bad_marker):
    def _tokenize_fields(self, data):
        
        CHAT_COL = 'chat_rounds'
        ROLE_COL = 'role'
        CONTENT_COL = 'content'
        HUMAN = 'human'
        BOT = 'bot'
        SYSTEM = 'system'
        ROLE_START_MARKER = '<|role_start|>'
        ROLE_END_MARKER = '<|role_end|>'
        human_marker_ids = self.tokenizer.encode(f"{ROLE_START_MARKER}{HUMAN}{ROLE_END_MARKER}")
        bot_marker_ids = self.tokenizer.encode(f"{ROLE_START_MARKER}{BOT}{ROLE_END_MARKER}")
        system_marker_ids = self.tokenizer.encode(f"{ROLE_START_MARKER}{SYSTEM}{ROLE_END_MARKER}")
        sft_end_marker_ids =  [self.tokenizer.eod_id]
        # 这个sft的eod要加在两个地方，第一个是prompt后面，第二个是answer后面

        input_ids = []
        loss_mask = []
        # labels = []

        chat = data[CHAT_COL]
        for r in chat:
            role = r[ROLE_COL]
            content = r[CONTENT_COL]
            content = self.punctuation_format(content)
            if not content.endswith('\n'):  # chatML格式
                content = content + '\n'
            if role == HUMAN:
                role_marker_ids = human_marker_ids
                content_ids = self.tokenizer.encode(content)
            elif role == BOT:
                role_marker_ids = bot_marker_ids
                content_ids = self.tokenizer.encode(content)
            elif role == SYSTEM:
                role_marker_ids = system_marker_ids
                content_ids = self.tokenizer.encode(content)
            else:
                raise ValueError(f"Role {role} not supported.")
            
            input_ids += role_marker_ids + content_ids + sft_end_marker_ids
            # 每一个bot输出结尾的eod,计算loss, 学会在哪里停， human和system的eod不需要计算loss
            masklet = [1] if role == BOT else [0]
            loss_mask += [0] * len(role_marker_ids) + masklet * len(content_ids) + masklet * len(sft_end_marker_ids)
            # masklet_labels = [1] if role == BOT else [-100]
            # labels += [-100] * len(role_marker_ids)

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
        # print(self.args.padding)
        if self.args.padding:
            if len(input_ids) <= self.max_seq_length:  # 实际长度2048 + 1 padding
                # yield self.padding(input_ids, loss_mask, labels)
                yield self.padding(input_ids, loss_mask)

            # 如果超长，直接使用seq_length窗口滑动采样
            else:
                # cursor = 0
                # while cursor < len(input_ids):
                #     end_idx = min(cursor + self.seq_length, len(input_ids))
                #     yield self.padding(input_ids[cursor: end_idx], loss_mask[cursor: end_idx])
                #     cursor = end_idx
                yield {}
        else:
            yield {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                # "labels": labels
            }

    def padding(self, input_ids, loss_mask, labels=None):
        # Pretrain阶段没加Padding，随便用一个Special Token，反正不算loss
        # pad_id = self.tokenizer.encode("<|extratoken_1|>")[0]
        pad_id = self.tokenizer.pad_id
        assert len(input_ids) <= self.max_seq_length, f"padding sequence: {len(input_ids)} > {self.max_seq_length}"
        input_ids += [pad_id] * (self.max_seq_length - len(input_ids))
        loss_mask += [0] * (self.max_seq_length - len(loss_mask))
        # labels += [-100] * (self.max_seq_length - len(labels))
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            # "labels": labels
        }

    def get_split(self):
        splits = []
        splits_string = self.args.data_split
        if splits_string.find(",") != -1:
            splits = [float(s) for s in splits_string.split(",")]
        elif splits_string.find("/") != -1:
            splits = [float(s) for s in splits_string.split("/")]
        else:
            splits = [float(splits_string)]
        while len(splits) < 3:
            splits.append(0.0)
        splits = splits[:3]
        return splits
    
    def _load_dataset_from_jsonl(self):
        data_prefixes = list(self.data_paths[1:-1].split(','))

        all_datasets = []
        all_datasets_length = []
        # 每个数据集的有效token数
        num_tokens = []
        effective_token_rate = []
        # 每个数据集的样本数
        total_sample_cnt = []

        self.global_num_samples = 0
        self.local_num_samples = 0

        # 不同数据集在不同文件夹下
        for dataset_index in range(len(data_prefixes)):
            if self.args.split_before_read:
                files = os.listdir(data_prefixes[dataset_index] + '/' + self.dataset_type)
            else:
                files = os.listdir(data_prefixes[dataset_index])
            cur_dataset_input_ids = []
            cur_dataset_loss_mask = []
            cur_dataset_labels = []
            cur_dataset_global_num_samples = 0
            cur_dataset_num_tokens = 0
            # 同一数据集下可能有多个jsonl文件
            for file in files:
                if self.args.split_before_read:
                    file_name = data_prefixes[dataset_index] + '/' + self.dataset_type + '/' + file
                else:
                    file_name = data_prefixes[dataset_index] + '/' + file
                    if os.path.isdir(file_name):
                        continue
                # fin = open(file_name, 'r')
                with open(file_name, 'r', encoding='utf-8', errors='replace') as fin:
                    lines = fin.readlines()
                line_num = len(lines)
                print_rank_0(f'lines of {file_name} are {line_num}')
                print_rank_0(f'open file {file_name}')
                if self.args.split_before_read:
                    start_index = 0
                    end_index = line_num
                else:
                    splits = self.get_split()
                    train_ratio = splits[0] / 100.0
                    train_num = int(math.ceil(train_ratio * line_num))
                    if self.dataset_type == 'train':
                        start_index = 0
                        end_index = train_num
                    if self.dataset_type == 'valid':
                        start_index = train_num
                        end_index = line_num
                for i in range(start_index, end_index):
                # for i, line in enumerate(lines):
                    line = lines[i]
                    self.global_num_samples += 1
                    if self.shard_data and i % self.world_size != self.global_rank:
                        continue
                    self.local_num_samples += 1
                    data = json.loads(line.rstrip('\n\r'))
                    features, length = self.encode(data)
                    
                    # 一个document可能编码不出sample
                    if len(features['input_ids']) == 0:
                        # print('no sample in this line!!!!!!')
                        self.global_num_samples -= 1
                        self.local_num_samples -= 1
                        continue
                    for idx in range(len(features['input_ids'])):
                        cur_dataset_input_ids.append(features['input_ids'][idx])
                        cur_dataset_loss_mask.append(features['loss_mask'][idx])
                        # cur_dataset_labels.append(features['labels'][idx])
                        # cur_dataset_num_tokens += sum(features['loss_mask'][idx])
                        if idx > 0:
                            # print(f'more than one sample in this line!!!!!! {idx + 1}')
                            self.global_num_samples += 1
                            self.local_num_samples += 1


                # print(f'features: {features}')
                fin.close()
            
            cur_dataset_input_ids = np.array(cur_dataset_input_ids, dtype=np.int32)
            cur_dataset_loss_mask = np.array(cur_dataset_loss_mask, dtype=np.int32)
            # cur_dataset_labels = np.array(cur_dataset_labels, dtype=np.int32)
            cur_dataset_num_tokens = np.sum(cur_dataset_loss_mask, dtype=np.int32)
            cur_dataset_sample_num = len(cur_dataset_input_ids)
            num_tokens.append(cur_dataset_num_tokens)
            total_sample_cnt.append(cur_dataset_sample_num)
            effective_token_rate.append(cur_dataset_num_tokens / (cur_dataset_sample_num * self.args.seq_length))

            # "task_id"字段会在getitem的时候获取
            cur_dataset = {'input_ids': cur_dataset_input_ids,
                           'loss_mask': cur_dataset_loss_mask,
                        #    'labels': cur_dataset_labels
                          }
            print(f"shape of cur {self.dataset_type} dataset in rank {self.global_rank}: {cur_dataset['input_ids'].shape}")
            # print(f"shape of cur valid dataset: {cur_valid_dataset['input_ids'].shape}")

            cur_ds = GPT2FromRawDataset(
                self.dataset_type,
                data_prefixes[dataset_index],
                cur_dataset,
                self.args.seq_length,
                weighted_loss_mode=self.args.weighted_loss_mode,
                # ds_weight=splits[0]
            )
            
            all_datasets.append(cur_ds)
            all_datasets_length.append(len(cur_ds))
        
        # 重写self.global_sample_num
        torch.distributed.barrier()
        if self.shard_data:
            device = f"cuda:{get_local_rank()}"
            global_num_samples_tensor = torch.tensor(self.local_num_samples, dtype=torch.int32)
            global_num_samples_tensor = global_num_samples_tensor.to(device)
            torch.distributed.all_reduce(global_num_samples_tensor, op=torch.distributed.ReduceOp.SUM)
            self.global_num_samples = global_num_samples_tensor.item()
            torch.distributed.barrier()
            print(f'global sample num of {self.dataset_type} in rank {self.global_rank}: {self.global_num_samples}')
            print(f'local sample num of {self.dataset_type} in rank {self.global_rank}: {self.local_num_samples}')
            print(f'num tokens of {self.dataset_type} in rank {self.global_rank}: {num_tokens}')
            print(f'effective token rate of {self.dataset_type} in rank {self.global_rank}: {effective_token_rate}')

        # weighted_loss_mode = self.args.weighted_loss_mode
        # ds_fn = partial(ds_weights_by_num_docs_sft)
        # if weighted_loss_mode == "token" or weighted_loss_mode == "random":
        #     ds_weights = ds_fn(num_tokens)
        #     loss_weights = ds_weights
        # elif weighted_loss_mode == "sample":
        #     loss_weights = ds_fn(all_datasets_length)
        # else:
        #     raise ValueError(f"weighted loss mode {weighted_loss_mode} is not supported.")
        num_tokens = []
        ds_fn = partial(ds_weights_by_num_docs_sft)
        loss_weights = ds_fn(all_datasets_length)
        
        print(f"> {self.dataset_type} loss weights in rank {self.global_rank}: {loss_weights}")

        factor = 1
        # calcualte common factor based on token cnt and total sample cnt
        if num_tokens:
            factor = sum(num_tokens) / (sum(total_sample_cnt) * self.args.seq_length)
            factor /= sum([1.0 / w for w in loss_weights]) / len(loss_weights)
        print(f"> common denomination factor for CE loss of {self.dataset_type} in rank {self.global_rank}: {factor}")
        
        sample_weights = [x / sum(all_datasets_length) for x in all_datasets_length]
        print(f"> {self.dataset_type} sample weights in rank {self.global_rank}: {sample_weights}")

        for i in range(len(all_datasets)):
            print(f'loss weight of {self.dataset_type} dataset {i} before update in rank {self.global_rank}: {all_datasets[i].ds_weight}')
        # train_dataset = None
        if all_datasets:
            for i in range(len(all_datasets)):
                all_datasets[i].update_ds_weight(loss_weights[i] / factor)
                print(f'loss weight of {self.dataset_type} dataset {i} after update in rank {self.global_rank}: {all_datasets[i].ds_weight}')
            self.datasets = all_datasets
            self.weights = sample_weights
    
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


def load_dataset_from_jsonl(args, tokenizer=None, shard_data=False, world_size=1, global_rank=0):

    # tokenization编码器
    encoder = UniformEncoder(args, args.tokenize_mode, tokenizer)
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
        cur_dataset_global_num_samples = 0
        cur_dataset_num_tokens = 0
        # 同一数据集下可能有多个jsonl文件
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
                    # 一个document可能编码出多个sample
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
                        

                # print(f'features: {features}')
                fin.close()

        # num_tokens.append(cur_dataset_num_tokens)
        # effective_token_rate.append(cur_dataset_num_tokens / (len(cur_dataset_loss_mask) * args.seq_length))
        
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

        # "weight"字段会在getitem的时候获取
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

    # 重写global_train_num和global_valid_num
    
    torch.distributed.barrier()
    device = f"cuda:{get_local_rank()}"
    
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
