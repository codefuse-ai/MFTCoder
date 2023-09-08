import sys
sys.path.append("..")
import torch
import atorch
from utils.common_utils import print_rank_0, TASK2ID, ID2TASK
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

torch.set_printoptions(threshold=np.inf)


def get_task_mask(task_id):
    task_num = len(TASK2ID)
    task_mask = torch.zeros(task_id.shape[0], task_num)
    task_mask[torch.arange(task_id.size(0)).unsqueeze(1), task_id] = 1
    
    return task_mask


def get_task_loss(task_losses, task_id):  # TODO
    # Fix a printing order.
    task_loss_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    # Count the occurrences of each task.
    task_num_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    for i in range(len(task_id)):
        task_num_per_batch[task_id[i][0]] += 1
        task_loss_per_batch[task_id[i][0]] = task_losses[task_id[i][0]]

    return task_loss_per_batch, task_num_per_batch


def loss_func_mft(outputs, inputs, weighted_loss_mode):
    # task_id shape: [[1], [2], [4], [3], ..., [1]]
    labels, task_mask, task_id, loss_mask, weights = inputs['labels'], inputs['task_mask'], inputs['task_id'], inputs['loss_mask'], inputs['weights']
    weighted = weighted_loss_mode
    lm_logits = outputs["logits"]
    labels = labels.to(device=lm_logits.device)
    task_mask = task_mask.to(device=lm_logits.device)
    task_id = task_id.to(device=lm_logits.device)
    bsz, seq_len = labels.shape
    if loss_mask is None:
        ineffective_tokens_per_sample = (labels==-100).sum(dim=1)
        effective_tokens_per_sample = - (ineffective_tokens_per_sample - seq_len)
        effective_tokens = bsz * seq_len - ineffective_tokens_per_sample.sum()
        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
    else:
        loss_mask = loss_mask.to(device=lm_logits.device)
        effective_tokens_per_sample = torch.sum(loss_mask, dim=1, dtype=torch.int)
        effective_tokens = torch.sum(loss_mask).item()
        loss_fct = CrossEntropyLoss(reduction='none')
    if weighted_loss_mode.endswith('focalloss'):
        losses = loss_fct(lm_logits, labels)
    else:
        losses = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))  # [B * L, 1]
    # losses = losses.contiguous().view(bsz, -1)
    losses = losses.view(bsz, -1)
    token_losses = losses.clone().detach().float() if loss_mask is None else losses.clone().detach().float()  # [B, L]
    task_mask_trans = torch.transpose(task_mask, 0, 1)
    unique_weights = torch.unique(weights)
    unique_id = torch.unique(task_id)
    if weighted_loss_mode == "case3" or weighted_loss_mode == "case4":
        loss = 0.0
        for i, w in enumerate(unique_weights):
            row_idx = torch.squeeze(weights) == w.item()
            if weighted_loss_mode == "case3":
                if loss_mask is None:
                    loss += torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
                else:
                    loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
            elif weighted_loss_mode == "case4":
                if loss_mask is None:
                    loss += torch.mean(torch.sum(losses, dim=1)[row_idx] / effective_tokens_per_sample[row_idx])
                else:
                    loss += torch.mean(torch.sum(losses * loss_mask, dim=1)[row_idx] / torch.sum(loss_mask, dim=1)[row_idx])
        loss /= len(unique_weights)
    elif weighted_loss_mode == "case2":
        if loss_mask is None:
            loss = torch.mean(torch.sum(losses, dim=1) / effective_tokens_per_sample)
        else:
            loss = torch.mean(torch.sum(losses * loss_mask, dim=1) / torch.sum(loss_mask, dim=1))
    elif weighted_loss_mode == "case1":
        # flatten losses & loss_mask tensor
        if loss_mask is None:
            loss = torch.sum(losses.view(-1)) / effective_tokens
        else:
            loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.sum()

    # Fix a printing order.
    task_loss = torch.zeros(len(ID2TASK)).to(device=task_id.device)  # per task loss
    task_num = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    # unique_id = torch.unique(task_id)
    for i, w in enumerate(unique_id):
        row_idx = torch.squeeze(task_id) == w.item()
        if loss_mask is None:
            task_loss[w] = torch.sum(token_losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
            task_num[w] = len(effective_tokens_per_sample[row_idx])
        else:  # TODO:
            task_loss[w] = torch.sum((token_losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
            task_num[w] = len(torch.sum(loss_mask, dim=1)[row_idx])
    
    return loss, task_loss, task_num


def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )

    # convert to binary
    return mask < 0.5


def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    # attention_mask = get_attn_mask(
    #     seq_length=seq_length,
    #     device=data.device,
    # )
    attention_mask = torch.ones((batch_size, seq_length), device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data).clone()

    return attention_mask, position_ids


def prepare_gpt_input(batch, device):
    batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}

    if 'loss_mask' in batch and 'labels' not in batch:
    # if 'loss_mask' in batch:
        print_rank_0('loss mask in batch')
        input_ids = batch['input_ids'].long()
        batch['input_ids'] = input_ids[:, :-1].contiguous().to(device=device)
        batch['labels'] = input_ids[:, 1:].contiguous().to(device=device)
        batch['loss_mask'] = batch['loss_mask'].float()[:, 1:].contiguous()
    else:
        batch['input_ids'] = batch['input_ids'].long()
        batch['labels'] = batch['labels'].long()
        batch['loss_mask'] = None

    # Get the masks and position ids.
    batch['attention_mask'], batch['position_ids'] = get_ltor_masks_and_position_ids(data=batch['input_ids'])

    if self.args.weighted_loss_mode:
        weights = batch['weight'].float().to(device=device)  # [2, 4, 6, 3, ..., 2]
        # batch['loss_mask'] *= weights
    
    if 'task_id' in batch and batch['task_id'] is not None:  # task_id: bsz * 1, [[1], [2], [4], [3], ..., [1]]
        batch['task_mask'] = get_task_mask(batch['task_id']).to(device=device)  # bsz * task_num

    return batch


@dataclass
class DataCollatorForMFTDataset(object):
    def __init__(self, model_type, weighted_loss_mode, use_dynamic_padding):
        self.model_type = model_type
        self.weighted_loss_mode = weighted_loss_mode
        self.use_dynamic_padding = use_dynamic_padding

    # tokenizer: None

    def __call__(self, instances):
        input_ids, attention_mask, position_ids, labels, loss_mask, weights, task_id = tuple(
            [instance[key] if key in instance else None for instance in instances] for key in
            ("input_ids", "attention_mask", "position_ids", "labels", "loss_mask", "weight", "task_id"))
        # input_ids, loss_mask, weights, task_id = tuple(instances[key] for key in ("input_ids", "loss_mask", "weight", "task_id"))

        result_batch = {}
        '''
        outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                # labels=(batch['labels'], batch['loss_mask'], batch['task_mask']),
                # labels=(batch['labels'], batch['loss_mask']),
                position_ids=batch['position_ids'],
            )
        '''

        input_ids = torch.tensor(np.array(input_ids)).long()
        # input_ids = input_ids.long()
        if loss_mask[0] is None:
            result_batch['input_ids'] = input_ids.contiguous()
            labels = torch.tensor(np.array(labels)).long()
            result_batch['labels'] = labels.contiguous()
            result_batch['loss_mask'] = None
        else:
            loss_mask = torch.tensor(np.array(loss_mask))
            if self.use_dynamic_padding:
                last_one_pos = (loss_mask == 1).long().cumsum(dim=1).argmax(dim=1)
                max_pos = last_one_pos.max().item() + 1
            else:
                max_pos = loss_mask.shape[-1]
            result_batch['input_ids'] = input_ids[:, :max_pos-1].contiguous()  # [B, L + 1] -> [B, L]
            result_batch['labels'] = input_ids[:, 1:max_pos].contiguous()
            result_batch['loss_mask'] = loss_mask.float()[:, 1:max_pos].contiguous()

        if self.weighted_loss_mode and weights is not None:
            weights = torch.tensor(np.array(weights))
            result_batch['weights'] = weights
            # if result_batch['loss_mask'] is not None:
            #     result_batch['loss_mask'] *= weights
        
        # Get the masks and position ids.
        result_batch['attention_mask'], result_batch['position_ids'] = get_ltor_masks_and_position_ids(data=result_batch['input_ids'])

        if task_id is not None:
            task_id = torch.tensor(np.array(task_id))
            result_batch['task_mask'] = get_task_mask(task_id) # bsz * task_num
            result_batch['task_id'] = task_id

        return result_batch
