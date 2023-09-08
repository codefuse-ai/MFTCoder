import sys
sys.path.append("..")
import torch
from utils.common_utils import print_rank_0, TASK2ID, ID2TASK
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
import numpy as np


def get_task_mask(task_id):
    task_num = len(TASK2ID)
    task_mask = torch.zeros(task_id.shape[0], task_num)
    task_mask[torch.arange(task_id.size(0)).unsqueeze(1), task_id] = 1
    
    return task_mask


def get_task_loss(task_losses, task_id):  # TODO
    # fix task order
    task_loss_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    # count task samples
    task_num_per_batch = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    for i in range(len(task_id)):
        task_num_per_batch[task_id[i][0]] += 1
        task_loss_per_batch[task_id[i][0]] = task_losses[task_id[i][0]]

    return task_loss_per_batch, task_num_per_batch

  
def loss_func_mft(outputs, labels, task_mask, task_id, weighted_loss_mode, loss_mask=None):
    """
    loss function for MFT loss
    :param outputs:
    :param labels:
    :param task_mask:
    :param task_id:
    :param weighted_loss_mode:
    :param loss_mask:
    :return:
    """
    # task_id shape: [[1], [2], [4], [3], ..., [1]]
    weighted = weighted_loss_mode
    lm_logits = outputs["logits"]
    labels = labels.to(device=lm_logits.device)
    task_mask = task_mask.to(device=lm_logits.device)
    task_id = task_id.to(device=lm_logits.device)
    shift_logits = lm_logits.contiguous()
    labels = labels.contiguous()

    bsz, seq_len = labels.shape
    # loss_mask = None
    if loss_mask is None:
        ineffective_tokens_per_sample = (labels==-100).sum(dim=1)
        effective_tokens_per_sample = - (ineffective_tokens_per_sample - seq_len)
        effective_tokens = bsz * seq_len - ineffective_tokens_per_sample.sum()
        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
    else:
        loss_mask = loss_mask.to(device=lm_logits.device)
        loss_fct = CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))  # [B * L, 1]
    losses = losses.contiguous().view(bsz, -1)
    token_losses = losses.clone().detach().float() if loss_mask is None else losses.clone().detach().float() * loss_mask  # [B, L]
    task_mask_trans = torch.transpose(task_mask, 0, 1)
    if weighted_loss_mode == "case3" or weighted_loss_mode == "case4":
        unique_id = torch.unique(task_id)
        loss = 0.0

        for i, w in enumerate(unique_id):
            row_idx = torch.squeeze(task_id) == w.item()
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

        loss /= len(unique_id)

    elif weighted_loss_mode == "case2":
        if loss_mask is None:
            loss = torch.mean(torch.sum(losses, dim=1) / effective_tokens_per_sample)
        else:
            loss = torch.mean(torch.sum(losses * loss_mask, dim=1) / torch.sum(loss_mask, dim=1))
    elif weighted_loss_mode == "case1":
        # flatten losses & loss_mask tensor
        if loss_mask is None:
            losses = losses.view(-1)
            loss = torch.sum(losses) / effective_tokens
        else:
            loss_mask = loss_mask.view(-1)
            losses = losses.view(-1)
            loss = torch.sum(losses * loss_mask) / loss_mask.sum()

    # fix task order
    task_loss = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    task_num = torch.zeros(len(ID2TASK)).to(device=task_id.device)
    for i, w in enumerate(unique_id):
        row_idx = torch.squeeze(task_id) == w.item()
        if loss_mask is None:
            task_loss[w] = torch.sum(token_losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx])
            task_num[w] = len(effective_tokens_per_sample[row_idx])
        else:
            task_loss[w] = torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :])
    
    return loss, task_loss, task_num
