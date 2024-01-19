import sys
import torch
from utils.common_utils import print_rank_0, TASK2ID, ID2TASK
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
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


def loss_func_mft(outputs, labels, task_mask, task_id, weighted_loss_mode, loss_mask=None, task_weights=None):
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
    if task_weights is None:
        task_weights = torch.ones(len(ID2TASK)).to(device=lm_logits.device) / len(ID2TASK)

    bsz, seq_len = labels.shape
    # loss_mask = None
    if loss_mask is None:
        ineffective_tokens_per_sample = (labels == -100).sum(dim=1)
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
    unique_id = torch.unique(task_id)
    if weighted_loss_mode == "case3" or weighted_loss_mode == "case4" or weighted_loss_mode == "selfpaced":
        loss = 0.0
        weights_sum = 0.0
        for i, w in enumerate(unique_id):
            row_idx = torch.squeeze(task_id) == w.item()
            task_weight = float(task_weights[w.item()])
            weights_sum += task_weight
            if weighted_loss_mode == "case3" or weighted_loss_mode == "selfpaced":
                if loss_mask is None:
                    loss += torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx]) * task_weight
                else:
                    loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :]) * task_weight
            elif weighted_loss_mode == "case4":
                if loss_mask is None:
                    loss += torch.mean(torch.sum(losses, dim=1)[row_idx] / effective_tokens_per_sample[row_idx]) * task_weight
                else:
                    loss += torch.mean(torch.sum(losses * loss_mask, dim=1)[row_idx] / torch.sum(loss_mask, dim=1)[row_idx]) * task_weight

        # loss /= len(unique_id)
        loss /= weights_sum

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


class MFTLossStatus:
    def __init__(self):
        super(MFTLossStatus, self).__init__()


class SelfpacedStatus(MFTLossStatus):
    def __init__(self,
                 selfpaced_scale_factor=50,
                 selfpaced_interval=1,
                 selfpaced_history_length=100,
                 selfpaced_sample_valid_num=1,
                 valid_dataloader=None
                 ):

        super(SelfpacedStatus, self).__init__()
        self.selfpaced_scale_factor = selfpaced_scale_factor
        self.selfpaced_interval = selfpaced_interval
        self.selfpaced_history_length = selfpaced_history_length
        self.selfpaced_sample_valid_num = selfpaced_sample_valid_num
        self.valid_dataloader = valid_dataloader
        self.valid_dataloader_length = len(valid_dataloader)
        self.valid_iterator = iter(valid_dataloader)
        self.valid_task_loss_accumulated = torch.zeros(len(ID2TASK))
        self.history_task_valid_loss = torch.zeros((selfpaced_history_length, len(ID2TASK)))
        self.log_per_task_weight = torch.zeros(len(ID2TASK))

    def selfpaced_evaluate(self, model, v_batch, per_task_weight=None, selfpaced_status=None):
        model.eval()
        with torch.no_grad():
            valid_outputs = model(
                input_ids=v_batch['input_ids'],
                attention_mask=v_batch['attention_mask'],
                position_ids=v_batch['position_ids']
            )

            _, valid_task_loss, valid_task_num = loss_func_mft(
                outputs=valid_outputs,
                labels=v_batch['labels'],
                task_mask=v_batch['task_mask'],
                task_id=v_batch['task_id'],
                weighted_loss_mode='selfpaced',
                loss_mask=v_batch['loss_mask'],
                task_weights=None
            )

            torch.distributed.all_reduce(valid_task_loss, op=torch.distributed.ReduceOp.SUM)
            valid_task_loss /= torch.distributed.get_world_size()
        model.train()
        return valid_task_loss
    
    def compute_per_task_weight(self, completed_steps=None):
        task_slope_fitting = torch.ones(len(ID2TASK))
        history_steps = torch.arange(completed_steps - self.selfpaced_history_length, completed_steps, 1)  # DEBUG: step < 0
        transpose_history_task_valid_loss = self.history_task_valid_loss.transpose(0, 1)
        for i in range(len(ID2TASK)):
            per_history_task_valid_loss = transpose_history_task_valid_loss[i]
            task_slope_fitting[i] = self.fit_window_point(history_steps, per_history_task_valid_loss,
                                                          history=self.selfpaced_history_length, type='slope')
        slope_sum_abs = torch.sum(torch.abs(task_slope_fitting))

        if slope_sum_abs == 0:
            per_task_weight = torch.ones(len(ID2TASK)) / len(ID2TASK)
        else:
            # print_rank_0(f"[step={completed_steps}][slope sum abs={slope_sum_abs}]")
            normalize_slope = len(ID2TASK) * task_slope_fitting / slope_sum_abs
            print_rank_0(f'normalize_slope: {normalize_slope}')
            score = F.softmax(normalize_slope, dim=-1) * (-1 * normalize_slope)
            print_rank_0(f'score: {score}')
            per_task_weight = F.softmax(self.selfpaced_scale_factor * score, dim=-1)
            print_rank_0(f'per_task_weight: {per_task_weight}')
        
        return per_task_weight
    
    def fit_window_point(self, x, y, history=10, type='slope'):

        nonzero_index = torch.squeeze(torch.nonzero(y), dim=1)
        y = torch.index_select(y, 0, nonzero_index)
        x = torch.index_select(x, 0, nonzero_index)

        ws = torch.flip(1 ** torch.arange(len(y)), dims=[0])
        ws = ws.float()

        if len(y) >= 2:
            if type == 'slope':
                X = torch.stack((x, torch.ones_like(x))).T
                X = X.float()
            else:
                X = torch.stack((x ** 2, x, torch.ones_like(x))).T
            w = torch.linalg.solve(X.T @ (ws[:, None] * X), X.T @ (ws * y))

            result = w[0]
        else:
            result = 0.0

        return result
    
    def sample_valid_batch(self, model, completed_steps):
        self.valid_task_loss_accumulated = torch.zeros(len(ID2TASK))
        for i in range(self.selfpaced_sample_valid_num):
            if (self.selfpaced_sample_valid_num * completed_steps // self.selfpaced_interval + i) % self.valid_dataloader_length == 0:
                self.valid_iterator = iter(self.valid_dataloader)
            v_batch = next(self.valid_iterator)
            valid_task_loss = self.selfpaced_evaluate(model, v_batch)
            self.valid_task_loss_accumulated += valid_task_loss.detach().cpu()
        self.valid_task_loss_accumulated /= self.selfpaced_sample_valid_num
        self.history_task_valid_loss = torch.cat((self.history_task_valid_loss, torch.unsqueeze(self.valid_task_loss_accumulated, dim=0)))
        if len(self.history_task_valid_loss) > self.selfpaced_history_length:
            self.history_task_valid_loss = self.history_task_valid_loss[len(self.history_task_valid_loss) - self.selfpaced_history_length:]
