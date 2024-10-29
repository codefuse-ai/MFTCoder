import sys
import torch
from utils.common_utils import print_rank_0, TASK2ID, ID2TASK
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Tuple, Union


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
        effective_tokens_per_sample = -(ineffective_tokens_per_sample - seq_len)
        effective_tokens = bsz * seq_len - ineffective_tokens_per_sample.sum()
        loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
    else:
        loss_mask = loss_mask.to(device=lm_logits.device)
        loss_fct = CrossEntropyLoss(reduction="none")
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))  # [B * L, 1]
    losses = losses.contiguous().view(bsz, -1)
    token_losses = (
        losses.clone().detach().float() if loss_mask is None else losses.clone().detach().float() * loss_mask
    )  # [B, L]
    task_mask_trans = torch.transpose(task_mask, 0, 1)
    unique_id = torch.unique(task_id)
    if weighted_loss_mode == "case3" or weighted_loss_mode == "case4" or weighted_loss_mode == "coba":
        loss = 0.0
        weights_sum = 0.0
        for i, w in enumerate(unique_id):
            row_idx = torch.squeeze(task_id) == w.item()
            task_weight = float(task_weights[w.item()])
            weights_sum += task_weight
            if weighted_loss_mode == "case3" or weighted_loss_mode == "coba":
                if loss_mask is None:
                    loss += (
                        torch.sum(losses[row_idx, :]) / torch.sum(effective_tokens_per_sample[row_idx]) * task_weight
                    )
                else:
                    loss += torch.sum((losses * loss_mask)[row_idx, :]) / torch.sum(loss_mask[row_idx, :]) * task_weight
            elif weighted_loss_mode == "case4":
                if loss_mask is None:
                    loss += (
                        torch.mean(torch.sum(losses, dim=1)[row_idx] / effective_tokens_per_sample[row_idx])
                        * task_weight
                    )
                else:
                    loss += (
                        torch.mean(torch.sum(losses * loss_mask, dim=1)[row_idx] / torch.sum(loss_mask, dim=1)[row_idx])
                        * task_weight
                    )

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
            # losses = losses.view(-1)
            loss = torch.sum(losses.view(-1)) / effective_tokens
        else:
            # loss_mask = loss_mask.view(-1)
            # losses = losses.view(-1)
            loss = torch.sum(losses.view(-1) * loss_mask.view(-1)) / loss_mask.view(-1).sum()

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


def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MFTLossStatus:
    def __init__(self):
        super(MFTLossStatus, self).__init__()


class CoBaStatus(MFTLossStatus):
    def __init__(
        self,
        coba_warmup_steps=100,
        coba_history_length=200,
        coba_tau=5,
        coba_update_interval=1,
        coba_sample_valid_num=1,
        valid_dataloader=None,
    ):

        super(CoBaStatus, self).__init__()
        self.coba_warmup_steps = coba_warmup_steps
        self.coba_history_length = coba_history_length
        self.coba_tau = coba_tau
        self.coba_update_interval = coba_update_interval
        self.coba_sample_valid_num = coba_sample_valid_num
        self.valid_dataloader = valid_dataloader
        self.valid_dataloader_length = len(valid_dataloader)
        self.valid_iterator = iter(valid_dataloader)
        self.valid_task_loss_accumulated = torch.zeros(len(ID2TASK))
        self.history_task_valid_loss = None
        self.per_task_slope_list = None
        self.total_slope_list = None
        self.minimum_weight = 1 / (len(ID2TASK) * 10)
        self.valid_task_loss_begining = torch.ones(len(ID2TASK), dtype=torch.float64)
        self.log_per_task_weight = torch.zeros(len(ID2TASK))

    def coba_evaluate(self, model, v_batch, per_task_weight=None, coba_status=None):
        model.eval()
        with torch.no_grad():
            valid_outputs = model(
                input_ids=v_batch["input_ids"],
                attention_mask=v_batch["attention_mask"],
                position_ids=v_batch["position_ids"],
            )

            _, valid_task_loss, valid_task_num = loss_func_mft(
                outputs=valid_outputs,
                labels=v_batch["labels"],
                task_mask=v_batch["task_mask"],
                task_id=v_batch["task_id"],
                weighted_loss_mode="coba",
                loss_mask=v_batch["loss_mask"],
                task_weights=None,
            )

            task_exist = (valid_task_loss != 0.0).float()
            torch.distributed.all_reduce(valid_task_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(task_exist, op=torch.distributed.ReduceOp.SUM)
            valid_task_loss /= task_exist.clamp_(1.0)
            valid_task_loss /= self.valid_task_loss_begining
        model.train()
        return valid_task_loss

    def compute_per_task_weight(self, completed_steps=None):
        task_num = len(ID2TASK)
        task_slope_fitting = torch.ones(task_num, dtype=torch.float64)
        start_step = max(0, completed_steps // self.coba_update_interval - self.coba_history_length)
        history_steps = torch.arange(start_step, completed_steps, 1)
        for i in range(task_num):
            per_task_history_valid_loss = self.history_task_valid_loss[i][-len(history_steps):]
            task_slope_fitting[i] = self.fit_window_slope(
                history_steps, per_task_history_valid_loss, type="slope"
            )
        history_total_valid_loss, index = torch.max(self.history_task_valid_loss[:, -len(history_steps):], dim=0)
        total_slope_fitting = self.fit_window_slope(
            history_steps, history_total_valid_loss, type="slope"
        )
        if completed_steps == self.coba_warmup_steps:
            self.per_task_slope_list = task_slope_fitting.unsqueeze(1)
            self.total_slope_list = total_slope_fitting.unsqueeze(0)
        else:
            self.per_task_slope_list = torch.cat((self.per_task_slope_list, task_slope_fitting.unsqueeze(1)), dim=-1)
            self.total_slope_list =  torch.cat((self.total_slope_list, total_slope_fitting.unsqueeze(0)), dim=0)
        
        # Relative Convergence Score
        normalize_task_slope = task_num * task_slope_fitting / task_slope_fitting.abs().sum()
        rcs = F.softmax(normalize_task_slope, dim=-1)
        
        # Absolute Convergence Score
        history_per_task_slope_list = self.per_task_slope_list[:, start_step:]
        reverse_normailize_iter_slope = -len(history_per_task_slope_list[0]) * history_per_task_slope_list \
                                        / history_per_task_slope_list.abs().sum(dim=-1, keepdim=True)

        flatten_rn_iter_slope = reverse_normailize_iter_slope.T.reshape(-1)
        current_step_rn_slope = flatten_rn_iter_slope[-task_num:]
        acs = F.softmax(current_step_rn_slope, dim=-1)

        # Divergence Factor
        normalize_total_iter_slope = - len(self.total_slope_list) * self.total_slope_list \
                                     / self.total_slope_list.abs().sum()
        divergence_factor = F.softmax(normalize_total_iter_slope * self.coba_tau, dim=-1)[-1] \
                          * len(self.total_slope_list)

        weight_logits = divergence_factor * rcs + (1 - divergence_factor) * acs
        per_task_weight = F.softmax(weight_logits * task_num, dim=-1)

        if len((per_task_weight < self.minimum_weight).nonzero().squeeze(0)) > 0:
            per_task_weight = per_task_weight * (1 - self.minimum_weight * task_num)
            per_task_weight += self.minimum_weight

        return per_task_weight
    
    def fit_window_slope(self, x, y, type="slope"):

        y = y[y != 0]
        x = x[:len(y)]
        
        nonzero_index = torch.squeeze(torch.nonzero(y), dim=1)
        y = torch.index_select(y, 0, nonzero_index)
        x = torch.index_select(x, 0, nonzero_index)

        ws = torch.flip(1 ** torch.arange(len(y)), dims=[0])
        ws = ws.double()

        if len(y) >= 2:
            if type == "slope":
                X = torch.stack((x, torch.ones_like(x, dtype=torch.float64))).T
                X = X.double()
            else:
                X = torch.stack((x ** 2, x, torch.ones_like(x, dtype=torch.float64))).T

            # implementation for numpy
            # X_np = X.T @ (ws[:, None] * X)
            # Y_np = X.T @ (ws * y)
            # w = torch.from_numpy(np.linalg.solve(X_np.numpy(), Y_np.numpy()))

            # implementation for torch
            w = torch.linalg.solve(X.T @ (ws[:, None] * X), X.T @ (ws * y))

            result = w[0]
        else:
            result = 0.0

        return result

    def sample_valid_batch(self, model, completed_steps):
        self.valid_task_loss_accumulated = torch.zeros(len(ID2TASK), dtype=torch.float64)
        for i in range(self.coba_sample_valid_num):
            if (
                self.coba_sample_valid_num * completed_steps // self.coba_update_interval + i
            ) % self.valid_dataloader_length == 0:
                self.valid_iterator = iter(self.valid_dataloader)
                v_batch = next(self.valid_iterator)
            else:
                v_batch = next(self.valid_iterator)
            valid_task_loss = self.coba_evaluate(model, v_batch)
            self.valid_task_loss_accumulated += valid_task_loss.detach().cpu().double()

        self.valid_task_loss_accumulated /= self.coba_sample_valid_num
        if self.history_task_valid_loss is None and completed_steps >= 1:
            self.history_task_valid_loss = self.valid_task_loss_accumulated.unsqueeze(1)
        elif self.history_task_valid_loss is not None:
            self.history_task_valid_loss = torch.cat(
                (self.history_task_valid_loss, self.valid_task_loss_accumulated.unsqueeze(1)), dim=-1
            )
