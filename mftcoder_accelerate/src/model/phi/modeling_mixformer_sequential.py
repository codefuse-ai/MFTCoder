# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# BSD 3-Clause License
# 
# Copyright (c) 2022, Tri Dao, trid@cs.stanford.edu.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import math
import copy
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from einops import rearrange
from transformers.activations import ACT2FN
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_mixformer_sequential import MixFormerSequentialConfig

import xformers.ops

@dataclass
class InferenceParams:
    """Inference parameters passed to model to efficiently calculate
    and store context during inference.
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py.
    Args:
        max_sequence_len: Maximum sequence length.
        max_batch_size: Maximum batch size.
        sequence_len_offset: Sequence length offset.
        batch_size_offset: Batch size offset.
        key_value_memory_dict: Key value memory dictionary.
        fused_ft_kernel: Whether to use fused kernel for fast inference.
        lengths_per_sample: Lengths per sample.
    """

    max_sequence_len: int = field(metadata={"help": "Maximum sequence length."})

    max_batch_size: int = field(metadata={"help": "Maximum batch size."})

    sequence_len_offset: int = field(default=0, metadata={"help": "Sequence length offset."})

    batch_size_offset: int = field(default=0, metadata={"help": "Batch size offset."})

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Key value memory dictionary."}
    )

    fused_ft_kernel: bool = field(default=False, metadata={"help": "Whether to use fused kernel for fast inference."})

    lengths_per_sample: torch.Tensor = field(default=None, metadata={"help": "Lengths per sample."})


class Embedding(nn.Module):
    """Token embedding with dropout."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        return hidden_states


class RotaryEmbedding(nn.Module):
    """Rotary embeddings.
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py.
    
    """

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        scale_base: Optional[float] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if scale_base is not None:
            raise NotImplementedError

        # Generate and save the inverse frequency buffer (non-trainable)
        self.dim = dim
        self.base = base
        self.scale_base = scale_base
        self.device = device

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x: torch.FloatTensor, seqlen_offset: int = 0) -> None:
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        seqlen = x.shape[1] + seqlen_offset

        # Re-generate the inverse frequency buffer if it's not fp32
        # (for instance if model.half() was called)
        if self.inv_freq.dtype != "torch.float32":
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32) / self.dim)
            )

        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=torch.float32)

            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device, dtype=torch.float32))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")

                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def _apply_rotary_emb_qkv(
        self,
        qkv: torch.FloatTensor,
        sin: torch.FloatTensor,
        cos: torch.FloatTensor,
        sin_k: Optional[torch.FloatTensor] = None,
        cos_k: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        _, seqlen, three, _, headdim = qkv.shape
        assert three == 3

        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen

        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)

        q_rot = qkv[:, :, 0, :, :rotary_dim]
        q_pass = qkv[:, :, 0, :, rotary_dim:]

        k_rot = qkv[:, :, 1, :, :rotary_dim]
        k_pass = qkv[:, :, 1, :, rotary_dim:]

        # Splits the queries and keys in half
        q1, q2 = q_rot.chunk(2, dim=-1)
        k1, k2 = k_rot.chunk(2, dim=-1)
        c, s = rearrange(cos[:seqlen], "s d -> s 1 d"), rearrange(sin[:seqlen], "s d -> s 1 d")

        # Casts to fp32 are necessary to prevent fp16 overflow issues
        q1, q2, k1, k2, c, s = [t.to(dtype=torch.float32) for t in [q1, q2, k1, k2, c, s]]

        # Computes the new keys and queries, recasting to original dtype
        q_rot = torch.cat([q1 * c - q2 * s, q1 * s + q2 * c], axis=-1).to(qkv.dtype)
        k_rot = torch.cat([k1 * c - k2 * s, k1 * s + k2 * c], axis=-1).to(qkv.dtype)

        return torch.cat(
            [
                torch.cat([q_rot, q_pass], axis=-1).unsqueeze(2),
                torch.cat([k_rot, k_pass], axis=-1).unsqueeze(2),
                qkv[:, :, 2:3, :, :],
            ],
            axis=2,
        )

    def forward(self, qkv: torch.Tensor, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        # `qkv` is of shape (batch, seqlen, 3, nheads, headdim)
        self._update_cos_sin_cache(qkv, seqlen_offset)
        return self._apply_rotary_emb_qkv(qkv, self._sin_cached[seqlen_offset:], self._cos_cached[seqlen_offset:])


class MLP(nn.Module):
    """Multi-Layer Perceptron.
    Reference:
        Attention Is All You Need.
        https://arxiv.org/pdf/1706.03762.pdf.
    """

    def __init__(self, config: PretrainedConfig, n_inner: Optional[int] = None, act_fn: Optional[str] = None) -> None:
        super().__init__()

        act_fn = config.activation_function if act_fn is None else act_fn
        assert act_fn in ACT2FN.keys(), f"`act_fn` must be one of: {ACT2FN.keys()}."

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.fc1 = nn.Linear(config.n_embd, n_inner)
        self.fc2 = nn.Linear(n_inner, config.n_embd)
        self.act = ACT2FN[act_fn]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SelfAttention(nn.Module):
    """Self-attention layer (compatible with PyTorch).
    
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py.
    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        qkv: torch.FloatTensor,
        causal: bool = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        causal = self.causal if causal is None else causal
        batch_size, seq_len = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)

        # flash attention
        output = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=xformers.ops.LowerTriangularMask(),
                                                         op=xformers.ops.MemoryEfficientAttentionFlashAttentionOp)

        # softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        # scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        # if attention_mask is not None:
        #     padding_mask = torch.full((batch_size, seq_len), -10000.0, dtype=scores.dtype, device=scores.device)
        #     padding_mask.masked_fill_(attention_mask, 0.0)

        #     scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        # if causal:
        #     causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
        #     scores = scores + causal_mask.to(dtype=scores.dtype)

        # attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        # attention = self.drop(attention)

        # output = torch.einsum("bhts,bshd->bthd", attention, v)

        return output


class CrossAttention(nn.Module):
    """Cross-attention layer (compatible with PyTorch).
    
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py.
    
    """

    def __init__(
        self,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        causal: bool = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        causal = self.causal if causal is None else causal
        batch_size, seq_len_q = q.shape[0], q.shape[1]
        assert kv.shape[0] == batch_size and kv.shape[3] == q.shape[2] and kv.shape[4] == q.shape[3]

        seq_len_k = kv.shape[1]
        k, v = kv.unbind(dim=2)

        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if attention_mask is not None:
            padding_mask = torch.full((batch_size, seq_len_k), -10000.0, dtype=scores.dtype, device=scores.device)
            padding_mask.masked_fill_(attention_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            causal_mask = torch.triu(torch.full((seq_len_q, seq_len_k), -10000.0, device=scores.device), 1)
            scores = scores + causal_mask.to(dtype=scores.dtype)

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.drop(attention)

        output = torch.einsum("bhts,bshd->bthd", attention, v)

        return output


def find_mha_dims(
    config: PretrainedConfig, n_head: Optional[int] = None, head_dim: Optional[int] = None
) -> Tuple[int, int]:
    """Validate and return the number of heads and head dimension for multi-head attention.
    Args:
        config: Model configuration.
        n_head: Number of heads.
        head_dim: Head dimension.
    Returns:
        Number of heads and head dimension.
    """

    assert all(
        hasattr(config, attr) for attr in ["n_embd", "n_head"]
    ), "`config` must have `n_embd` and `n_head` attributes."

    if head_dim is None:
        assert (
            config.n_embd % config.n_head == 0
        ), f"Hidden size ({config.n_embd}) must be divisible by the number of heads ({config.n_head})."

    if n_head is None and head_dim is None:
        head_dim = config.n_embd // config.n_head
        n_head = config.n_head
    elif n_head is None or head_dim is None:
        raise ValueError("`n_head` and `head_dim` must be both specified or `None`.")

    return n_head, head_dim


def update_kv_cache(kv: torch.FloatTensor, inference_params: InferenceParams, layer_idx: int) -> torch.FloatTensor:
    """Update the key-value cache for inference.
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py.
    Args:
        kv: Key-value tensor.
        inference_params: Inference parameters.
        layer_idx: Layer index.
    Returns:
        Updated key-value tensor.
    """

    num_heads, head_dim = kv.shape[-2:]

    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_sequence_len,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        if not inference_params.fused_ft_kernel:
            kv_cache = inference_params.key_value_memory_dict[layer_idx]
        else:
            k_cache, v_cache = inference_params.key_value_memory_dict[layer_idx]
            kv_cache = None

    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    assert batch_end <= (kv_cache.shape[0] if kv_cache is not None else v_cache.shape[0])

    sequence_start = inference_params.sequence_len_offset
    sequence_end = sequence_start + kv.shape[1]
    assert sequence_end <= (kv_cache.shape[1] if kv_cache is not None else v_cache.shape[2])

    if not inference_params.fused_ft_kernel:
        assert kv_cache is not None

        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        kv = kv_cache[batch_start:batch_end, :sequence_end, ...]

        return kv

    assert inference_params.sequence_len_offset == 0
    assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]

    packsize = 4 if kv.dtype == torch.float32 else 8

    if kv_cache is not None:
        kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
        k_cache = rearrange(kv_cache[:, :, 0], "b s h (d packsize) -> b h d s packsize", packsize=packsize).contiguous()
        v_cache = rearrange(kv_cache[:, :, 1], "b s h d -> b h s d").contiguous()
        inference_params.key_value_memory_dict[layer_idx] = (k_cache, v_cache)
    else:
        k_cache[batch_start:batch_end, :, :, :sequence_end, :] = rearrange(
            kv[:, :, 0], "b s h (d packsize) -> b h d s packsize", packsize=packsize
        )
        v_cache[batch_start:batch_end, :, :sequence_end, :] = rearrange(kv[:, :, 1], "b s h d -> b h s d")

    return kv


class MHA(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        rotary_dim: Optional[int] = None,
        rotary_emb_scale_base: Optional[float] = None,
        n_head: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        causal: bool = True,
        softmax_scale: Optional[float] = None,
        dropout: float = 0.0,
        layer_idx: Optional[int] = None,
        return_residual: bool = False,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        # Rotary embedding
        self.rotary_emb_dim = rotary_dim if rotary_dim is not None else getattr(config, "rotary_dim", 0)
        if self.rotary_emb_dim > 0:
            rotary_kwargs = {"device": device}
            if rotary_emb_scale_base is not None and rotary_emb_scale_base > 0.0:
                rotary_kwargs["scale_base"] = rotary_emb_scale_base
            self.rotary_emb = RotaryEmbedding(self.rotary_emb_dim, **rotary_kwargs)
        
        # MLP
        self.n_head, self.head_dim = find_mha_dims(config, n_head, head_dim)
        op_size = self.n_head * self.head_dim
        hidden_size = config.n_embd

        self.Wqkv = nn.Linear(hidden_size, 3 * op_size, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(op_size, hidden_size, bias=bias, device=device, dtype=dtype)

        # Attention
        self.inner_attn = SelfAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)
        self.inner_cross_attn = CrossAttention(causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout)

        self.layer_idx = layer_idx
        self.return_residual = return_residual
        # self.checkpointing = checkpointing
        self.checkpointing = True

    def forward(
        self,
        x: torch.FloatTensor,
        past_key_values: Optional[InferenceParams] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)

        seqlen_offset = past_key_values.sequence_len_offset if past_key_values is not None else 0
        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset)

        if past_key_values is not None:
            kv = update_kv_cache(qkv[:, :, 1:], past_key_values, self.layer_idx)

        if attention_mask is not None:
            attention_mask = attention_mask[0] if isinstance(attention_mask, tuple) else attention_mask
            attention_mask = attention_mask.bool().to(qkv.device)

        attention_kwargs = {"attention_mask": attention_mask}

        if past_key_values is None or seqlen_offset == 0:
            if self.checkpointing:
                # attn_output = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **attention_kwargs)
                attn_output = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv)
            else:
                attn_output = self.inner_attn(qkv, **attention_kwargs)
        else:
            q = qkv[:, :, 0]
            causal = None if past_key_values.sequence_len_offset == 0 else False
            attn_output = self.inner_cross_attn(q, kv, causal=causal, **attention_kwargs)

        output = rearrange(attn_output, "... h d -> ... (h d)")
        output = self.out_proj(output)

        return output if not self.return_residual else (output, x)


class ParallelBlock(nn.Module):
    """Parallel block.
    This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        block_idx: Optional[int] = None,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.block_idx = block_idx

        self.mixer = MHA(config, layer_idx=block_idx, checkpointing=checkpointing)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.ln(hidden_states)

        attn_outputs = self.mixer(hidden_states, past_key_values=past_key_values, attention_mask=attention_mask)
        if isinstance(attn_outputs, tuple):
            attn_outputs = attn_outputs[0]

        attn_outputs = self.resid_dropout(attn_outputs)
        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))

        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states


class CausalLMHead(nn.Module):
    """Causal Language Modeling head.
    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.ln(hidden_states)
        logits = self.linear(hidden_states).to(torch.float32)

        return logits


class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss.
    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.
    """

    def __init__(self, shift_labels: bool = True) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss


class MixFormerSequentialPreTrainedModel(PreTrainedModel):
    """MixFormer (sequential for DeepSpeed) pre-trained model."""

    config_class = MixFormerSequentialConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if attention_mask is not None and torch.any(~attention_mask.bool()):
            total_seq_len = torch.sum(attention_mask, dim=1)
            max_seq_len = torch.max(total_seq_len)

            total_seq_len = torch.cat((torch.tensor([0], device=attention_mask.device), total_seq_len)).unsqueeze(1)
            cumulative_seq_len = torch.cumsum(total_seq_len, dim=0).squeeze(1).to(torch.int32)
            attention_mask = (attention_mask.bool(), cumulative_seq_len, max_seq_len.item())
        else:
            attention_mask = None

        if past_key_values is None or not (isinstance(past_key_values, InferenceParams)):
            past_key_values = InferenceParams(
                max_batch_size=input_ids.shape[0],
                max_sequence_len=self.config.n_positions,
                sequence_len_offset=0,
                batch_size_offset=0,
                fused_ft_kernel=False,
                key_value_memory_dict={},
            )
        else:
            # Assume that `past_key_values` has cached all tokens up to the last token in `input_ids`
            past_key_values.sequence_len_offset = len(input_ids[0]) - 1
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
    
    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2PreTrainedModel._set_gradient_checkpointing with GPT2->GPTBigCode
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PreTrainedModel):
            module.gradient_checkpointing = value


class MixFormerSequentialForCausalLM(MixFormerSequentialPreTrainedModel):
    """MixFormer (sequential for DeepSpeed) for Causal Language Modeling."""

    _keys_to_ignore_on_load_missing = [""]
    _keys_to_ignore_on_load_unexpected = [r"layers\.\d+\.mlp.(fc_in|fc_out)\.(weight|bias)"]
    _no_split_modules = ["ParallelBlock"]

    def __init__(self, config: MixFormerSequentialConfig) -> None:
        super().__init__(config)

        modules = [Embedding(config)]
        modules += [ParallelBlock(config, block_idx=i) for i in range(config.n_layer)]
        modules.append(CausalLMHead(config))

        self.layers = nn.Sequential(*modules)
        self.loss = CausalLMLoss()

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.layers[0].wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding) -> None:
        self.layers[0].wte = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.layers[-1].linear

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.layers[-1].linear = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[torch.FloatTensor, InferenceParams]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is not None and self.training:
            print("`attention_mask` is not supported during training. Using it might lead to unexpected results.")

        if past_key_values is None and attention_mask is None:
            lm_logits = self.layers(input_ids)
        else:
            hidden_layer = self.layers[0](input_ids)
            for module in self.layers[1:-1]:
                hidden_layer = module(hidden_layer, past_key_values=past_key_values, attention_mask=attention_mask)
            lm_logits = self.layers[-1](hidden_layer)

        loss = None
        if labels is not None:
            loss = self.loss(lm_logits, labels)

        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=past_key_values)
