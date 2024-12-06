"""An adaptation of ESPNET VALL-E
Originally by Jinchuan Tian 

https://github.com/espnet/espnet

Authors
 * Artem Ploujnikov 2024
"""

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Implementation of Vall-E: https://arxiv.org/abs/2301.02111

import logging
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from speechbrain.dataio.dataio import length_to_mask

from torch import Tensor
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class SpeechLMInferenceOptions:
    device: str = None
    search_algo: str = "sampling"
    nbest: int = 1
    sampling_temperature: float = 1.0
    top_k: int = 20
    maxlenratio: float = 0.0
    minlenratio: float = 0.0
    eos: int = 5
    start: int = 1
    masks: torch.Tensor = None
    nq: int = None
    allow_invalid: bool = True


class ValleLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        nq: int,
        share_emb: bool = True,
        att_unit: int = 256,
        head: int = 2,
        ar_layer: int = 4,
        nar_layer: int = 4,
        n_ctx: int = 3000,
        residual: bool = True
    ):
        """Initialize Vall-E model

        Args:
            vocab_size (int): Dimention of vocabulary.
            nq (int): Number of codes for each token / frame, usually for speech codec.
            share_emb (bool): If true, share the embedding and lm_head weight.
            att_unit (int): Dimention of Transformer attention.
            head (int): Number of heads in Transformer attention.
            ar_layer (int): Number of layers in AR Transformer.
            nar_layer (int): Number of layers in NAR Transformer.
            n_ctx (int): maximum context length of AR & NAR Transformer.
        """
        super(ValleLM, self).__init__()

        self.emb = torch.nn.Embedding(vocab_size, att_unit)
        self.lm_head = torch.nn.Linear(att_unit, vocab_size, bias=False)
        if share_emb:
            self.lm_head.weight = self.emb.weight

        self.ar_decoder = TransformerDecoder(
            n_ctx=n_ctx, n_state=att_unit, n_head=head, n_layer=ar_layer, causal=True
        )

        self.nar_decoder = ValleNARDecoder(
            n_level=nq - 1,
            n_ctx=n_ctx,
            n_state=att_unit,
            n_head=head,
            n_layer=nar_layer,
            causal=False,
        )

        self.nq = nq
        self.residual = residual

    def forward(
        self,
        dec_seq: torch.Tensor,
        dec_seq_lengths: torch.Tensor = None,
        enc_seq: torch.Tensor = None,
        enc_seq_lengths: torch.Tensor = None,
        prefix_len: torch.Tensor = None,
        nar_level_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Vall-E forward for training

        Args:
            dec_seq (LongTensor): Batch of decoder sequences (B, T, nq).
            dec_seq_lengths (Tensor): Lengths of batched decoder sequences (B,).
            enc_seq (LongTensor): Batch of encoder sequences (B, T, nq), keep
                the interface, may not be used.
            enc_seq_lengths (LongTensor): Lengths of batched encoder sequences (B,),
                keep the interface, may not be used.
            prefix_len (LongTensor): Lengths of condition part in dec_seq (B,).
        """

        assert dec_seq.dim() == 3
        prefix_len = (prefix_len * dec_seq.size(1)).long()

        batch_size = dec_seq.size(0)
        dec_seq_emb = self.emb(dec_seq)  # [B, T, nq, D]

        # Auto-Regressive part
        input_ar_emb = self.prepare_input(dec_seq_emb, prefix_len, 1)[
            :, :-1
        ]  # [B, T, D]
        h_ar = self.ar_decoder(input_ar_emb)
        logits_ar = self.lm_head(h_ar)  # [B, T, V]

        # Non-Auto-Regressive part
        if nar_level_idx is None:
            nar_level_idx = torch.randint(
                1, self.nq, (batch_size,), device=dec_seq.device
            ).long()
        input_nar_emb = self.prepare_input(dec_seq_emb, prefix_len, nar_level_idx)[
            :, 1:
        ]  # [B, T, V]
        h_nar = self.nar_decoder(input_nar_emb, nar_level_idx - 1)

        logits_nar = self.lm_head(h_nar)  # [B, T, V]

        return logits_ar, logits_nar

    def prepare_input(self, dec_seq_emb, prefix_len, level):
        # (1) level mask, [B, 1, nq, 1], True is to include
        if isinstance(level, int):
            level = torch.ones_like(dec_seq_emb[:, 0, 0, 0]) * level
        level_mask = length_to_mask(level, self.nq).bool()
        level_mask = level_mask.unsqueeze(1).unsqueeze(3)

        # (2) prefix mask, [B, T, 1, 1], True is the prefix
        prefix_mask = length_to_mask(prefix_len, dec_seq_emb.size(1)).bool()
        prefix_mask = prefix_mask.unsqueeze(2).unsqueeze(3)

        # (3) mask and then sum in nq-axis.
        mask = torch.logical_or(level_mask, prefix_mask)
        return dec_seq_emb.masked_fill(~mask, 0.0).sum(2)

    @torch.no_grad()
    def inference(
        self,
        prefix: torch.Tensor,
        opts: SpeechLMInferenceOptions,
        enc_seq: torch.Tensor = None,
        suffix: torch.Tensor = None,
    ):
        """Vall-E Inference.

        Args:
            prefix (LongTensor): Prefix part of dec_seq (B, T, nq).
            opts (SpeechLMInferenceOptions): inference options.
            enc_seq (LongTensor): Encoder token sequence (B, T, nq).
            suffix (LongTensor): suffix part of dec_seq (B, T, nq),
                usually the target sequence for teacher-forcing.
        """
        device = opts.device or prefix.device

        # (1) initialization
        cache, hooks = install_kv_cache_hook(self.ar_decoder, {})

        # (2) auto-regressive prefix forward on first code layer
        prefix = prefix.expand(opts.nbest, -1, -1)
        if suffix is not None:
            suffix = suffix.expand(opts.nbest, -1, -1)
        prefix_emb = self.emb(prefix).sum(dim=2)  # [B, T, D]
        _ = self.ar_decoder(prefix_emb, kv_cache=cache)

        # (3) auto-regressive loop on first code layer
        # (3.1) AR initialization
        minlen = int(prefix.size(1) * opts.minlenratio) if opts.minlenratio > 0 else 0
        maxlen = int(prefix.size(1) * opts.maxlenratio)
        if opts.search_algo == "teacher_force":
            assert suffix is not None
            minlen = suffix.size(1)
            maxlen = suffix.size(1)

        generated = {"token": [], "score": []}
        finish_idx = torch.tensor([-1], device=device).expand(opts.nbest).long()
        prev_tok = torch.tensor([opts.start], device=device).tile(opts.nbest, 1).long()
        for step in range(maxlen):
            #  (3.2) AR loop
            prev_emb = self.emb(prev_tok)  # [B, 1, D]
            h_ar = self.ar_decoder(prev_emb, kv_cache=cache)
            logits = self.lm_head(h_ar)  # [B, 1, V]
            gen_tok, gen_score = logits_to_tokens(
                logits.unsqueeze(2),
                opts,
                allow_eos=step >= minlen,
                nq_level=0,
            )
            # [B, 1, 1] -> [B, 1]
            gen_tok, gen_score = gen_tok.squeeze(2), gen_tok.squeeze(2)

            generated["token"].append(gen_tok)
            generated["score"].append(gen_score)

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, step : step + 1, 0]
            else:
                prev_tok = gen_tok  # [B, 1]

            # (3.3) detect ended hypotheses.
            finish_idx = torch.where(
                torch.logical_and(prev_tok[:, 0] == opts.eos, finish_idx == -1),
                step,
                finish_idx,
            )

            if torch.all(torch.ge(finish_idx, 0)):
                break

        logging.info(f"Terminate at steps: {finish_idx.cpu().tolist()}")

        # (3.4) finalize auto-regressive
        if not opts.allow_invalid:
            valid_idx = finish_idx.ne(-1).nonzero(as_tuple=True)[0]
            if len(valid_idx) < prefix.size(0):
                logging.info(f"Only {len(valid_idx)} of {prefix.size(0)} are valid")
            elif len(valid_idx) == 0:
                logging.warning("No valid examples. Return None")
                return None, None
        else:
            valid_idx = torch.ones(len(finish_idx), device=finish_idx.device).bool()
        finish_idx = finish_idx[valid_idx]
        prefix_emb = prefix_emb[valid_idx]

        if suffix is not None:
            suffix = suffix[valid_idx]
        gen_tokens_ar = torch.cat(generated["token"], dim=1)[valid_idx].unsqueeze(
            2
        )  # [B, T, 1]
        gen_scores_ar = torch.cat(generated["score"], dim=1)[valid_idx].unsqueeze(2)
        end = finish_idx.max()
        if end > -1:
            gen_tokens_ar = gen_tokens_ar[:, : finish_idx.max() + 1]  # to include <sos>
            gen_scores_ar = gen_scores_ar[:, : finish_idx.max() + 1]

        for hook in hooks:
            hook.remove()
        cache = {}

        # (4) non-auto-regressive loop on the remained code layers
        # (4.1) NAR initialization
        if opts.search_algo == "teacher_force":
            prev_tok = suffix[:, :, 0]
        else:
            prev_tok = gen_tokens_ar[:, :, 0]
        start_emb = self.emb.weight[opts.start].tile(opts.nbest, 1, 1)  # [B, 1, D]
        prev_emb = torch.cat(
            [prefix_emb[:, 1:], start_emb, self.emb(prev_tok)], dim=1
        )  # [B, T, D]

        ones = torch.ones_like(valid_idx)
        generated = {"token": [], "score": []}
        # (4.2) NAR loop
        for step in range(1, opts.nq):
            h_nar = self.nar_decoder(prev_emb, ones * step - 1)  # [B, T, D]
            logits = self.lm_head(h_nar)  # [B, T, V]
            gen_tok, gen_score = logits_to_tokens(
                logits.unsqueeze(2),
                opts,
                allow_eos=False,
                nq_level=step,
            )
            gen_tok, gen_score = gen_tok.squeeze(2), gen_score.squeeze(2)  # [B, T]

            generated["token"].append(gen_tok[:, prefix.size(1) :])
            generated["score"].append(gen_score[:, prefix.size(1) :])

            if opts.search_algo == "teacher_force":
                prev_tok = suffix[:, :, step]
            else:
                prev_tok = gen_tok[:, prefix.size(1):]
            prev_emb[:, prefix.size(1) :] += self.emb(prev_tok)  # [B, T, D]
            prev_emb[:, prefix.size(1) - 1 : prefix.size(1)] += start_emb

        # (5) combine AR and NAR results
        gen_tokens_nar = torch.stack(generated["token"], dim=2)  # [B, T, nq]
        gen_scores_nar = torch.stack(generated["score"], dim=2)

        gen_tokens = torch.cat([gen_tokens_ar, gen_tokens_nar], dim=2)  # [B, T, nq]
        gen_scores = torch.cat([gen_scores_ar, gen_scores_nar], dim=2)

        gen_tokens_list, gen_scores_list = [], []
        for b in range(len(valid_idx)):
            gen_tokens_list.append(gen_tokens[b][: finish_idx[b]])
            gen_scores_list.append(gen_scores[b][: finish_idx[b]])

        return gen_tokens_list, gen_scores_list


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        causal: bool = True,
        layer_class=ResidualAttentionBlock,
    ):
        super().__init__()

        self.pos_emb = nn.Embedding(n_ctx, n_state)

        self.blocks = nn.ModuleList(
            [
                layer_class(n_state, n_head, cross_attention=False)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        if causal:
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

    def forward(self, x: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class ResidualAttentionBlockAdaLM(ResidualAttentionBlock):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super(ResidualAttentionBlockAdaLM, self).__init__(
            n_state=n_state,
            n_head=n_head,
            cross_attention=cross_attention,
        )

        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                setattr(self, name, AdaLN(n_state))

    def forward(
        self,
        x: Tensor,
        level: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x, level), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = (
                x
                + self.cross_attn(self.cross_attn_ln(x, level), xa, kv_cache=kv_cache)[
                    0
                ]
            )
        x = x + self.mlp(self.mlp_ln(x, level))
        return x


class ValleNARDecoder(TransformerDecoder):
    def __init__(
        self,
        n_level: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        causal: bool = True,
        layer_class=ResidualAttentionBlockAdaLM,
    ):
        super(ValleNARDecoder, self).__init__(
            n_ctx=n_ctx,
            n_state=n_state,
            n_head=n_head,
            n_layer=n_layer,
            causal=causal,
            layer_class=layer_class,
        )
        self.level_emb = nn.Embedding(n_level, n_state)
        self.ln = AdaLN(n_state)

    def forward(self, x: Tensor, level: Tensor, kv_cache: Optional[dict] = None):
        level = self.level_emb(level)

        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, level=level, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x, level)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None)
            # will prepend the cached kv tensors;
            # otherwise, perform key/value projections for
            # self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and
            # reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class AdaLN(nn.Module):
    def __init__(self, n_state, eps=1e-5):
        super().__init__()
        self.weight = nn.Linear(n_state, n_state, bias=False)
        self.bias = nn.Linear(n_state, n_state, bias=False)
        nn.init.constant_(self.weight.weight, 1.0)
        nn.init.constant_(self.bias.weight, 0.0)

        self.n_state = n_state
        self.eps = eps

    def forward(self, x: Tensor, level_emb: Tensor):
        w = self.weight(level_emb).unsqueeze(1)
        b = self.bias(level_emb).unsqueeze(1)
        x = nn.functional.layer_norm(x, (self.n_state,), eps=self.eps)
        x = w * x + b
        return x


def install_kv_cache_hook(model, cache):
    cache = {**cache} if cache is not None else {}
    hooks = []

    def save_to_cache(module, _, output):
        if module not in cache:
            # save as-is, for the first token or cross attention
            cache[module] = output
        else:
            cache[module] = torch.cat([cache[module], output], dim=1).detach()
        return cache[module]

    def install_hooks(layer: torch.nn.Module):
        if isinstance(layer, MultiHeadAttention):
            hooks.append(layer.key.register_forward_hook(save_to_cache))
            hooks.append(layer.value.register_forward_hook(save_to_cache))

    model.apply(install_hooks)
    return cache, hooks


def logits_to_tokens(
    logits: torch.Tensor,
    opts: SpeechLMInferenceOptions,
    allow_eos: bool = True,
    nq_level: int = None,
):
    assert logits.dim() == 4

    # (1) Apply mask
    mask = opts.masks
    if allow_eos:  # only predict eos in the first code
        mask[..., 0, opts.eos] = False
    if nq_level is not None:
        mask = mask[nq_level : nq_level + 1]
    mask = mask.unsqueeze(0).unsqueeze(0)
    logits = logits.masked_fill_(mask, -1e20)

    # (2) token selection
    topk_values, topk_indices = torch.topk(logits, opts.top_k, dim=-1)

    if opts.search_algo in ["sampling"]:
        logp = torch.softmax(topk_values / opts.sampling_temperature, dim=-1)
        inner_indices = torch.multinomial(logp.flatten(end_dim=-2), num_samples=1).view(
            logp[..., :1].size()
        )
        gen_token_idx = torch.gather(topk_indices, -1, inner_indices).squeeze(-1)
        gen_token_score = torch.gather(topk_values, -1, inner_indices).squeeze(-1)

    elif opts.search_algo in ["greedy_search", "teacher_force"]:
        gen_token_idx = topk_indices[:, :, :, 0]
        gen_token_score = topk_values[:, :, :, 0]

    else:
        raise NotImplementedError(f"opts.search_algo={opts.search_algo}")

    return gen_token_idx, gen_token_score
