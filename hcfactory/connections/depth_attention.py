"""Attention over depth: AttnRes (Kimi), MHAR, Delta AttnRes (DAR).

All three keep a list of *sources* and let every sublayer read a softmax
mixture of them, with a learned per-site pseudo-query over RMS-normalised
sources:

    alpha_i = softmax_i( q_l . RMSNorm(s_i) )           (per head, per token)

AttnRes / MHAR  (replacement):  input_l = sum_i alpha_i s_i
                sources = [embedding, out_1, out_2, ...]      (full)
                sources = [embedding, block sums..., partial]  (block)
DAR  (additive):  input_l = h_l + g_l * sum_i alpha_i v_i
                  with h the ordinary residual stream and v_i the sublayer
                  outputs ("deltas").
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Connection, rms


class DepthRouter(nn.Module):
    """One routing site: H-head softmax over sources, zero-init query."""

    def __init__(self, d_model: int, heads: int = 1):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.query = nn.Parameter(torch.zeros(d_model))       # zero init -> uniform mix
        self.key_norm_weight = nn.Parameter(torch.ones(d_model))  # affine of RMSNorm(keys)

    def weights(self, keys):
        """keys: list of RMS-normalised sources [B,T,D] -> alpha [B,T,H,N]."""
        # elementwise product + per-head sum (fuses well under torch.compile;
        # an einsum here lowers to a slow batched matmul for heads > 1)
        q = self.query * self.key_norm_weight
        logits = torch.stack(
            [(k * q).unflatten(-1, (self.heads, -1)).sum(-1) for k in keys], dim=-1
        )  # [B, T, H, N]: sources last, so the softmax reduces a contiguous dim
        return logits.float().softmax(-1)

    def forward(self, sources, keys):
        alpha = self.weights(keys).to(sources[0].dtype)
        e = sources[0].shape[-1] // self.heads
        out = 0
        for i, s in enumerate(sources):
            # [B,T,H] -> [B,T,D]: head weight repeated over its e channels
            out = out + alpha[..., i].repeat_interleave(e, dim=-1) * s
        return out, alpha


class AttnRes(Connection):
    """Attention Residuals (Kimi / Moonshot).

    kwargs:
        heads:      depth-softmax heads (1 = AttnRes, >1 = MHAR)
        block_size: 0 = Full AttnRes (every sublayer output is a source);
                    k > 0 = Block AttnRes, k sublayers are summed into one source.
    """

    def __init__(self, cfg, heads: int = 1, block_size: int = 0):
        super().__init__(cfg)
        self.heads, self.block_size = heads, block_size
        # one router per sublayer + one for the final output
        self.routers = nn.ModuleList(DepthRouter(cfg.d_model, heads) for _ in range(self.n_sub + 1))
        self.record_alpha = False
        self.last_alpha = []

    FAST_MIN_SUBLAYERS = 40

    def forward(self, x0, sublayers):
        # the one-GEMM logit path pays off only in deep stacks: on one H100 it is 16% faster
        # at 48 sublayers (330M) but 15% slower at 32 (142M), where the per-key re-reads are cheap
        if self.block_size == 0 and not self.record_alpha and self.n_sub >= self.FAST_MIN_SUBLAYERS:
            return self._forward_full_fast(x0, sublayers)
        return self._forward_generic(x0, sublayers)

    def _forward_full_fast(self, x0, sublayers):
        """Full AttnRes with each source's logits for *all* routers computed once, when the
        source is created (one [BT,D] x [D,R*H] GEMM), instead of re-reading every source's
        normalised key at every later router. Same function as _forward_generic."""
        R, H = len(self.routers), self.heads
        D = x0.shape[-1]
        # Q[r] = query_r * key_norm_weight_r; logits[..., r, h] = <rms(s)_h, Q[r]_h>
        Q = torch.stack([r.query * r.key_norm_weight for r in self.routers]).to(x0.dtype)  # [R, D]
        if H == 1:
            def logits_of(s):  # [B,T,D] -> [B,T,R,1]
                return (rms(s) @ Q.t()).unsqueeze(-1)
        else:
            Qh = Q.view(R, H, D // H)

            def logits_of(s):  # [B,T,D] -> [B,T,R,H]
                return torch.einsum("bthe,rhe->btrh", rms(s).unflatten(-1, (H, D // H)), Qh)

        sources, logits = [x0], [logits_of(x0)]
        e = D // H
        for i, f in enumerate(sublayers):
            lg = torch.stack([l[..., i, :] for l in logits], dim=-1)          # [B,T,H,N]
            alpha = lg.float().softmax(-1).to(x0.dtype)
            h = 0
            for j, s in enumerate(sources):
                h = h + alpha[..., j].repeat_interleave(e, dim=-1) * s
            out = f(h)
            sources.append(out)
            logits.append(logits_of(out))
        lg = torch.stack([l[..., R - 1, :] for l in logits], dim=-1)
        alpha = lg.float().softmax(-1).to(x0.dtype)
        h = 0
        for j, s in enumerate(sources):
            h = h + alpha[..., j].repeat_interleave(e, dim=-1) * s
        return h

    def _forward_generic(self, x0, sublayers):
        self.last_alpha = []
        sources, keys = [x0], [rms(x0)]
        partial = None  # running sum of the current (unfinished) block
        for i, f in enumerate(sublayers):
            srcs, ks = sources, keys
            if partial is not None:
                srcs, ks = sources + [partial], keys + [rms(partial)]
            h, alpha = self.routers[i](srcs, ks)
            if self.record_alpha:
                self.last_alpha.append(alpha.detach().mean((0, 1)))
            out = f(h)
            if self.block_size == 0:
                sources.append(out)
                keys.append(rms(out))
            else:
                partial = out if partial is None else partial + out
                if (i + 1) % self.block_size == 0:
                    sources.append(partial)
                    keys.append(rms(partial))
                    partial = None
        if partial is not None:
            sources.append(partial)
            keys.append(rms(partial))
        h, _ = self.routers[-1](sources, keys)
        return h

    def extra_repr(self):
        return f"heads={self.heads}, block_size={self.block_size}"


class MHAR(AttnRes):
    """Multi-Head Attention Residuals: AttnRes with H independent depth softmaxes."""

    def __init__(self, cfg, heads: int = 4, block_size: int = 0):
        super().__init__(cfg, heads=heads, block_size=block_size)


class DeltaAttnRes(Connection):
    """Delta Attention Residuals (additive routing over sublayer outputs).

    kwargs:
        heads: depth-softmax heads
        block: False = Delta AttnRes (every sublayer output is a source)
               True  = Delta Block (one source per transformer block = its delta)
        gate:  "none" (g=1, for pre-training) or "zero" (learned scalar g, init 0,
               the safe-conversion setting for pretrained checkpoints)
    """

    def __init__(self, cfg, heads: int = 1, block: bool = False, gate: str = "none"):
        super().__init__(cfg)
        self.heads, self.block, self.gate_mode = heads, block, gate
        self.routers = nn.ModuleList(DepthRouter(cfg.d_model, heads) for _ in range(self.n_sub))
        if gate == "zero":
            self.gates = nn.Parameter(torch.zeros(self.n_sub))
        else:
            self.register_parameter("gates", None)

    def forward(self, x0, sublayers):
        h = x0
        sources, keys = [x0], [rms(x0)]
        block_start = x0
        for i, f in enumerate(sublayers):
            routed, _ = self.routers[i](sources, keys)
            if self.gates is not None:
                routed = self.gates[i] * routed
            out = f(h + routed)
            h = h + out
            if not self.block:
                sources.append(out)
                keys.append(rms(out))
            elif i % 2 == 1:  # end of a transformer block (attn + mlp)
                delta = h - block_start
                sources.append(delta)
                keys.append(rms(delta))
                block_start = h
        return h

    def extra_repr(self):
        return f"heads={self.heads}, block={self.block}, gate={self.gate_mode}"
