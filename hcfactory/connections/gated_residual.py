"""Gated Residual (GR), the 4-stream residual of Qwen3.8-Flash-Next.

Reference: Qwen3.8-Flash-Next model card ("element-wise, data-dependent read gate and a per-branch
scalar write gate", 4 branches, bottleneck rank 320 at d = 2560) and the open MaxText port
(AI-Hypercomputer/maxtext PR #5736, qwen38_flash_next/hyper_connection.py), which this follows
operator by operator. Per sublayer, with H the n x d residual state (streams start as copies of
the token embedding):

    N    = rmsnorm_per_stream(H) * (1 + s)                      zero-centred scale s in R^{n d}
    read = sigmoid(U silu(D vec(N) / n))                        D: nd -> r, U: r -> nd
    x    = mean_k read_k * N_k                                  sublayer input, no extra pre-norm
    w    = 2 sigmoid(P vec(N) / n)                              P: nd -> n, one scalar per stream
    H_k <- H_k + w_k f(x)

The final read-out is one more read (same form, no write) and replaces the plain stream sum.
Deviation: GPT still applies its final RMSNorm after it (Qwen feeds the mixer to lm_head).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Connection


class _GRSite(nn.Module):
    def __init__(self, n: int, d: int, rank: int, write: bool):
        super().__init__()
        self.n, self.d = n, d
        self.scale = nn.Parameter(torch.zeros(n * d))  # zero-centred RMSNorm scale
        self.down = nn.Linear(n * d, rank, bias=False)
        self.up = nn.Linear(rank, n * d, bias=False)
        self.inject = nn.Linear(n * d, n, bias=False) if write else None

    def reset_parameters(self):
        # nnx.Linear default (LeCun normal, std 1/sqrt(fan_in)); GPT._init_weights skips connections
        for m in (self.down, self.up, self.inject):
            if m is not None:
                nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.in_features))

    def norm(self, H):  # H [B, T, n, d] -> vec(N) [B, T, n d]
        N = F.rms_norm(H.float(), (self.d,), eps=1e-6).flatten(-2) * (1.0 + self.scale.float())
        return N.to(H.dtype)

    def read(self, N):
        g = torch.sigmoid(self.up(F.silu(self.down(N) / self.n)))
        return (g * N).unflatten(-1, (self.n, self.d)).mean(-2)

    def forward(self, H, f):
        N = self.norm(H)
        w = 2.0 * torch.sigmoid(self.inject(N).float() / self.n)  # [B, T, n]
        y = f(self.read(N))
        return H + w.to(H.dtype).unsqueeze(-1) * y.unsqueeze(-2)


class GatedResidual(Connection):
    """Qwen3.8-Flash-Next Gated Residual. kwargs: n (streams, 4), rank (read bottleneck; default
    d / 8, the 320 / 2560 ratio of the released model)."""

    sublayer_prenorm = False  # the read output is already normalised; Qwen blocks have no input norm

    def __init__(self, cfg, n: int = 4, rank: int | None = None):
        super().__init__(cfg)
        self.n = n
        self.rank = rank or cfg.d_model // 8
        self.sites = nn.ModuleList(_GRSite(n, cfg.d_model, self.rank, True) for _ in range(self.n_sub))
        self.mixer = _GRSite(n, cfg.d_model, self.rank, False)
        self.reset_parameters()

    def reset_parameters(self):
        for s in (*self.sites, self.mixer):
            s.reset_parameters()

    def forward(self, x0, sublayers):
        H = x0.unsqueeze(-2).expand(*x0.shape[:-1], self.n, x0.shape[-1])
        for site, f in zip(self.sites, sublayers):
            H = site(H, f)
        return self.mixer.read(self.mixer.norm(H))

    def extra_repr(self):
        return f"n={self.n}, rank={self.rank}"
