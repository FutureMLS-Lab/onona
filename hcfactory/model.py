"""A small Llama-style decoder whose residual wiring is pluggable.

The model is split into three parts:

* ``embed``     token embedding -> x0            (shared by every variant)
* ``sublayers`` 2L callables, alternating Attention / MLP, each computing
                ``f(norm(x))``                   (shared by every variant)
* ``connection`` decides *what each sublayer reads* and *how its output is
                written back* -- this is the only thing that differs between
                pre-norm, hyper-connections, mHC, attention residuals, ...

Keeping the sublayers identical means that any difference in the results is
due to the connection, not to incidental changes in the transformer body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    mlp_ratio: float = 8 / 3  # SwiGLU hidden = mlp_ratio * d_model (rounded)
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    # Qwen3-style attention: grouped KV heads + per-head RMSNorm on q/k
    n_kv_head: int | None = None
    qk_norm: bool = False
    # "single" | "twin_avg" (two tables, always averaged) |
    # "drop_token" (two tables; each token picks one w.p. 0.5 in training) |
    # "drop_token_tied" (like drop_token, but the lm_head at position t uses the same
    # table that position t's input picked: one coherent A- or B-path per position).
    # "drop_seq" (one coin per sequence). Eval / lm_head use 0.5*(A+B), so after
    # training the two tables merge into one: inference cost == "single".
    # Controls: "drop_head" (input uses the mean, only the head is routed per position);
    # "drop_token_tied_fixed" (path-coupled, but the coin is a hash of the token's absolute
    # position in the training data, so every occurrence keeps its table across epochs:
    # a fixed half/half data split instead of fresh routing); "grad_mask_tied" (ONE tied
    # table; each position passes input and output gradient to it w.p. 0.5, i.e. the
    # data subsampling of path coupling without a second table).
    embed_mode: str = "single"
    # tgr* modes: bottleneck rank of the GR-style read gate (0: d_model // 8, the Qwen3.8 ratio) and
    # init of its up projection ("zero": g = 1/2 at step 0; "lecun": random gates, as in Qwen3.8)
    tgr_rank: int = 0
    tgr_up_init: str = "zero"
    # tgr_sel: how the per-position head selector is trained. "st": hard top-1 table, straight-through
    # gradient; "mos": soft mixture of the two softmaxes; "mos_st": mos until the cooldown, then st.
    # Evaluation always selects ONE table per position (the served model: Standard head FLOPs).
    tgr_sel_train: str = "st"
    # standard regularizers (baselines): dropout on embedding coordinates, word dropout
    # (input token replaced by UNK), dropout on sublayer outputs
    emb_dropout: float = 0.0
    word_dropout: float = 0.0
    resid_dropout: float = 0.0
    embed_init_std: float = 0.02
    connection: str = "prenorm"
    connection_kwargs: dict = field(default_factory=dict)
    # token mixer: "attn" (every layer softmax attention) | "gdn_hybrid" (Gated DeltaNet,
    # with softmax attention in every `hybrid_attn_every`-th layer, i.e. 3:1 at 4)
    mixer: str = "attn"
    hybrid_attn_every: int = 4
    gdn_head_dim: int = 128
    # MoE MLP (0 = dense). Experts default to dense_hidden / top_k wide, so the ACTIVE
    # MLP parameters match the dense model; tokens over capacity skip the MLP.
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_capacity: float = 1.25
    moe_aux_coef: float = 0.01


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).type_as(x)
        return y * self.weight if self.weight is not None else y


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):  # x: [B, H, T, hd]
        T = x.shape[-2]
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class Attention(nn.Module):
    """Causal self-attention with its own pre-norm.

    ``forward(x)`` reads one stream. ``forward(x, xk, xv)`` lets connections
    such as MUDDFormer feed different streams into Q, K and V.
    """

    is_attention = True

    def __init__(self, cfg: ModelConfig, rotary: Rotary, prenorm: bool = True,
                 qkv_streams: bool = False):
        super().__init__()
        d, h = cfg.d_model, cfg.n_head
        self.n_head, self.head_dim = h, d // h
        self.n_kv = cfg.n_kv_head or h
        kv = self.n_kv * self.head_dim
        norm = (lambda: RMSNorm(d)) if prenorm else nn.Identity
        self.norm_q = norm()
        # separate K/V norms only exist when a connection feeds distinct streams
        self.norm_k = norm() if qkv_streams else None
        self.norm_v = norm() if qkv_streams else None
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, kv, bias=False)
        self.wv = nn.Linear(d, kv, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else None
        self.rotary = rotary
        self.resid_p = cfg.resid_dropout

    def forward(self, x, xk=None, xv=None):
        B, T, D = x.shape
        xq = self.norm_q(x)
        xk = xq if xk is None else self.norm_k(xk)
        xv = xq if xv is None else self.norm_v(xv)
        q = self.wq(xq).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(xk).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(xv).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           enable_gqa=self.n_kv != self.n_head)
        return F.dropout(self.wo(y.transpose(1, 2).reshape(B, T, D)), self.resid_p, self.training)


class MLP(nn.Module):
    is_attention = False

    def __init__(self, cfg: ModelConfig, prenorm: bool = True):
        super().__init__()
        d = cfg.d_model
        hidden = int(cfg.mlp_ratio * d)
        hidden = 64 * ((hidden + 63) // 64)
        self.norm = RMSNorm(d) if prenorm else nn.Identity()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)
        self.resid_p = cfg.resid_dropout

    def forward(self, x):
        x = self.norm(x)
        return F.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)), self.resid_p, self.training)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (Yang et al., 2025) on the fla chunk kernel.

    q/k/v projections with a causal depthwise short convolution and SiLU, per-head
    decay g = -exp(A_log) * softplus(a + dt_bias), write strength beta = sigmoid(b),
    L2-normalised q/k inside the kernel, per-head RMSNorm gated by SiLU(W_g x), output
    projection. Same interface as Attention (own pre-norm, returns the sublayer output).
    """

    is_attention = True

    def __init__(self, cfg: ModelConfig, prenorm: bool = True, conv_size: int = 4):
        super().__init__()
        d = cfg.d_model
        self.head_dim = cfg.gdn_head_dim
        assert d % self.head_dim == 0
        self.n_head = d // self.head_dim
        self.norm = RMSNorm(d) if prenorm else nn.Identity()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wa = nn.Linear(d, self.n_head, bias=False)
        self.wb = nn.Linear(d, self.n_head, bias=False)
        self.wg = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.conv = nn.ModuleList(
            nn.Conv1d(d, d, conv_size, groups=d, padding=conv_size - 1, bias=False) for _ in range(3))
        A = torch.empty(self.n_head).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(self.n_head) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus^-1(dt)
        self.o_norm = RMSNorm(self.head_dim)
        self.resid_p = cfg.resid_dropout

    def forward(self, x, xk=None, xv=None):
        assert xk is None and xv is None, "GatedDeltaNet reads one stream"
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        B, T, D = x.shape
        H, hd = self.n_head, self.head_dim
        x = self.norm(x)
        # one GEMM for all six input projections (same parameters as the separate Linears)
        W = torch.cat([self.wq.weight, self.wk.weight, self.wv.weight, self.wg.weight,
                       self.wa.weight, self.wb.weight])
        qkv, gate, a, b = F.linear(x, W).split([3 * D, D, H, H], dim=-1)
        # one depthwise causal conv over the q|k|v channels, then SiLU
        cw = torch.cat([c.weight for c in self.conv])                      # [3D, 1, K]
        qkv = F.conv1d(qkv.transpose(1, 2), cw, padding=cw.shape[-1] - 1, groups=3 * D)
        q, k, v = F.silu(qkv[..., :T].transpose(1, 2)).split(D, dim=-1)
        q, k, v = (t.reshape(B, T, H, hd) for t in (q, k, v))
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        beta = b.float().sigmoid()
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        o = self.o_norm(o) * F.silu(gate).view(B, T, H, hd)
        return F.dropout(self.wo(o.reshape(B, T, D)), self.resid_p, self.training)


class MoE(nn.Module):
    """Top-k routed SwiGLU experts with a fixed capacity (static shapes for compile).

    Each expert is ``dense_hidden / top_k`` wide, so the active MLP parameters per token
    equal the dense MLP. Router: softmax over experts, top-k, renormalised weights;
    Switch-style load-balancing loss (stored in ``self.aux``). Assignments beyond
    ``capacity * N * k / E`` per expert (first choices first) are dropped for that expert.
    """

    is_attention = False

    def __init__(self, cfg: ModelConfig, prenorm: bool = True):
        super().__init__()
        d, E, k = cfg.d_model, cfg.moe_experts, cfg.moe_top_k
        hidden = 64 * ((int(cfg.mlp_ratio * d) + 63) // 64)
        h = hidden // k
        self.E, self.k, self.cap = E, k, cfg.moe_capacity
        self.norm = RMSNorm(d) if prenorm else nn.Identity()
        self.router = nn.Linear(d, E, bias=False)
        out_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        self.w1 = nn.Parameter(torch.randn(E, d, h) * 0.02)
        self.w3 = nn.Parameter(torch.randn(E, d, h) * 0.02)
        self.w2 = nn.Parameter(torch.randn(E, h, d) * out_std)
        self.resid_p = cfg.resid_dropout
        self.aux = None

    def forward(self, x):
        B, T, D = x.shape
        E, k = self.E, self.k
        x = self.norm(x).reshape(-1, D)
        N = x.shape[0]
        C = max(1, int(math.ceil(self.cap * N * k / E)))
        probs = self.router(x).float().softmax(-1)  # [N, E]
        topv, topi = probs.topk(k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        # k-major order: every token's first choice is placed before any second choice
        onehot = F.one_hot(topi.t().reshape(-1), E)  # [kN, E]
        pos = ((onehot.cumsum(0) - 1) * onehot).sum(-1)  # slot within the chosen expert
        e = topi.t().reshape(-1)
        keep = pos < C
        slot = torch.where(keep, e * C + pos, torch.full_like(pos, E * C))
        xin = x.repeat(k, 1)
        buf = x.new_zeros(E * C + 1, D).index_add(0, slot, xin)[:E * C].view(E, C, D)
        hmid = F.silu(torch.bmm(buf, self.w1)) * torch.bmm(buf, self.w3)
        out = torch.cat([torch.bmm(hmid, self.w2).view(E * C, D), x.new_zeros(1, D)])
        w = (topv.t().reshape(-1) * keep).to(x.dtype).unsqueeze(-1)
        y = (out[slot] * w).view(k, N, D).sum(0)
        if self.training:
            frac = onehot.float().mean(0) * E  # share of assignments per expert, x E
            self.aux = (frac * probs.mean(0)).sum()
        return F.dropout(y.view(B, T, D), self.resid_p, self.training)


class _TwinGR(nn.Module):
    """Qwen3.8 Gated Residual read over the two twin tables (two streams of width d), per token row:
    [g_A; g_B] = sigmoid(U silu(D (rmsnorm_rows([A; B]) * (1 + s)) / 2)). Returns (g_A, g_B), [V, d]."""

    def __init__(self, d: int, rank: int):
        super().__init__()
        self.d = d
        self.scale = nn.Parameter(torch.zeros(2 * d))  # zero-centred RMSNorm scale
        self.down = nn.Linear(2 * d, rank, bias=False)
        self.up = nn.Linear(rank, 2 * d, bias=False)

    def reset_parameters(self, up_init: str = "zero"):
        nn.init.normal_(self.down.weight, std=1.0 / math.sqrt(self.down.in_features))  # LeCun
        if up_init == "zero":
            nn.init.zeros_(self.up.weight)
        else:
            assert up_init == "lecun", up_init
            nn.init.normal_(self.up.weight, std=1.0 / math.sqrt(self.up.in_features))
        nn.init.zeros_(self.scale)

    def forward(self, A, Bt):
        N = torch.cat([F.rms_norm(A.float(), (self.d,), eps=1e-6),
                       F.rms_norm(Bt.float(), (self.d,), eps=1e-6)], -1) * (1.0 + self.scale.float())
        g = torch.sigmoid(self.up(F.silu(self.down(N) / 2))).to(A.dtype)
        return g[:, : self.d], g[:, self.d :]


class _PairSel(nn.Module):
    """twin_pair route score per token row: s_v = w . silu(D (rmsnorm_rows([A; B]) * (1 + s)) / 2), [V]."""

    def __init__(self, d: int, rank: int):
        super().__init__()
        self.d = d
        self.scale = nn.Parameter(torch.zeros(2 * d))
        self.down = nn.Linear(2 * d, rank, bias=False)
        self.out = nn.Linear(rank, 1, bias=False)

    def reset_parameters(self):  # LeCun: a random ~half / half split of the vocabulary at step 0
        for m in (self.down, self.out):
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(m.in_features))
        nn.init.zeros_(self.scale)

    def forward(self, A, Bt):
        N = torch.cat([F.rms_norm(A.float(), (self.d,), eps=1e-6),
                       F.rms_norm(Bt.float(), (self.d,), eps=1e-6)], -1) * (1.0 + self.scale.float())
        return self.out(F.silu(self.down(N) / 2)).squeeze(-1)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from .connections import build_connection

        self.cfg = cfg
        self.connection = build_connection(cfg)
        prenorm = self.connection.sublayer_prenorm
        rotary = Rotary(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        twin = cfg.embed_mode not in ("single", "grad_mask_tied", "mos2")
        self.embed_b = nn.Embedding(cfg.vocab_size, cfg.d_model) if twin else None
        # learned head router (embed_mode router_*): g = sigmoid(w . h) from the final hidden
        # state picks or mixes the two tied tables; deterministic, same in training and eval
        self.head_router = (nn.Linear(cfg.d_model, 1, bias=False)
                            if cfg.embed_mode.startswith("router_") else None)
        # router_mos_gin: the input table is gated too, by its own vector. Row v reads
        # E[v] = g_v A[v] + (1 - g_v) B[v], g_v = sigmoid(u . rmsnorm((A[v] + B[v]) / 2)); the head is
        # the router_mos mixture of softmaxes. Extra parameters: 2d instead of d.
        # gin_only: the router_mos_gin input gate alone; the head is the plain mean (A + B) / 2 (no
        # routing, one matmul). Both ends fold to one table each: Standard inference cost.
        self.emb_router = (nn.Linear(cfg.d_model, 1, bias=False)
                           if cfg.embed_mode in ("router_mos_gin", "gin_only") else None)
        # router_mos_tok: the router_mos head, but the gate reads the current input token's row
        # instead of h, g_t = sigmoid(w . rmsnorm(M[v_t])): token-only, a V-vector precomputable at
        # serving. Ablation: is router_mos's gain the context in the gate or the mixture itself?
        # twin gate (embed_mode tgate*): token-conditioned elementwise gate over the two tables,
        # W[v] = g_v * A[v] + (1 - g_v) * B[v], g_v = sigmoid(W_g rmsnorm((A[v] + B[v]) / 2) + b).
        # The embedding reads W and the head scores h . W^T; the gate depends only on the token, so
        # W folds into one tied V x d table at serving (inference cost == "single"). tgate: one gate
        # (vector, d x d) shared by embedding and head; tgate_io: same form, separate input / output
        # gates (two folded tables); tgate_row_io: scalar per-row gates (d x 1), separate in / out.
        # Zero init: g = 1/2, W = (A + B) / 2 at step 0.
        self.tgate = cfg.embed_mode in ("tgate", "tgate_io", "tgate_row_io",
                                        "tgr", "tgr_io", "tgr_ctx", "tgr_sel")
        # twin gated residual (embed_mode tgr*): the Qwen3.8 Gated Residual read applied to the two
        # tables as two streams, nothing else in the model changes. Per token v,
        # N = rmsnorm_rows([A[v]; B[v]]) * (1 + s), [g_A; g_B] = sigmoid(U silu(D N / 2)) (rank r),
        # W[v] = g_A * A[v] + g_B * B[v]: independent element-wise gates per table (tgate ties
        # g_B = 1 - g_A). Token-conditioned, so W folds into one table at serving. tgr: one gate for
        # embedding and head; tgr_io: separate in / out gates; tgr_ctx: tgr_io plus the GR write gate
        # at the head, a per-position scalar per table w = 2 sigmoid(P h / 2) (zero init: w = 1),
        # logits = w_A h.(g_A A)^T + w_B h.(g_B B)^T, which costs two head matmuls (upper bound only).
        # tgr_sel: input reads the tgr table W; the head SELECTS one of the two tables per position,
        # k = [w . h > 0], logits = h . T_k^T with T_A = A, T_B = B: one head matmul per token at serving
        # (Standard FLOPs, two V x d head tables in memory). Training per cfg.tgr_sel_train.
        self.tgr = cfg.embed_mode.startswith("tgr")
        if self.tgr:
            r = cfg.tgr_rank or cfg.d_model // 8
            self.tw_gate_in = _TwinGR(cfg.d_model, r)
            self.tw_gate_out = _TwinGR(cfg.d_model, r) if cfg.embed_mode in ("tgr_io", "tgr_ctx") else None
            self.tw_gate_write = nn.Linear(cfg.d_model, 2) if cfg.embed_mode == "tgr_ctx" else None
        elif self.tgate:
            gd = 1 if cfg.embed_mode == "tgate_row_io" else cfg.d_model
            self.tw_gate_in = nn.Linear(cfg.d_model, gd)
            self.tw_gate_out = nn.Linear(cfg.d_model, gd) if cfg.embed_mode != "tgate" else None
        self.tw_gate_sel = (nn.Linear(cfg.d_model, 1, bias=False)
                            if cfg.embed_mode == "tgr_sel" else None)
        # twin_pair: path-consistent selection. A GR-style scorer over the token's own two rows picks one
        # table per token, s_v = w . silu(D rmsnorm_rows([A[v]; B[v]]) / 2); position t with input v reads
        # T_k[v] AND scores the next token with head T_k (k = A if s_v > 0): the route is known before the
        # body runs, so serving is one lookup + one head matmul (Standard FLOPs, two tables in memory).
        # Trained per cfg.tgr_sel_train (st / mos / mos_st), with the SAME p_v = sigmoid(s_v) on both ends.
        self.tw_gate_pair = (_PairSel(cfg.d_model, cfg.tgr_rank or cfg.d_model // 8)
                             if cfg.embed_mode == "twin_pair" else None)
        self.sel_hard = cfg.tgr_sel_train == "st"  # training-time head mode (train.py flips mos_st)
        self.sel_eval = "hard"                     # hard | soft (diagnostic)
        # attention embedding (embed_mode attnemb_*): AttnRes-style softmax over the two tables,
        # alpha_k(v) = softmax_k(q . rmsnorm(T_k[v])), E = sum_k alpha_k T_k; zero-init q = average
        self.attnemb = cfg.embed_mode.startswith("attnemb_")
        self.emb_q = nn.Parameter(torch.zeros(cfg.d_model)) if self.attnemb else None
        self.head_q = (nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                       if cfg.embed_mode == "attnemb_ctx" else None)
        # standard Mixture of Softmaxes, K = 2 (Yang et al. 2018; embed_mode mos2): one tied table,
        # two tanh-projected contexts h_k = tanh(W_k h) and a softmax prior over k. Same head cost as
        # router_mos; the extra parameters are 2 d^2 + 2 d instead of a second V x d table.
        self.mos_latent = (nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
                           if cfg.embed_mode == "mos2" else None)
        self.mos_prior = nn.Linear(cfg.d_model, 2, bias=False) if cfg.embed_mode == "mos2" else None
        self.eval_embed = "avg"  # avg | A | B  (eval-time diagnostic for twin modes)
        subs = []
        for i in range(cfg.n_layer):
            if cfg.mixer == "gdn_hybrid" and (i + 1) % cfg.hybrid_attn_every != 0:
                subs.append(GatedDeltaNet(cfg, prenorm))
            else:
                assert cfg.mixer in ("attn", "gdn_hybrid"), cfg.mixer
                subs.append(Attention(cfg, rotary, prenorm, self.connection.qkv_streams))
            subs.append(MoE(cfg, prenorm) if cfg.moe_experts else MLP(cfg, prenorm))
        self.sublayers = nn.ModuleList(subs)
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()
        if self.head_q is not None:
            nn.init.zeros_(self.head_q.weight)  # starts as the static query
        if self.tw_gate_pair is not None:
            self.tw_gate_pair.reset_parameters()
        if self.tgr:
            for m in (self.tw_gate_in, self.tw_gate_out):
                if m is not None:
                    m.reset_parameters(cfg.tgr_up_init)
            if self.tw_gate_write is not None:
                nn.init.zeros_(self.tw_gate_write.weight); nn.init.zeros_(self.tw_gate_write.bias)
        elif self.tgate:
            for m in (self.tw_gate_in, self.tw_gate_out):
                if m is not None:
                    nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        if cfg.tie_embeddings and not twin:
            self.lm_head.weight = self.embed.weight
        if twin:
            assert cfg.tie_embeddings, "twin embeddings are tied to lm_head via 0.5*(A+B)"
            del self.lm_head
        # Connections may need custom init after the body is initialised.
        self.connection.reset_parameters()

    def _init_weights(self):
        std = 0.02
        # GPT-2 style: scale output projections by depth.
        out_std = std / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if name.startswith("connection."):
                continue
            if p.dim() == 2:
                is_out = name.endswith("wo.weight") or name.endswith("w2.weight")
                is_emb = name.startswith(("embed.", "embed_b."))
                nn.init.normal_(p, mean=0.0, std=self.cfg.embed_init_std if is_emb
                                else out_std if is_out else std)

    def embed_tokens(self, idx, pos=None):
        """Returns (x0, pick, head); pick is the per-position table choice the head must
        follow (None: head uses head_weight()); head is "A/B" or "mask" (grad_mask_tied)."""
        mode = self.cfg.embed_mode
        if mode == "grad_mask_tied" and self.training:
            pick = torch.rand(idx.shape, device=idx.device) < 0.5
            e = self.embed(idx)
            return torch.where(pick.unsqueeze(-1), e, e.detach()), pick, "mask"
        if self.embed_b is None:
            return self.embed(idx), None, None
        if self.attnemb:
            if self.eval_embed in ("A", "B") and not self.training:
                t = self.embed if self.eval_embed == "A" else self.embed_b
                return t(idx), None, None
            E = self.attn_table()
            x0 = F.embedding(idx, E)
            if mode == "attnemb_drop" and self.training:
                # with prob 1/2 a position keeps only one (random) table, in input and head alike
                drop = torch.rand(idx.shape, device=idx.device) < 0.5
                coin = torch.rand(idx.shape, device=idx.device) < 0.5
                one = torch.where(coin.unsqueeze(-1), self.embed(idx), self.embed_b(idx))
                x0 = torch.where(drop.unsqueeze(-1), one, x0)
                return x0, (drop, coin, E), "attnemb"
            return x0, (None, None, E), "attnemb"
        if self.tw_gate_pair is not None:
            if self.eval_embed in ("A", "B") and not self.training:
                t = self.embed if self.eval_embed == "A" else self.embed_b
                return t(idx), None, None
            z = self.tw_gate_pair(self.embed.weight, self.embed_b.weight)[idx].unsqueeze(-1)  # [B, T, 1]
            p = torch.sigmoid(z)
            hard = self.sel_hard if self.training else self.sel_eval == "hard"
            if hard:
                m = (z > 0).float()
                if self.training:
                    m = m + p - p.detach()  # straight-through
                w = m
            else:
                w = p
            ea, eb = self.embed(idx), self.embed_b(idx)
            wd = w.to(ea.dtype)
            return wd * ea + (1 - wd) * eb, (w, z, hard), "pair"
        if self.tgate:
            if self.eval_embed in ("A", "B") and not self.training:
                t = self.embed if self.eval_embed == "A" else self.embed_b
                return t(idx), None, None
            return F.embedding(idx, self.tgate_table(self.tw_gate_in)), None, None
        if mode.startswith("router_"):  # input reads M = (A + B) / 2; the head is routed
            if self.eval_embed in ("A", "B") and not self.training:
                t = self.embed if self.eval_embed == "A" else self.embed_b
                return t(idx), None, None
            if self.emb_router is not None:  # router_mos_gin: per-row learned input gate
                return F.embedding(idx, self.gated_table()), None, None
            return 0.5 * (self.embed(idx) + self.embed_b(idx)), None, None
        if mode == "gin_only":  # gated input table, head_weight() = (A + B) / 2
            if self.eval_embed in ("A", "B") and not self.training:
                t = self.embed if self.eval_embed == "A" else self.embed_b
                return t(idx), None, None
            return F.embedding(idx, self.gated_table()), None, None
        if mode == "twin_avg" or not self.training:
            which = "avg" if self.training else self.eval_embed
            if which == "A":
                return self.embed(idx), None, None
            if which == "B":
                return self.embed_b(idx), None, None
            return 0.5 * (self.embed(idx) + self.embed_b(idx)), None, None
        if mode in ("drop_token", "drop_token_tied", "drop_head"):
            pick = torch.rand(idx.shape, device=idx.device) < 0.5
        elif mode == "drop_token_tied_fixed":
            assert pos is not None, "drop_token_tied_fixed needs absolute data positions"
            pick = ((pos * 2654435761) % 4294967296 >> 16) % 2 == 1  # fixed hash of position
        elif mode == "drop_seq":
            pick = torch.rand(idx.shape[0], 1, device=idx.device) < 0.5
        else:
            raise ValueError(mode)
        if mode == "drop_head":  # input uses the mean; only the head is routed
            return 0.5 * (self.embed(idx) + self.embed_b(idx)), pick, "A/B"
        x0 = torch.where(pick.unsqueeze(-1), self.embed(idx), self.embed_b(idx))
        coupled = mode in ("drop_token_tied", "drop_token_tied_fixed")
        return x0, (pick if coupled else None), ("A/B" if coupled else None)

    def gated_table(self):
        """router_mos_gin input table: E = g A + (1 - g) B, g_v = sigmoid(u . rmsnorm(M[v]))."""
        A, Bt = self.embed.weight, self.embed_b.weight
        M = 0.5 * (A.float() + Bt.float())
        g = torch.sigmoid(self.emb_router(F.rms_norm(M, (M.shape[-1],))).float()).to(A.dtype)  # [V, 1]
        return g * A + (1 - g) * Bt

    def tgate_table(self, gate):
        """Folded twin table W = g * A + (1 - g) * B, g = sigmoid(gate(rmsnorm((A + B) / 2)));
        tgr: W = g_A * A + g_B * B with the GR read gate."""
        A, Bt = self.embed.weight, self.embed_b.weight
        if self.tgr:
            ga, gb = gate(A, Bt)
            return ga * A + gb * Bt
        M = 0.5 * (A.float() + Bt.float())
        g = torch.sigmoid(gate(F.rms_norm(M, (M.shape[-1],))).float()).to(A.dtype)  # [V, d] or [V, 1]
        return Bt + g * (A - Bt)

    def attn_table(self):
        """E = alpha_A * A + alpha_B * B with alpha = softmax over tables of q . rmsnorm(row)."""
        A, Bt = self.embed.weight, self.embed_b.weight
        q = self.emb_q.float()
        za = F.rms_norm(A.float(), (A.shape[-1],)) @ q
        zb = F.rms_norm(Bt.float(), (Bt.shape[-1],)) @ q
        wa = torch.sigmoid(za - zb).unsqueeze(-1).to(A.dtype)   # 2-way softmax
        return wa * A + (1 - wa) * Bt

    def head_weight(self):
        if self.embed_b is None:
            return self.lm_head.weight
        if not self.training and self.eval_embed in ("A", "B"):
            return self.embed.weight if self.eval_embed == "A" else self.embed_b.weight
        if self.tgate:
            return self.tgate_table(self.tw_gate_in if self.tw_gate_out is None else self.tw_gate_out)
        return 0.5 * (self.embed.weight + self.embed_b.weight)

    def forward(self, idx, targets=None, pos=None):
        if self.training and self.cfg.word_dropout > 0:
            # word dropout: replace dropped input tokens by an unused UNK row (the last id,
            # a padding slot of the GPT-2 vocab). Zeroing the row instead gives an all-zero
            # residual stream that RMSNorm's backward amplifies by 1/sqrt(eps) -> NaN in bf16.
            drop = torch.rand(idx.shape, device=idx.device) < self.cfg.word_dropout
            idx = torch.where(drop, torch.full_like(idx, self.cfg.vocab_size - 1), idx)
        x0, pick, head = self.embed_tokens(idx, pos)
        if self.training and self.cfg.emb_dropout > 0:
            x0 = F.dropout(x0, self.cfg.emb_dropout, True)
        h = self.final_norm(self.connection(x0, self.sublayers))
        if head == "attnemb":
            drop, coin, E = pick
            A, Bt = self.embed.weight, self.embed_b.weight
            if self.head_q is not None:  # contextual query: alpha depends on (h_t, v)
                qt = self.head_q(h) + self.emb_q.to(h.dtype)
                ra = F.linear(qt, F.rms_norm(A, (A.shape[-1],)))
                rb = F.linear(qt, F.rms_norm(Bt, (Bt.shape[-1],)))
                w = torch.sigmoid((ra - rb).float())
                logits = w * F.linear(h, A).float() + (1 - w) * F.linear(h, Bt).float()
            elif drop is None:
                logits = F.linear(h, E)
            else:
                d = drop.unsqueeze(-1).to(h.dtype); c = coin.unsqueeze(-1).to(h.dtype)
                logits = (F.linear(h * (1 - d), E) + F.linear(h * d * c, A)
                          + F.linear(h * d * (1 - c), Bt))
        elif head == "pair":  # same route as the input: hard pick, or the soft mixture of softmaxes
            w, z, hard = pick
            la = F.linear(h, self.embed.weight).float()
            lb = F.linear(h, self.embed_b.weight).float()
            if hard:
                logits = w * la + (1 - w) * lb
            else:
                logits = torch.logaddexp(F.log_softmax(la, -1) + F.logsigmoid(z),
                                         F.log_softmax(lb, -1) + F.logsigmoid(-z))
        elif self.tw_gate_sel is not None:
            A, Bt = self.embed.weight, self.embed_b.weight
            if not self.training and self.eval_embed in ("A", "B"):
                logits = F.linear(h, A if self.eval_embed == "A" else Bt)
            else:
                z = self.tw_gate_sel(h).float()                                  # [B, T, 1]
                la, lb = F.linear(h, A).float(), F.linear(h, Bt).float()
                hard = self.sel_hard if self.training else self.sel_eval == "hard"
                if hard:
                    pick = (z > 0).float()
                    if self.training:  # straight-through: forward = selected table, backward via sigmoid
                        g = torch.sigmoid(z)
                        pick = pick + g - g.detach()
                    logits = pick * la + (1 - pick) * lb
                else:  # mixture of the two softmaxes, p_A = sigmoid(z)
                    logits = torch.logaddexp(F.log_softmax(la, -1) + F.logsigmoid(z),
                                             F.log_softmax(lb, -1) + F.logsigmoid(-z))
        elif (self.tgr and self.tw_gate_write is not None
              and not (not self.training and self.eval_embed in ("A", "B"))):
            ga, gb = self.tw_gate_out(self.embed.weight, self.embed_b.weight)
            w = (2.0 * torch.sigmoid(self.tw_gate_write(h).float() / 2)).to(h.dtype)  # [B, T, 2]
            logits = (w[..., :1] * F.linear(h, ga * self.embed.weight)
                      + w[..., 1:] * F.linear(h, gb * self.embed_b.weight))
        elif self.head_router is not None and not (not self.training and self.eval_embed in ("A", "B")):
            mode = self.cfg.embed_mode
            src = F.rms_norm(x0, (x0.shape[-1],)) if mode == "router_mos_tok" else h  # x0 = M[v_t]
            z = self.head_router(src).float()                                   # [B, T, 1]
            g = torch.sigmoid(z)
            A, Bt = self.embed.weight, self.embed_b.weight
            if mode in ("router_mos", "router_mos_gin", "router_mos_tok"):  # mixture of the two tables' softmaxes
                la = F.log_softmax(F.linear(h, A).float(), -1)
                lb = F.log_softmax(F.linear(h, Bt).float(), -1)
                # log g = logsigmoid(z), log(1 - g) = logsigmoid(-z): finite when g saturates
                logits = torch.logaddexp(la + F.logsigmoid(z), lb + F.logsigmoid(-z))  # log p
            else:
                if mode == "router_hard":  # top-1 table, straight-through gradient to g
                    g = (g > 0.5).float() + g - g.detach()
                elif mode != "router_mix":
                    raise ValueError(mode)
                g = g.to(h.dtype)
                logits = F.linear(h * g, A) + F.linear(h * (1 - g), Bt)
        elif self.mos_latent is not None:  # mos2: p = sum_k pi_k softmax(tanh(W_k h) . E)
            la, lb = torch.tanh(self.mos_latent(h)).chunk(2, -1)
            W = self.head_weight()
            lpa = F.log_softmax(F.linear(la, W).float(), -1)
            lpb = F.log_softmax(F.linear(lb, W).float(), -1)
            lpi = F.log_softmax(self.mos_prior(h).float(), -1)                  # [B, T, 2]
            logits = torch.logaddexp(lpa + lpi[..., :1], lpb + lpi[..., 1:])     # log p
        elif pick is None:
            logits = F.linear(h, self.head_weight())
        elif head == "mask":  # one table; the head gradient reaches it only where pick
            m = pick.unsqueeze(-1).to(h.dtype)
            w = self.head_weight()
            logits = F.linear(h * m, w) + F.linear(h * (1 - m), w.detach())
        else:  # position t scores the vocab with the table its own input used
            m = pick.unsqueeze(-1).to(h.dtype)
            logits = F.linear(h * m, self.embed.weight) + F.linear(h * (1 - m), self.embed_b.weight)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        if self.training and self.cfg.moe_experts:
            aux = sum(m.aux for m in self.sublayers if isinstance(m, MoE))
            loss = loss + self.cfg.moe_aux_coef * aux / self.cfg.n_layer
        return logits, loss

    def num_params(self, exclude_embedding: bool = True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.embed.weight.numel()
            if self.embed_b is not None:
                n -= self.embed_b.weight.numel()
            if self.embed_b is None and not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()  # untied head is an embedding-sized table too
        return n

    def active_params(self, exclude_embedding: bool = True):
        """Parameters used per token (MoE: router + top_k of the E experts)."""
        n = self.num_params(exclude_embedding)
        for m in self.sublayers:
            if isinstance(m, MoE):
                n -= (m.w1.numel() + m.w3.numel() + m.w2.numel()) * (m.E - m.k) // m.E
        return n

    def connection_params(self):
        return sum(p.numel() for p in self.connection.parameters())
