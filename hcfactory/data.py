"""Token-shard data pipeline (nanoGPT style: flat uint16 .bin files)."""
from __future__ import annotations

import os

import numpy as np
import torch


class TokenLoader:
    """Samples random contiguous windows from a memory-mapped token file.

    Under DDP each rank uses a different seed, so ranks see different windows.

    ``order="sequential"`` is the single-pass alternative: the file is cut into
    non-overlapping seq_len windows, shuffled once by ``perm_seed`` (identical on every
    rank), and consumed without replacement; rank r takes slot ``cursor * world + r``.
    ``cursor`` counts micro-batches, so a resumed run sets it to step * grad_accum.
    Running past the end wraps around to the same permutation (a second epoch).
    """

    def __init__(self, path: str, batch_size: int, seq_len: int, seed: int = 0, device="cpu",
                 limit: int = 0, order: str = "random", rank: int = 0, world: int = 1,
                 perm_seed: int = 0):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run `python -m hcfactory.prepare_data` first")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if limit:  # data-constrained regime: only the first `limit` tokens are ever seen
            self.data = self.data[:limit]
        self.batch_size, self.seq_len = batch_size, seq_len
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        if order not in ("random", "sequential"):
            raise ValueError(f"unknown data order {order!r}")
        self.order, self.rank, self.world, self.cursor = order, rank, world, 0
        if order == "sequential":
            n_win = (len(self.data) - 1) // seq_len
            if n_win < batch_size * world:
                raise ValueError(f"{path}: {n_win} windows < one global micro-batch")
            self.perm = np.random.default_rng(perm_seed).permutation(n_win)

    def __len__(self):
        return len(self.data)

    def next(self):
        if self.order == "sequential":
            slot = (self.cursor * self.world + self.rank) * self.batch_size
            ix = self.perm[(slot + np.arange(self.batch_size)) % len(self.perm)] * self.seq_len
            self.cursor += 1
        else:
            ix = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        buf = np.stack([self.data[i : i + self.seq_len + 1].astype(np.int64) for i in ix])
        t = torch.from_numpy(buf)
        # absolute data position of every input token (for position-hashed routing)
        self.last_pos = torch.from_numpy(ix[:, None] + np.arange(self.seq_len)).to(self.device)
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        else:
            t = t.to(self.device)
        return t[:, :-1], t[:, 1:]

    def fixed_batches(self, n: int, seed: int = 1234):
        """Deterministic evaluation batches (identical for every variant)."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            ix = rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
            buf = np.stack([self.data[i : i + self.seq_len + 1].astype(np.int64) for i in ix])
            t = torch.from_numpy(buf).to(self.device)
            out.append((t[:, :-1], t[:, 1:]))
        return out
