"""Training template shared by every connection variant.

Single device:
    python -m hcfactory.train --config configs/tiny.yaml
Multi-GPU:
    torchrun --nproc_per_node=8 -m hcfactory.train --config configs/142m.yaml

Any config key can be overridden as ``key=value`` (``connection_kwargs.n=4`` for
nested keys). Every run writes ``<out_dir>/<run_name>/{config.json,log.jsonl,summary.json}``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from .data import TokenLoader
from .model import GPT, ModelConfig

DEFAULTS = dict(
    # model
    n_layer=8, n_head=8, d_model=512, mlp_ratio=8 / 3, vocab_size=50304,
    connection="prenorm", connection_kwargs={},
    n_kv_head=None, qk_norm=False, rope_theta=10000.0, embed_mode="single", tie_embeddings=True,
    tgr_rank=0, tgr_up_init="zero", tgr_sel_train="st",
    emb_dropout=0.0, word_dropout=0.0, resid_dropout=0.0, embed_init_std=0.02,
    embed_weight_decay=None,  # None: same as weight_decay
    mixer="attn", hybrid_attn_every=4, gdn_head_dim=128,
    moe_experts=0, moe_top_k=2, moe_capacity=1.25, moe_aux_coef=0.01,
    # data
    data_dir="data/fineweb-edu", train_tokens=0, seq_len=1024, batch_size=16, grad_accum=1,
    data_order="random",  # "random": windows sampled with replacement; "sequential": one shuffled pass
    # optimisation
    max_steps=2000, lr=3e-3, min_lr_ratio=0.1, warmup_steps=100, schedule="cosine",
    cooldown_frac=0.2, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    connection_lr_mult=1.0,
    # eval / logging
    eval_every=200, eval_batches=20, eval_batch_size=None, log_every=10,
    out_dir="runs", run_name=None, seed=0, compile=False, dtype="auto", device="auto",
    wandb_project=None,
    # checkpointing: every `ckpt_every` steps rank 0 writes <ckpt_dir>/<run_name>/latest.pt;
    # a restarted run with the same run_name resumes from it automatically
    ckpt_dir=None, ckpt_every=2000,
    # >0: also keep a model-only bf16 snapshot <ckpt_dir>/<run_name>/step_XXXXXX.pt every
    # `ckpt_keep_every` steps (and at the last step); latest.pt stays the resume point
    ckpt_keep_every=0,
    save_embeddings=False,  # write <run_dir>/embeddings.pt (fp16 embedding tables) at the end
)


def parse_value(v: str):
    # YAML 1.1 reads "3e-4" as a string, so try numbers first.
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("overrides", nargs="*", help="key=value")
    args = ap.parse_args(argv)
    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        d = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = parse_value(v)
    if cfg["run_name"] is None:
        kw = "".join(f"_{k}{v}" for k, v in sorted(cfg["connection_kwargs"].items()))
        cfg["run_name"] = f"{cfg['connection']}{kw}_s{cfg['seed']}"
    return cfg


def lr_at(step: int, cfg) -> float:
    base, warm, total = cfg["lr"], cfg["warmup_steps"], cfg["max_steps"]
    lo = base * cfg["min_lr_ratio"]
    if step < warm:
        return base * (step + 1) / warm
    if cfg["schedule"] == "wsd":  # warmup-stable-decay
        start = int(total * (1 - cfg["cooldown_frac"]))
        if step < start:
            return base
        return lo + (base - lo) * (1 - (step - start) / max(1, total - start))
    t = (step - warm) / max(1, total - warm)
    return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * min(t, 1.0)))


FREQ_EDGES = [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, float("inf")]


@torch.no_grad()
def freq_bin_losses(model, train, val_batches, amp, cfg):
    """Mean per-token loss on val and fixed train batches, binned by how often the TARGET
    token occurs in the training data actually seen (train.data, i.e. after train_tokens)."""
    freq = np.zeros(cfg["vocab_size"], dtype=np.int64)
    for i in range(0, len(train.data), 100_000_000):  # chunked: train.data may be GBs
        freq += np.bincount(train.data[i : i + 100_000_000], minlength=cfg["vocab_size"])
    edges = np.array(FREQ_EDGES)
    bin_of = torch.from_numpy(np.searchsorted(edges, freq, side="right") - 1)
    nb = len(edges) - 1
    out = dict(edges=[e if math.isfinite(e) else None for e in FREQ_EDGES])
    model.eval()
    model.eval_embed = "avg"
    for split, batches in (("val", val_batches), ("train", train.fixed_batches(cfg["eval_batches"]))):
        tot, cnt = torch.zeros(nb, dtype=torch.float64), torch.zeros(nb, dtype=torch.float64)
        for x, y in batches:
            with amp:
                logits, _ = model(x)
            l = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1),
                                reduction="none").cpu().double()
            b = bin_of[y.reshape(-1).cpu()]
            tot.index_add_(0, b, l)
            cnt.index_add_(0, b, torch.ones_like(l))
        out[f"{split}_loss"] = (tot / cnt.clamp(min=1)).tolist()
        out[f"{split}_count"] = cnt.tolist()
    model.train()
    return out


def main(argv=None):
    cfg = load_config(argv)

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        if cfg["device"] != "auto":
            device = torch.device(cfg["device"])
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    master = rank == 0
    torch.manual_seed(cfg["seed"])

    dtype = cfg["dtype"]
    if dtype == "auto":
        dtype = "bfloat16" if device.type == "cuda" else "float32"
    amp = (
        torch.autocast(device.type, dtype=getattr(torch, dtype))
        if dtype != "float32"
        else contextlib.nullcontext()
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mcfg = ModelConfig(
        vocab_size=cfg["vocab_size"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
        d_model=cfg["d_model"], mlp_ratio=cfg["mlp_ratio"], max_seq_len=cfg["seq_len"],
        connection=cfg["connection"], connection_kwargs=cfg["connection_kwargs"],
        n_kv_head=cfg["n_kv_head"], qk_norm=cfg["qk_norm"], rope_theta=cfg["rope_theta"],
        embed_mode=cfg["embed_mode"], tie_embeddings=cfg["tie_embeddings"],
        tgr_rank=cfg["tgr_rank"], tgr_up_init=cfg["tgr_up_init"], tgr_sel_train=cfg["tgr_sel_train"],
        emb_dropout=cfg["emb_dropout"], word_dropout=cfg["word_dropout"],
        resid_dropout=cfg["resid_dropout"], embed_init_std=cfg["embed_init_std"],
        mixer=cfg["mixer"], hybrid_attn_every=cfg["hybrid_attn_every"],
        gdn_head_dim=cfg["gdn_head_dim"], moe_experts=cfg["moe_experts"],
        moe_top_k=cfg["moe_top_k"], moe_capacity=cfg["moe_capacity"],
        moe_aux_coef=cfg["moe_aux_coef"],
    )
    model = GPT(mcfg).to(device)
    n_params = model.num_params()
    n_conn = model.connection_params()

    # Connection parameters (gates, routing queries, mixing matrices) get no
    # weight decay: decaying them pulls every variant back towards "no mixing".
    decay, no_decay, conn, emb = [], [], [], []
    emb_wd = cfg["embed_weight_decay"]
    for name, p in model.named_parameters():
        if name.startswith("connection."):
            conn.append(p)
        elif name.startswith("tw_gate"):  # twin gates: no decay (decay pulls g back to 1/2)
            no_decay.append(p)
        elif emb_wd is not None and name.startswith(("embed.", "embed_b.")):
            emb.append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        dict(params=decay, weight_decay=cfg["weight_decay"], lr_mult=1.0),
        dict(params=no_decay, weight_decay=0.0, lr_mult=1.0),
        dict(params=conn, weight_decay=0.0, lr_mult=cfg["connection_lr_mult"]),
    ]
    if emb:
        groups.append(dict(params=emb, weight_decay=emb_wd, lr_mult=1.0))
    opt = torch.optim.AdamW(
        groups, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
        fused=device.type == "cuda",
    )

    fwd = torch.compile(model) if cfg["compile"] else model
    if ddp:
        fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[device.index])

    train = TokenLoader(os.path.join(cfg["data_dir"], "train.bin"), cfg["batch_size"],
                        cfg["seq_len"], seed=cfg["seed"] * 1000 + rank, device=device,
                        limit=cfg["train_tokens"], order=cfg["data_order"], rank=rank,
                        world=world, perm_seed=cfg["seed"])
    eval_batch_size = cfg["eval_batch_size"] or cfg["batch_size"]
    val = TokenLoader(os.path.join(cfg["data_dir"], "val.bin"), eval_batch_size,
                      cfg["seq_len"], device=device)
    val_batches = val.fixed_batches(cfg["eval_batches"])
    tokens_per_step = cfg["batch_size"] * cfg["seq_len"] * cfg["grad_accum"] * world

    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    ckpt_path = (os.path.join(cfg["ckpt_dir"], cfg["run_name"], "latest.pt")
                 if cfg["ckpt_dir"] else None)
    resume = torch.load(ckpt_path, map_location=device, weights_only=False) \
        if ckpt_path and os.path.exists(ckpt_path) else None
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        # fresh, rank- and step-dependent sampling stream after a restart
        train.rng = np.random.default_rng([cfg["seed"], rank, resume["step"]])
        train.cursor = resume["step"] * cfg["grad_accum"]  # sequential order: skip consumed windows
        if master:
            print(f"resumed {cfg['run_name']} from step {resume['step']}")
    log_f = None
    if master:
        os.makedirs(run_dir, exist_ok=True)
        meta = dict(cfg=cfg, model=asdict(mcfg), params_non_embedding=n_params,
                    params_active_non_embedding=model.active_params(),
                    connection_params=n_conn, device=str(device), world_size=world,
                    tokens_per_step=tokens_per_step, initial_hashes=getattr(model,"init_hashes",{}), validation_windows=cfg["eval_batches"]*eval_batch_size, validation_seed=1234)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        log_f = open(os.path.join(run_dir, "log.jsonl"), "a" if resume else "w")
        print(f"[{cfg['run_name']}] params(non-emb)={n_params / 1e6:.2f}M "
              f"connection={n_conn / 1e3:.1f}K tokens/step={tokens_per_step} device={device}")
        if cfg["wandb_project"]:
            import wandb
            wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=meta)

    def log(rec):
        if not master:
            return
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if cfg["wandb_project"]:
            import wandb
            wandb.log(rec, step=rec["step"])

    @torch.no_grad()
    def evaluate(which="avg"):
        model.eval()
        model.eval_embed = which
        tot = torch.zeros((), device=device)
        for x, y in val_batches:
            with amp:
                _, loss = model(x, y)
            tot += loss.float()
        model.train()
        model.eval_embed = "avg"
        tot /= len(val_batches)
        if ddp:
            dist.all_reduce(tot, op=dist.ReduceOp.AVG)
        return tot.item()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    best_val, diverged = float("inf"), False
    t_train, tok_seen, n_timed, step0 = 0.0, 0, 0, 0
    if resume is not None:
        step0, tok_seen, t_train, n_timed, best_val = (
            resume["step"], resume["tok_seen"], resume["t_train"], resume["n_timed"],
            resume["best_val"])
        del resume

    def save_ckpt(step):
        if master:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            tmp = ckpt_path + ".tmp"
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                            tok_seen=tok_seen, t_train=t_train, n_timed=n_timed,
                            best_val=best_val), tmp)
            os.replace(tmp, ckpt_path)  # atomic: never leave a half-written latest.pt
            k = cfg["ckpt_keep_every"]
            if k and (step % k == 0 or step == cfg["max_steps"]):
                snap = os.path.join(os.path.dirname(ckpt_path), f"step_{step:06d}.pt")
                torch.save(dict(model={n: (v.to(torch.bfloat16) if v.is_floating_point() else v)
                                       for n, v in model.state_dict().items()}, step=step), snap)
        if ddp:
            dist.barrier()

    for step in range(step0, cfg["max_steps"] + 1):
        if ckpt_path and step > step0 and (step % cfg["ckpt_every"] == 0 or step == cfg["max_steps"]):
            save_ckpt(step)
        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            vl = evaluate()
            best_val = min(best_val, vl)
            from .model import MoE
            layers=[m for m in model.modules() if isinstance(m,MoE)]
            caps=[m.cap for m in layers]
            for m in layers:m.cap=m.E/m.k
            vl_nodrop=evaluate()
            for m,c in zip(layers,caps):m.cap=c
            log(dict(step=step, tokens=tok_seen, val_loss=vl, val_loss_no_drop=vl_nodrop))
            if master:
                print(f"step {step:6d} | val {vl:.4f}")
            if not math.isfinite(vl):
                diverged = True
                break
        if step == cfg["max_steps"]:
            break

        lr = lr_at(step, cfg)
        if cfg["tgr_sel_train"] == "mos_st":  # soft head until the cooldown, then the served hard head
            model.sel_hard = step >= int(cfg["max_steps"] * (1 - cfg["cooldown_frac"]))
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]
        sync()
        t0 = time.perf_counter()
        loss_acc = 0.0
        for micro in range(cfg["grad_accum"]):
            x, y = train.next()
            kw = {}
            if cfg["embed_mode"] == "drop_token_tied_fixed":
                kw["pos"] = train.last_pos
            ctx = (fwd.no_sync() if ddp and micro < cfg["grad_accum"] - 1
                   else contextlib.nullcontext())
            with ctx, amp:
                _, loss = fwd(x, y, **kw)
                loss = loss / cfg["grad_accum"]
            loss.backward()
            loss_acc += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        dt = time.perf_counter() - t0
        if step > step0 + 5:  # skip compile / warmup steps in the throughput figure
            t_train += dt
            n_timed += 1
        tok_seen += tokens_per_step

        if step % cfg["log_every"] == 0:
            lv = float(loss_acc)
            rec = dict(step=step, tokens=tok_seen, loss=lv, lr=lr, grad_norm=float(gnorm),
                       tok_per_s=tokens_per_step / dt)
            if device.type == "cuda":
                rec["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            log(rec)
            if master and step % (cfg["log_every"] * 10) == 0:
                print(f"step {step:6d} | loss {lv:.4f} | lr {lr:.2e} | gnorm {float(gnorm):.2f}"
                      f" | {tokens_per_step / dt / 1e3:.1f}K tok/s")
            if not math.isfinite(lv):
                diverged = True
                break

    extra_eval = {}
    if model.embed_b is not None and not diverged:
        # Diagnostic: replace both input and output with table A or B.
        # For router_mos_tok, avg retains the full mixture head.
        for which in ("A", "B"):
            extra_eval[f"val_loss_{which}_only"] = evaluate(which)
        if cfg["embed_mode"] in ("tgr_sel", "twin_pair"):  # final_val_loss is the hard (served) head; soft for reference
            model.sel_eval = "soft"
            extra_eval["val_loss_soft_head"] = evaluate()
            model.sel_eval = "hard"
        # train-set loss on fixed batches -> generalisation gap (matters when data repeats)
    if not diverged:
        tb = train.fixed_batches(cfg["eval_batches"])
        vb, val_batches = val_batches, tb
        extra_eval["train_fixed_loss"] = evaluate()
        val_batches = vb
    if master and cfg["save_embeddings"]:
        sd = model.state_dict()
        torch.save({k: v.detach().to("cpu", torch.float16) for k, v in sd.items()
                    if k.startswith(("embed", "lm_head"))},
                   os.path.join(run_dir, "embeddings.pt"))
    if master:
        summary = dict(
            run_name=cfg["run_name"], connection=cfg["connection"],
            connection_kwargs=cfg["connection_kwargs"], seed=cfg["seed"],
            params_non_embedding=n_params, connection_params=n_conn,
            final_val_loss=vl, final_val_loss_no_drop=vl_nodrop, best_val_loss=best_val, steps=step, tokens=tok_seen, diverged=diverged,
            tok_per_s=(tokens_per_step * n_timed / t_train if n_timed and t_train > 0 else None),
            peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9
                         if device.type == "cuda" else None),
            device=str(device), world_size=world, embed_mode=cfg["embed_mode"],
            train_tokens=cfg["train_tokens"], **extra_eval,
        )
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log_f.close()
        print(json.dumps(summary))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
