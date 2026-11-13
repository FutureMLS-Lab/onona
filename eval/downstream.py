"""Zero-shot downstream evaluation for hcfactory checkpoints (onóna vs Standard).

Adapted from LoopResidual/eval/downstream.py (lm-evaluation-harness conventions):
  * zero-shot multiple choice: score each candidate continuation by its summed token
    log-likelihood given the context; acc = argmax of the raw sum, acc_norm = argmax of
    the sum divided by the continuation's byte length (headline for hellaswag / arc /
    openbookqa); piqa, boolq, winogrande report acc. winogrande scores only the shared
    suffix after the blank, under the two filled-in prefixes.
  * lambada_openai: last-word prediction; acc = every last-word token is the greedy
    argmax, plus perplexity of the last word.
  * sciq (harness prompt: support, question, four answers) reports acc; boolq also reports
    acc_balanced (mean per-class recall, free of the yes/no prior); mmlu is 5-shot from each
    subject's dev split with letter answers, acc. Prompts longer than max_seq_len keep their
    last max_seq_len tokens, which for mmlu can drop the header and the first shots.
  * wikitext (--task wikitext, not in the default list): wikitext-2-raw-v1 test, one
    document at a time, disjoint rolling windows of max_seq_len targets (the last window
    keeps a full context), each document starting after an end-of-text token, as in
    lm-evaluation-harness; reports word perplexity, token perplexity and bits per byte.
Only continuation tokens are scored; context tokens are masked. Inputs longer than the
model's max_seq_len keep their last max_seq_len tokens.

For router_mos_tok, eval_embed="avg" uses averaged input embeddings and the full
mixture of the two tied softmaxes. It does not merge the output head.

Usage from the repository root:
  python eval/downstream.py --run onona=runs/onona:checkpoints/onona/latest.pt \
      --moe_capacity 4.0 --out evaluation.json
"""

import argparse
import contextlib
from pathlib import Path
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hcfactory.model import GPT, ModelConfig  # noqa: E402

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "boolq", "sciq", "lambada_openai", "mmlu"]
HEADLINE = {"hellaswag": "acc_norm", "arc_easy": "acc_norm", "arc_challenge": "acc_norm",
            "openbookqa": "acc_norm", "piqa": "acc", "winogrande": "acc", "boolq": "acc",
            "sciq": "acc", "lambada_openai": "acc", "mmlu": "acc"}
MMLU_SHOTS = 5


# --------------------------------------------------------------------------- data
def _norm(t):
    return " ".join(t.strip().split())


def load_task(name):
    """-> list of (context or [contexts], [continuations], gold_index)."""
    from datasets import load_dataset
    items = []
    if name == "hellaswag":
        for r in load_dataset("Rowan/hellaswag", split="validation"):
            ctx = _norm(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"])
            items.append((ctx, [" " + _norm(e) for e in r["endings"]], int(r["label"])))
    elif name in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if name == "arc_easy" else "ARC-Challenge"
        for r in load_dataset("allenai/ai2_arc", cfg, split="test"):
            labels = r["choices"]["label"]
            if r["answerKey"] in labels:
                items.append(("Question: " + _norm(r["question"]) + "\nAnswer:",
                              [" " + _norm(t) for t in r["choices"]["text"]],
                              labels.index(r["answerKey"])))
    elif name == "openbookqa":
        for r in load_dataset("allenai/openbookqa", split="test"):
            labels = r["choices"]["label"]
            if r["answerKey"] in labels:
                items.append((_norm(r["question_stem"]),
                              [" " + _norm(t) for t in r["choices"]["text"]],
                              labels.index(r["answerKey"])))
    elif name == "piqa":
        for r in load_dataset("baber/piqa", split="validation"):
            items.append(("Question: " + _norm(r["goal"]) + "\nAnswer:",
                          [" " + _norm(r["sol1"]), " " + _norm(r["sol2"])], int(r["label"])))
    elif name == "winogrande":
        for r in load_dataset("allenai/winogrande", "winogrande_xl", split="validation"):
            i = r["sentence"].index("_")
            suffix = r["sentence"][i + 1:]
            ctxs = [_norm(r["sentence"][:i] + r[f"option{k}"]) for k in (1, 2)]
            items.append((ctxs, [suffix, suffix], int(r["answer"]) - 1))
    elif name == "boolq":
        for r in load_dataset("google/boolq", split="validation"):
            ctx = r["passage"].strip() + "\nQuestion: " + r["question"].strip() + "?\nAnswer:"
            items.append((ctx, [" no", " yes"], int(bool(r["answer"]))))
    elif name == "lambada_openai":
        for r in load_dataset("EleutherAI/lambada_openai", "default", split="test"):
            ctx, last = r["text"].rsplit(" ", 1)
            items.append((ctx, [" " + last], 0))
    elif name == "sciq":  # lm-evaluation-harness sciq: support + question, correct answer last
        for r in load_dataset("allenai/sciq", split="test"):
            ctx = r["support"].lstrip() + "\nQuestion: " + r["question"] + "\nAnswer:"
            conts = [" " + r[k] for k in ("distractor1", "distractor2", "distractor3", "correct_answer")]
            items.append((ctx, conts, 3))
    elif name == "mmlu":  # harness mmlu, MMLU_SHOTS-shot from each subject's dev split, letter answers
        letters = "ABCD"

        def fmt(r):
            return (r["question"].strip() + "\n" + "".join(f"{l}. {c}\n" for l, c in zip(letters, r["choices"]))
                    + "Answer:")
        dev = {}
        for r in load_dataset("cais/mmlu", "all", split="dev"):
            dev.setdefault(r["subject"], []).append(r)
        for r in load_dataset("cais/mmlu", "all", split="test"):
            s = r["subject"]
            ctx = f"The following are multiple choice questions (with answers) about {s.replace('_', ' ')}.\n\n"
            ctx += "".join(fmt(d) + f" {letters[d['answer']]}\n\n" for d in dev[s][:MMLU_SHOTS])
            items.append((ctx + fmt(r), [" " + l for l in letters], int(r["answer"])))
    else:
        raise ValueError(name)
    return items


# ---------------------------------------------------------------------- inference
def encode_rows(enc, items, max_len):
    """-> list of (item, choice, ids, n_cont, nbytes); one row per (context, choice)."""
    rows = []
    for i, (ctx, conts, _gold) in enumerate(items):
        for j, cont in enumerate(conts):
            c = ctx[j] if isinstance(ctx, list) else ctx
            cid, tid = enc.encode_ordinary(c), enc.encode_ordinary(cont)
            ids = (cid + tid)[-max_len:]
            n_cont = min(len(tid), len(ids) - 1)
            if n_cont <= 0:
                ids, n_cont = (ids or [enc.eot_token]) + [enc.eot_token], 1
            rows.append((i, j, ids, n_cont, max(len(cont.encode("utf-8")), 1)))
    return rows


@torch.no_grad()
def score_rows(model, rows, device, batch_tokens=16384, max_batch=128):
    """Summed continuation log-prob and all-greedy flag per row, in input order.
    Rows are length-sorted and right-padded; only continuation positions are read."""
    out, greedy = [0.0] * len(rows), [False] * len(rows)
    order = sorted(range(len(rows)), key=lambda k: len(rows[k][2]))
    b = 0
    while b < len(order):
        L = len(rows[order[b]][2])
        idx = order[b:b + max(1, min(max_batch, batch_tokens // max(L, 1)))]
        L = max(len(rows[k][2]) for k in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long, device=device)
        for r, k in enumerate(idx):
            x[r, :len(rows[k][2])] = torch.tensor(rows[k][2], device=device)
        with (torch.autocast("cuda", dtype=torch.bfloat16)
              if torch.device(device).type == "cuda" else contextlib.nullcontext()):
            logits, _ = model(x)
        lg = logits[:, :-1].float()
        tok_lp = lg.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, -1)
        am = lg.argmax(-1)
        for r, k in enumerate(idx):
            n_ids, n_cont = len(rows[k][2]), rows[k][3]
            sl = slice(n_ids - 1 - n_cont, n_ids - 1)
            out[k] = tok_lp[r, sl].sum().item()
            greedy[k] = bool((am[r, sl] == x[r, 1:][sl]).all().item())
        b += len(idx)
    return out, greedy


def evaluate(model, enc, task, items, device, max_len):
    rows = encode_rows(enc, items, max_len)
    scores, greedy = score_rows(model, rows, device)
    per = {}
    for (i, j, ids, nc, nb), sc, g in zip(rows, scores, greedy):
        per.setdefault(i, {})[j] = (sc, nb, g, nc)
    norm = HEADLINE[task] == "acc_norm"
    hit, hit_norm, margin = [], [], []
    nll_sum = ntok = 0
    for i, (_ctx, conts, gold) in enumerate(items):
        d = per[i]
        if task == "lambada_openai":
            sc, _nb, g, nc = d[0]
            hit.append(int(g)); hit_norm.append(int(g)); margin.append(sc)
            nll_sum -= sc; ntok += nc
            continue
        raw = {k: d[k][0] for k in d}
        byt = {k: d[k][0] / d[k][1] for k in d}
        hit.append(int(max(raw, key=raw.get) == gold))
        hit_norm.append(int(max(byt, key=byt.get) == gold))
        s = byt if norm else raw
        margin.append(s[gold] - max(v for k, v in s.items() if k != gold))
    n = len(items)
    acc, acc_norm = sum(hit) / n, sum(hit_norm) / n
    head = acc_norm if norm else acc
    res = dict(n=n, metric=HEADLINE[task], acc=acc, acc_norm=acc_norm, headline=head,
               stderr=math.sqrt(head * (1 - head) / n),
               per_item_hit=(hit_norm if norm else hit), per_item_margin=margin)
    if task == "lambada_openai":
        res["last_word_ppl"] = math.exp(nll_sum / max(ntok, 1))
    if task == "boolq":  # mean per-class recall: removes the model's yes/no prior
        gold = [g for _c, _k, g in items]
        rec = [sum(h for h, g in zip(hit, gold) if g == c) / max(gold.count(c), 1) for c in (0, 1)]
        res["acc_balanced"] = sum(rec) / 2
    return res


def build(run_dir, ckpt_path, device):
    saved = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = saved.get("cfg", dict(embed_mode=saved["model"]["embed_mode"],
                                 resid_dropout=saved["model"].get("resid_dropout", 0.0),
                                 seq_len=saved["model"]["max_seq_len"]))
    model = GPT(ModelConfig(**saved["model"]))   # the exact ModelConfig train.py built
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), ck.get("step"), cfg


@torch.no_grad()
def wikitext_ppl(model, enc, device, max_len, batch=16):
    """Rolling log-likelihood of every wikitext-2 test document -> word/token PPL and BPB."""
    import re
    from datasets import load_dataset
    docs = [r["page"] for r in load_dataset("EleutherAI/wikitext_document_level",
                                            "wikitext-2-raw-v1", split="test")]
    wins, nll, ntok, nwords, nbytes = [], 0.0, 0, 0, 0
    for doc in docs:
        ids = [enc.eot_token] + enc.encode_ordinary(doc)
        nwords += len(re.split(r"\s+", doc))
        nbytes += len(doc.encode("utf-8"))
        ntok += len(ids) - 1
        for s0 in range(1, len(ids), max_len):        # targets ids[s0:e]
            e = min(s0 + max_len, len(ids))
            a = max(0, e - 1 - max_len)               # model reads ids[a:e-1]
            wins.append((ids[a:e], e - s0))
    wins.sort(key=lambda w: len(w[0]))
    for b in range(0, len(wins), batch):
        chunk = wins[b:b + batch]
        L = max(len(w) for w, _ in chunk)
        x = torch.zeros(len(chunk), L, dtype=torch.long, device=device)
        for r, (w, _) in enumerate(chunk):
            x[r, :len(w)] = torch.tensor(w, device=device)
        with (torch.autocast("cuda", dtype=torch.bfloat16)
              if torch.device(device).type == "cuda" else contextlib.nullcontext()):
            logits, _ = model(x[:, :-1])
        lg = logits.float()
        tok_lp = lg.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, -1)
        for r, (w, n) in enumerate(chunk):
            nll -= tok_lp[r, len(w) - 1 - n:len(w) - 1].sum().item()
    return dict(n=len(docs), metric="word_ppl", word_ppl=math.exp(nll / nwords),
                token_ppl=math.exp(nll / ntok), bpb=nll / math.log(2) / nbytes,
                nll=nll, ntok=ntok, nwords=nwords, nbytes=nbytes)


# ---------------------------------------------------------------------- comparison
def paired_ci(a, b, n_boot=4000, seed=0):
    """Mean of (b - a) over items with a paired bootstrap 95% CI."""
    rng = random.Random(seed)
    n = len(a)
    d = [y - x for x, y in zip(a, b)]
    mean = sum(d) / n
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def compare(res, base, other, tasks):
    print(f"\n=== paired: {other} - {base} (95% bootstrap CI over items) ===")
    print(f"{'task':<16}{'metric':<10}{'d acc (pp)':>22}{'d margin':>30}")
    for t in tasks:
        ra, rb = res[base]["tasks"][t], res[other]["tasks"][t]
        m, lo, hi = paired_ci(ra["per_item_hit"], rb["per_item_hit"])
        mm, mlo, mhi = paired_ci(ra["per_item_margin"], rb["per_item_margin"])
        print(f"{t:<16}{ra['metric']:<10}{100*m:>+8.2f} [{100*lo:+.2f},{100*hi:+.2f}]"
              f"{mm:>+12.4f} [{mlo:+.4f},{mhi:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="name=run_dir:ckpt_path (run_dir holds config.json)")
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--moe_capacity", type=float, default=4.0,
                    help="override the MoE capacity factor at eval; experts/top_k never drops a "
                         "token, so padding and batch composition cannot change the scores")
    args = ap.parse_args()
    tasks = args.task or TASKS
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    loaded = {}
    for t in tasks:
        if t == "wikitext":
            continue
        items = load_task(t)
        loaded[t] = items[:args.limit] if args.limit else items
        print(f"  {t:<16} n={len(loaded[t])}", flush=True)

    res = {}
    for spec in args.run:
        name, paths = spec.split("=", 1)
        run_dir, ckpt = paths.split(":", 1)
        model, step, cfg = build(run_dir, ckpt, device)
        if args.moe_capacity is not None:
            for m in model.modules():
                if hasattr(m, "cap") and hasattr(m, "E"):
                    m.cap = args.moe_capacity
        print(f"\n=== {name} (step {step}, embed_mode={cfg['embed_mode']}, "
              f"resid_dropout={cfg['resid_dropout']}) ===", flush=True)
        res[name] = dict(step=step, run_dir=run_dir, ckpt=ckpt, tasks={})
        for t in tasks:
            t0 = time.time()
            if t == "wikitext":
                r = wikitext_ppl(model, enc, device, cfg["seq_len"])
                res[name]["tasks"][t] = r
                print(f"  wikitext        word ppl {r['word_ppl']:.2f}  token ppl "
                      f"{r['token_ppl']:.2f}  bpb {r['bpb']:.4f}  [{time.time()-t0:.0f}s]",
                      flush=True)
                continue
            r = evaluate(model, enc, t, loaded[t], device, cfg["seq_len"])
            res[name]["tasks"][t] = r
            extra = f" ppl {r['last_word_ppl']:.1f}" if "last_word_ppl" in r else ""
            extra += f" balanced {100*r['acc_balanced']:.2f}" if "acc_balanced" in r else ""
            print(f"  {t:<16}{r['metric']:<9}{100*r['headline']:6.2f}%  (acc {100*r['acc']:.2f}"
                  f" / norm {100*r['acc_norm']:.2f}, n={r['n']}, se={100*r['stderr']:.2f})"
                  f"{extra}  [{time.time()-t0:.0f}s]", flush=True)
        del model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"\nwrote {args.out}")
    names = list(res)
    tasks = [t for t in tasks if t != "wikitext"]
    if len(names) > 1 and tasks:
        hdr = f"{'run':<22}" + "".join(f"{t[:10]:>12}" for t in tasks) + f"{'MEAN':>9}"
        print("\n" + hdr)
        for nm in names:
            v = [res[nm]["tasks"][t]["headline"] for t in tasks]
            print(f"{nm:<22}" + "".join(f"{100*x:>12.2f}" for x in v) + f"{100*sum(v)/len(v):>9.2f}")
        for other in names[1:]:
            compare(res, names[0], other, tasks)


if __name__ == "__main__":
    main()
