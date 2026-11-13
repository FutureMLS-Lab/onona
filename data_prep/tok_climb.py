"""Tokenize one ClimbMix arrow shard read from stdin (HF save_to_disk IPC stream) into flat GPT-2 uint16 tokens.

Same convention as hcfactory/prepare_data.py: tiktoken gpt2 encode_ordinary, EOT (50256) after every document.
Appends at most --max-tokens tokens (whole documents, in shard order) to --out.
Usage: python data_prep/tok_climb.py --out train.bin --max-tokens N < shard.arrow
"""
import argparse, collections, sys
from multiprocessing import Pool
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


def enc_batch(texts):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return [np.asarray(t + [enc.eot_token], dtype=np.uint16) for t in enc.encode_ordinary_batch(texts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--stats", action="store_true", help="only print row/cluster stats, write nothing")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    tbl = ipc.open_stream(pa.input_stream(sys.stdin.buffer)).read_all()
    texts = tbl.column("text").to_pylist()
    cl = tbl.column("cluster_id").to_pylist()
    if a.stats:
        c = collections.Counter(cl)
        first = collections.Counter(cl[:2000])
        print(f"rows {len(texts)} clusters {len(c)} top {c.most_common(3)} first2000_clusters {len(first)} chars {sum(map(len, texts))}")
        return
    n, used = 0, 0
    with open(a.out, "ab") as f, Pool(a.workers) as pool:
        step = 4000
        for i in range(0, len(texts), step):
            parts = pool.map(enc_batch, [texts[j:j + 250] for j in range(i, min(i + step, len(texts)), 250)])
            for docs in parts:
                for d in docs:
                    if n + len(d) > a.max_tokens:
                        print(f"tokens {n} docs {used} clusters {len(set(cl[:used]))}")
                        return
                    d.tofile(f); n += len(d); used += 1
    print(f"tokens {n} docs {used} clusters {len(set(cl[:used]))} (shard exhausted)")


if __name__ == "__main__":
    main()
