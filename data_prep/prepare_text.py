"""Convert local JSONL documents into a flat GPT-2 uint16 token stream."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder = tiktoken.get_encoding("gpt2")
    digest = hashlib.sha256()
    documents = tokens = 0
    with args.input.open(encoding="utf-8") as source, args.out.open("xb") as dest:
        for line in source:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            ids = encoder.encode_ordinary(text) + [encoder.eot_token]
            raw = np.asarray(ids, dtype=np.uint16).tobytes()
            dest.write(raw)
            digest.update(raw)
            documents += 1
            tokens += len(ids)
    metadata = dict(tokenizer="gpt2", dtype="uint16", documents=documents,
                    tokens=tokens, sha256=digest.hexdigest())
    args.out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
