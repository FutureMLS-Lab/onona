# Data preparation

The loader expects `train.bin` and `val.bin` as flat native-endian `uint16` arrays. GPT-2 has 50,257 token IDs; the model pads the table to 50,304. Each text document is encoded using `tiktoken`'s `gpt2` `encode_ordinary`, followed by EOT (50256). Keep validation documents separate from training documents.

## Local text

For JSONL files with a `text` field:

```bash
python data_prep/prepare_text.py --input train.jsonl --out data/custom/train.bin
python data_prep/prepare_text.py --input val.jsonl --out data/custom/val.bin
```

The tool refuses to overwrite an existing output. It records token/document counts and a SHA-256 hash next to each output. This path supports your own corpora; it does not reconstruct the paper's ClimbMix selection.

## ClimbMix

The paper uses NVIDIA's [ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix), distributed under **CC BY-NC 4.0**. Follow NVIDIA's dataset card for obtaining the data and converting its released GPT-2 token sequences back to text if needed. Dataset licensing is separate from this repository's MIT code license.

The archived preprocessing utility accepts a Hugging Face Arrow IPC stream with `text` and `cluster_id` columns:

```bash
mkdir -p data/climbmix
python data_prep/tok_climb.py --out data/climbmix/train.bin \
  --max-tokens 1000000000 --workers 8 < text-shard.arrow
```

It appends whole encoded documents and stops before exceeding `--max-tokens` **for that invocation**. Start with a new destination and process each selected shard exactly once. Do not accidentally rerun an append command on the same data.

The historical pools were prepared from a local 3,584-shard text conversion. The approximately 7.18B-token pool selected 72 shards beginning at index 2 with stride 5; the approximately 23.9B-token pool used 240 shards from that sequence. These identify the local conversion, not a stable shard index into today's upstream dataset. Validation used a separate approximately 10M-token pool. The exact converted shards, pool hashes and a public download of those historical token bins are not part of this release, so a fresh upstream preparation need not reproduce their exact ordering or scores.

For a small installation check, `python scripts/make_smoke_data.py` creates synthetic token IDs without downloading any corpus. Never interpret results on those tokens as language-model performance.
