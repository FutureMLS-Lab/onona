# onóna: Twin Embeddings with Tied Unembeddings

[![arXiv](https://img.shields.io/badge/arXiv-2611.10556-b31b1b.svg)](https://arxiv.org/abs/2611.10556)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/FutureMLS-Lab/onona/actions/workflows/tests.yml/badge.svg)](https://github.com/FutureMLS-Lab/onona/actions/workflows/tests.yml)

Research code for [**onóna: Twin Embeddings with Tied Unembeddings**](https://arxiv.org/abs/2611.10556).

[Paper](https://arxiv.org/abs/2611.10556) · [Project page](https://futuremls.org/projects/onona.html) · [Code](https://github.com/FutureMLS-Lab/onona)

Cheng Luo · Xinnan Dai · Zhongzhu Zhou · Zefan Cai · Ruijie Zhu · Tianyu Zhang · Junjie Hu

onóna uses two vocabulary tables, **A** and **B**, for both input embeddings and output prediction. The input uses their average. The output mixes their two softmax distributions using a gate that depends only on the current input token.

```text
M       = (A + B) / 2
x_t     = M[token_t]
h_t     = Transformer(x_≤t)
g_t     = sigmoid(wᵀ RMSNorm(x_t))
p(·|≤t) = g_t softmax(A h_t) + (1 − g_t) softmax(B h_t)
```

The paper implementation is **`embed_mode=router_mos_tok`** in [`hcfactory/model.py`](hcfactory/model.py). It uses no dropout. The same model file retains earlier experimental modes for checkpoint compatibility; those modes are not the paper's onóna method.

## Installation

Use Python 3.10+ and install a PyTorch build appropriate for your hardware before installing this package.

```bash
git clone https://github.com/FutureMLS-Lab/onona.git
cd onona
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,data,eval]'
```

Dense, MoE and Attention Residual models use PyTorch. Gated DeltaNet also requires [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) and a supported GPU backend. Follow its installation instructions for your PyTorch/CUDA combination. The GDN path calls `fla.ops.gated_delta_rule.chunk_gated_delta_rule`.

## Quick start

Run a small CPU training job with synthetic tokens to check the installation. This is a functional example, not a language-model benchmark.

```bash
python scripts/make_smoke_data.py
python -m hcfactory.train --config configs/tiny.yaml
pytest -q
```

The trainer writes `config.json`, `log.jsonl` and `summary.json` under `runs/tiny/`, and a resumable checkpoint under `checkpoints/tiny/latest.pt`.

## Training

Training data consists of flat GPT-2 `uint16` token streams named `train.bin` and `val.bin`. See [data preparation](docs/data.md) for the format, ClimbMix preprocessing and the limits of reconstructing the historical data pools.

The following configurations encode the paper's 40-token-per-parameter budgets. Each uses **four GPUs**, per-GPU batch size 8 and sequence length 1,024.

| Config | Layers × width | Gradient accumulation | Steps | Global tokens/step |
|---|---:|---:|---:|---:|
| [`142m.yaml`](configs/142m.yaml) | 16 × 768 | 8 | 21,740 | 262,144 |
| [`330m.yaml`](configs/330m.yaml) | 24 × 1,024 | 8 | 50,349 | 262,144 |
| [`1b.yaml`](configs/1b.yaml) | 20 × 2,048 | 16 | 78,800 | 524,288 |

```bash
# onóna, Dense, 142M baseline scale
torchrun --standalone --nproc_per_node=4 -m hcfactory.train \
  --config configs/142m.yaml data_dir=data/climbmix_7b run_name=142m_onona

# Standard tied baseline, same data and token budget
torchrun --standalone --nproc_per_node=4 -m hcfactory.train \
  --config configs/142m.yaml data_dir=data/climbmix_7b \
  embed_mode=single run_name=142m_standard
```

Any configuration value can be overridden as `key=value`. Choose a distinct `run_name` for every experiment. Runs with an existing checkpoint resume automatically. If you change GPU count or batch size, adjust gradient accumulation to preserve the global token budget. Validation uses 1,024 fixed windows in batches of 8, independently of the training batch size.

Architecture overrides apply to either embedding method:

| Architecture | Overrides |
|---|---|
| Dense | none |
| Gated DeltaNet hybrid | `mixer=gdn_hybrid hybrid_attn_every=4 gdn_head_dim=128` |
| MoE | `moe_experts=8 moe_top_k=2 moe_capacity=1.25 moe_aux_coef=0.01` |
| Attention Residual | `connection=attnres connection_kwargs.block_size=0` |

Paper controls use `embed_mode=gin_only`, `embed_mode=mos2`, or `embed_mode=single tie_embeddings=false`. The 330M deeper baseline uses `embed_mode=single n_layer=28` with the 330M token budget. See [reproduction notes](docs/reproduction.md) before comparing results.

## Evaluation

```bash
python eval/downstream.py \
  --run onona=runs/142m_onona:checkpoints/142m_onona/latest.pt \
  --moe_capacity 4.0 --out downstream.json

python eval/downstream.py \
  --run onona=runs/142m_onona:checkpoints/142m_onona/latest.pt \
  --task wikitext --moe_capacity 4.0 --out wikitext.json
```

The default tasks are HellaSwag, ARC-Easy, ARC-Challenge, PIQA, BoolQ, SciQ, LAMBADA and five-shot MMLU. The scorer follows the paper's protocol, including byte-normalized `acc_norm` and left-truncated MMLU prompts. It is **not an exact replacement for lm-evaluation-harness**. CUDA evaluation uses BF16; `--device cpu` is available for small checks.

For `router_mos_tok`, evaluation keeps the full mixture head. `eval_embed="A"` or `"B"` replaces both input and output tables for the single-table diagnostic.

## Inference cost

The reference forward pass computes both vocabulary projections. Gate values can be precomputed per input token. At temperature 1 without top-k/top-p filtering, exact ancestral sampling can first sample a mixture component and then compute only that component's projection. Greedy decoding, likelihood scoring and general temperature/filtering transformations still require the full mixture. This repository does not yet implement a fused or cached serving path.

## Release contents

- Model implementations, training, token loading and paper-scale configurations.
- Data preparation utilities and downstream/WikiText evaluation.
- CPU tests for the mixture distribution, gradients, causality, controls and data loading.

Training data, pretrained checkpoints and historical run artifacts are not included. The release tests validate functionality; they do not rerun the paper's GPU training experiments.

## Citation

```bibtex
@article{luo2026onona,
  title={onóna: Twin Embeddings with Tied Unembeddings},
  author={Luo, Cheng and Dai, Xinnan and Zhou, Zhongzhu and Cai, Zefan and
          Zhu, Ruijie and Zhang, Tianyu and Hu, Junjie},
  journal={arXiv preprint arXiv:2611.10556},
  year={2026},
  url={https://arxiv.org/abs/2611.10556}
}
```

## License and acknowledgments

Code is released under the [MIT License](LICENSE). The model/trainer build on Cheng Luo's MIT-licensed `hyper-connection-factory` code; the scorer retains its LoopResidual evaluation provenance. Gated DeltaNet uses the separately distributed Flash Linear Attention package. ClimbMix and downstream datasets retain their own licenses; the code's MIT license does not relicense those datasets.
