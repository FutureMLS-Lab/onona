# Reproduction notes

## Source and scope

The model comes from the archived 40-TPP experiment source and is unchanged in this release. Its SHA-256 is recorded in [`source-manifest.json`](source-manifest.json). It is also identical to the model in the archived 1B MoE training snapshot. The trainer comes from that later snapshot; packaging changes add an explicit device override and an evaluation batch size independent of the training batch size. The evaluator retains the paper's task formatting and scoring, with portable imports, a CPU option and support for both archived configuration formats.

These are usable research implementations and configurations, not an assertion that every historical run is bitwise reproducible from this release. Earlier experiments used multiple launchers and runtime versions. Exact data snapshots, pretrained weights and archived per-run environments are not distributed here.

## Training recipe

All three scale configs use RMSNorm, SwiGLU, grouped-query attention, QK normalization, RoPE theta 1,000,000, GPT-2 tokenization and a padded vocabulary of 50,304. The input and output share vocabulary tables. onóna initializes A and B independently with standard deviation 0.02 and averages them for the input. There is no embedding or residual dropout.

AdamW uses betas (0.9, 0.95), epsilon 1e-8, matrix weight decay 0.1 and gradient clipping at 1.0. The peak learning rate is 1e-3 for 142M/330M and 6e-4 for 1B. WSD uses 1,000/2,000 warmup steps and linearly decays to 0.1 times the peak rate over the last 20% of training. CUDA configs enable BF16 autocast and `torch.compile`, retaining FP32 model parameters.

The 142M/330M configurations use the approximately 7.18B-token pool. The 1B configuration uses the separate approximately 23.9B-token pool. `data_order=sequential` creates one seed-0 permutation of nonoverlapping windows and repeats that permutation after pool exhaustion. These budgets are matched to each scale's Dense Standard baseline; MoE total parameter count is larger.

Validation uses seed 1234, 1,024 windows, batch size 8 and the training MoE capacity of 1.25. The trainer additionally records validation with capacity `experts/top_k`, but `final_val_loss` remains the original capacity-1.25 score. A-only/B-only diagnostics change both ends of the model.

Model initialization consumes random numbers differently across embedding variants. Equal seed alone does not imply identical body weights. Some archived runs used a separate matched-initialization launcher; the release's common trainer uses the original `GPT` initialization. Treat paired-initialization experiments separately when reproducing them.

## Evaluation protocol

The default eight-task average weights each task equally. HellaSwag and ARC report `acc_norm` divided by UTF-8 continuation bytes. PIQA, BoolQ, SciQ and MMLU use unnormalized multiple-choice accuracy. LAMBADA requires every token of the last word to be the greedy prediction. MMLU uses five development examples per subject, then left-truncates to the model's context limit. HellaSwag preprocessing is intentionally the simplified whitespace-normalization procedure used in the experiments.

WikiText evaluation uses the document-level WikiText-2 raw test set and rolling windows. It reports word PPL, token PPL and bits per byte separately. Results from different tokenization or word-count conventions should not be compared as if identical.

Downstream and WikiText evaluation set MoE capacity to 4.0 to prevent token dropping for 8 experts/top-2. The training capacity remains 1.25. Only load checkpoints you trust, because the archived checkpoint format uses Python pickle through `torch.load`.

## Runtime and validation

The CPU release checks use PyTorch 2.8.0. They do not validate CUDA compilation, the FLA kernel, large-scale training convergence or the paper's numerical results. The experiments encountered MoE instability on PyTorch 2.7.1; use and record a newer runtime for new MoE runs. FLA and CUDA version choices can affect kernel compatibility and numerics. Record the exact PyTorch, CUDA, Triton and FLA versions alongside any reproduction.
