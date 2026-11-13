import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from hcfactory.data import TokenLoader
from hcfactory.model import GPT, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("onona_eval", ROOT / "eval/downstream.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def test_sequential_rank_partition_and_resume(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(129, dtype=np.uint16).tofile(path)
    global_loader = TokenLoader(path, 4, 8, order="sequential")
    ranks = [TokenLoader(path, 2, 8, order="sequential", rank=r, world=2) for r in range(2)]
    for _ in range(6):  # includes a pass boundary
        expected = global_loader.next()[0]
        actual = torch.cat([r.next()[0] for r in ranks])
        torch.testing.assert_close(actual, expected)
    resumed = TokenLoader(path, 2, 8, order="sequential", rank=0, world=2)
    resumed.cursor = ranks[0].cursor
    torch.testing.assert_close(resumed.next()[0], ranks[0].next()[0])


def test_validation_windows_independent_of_batch_grouping(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(4096, dtype=np.uint16).tofile(path)
    a = TokenLoader(path, 8, 16).fixed_batches(4)
    b = TokenLoader(path, 4, 16).fixed_batches(8)
    torch.testing.assert_close(torch.cat([x for x, _ in a]), torch.cat([x for x, _ in b]))


def test_continuation_scoring_matches_individual_sequences():
    torch.set_num_threads(1)
    model = GPT(ModelConfig(vocab_size=31, n_layer=2, n_head=2, d_model=32,
                            max_seq_len=16, embed_mode="router_mos_tok")).eval()
    rows = [(0, 0, [1, 3, 5, 7], 2, 2), (0, 1, [1, 3, 8], 1, 1)]
    scores, greedy = scorer.score_rows(model, rows, "cpu")
    for row, score, hit in zip(rows, scores, greedy):
        ids = torch.tensor([row[2]])
        logits = model(ids)[0][:, :-1]
        lp = F.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        assert score == pytest.approx(lp[:, -row[3]:].sum().item(), abs=1e-5)
        assert hit == bool((logits.argmax(-1)[:, -row[3]:] == ids[:, -row[3]:]).all())


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_loads_with_both_config_formats(tmp_path, legacy):
    config = ModelConfig(vocab_size=31, n_layer=1, n_head=2, d_model=32,
                         max_seq_len=16, embed_mode="router_mos_tok")
    model = GPT(config)
    saved = dict(model=asdict(config))
    if not legacy:
        saved["cfg"] = dict(embed_mode=config.embed_mode, seq_len=16, resid_dropout=0.0)
    (tmp_path / "config.json").write_text(json.dumps(saved))
    checkpoint = tmp_path / "latest.pt"
    torch.save(dict(model=model.state_dict(), step=3), checkpoint)
    restored, step, _ = scorer.build(str(tmp_path), checkpoint, "cpu")
    assert step == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize("scale,expected", [
    ("142m", 5699010560), ("330m", 13198688256), ("1b", 41313894400),
])
def test_config_token_budgets(scale, expected):
    cfg = yaml.safe_load((ROOT / f"configs/{scale}.yaml").read_text())
    assert 4 * cfg["batch_size"] * cfg["grad_accum"] * cfg["seq_len"] * cfg["max_steps"] == expected
    assert cfg["eval_batches"] * cfg["eval_batch_size"] == 1024
