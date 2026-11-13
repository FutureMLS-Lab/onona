import copy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from hcfactory.model import GPT, ModelConfig


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(7)


def make(**kwargs):
    cfg = ModelConfig(vocab_size=31, n_layer=2, n_head=2, d_model=32,
                      max_seq_len=16, embed_mode="router_mos_tok")
    return GPT(replace(cfg, **kwargs))


def test_probability_mixture_and_loss():
    model = make()
    x, y = torch.randint(0, 31, (2, 8)), torch.randint(0, 31, (2, 8))
    actual, loss = model(x, y)
    e = (model.embed(x) + model.embed_b(x)) / 2
    h = model.final_norm(model.connection(e, model.sublayers))
    g = model.head_router(F.rms_norm(e, (32,))).sigmoid()
    expected = g * F.linear(h, model.embed.weight).softmax(-1)
    expected += (1 - g) * F.linear(h, model.embed_b.weight).softmax(-1)
    torch.testing.assert_close(actual.exp(), expected)
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones_like(x, dtype=torch.float))
    torch.testing.assert_close(loss, -expected.gather(-1, y[..., None]).log().mean())
    loss.backward()
    for p in (model.embed.weight, model.embed_b.weight, model.head_router.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0


def test_swapping_tables_and_complementing_gate_preserves_distribution():
    model = make().eval()
    swapped = copy.deepcopy(model)
    with torch.no_grad():
        swapped.embed.weight.copy_(model.embed_b.weight)
        swapped.embed_b.weight.copy_(model.embed.weight)
        swapped.head_router.weight.neg_()
    x = torch.randint(0, 31, (2, 8))
    torch.testing.assert_close(model(x)[0], swapped(x)[0])


def test_equal_tables_collapse_to_standard_tied_head():
    model = make().eval()
    with torch.no_grad():
        model.embed_b.weight.copy_(model.embed.weight)
    x = torch.randint(0, 31, (2, 8))
    h = model.final_norm(model.connection(model.embed(x), model.sublayers))
    expected = F.log_softmax(F.linear(h, model.embed.weight), -1)
    torch.testing.assert_close(model(x)[0], expected)


def test_causality_and_train_eval_consistency():
    model = make()
    x = torch.randint(0, 31, (2, 8))
    train = model(x)[0]
    model.eval()
    torch.testing.assert_close(model(x)[0], train)
    altered = x.clone()
    altered[:, 5:] = (altered[:, 5:] + 1) % 31
    torch.testing.assert_close(model(x)[0][:, :5], model(altered)[0][:, :5])


def test_saturated_gate_stays_finite():
    model = make()
    with torch.no_grad():
        model.head_router.weight.mul_(100000)
    x = torch.randint(0, 31, (2, 8))
    logits, loss = model(x, x)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.head_router.weight.grad).all()


@pytest.mark.parametrize("options", [
    dict(embed_mode="single"),
    dict(embed_mode="single", tie_embeddings=False),
    dict(embed_mode="gin_only"),
    dict(embed_mode="mos2"),
    dict(connection="attnres", connection_kwargs={"block_size": 0}),
    dict(moe_experts=4, moe_top_k=2, moe_capacity=2.0),
])
def test_controls_and_architectures_backward(options):
    model = make(**options)
    x = torch.randint(0, 31, (2, 8))
    _, loss = model(x, x)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize("which", ["A", "B"])
def test_single_table_diagnostic_changes_both_ends(which):
    model = make().eval()
    model.eval_embed = which
    table = model.embed if which == "A" else model.embed_b
    x = torch.randint(0, 31, (2, 8))
    h = model.final_norm(model.connection(table(x), model.sublayers))
    torch.testing.assert_close(model(x)[0], F.linear(h, table.weight))
