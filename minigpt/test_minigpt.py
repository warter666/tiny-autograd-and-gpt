"""Verification for minigpt: torch forward parity, finite-difference gradient
checks, and sampling smoke tests."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from minigpt.model import forward, init_params, next_token_distribution, generate
from minigpt.backward import loss_and_grad


def build_small(seed=0):
    rng = np.random.default_rng(seed)
    return init_params(vocab_size=11, n_layer=2, n_head=2, n_embd=16,
                       block_size=16, rng=rng)


def test_torch_forward_parity():
    params = build_small()
    ids = [1, 5, 3, 9, 2]

    def t(x):
        return torch.tensor(np.asarray(x), dtype=torch.float64)

    T = len(ids)
    x = t(params["wte"])[torch.tensor(ids)] + t(params["wpe"])[:T]
    for block in params["blocks"]:
        h = F.layer_norm(x, (x.shape[-1],), t(block["ln_1"]["g"]),
                         t(block["ln_1"]["b"]), 1e-5)
        qkv = h @ t(block["attn"]["c_attn"]["w"]) + t(block["attn"]["c_attn"]["b"])
        q, k, v = qkv.chunk(3, dim=-1)
        H = 2
        q, k, v = (z.view(T, H, -1).transpose(0, 1) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~mask, -1e10).softmax(-1)
        y = (att @ v).transpose(0, 1).reshape(T, -1)
        y = y @ t(block["attn"]["c_proj"]["w"]) + t(block["attn"]["c_proj"]["b"])
        x = x + y
        h2 = F.layer_norm(x, (x.shape[-1],), t(block["ln_2"]["g"]),
                          t(block["ln_2"]["b"]), 1e-5)
        h3 = h2 @ t(block["mlp"]["c_fc"]["w"]) + t(block["mlp"]["c_fc"]["b"])
        g = 0.5 * h3 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (h3 + 0.044715 * h3 ** 3)))
        h4 = g @ t(block["mlp"]["c_proj"]["w"]) + t(block["mlp"]["c_proj"]["b"])
        x = x + h4
    xf = F.layer_norm(x, (x.shape[-1],), t(params["ln_f"]["g"]),
                      t(params["ln_f"]["b"]), 1e-5)
    logits_torch = (xf @ t(params["wte"]).T).numpy()

    logits_np = forward(ids, params, n_head=2)
    assert np.allclose(logits_np, logits_torch, atol=1e-10), \
        np.max(np.abs(logits_np - logits_torch))


def _leaf_refs(d, out=None):
    if out is None:
        out = []
    if isinstance(d, dict):
        for k, v in d.items():
            _leaf_refs(v, out)
    elif isinstance(d, list):
        for v in d:
            _leaf_refs(v, out)
    else:
        out.append(d)
    return out


def test_gradient_check():
    params = build_small(seed=1)
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 11, size=10).tolist()

    loss0, grads = loss_and_grad(ids, params, n_head=2)

    leaves = _leaf_refs(params)
    garrs = _leaf_refs(grads)
    n_checked = 0
    for p_arr, g_arr in zip(leaves, garrs):
        flat_idx = rng.choice(p_arr.size, size=min(5, p_arr.size), replace=False)
        for idx in flat_idx:
            multi = np.unravel_index(idx, p_arr.shape)
            eps = 1e-5
            orig = p_arr[multi]
            p_arr[multi] = orig + eps
            lp = loss_and_grad(ids, params, n_head=2)[0]
            p_arr[multi] = orig - eps
            lm = loss_and_grad(ids, params, n_head=2)[0]
            p_arr[multi] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(abs(num), abs(g_arr[multi]), 1e-8)
            rel = abs(num - g_arr[multi]) / denom
            assert rel < 1e-4, (rel, num, g_arr[multi])
            n_checked += 1
    print(f"gradient check passed on {n_checked} entries, loss={loss0:.4f}")


def test_sampling():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=50)

    p = next_token_distribution(logits, top_k=1)
    assert p[np.argmax(logits)] == 1.0 and p.sum() == 1.0

    p = next_token_distribution(logits, top_p=1e-9)
    order = np.argsort(logits)
    # only the highest-logit token may survive
    assert p[order[-1]] > 0 and np.count_nonzero(p) == 1

    p = next_token_distribution(logits, temperature=0)
    assert p[np.argmax(logits)] == 1.0

    params = build_small(seed=4)
    ids = generate([1, 2], params, n_head=2, n_tokens_to_generate=5,
                   temperature=0.9, top_k=5, rng=rng)
    assert len(ids) == 5 and all(0 <= i < 11 for i in ids)


if __name__ == "__main__":
    test_torch_forward_parity()
    print("torch forward parity passed")
    test_gradient_check()
    test_sampling()
    print("all minigpt tests passed")
