"""GPT-2 forward pass and sampling, numpy only.

Parameter layout mirrors GPT-2 checkpoints (and picoGPT) so trained or
downloaded weights stay interchangeable:

    params = {
        "wte": (V, C), "wpe": (T_max, C),
        "blocks": [{"ln_1": {"g","b"},
                    "attn": {"c_attn": {"w","b"}, "c_proj": {"w","b"}},
                    "ln_2": {"g","b"},
                    "mlp": {"c_fc": {"w","b"}, "c_proj": {"w","b"}}}],
        "ln_f": {"g","b"},
    }
"""

import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    """d gelu / d x for the tanh approximation used above."""
    s = np.sqrt(2.0 / np.pi)
    u = s * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * s * (1.0 + 3.0 * 0.044715 * x ** 2)


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


def linear(x, w, b):
    return x @ w + b


def split_heads(x, n_head):
    """(T, C) -> (n_head, T, C/n_head)"""
    T, C = x.shape
    return x.reshape(T, n_head, C // n_head).transpose(1, 0, 2)


def merge_heads(x):
    """(n_head, T, hs) -> (T, C)"""
    H, T, hs = x.shape
    return x.transpose(1, 0, 2).reshape(T, H * hs)


def attention_forward(x, c_attn, c_proj, n_head):
    """Causal multi-head attention; returns output and a cache for backward."""
    T, C = x.shape
    qkv = linear(x, **c_attn)
    q, k, v = np.split(qkv, 3, axis=-1)
    qh, kh, vh = split_heads(q, n_head), split_heads(k, n_head), split_heads(v, n_head)
    hs = qh.shape[-1]
    scores = qh @ kh.transpose(0, 2, 1) / np.sqrt(hs)  # (H, T, T)
    causal = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(causal, scores, -1e10)
    probs = softmax(scores)
    heads_out = probs @ vh
    merged = merge_heads(heads_out)
    out = linear(merged, **c_proj)
    cache = {"x": x, "q": q, "k": k, "v": v, "probs": probs, "merged": merged}
    return out, cache


def mlp_forward(x, mlp):
    h = linear(x, **mlp["c_fc"])
    act = gelu(h)
    out = linear(act, **mlp["c_proj"])
    return out, {"x": x, "h": h}


def block_forward(x, block, n_head):
    attn_out, attn_cache = attention_forward(
        layer_norm(x, **block["ln_1"]),
        block["attn"]["c_attn"], block["attn"]["c_proj"], n_head,
    )
    x1 = x + attn_out
    mlp_out, mlp_cache = mlp_forward(layer_norm(x1, **block["ln_2"]), block["mlp"])
    x2 = x1 + mlp_out
    cache = {"x": x, "attn": attn_cache, "attn_out": attn_out,
             "x1": x1, "mlp": mlp_cache, "mlp_out": mlp_out}
    return x2, cache


def forward(ids, params, n_head, with_cache=False):
    """ids: list/array of token ids (single sequence). Returns logits (T, V)."""
    ids = np.asarray(ids)
    T = len(ids)
    x = params["wte"][ids] + params["wpe"][:T]
    block_caches = []
    for block in params["blocks"]:
        x, cache = block_forward(x, block, n_head)
        block_caches.append(cache)
    ln_f_out = layer_norm(x, **params["ln_f"])
    logits = ln_f_out @ params["wte"].T
    if with_cache:
        cache = {"ids": ids, "blocks": block_caches,
                 "ln_f_in": x, "ln_f_out": ln_f_out, "n_head": n_head}
        return logits, cache
    return logits


def next_token_distribution(logits, temperature=1.0, top_k=None, top_p=None):
    """Turn last-position logits into a probability vector over the vocab."""
    if temperature <= 0:
        p = np.zeros_like(logits, dtype=np.float64)
        p[np.argmax(logits)] = 1.0
        return p
    x = logits.astype(np.float64) / temperature
    if top_k is not None:
        k = min(int(top_k), x.size)
        thresh = np.sort(x)[-k]
        x[x < thresh] = -np.inf
    if top_p is not None:
        order = np.argsort(x)[::-1]
        ps = softmax(x[order])
        keep = np.cumsum(ps) - ps < top_p  # always keep the first token
        x[order[~keep]] = -np.inf
    return softmax(x)


def generate(ids, params, n_head, n_tokens_to_generate,
             temperature=1.0, top_k=None, top_p=None, rng=None):
    rng = rng or np.random.default_rng()
    ids = list(ids)
    out = []
    for _ in range(n_tokens_to_generate):
        logits = forward(ids, params, n_head)
        p = next_token_distribution(logits[-1], temperature, top_k, top_p)
        if temperature <= 0:
            nxt = int(np.argmax(p))
        else:
            nxt = int(rng.choice(len(p), p=p))
        ids.append(nxt)
        out.append(nxt)
    return out


def init_params(vocab_size, n_layer, n_head, n_embd, block_size, rng=None):
    """GPT-2-style init for building small models in tests and training."""
    rng = rng or np.random.default_rng()
    std = 0.02

    def ln():
        return {"g": np.ones(n_embd), "b": np.zeros(n_embd)}

    def lin(n_in, n_out):
        return {"w": rng.normal(0, std, (n_in, n_out)), "b": np.zeros(n_out)}

    blocks = []
    for _ in range(n_layer):
        blocks.append({
            "ln_1": ln(),
            "attn": {"c_attn": lin(n_embd, 3 * n_embd),
                     "c_proj": lin(n_embd, n_embd)},
            "ln_2": ln(),
            "mlp": {"c_fc": lin(n_embd, 4 * n_embd),
                    "c_proj": lin(4 * n_embd, n_embd)},
        })
    return {"wte": rng.normal(0, std, (vocab_size, n_embd)),
            "wpe": rng.normal(0, std, (block_size, n_embd)),
            "blocks": blocks,
            "ln_f": ln()}
