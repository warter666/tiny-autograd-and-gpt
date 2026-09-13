"""Hand-derived backward pass for minigpt's forward pass.

`loss_and_grad(ids, params, n_head)` computes next-token cross-entropy loss and
gradients with the same nested structure as `params`. Weight tying is handled by
accumulating both the output-projection and token-embedding gradients into `wte`.
"""

import numpy as np

from .model import forward, gelu, gelu_grad, split_heads, merge_heads


def zeros_like_params(params):
    if isinstance(params, dict):
        return {k: zeros_like_params(v) for k, v in params.items()}
    if isinstance(params, list):
        return [zeros_like_params(v) for v in params]
    return np.zeros_like(params)


def _layer_norm_backward(x, g, eps, dy):
    """y = g * xhat + b, xhat = (x - mean) / sqrt(var + eps); returns (dx, dg, db)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    sigma = np.sqrt(var + eps)
    xhat = (x - mean) / sigma
    dg = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
    db = np.sum(dy, axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    dx = (dxhat - np.mean(dxhat, axis=-1, keepdims=True)
          - xhat * np.mean(dxhat * xhat, axis=-1, keepdims=True)) / sigma
    return dx, dg, db


def _linear_backward(x, w, dy):
    """y = x @ w + b; returns (dx, dw, db)."""
    dx = dy @ w.T
    dw = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    db = dy.reshape(-1, dy.shape[-1]).sum(axis=0)
    return dx, dw, db


def _attention_backward(cache, c_attn, c_proj, n_head, dout):
    x = cache["x"]
    dmerged, dc_proj_w, dc_proj_b = _linear_backward(cache["merged"], c_proj["w"], dout)

    qh = split_heads(cache["q"], n_head)
    kh = split_heads(cache["k"], n_head)
    vh = split_heads(cache["v"], n_head)
    hs = qh.shape[-1]
    probs = cache["probs"]
    T = x.shape[0]
    causal = np.tril(np.ones((T, T), dtype=bool))

    dheads_out = split_heads(dmerged, n_head)
    dprobs = dheads_out @ vh.transpose(0, 2, 1)
    dvh = probs.transpose(0, 2, 1) @ dheads_out
    dscores = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))
    dscores = np.where(causal, dscores, 0.0)
    dqh = dscores @ kh / np.sqrt(hs)
    dkh = dscores.transpose(0, 2, 1) @ qh / np.sqrt(hs)

    dqkv = np.concatenate([merge_heads(dqh), merge_heads(dkh), merge_heads(dvh)], axis=-1)
    dx, dc_attn_w, dc_attn_b = _linear_backward(x, c_attn["w"], dqkv)
    return dx, {"c_attn": {"w": dc_attn_w, "b": dc_attn_b},
                "c_proj": {"w": dc_proj_w, "b": dc_proj_b}}


def _mlp_backward(cache, mlp, dout):
    act = gelu(cache["h"])
    dact, dc_proj_w, dc_proj_b = _linear_backward(act, mlp["c_proj"]["w"], dout)
    dh = dact * gelu_grad(cache["h"])
    dx, dc_fc_w, dc_fc_b = _linear_backward(cache["x"], mlp["c_fc"]["w"], dh)
    return dx, {"c_fc": {"w": dc_fc_w, "b": dc_fc_b},
                "c_proj": {"w": dc_proj_w, "b": dc_proj_b}}


def _block_backward(cache, block, n_head, dx):
    """dx: grad wrt block output. Returns grad wrt block input and param grads."""
    grads = zeros_like_params(block)

    # x2 = x1 + mlp(layer_norm(x1))
    d_ln2_out, mlp_grads = _mlp_backward(cache["mlp"], block["mlp"], dx)
    grads["mlp"] = mlp_grads  # zeros_like start, safe to assign
    dx1_via_ln2, dg2, db2 = _layer_norm_backward(
        cache["x1"], block["ln_2"]["g"], 1e-5, d_ln2_out)
    grads["ln_2"]["g"] += dg2
    grads["ln_2"]["b"] += db2
    dx1 = dx + dx1_via_ln2  # direct residual path + path through ln_2

    # x1 = x + attn(layer_norm(x))
    d_ln1_out, attn_grads = _attention_backward(
        cache["attn"], block["attn"]["c_attn"], block["attn"]["c_proj"], n_head, dx1)
    grads["attn"] = attn_grads
    dx_via_ln1, dg1, db1 = _layer_norm_backward(
        cache["x"], block["ln_1"]["g"], 1e-5, d_ln1_out)
    grads["ln_1"]["g"] += dg1
    grads["ln_1"]["b"] += db1
    dx_block = dx1 + dx_via_ln1
    return dx_block, grads


def backward(dlogits, ids, params, cache):
    """Accumulate parameter gradients given dL/dlogits of shape (T, V)."""
    grads = zeros_like_params(params)
    n_head = cache["n_head"]
    T = len(ids)

    # tied output projection: logits = ln_f_out @ wte.T
    grads["wte"] += dlogits.T @ cache["ln_f_out"]
    dx = dlogits @ params["wte"]

    dx, dg, db = _layer_norm_backward(
        cache["ln_f_in"], params["ln_f"]["g"], 1e-5, dx)
    grads["ln_f"]["g"] += dg
    grads["ln_f"]["b"] += db

    for i in reversed(range(len(params["blocks"]))):
        dx, block_grads = _block_backward(cache["blocks"][i], params["blocks"][i],
                                          n_head, dx)
        grads["blocks"][i] = block_grads

    # embeddings: x = wte[ids] + wpe[:T]
    np.add.at(grads["wte"], cache["ids"], dx)
    grads["wpe"][:T] += dx
    return grads


def loss_and_grad(ids, params, n_head):
    """Cross-entropy on next-token prediction; returns (loss, grads)."""
    logits, cache = forward(ids, params, n_head, with_cache=True)
    targets = np.asarray(ids[1:])
    logit_rows = logits[:-1]  # predict ids[1:] from positions [:-1]
    shifted = logit_rows - np.max(logit_rows, axis=-1, keepdims=True)
    logprobs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    n = len(targets)
    loss = float(-np.mean(logprobs[np.arange(n), targets]))

    probs = np.exp(logprobs)
    dlogits = np.zeros_like(logits)
    dlogits[:-1] = probs
    dlogits[np.arange(n), targets] -= 1.0
    dlogits[:-1] /= n
    grads = backward(dlogits, ids, params, cache)
    return loss, grads
