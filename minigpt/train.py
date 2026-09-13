"""Train a tiny GPT from scratch on a synthetic cyclic sequence with Adam.

The model must learn the repeating alphabet "abcdefg..." — a task with a
verifiable ground truth for generation.
"""

import numpy as np

from minigpt.model import generate, init_params
from minigpt.backward import loss_and_grad


def _leaf_refs(d, out=None):
    if out is None:
        out = []
    if isinstance(d, dict):
        for v in d.values():
            _leaf_refs(v, out)
    elif isinstance(d, list):
        for v in d:
            _leaf_refs(v, out)
    else:
        out.append(d)
    return out


def _path_leaf_map(d, prefix=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _path_leaf_map(v, prefix + (k,))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _path_leaf_map(v, prefix + (i,))
    else:
        yield prefix, d


def adam_update(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    for path, p in _path_leaf_map(params):
        g = grads[path[0]]
        for k in path[1:]:
            g = g[k]
        mp, vp = m[path], v[path]
        mp *= b1
        mp += (1 - b1) * g
        vp *= b2
        vp += (1 - b2) * g ** 2
        mhat = mp / (1 - b1 ** t)
        vhat = vp / (1 - b2 ** t)
        p -= lr * mhat / (np.sqrt(vhat) + eps)


def main():
    rng = np.random.default_rng(0)
    cycle = np.arange(26)
    data = np.tile(cycle, 60)  # 1560 tokens of "abcdefg..."

    params = init_params(vocab_size=26, n_layer=2, n_head=4, n_embd=64,
                         block_size=65, rng=rng)  # 65 = T + 1 forward positions
    m = {p: np.zeros_like(a) for p, a in _path_leaf_map(params)}
    v = {p: np.zeros_like(a) for p, a in _path_leaf_map(params)}

    T = 64
    lr = 3e-3
    for step in range(1, 601):
        start = int(rng.integers(0, len(data) - T - 1))
        ids = data[start:start + T + 1].tolist()
        loss, grads = loss_and_grad(ids, params, n_head=4)
        adam_update(params, grads, m, v, step, lr=lr)
        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss:.4f}")

    prompt = cycle[:8].tolist()  # "abcdefgh"
    out = generate(prompt, params, n_head=4, n_tokens_to_generate=18,
                   temperature=0, rng=rng)
    expected = ((np.arange(8, 26)) % 26).tolist()  # "ijkl...xyz"
    print("generated:", "".join(chr(97 + i) for i in out))
    print("expected :", "".join(chr(97 + i) for i in expected))
    final_loss, _ = loss_and_grad(data[:T + 1].tolist(), params, n_head=4)
    print(f"final loss on training prefix: {final_loss:.4f}")
    assert out == expected, "model failed to learn the cycle"
    print("training passed")


if __name__ == "__main__":
    main()
