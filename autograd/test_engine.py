"""Gradient verification for the Value engine: torch parity + finite differences."""

import math

import torch

from autograd.engine import Value


def test_torch_parity_basic():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()

    xt = torch.Tensor([-4.0]).double()
    xt.requires_grad = True
    zt = 2 * xt + 2 + xt
    qt = zt.relu() + zt * xt
    ht = (zt * zt).relu()
    yt = ht + qt + qt * xt
    yt.backward()

    assert y.data == yt.data.item()
    assert x.grad == xt.grad.item()


def test_torch_parity_more_ops():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f
    g.backward()

    at = torch.Tensor([-4.0]).double()
    bt = torch.Tensor([2.0]).double()
    at.requires_grad = True
    bt.requires_grad = True
    ct = at + bt
    dt = at * bt + bt ** 3
    ct = ct + ct + 1
    ct = ct + 1 + ct + (-at)
    dt = dt + dt * 2 + (bt + at).relu()
    dt = dt + 3 * dt + (bt - at).relu()
    et = ct - dt
    ft = et ** 2
    gt = ft / 2.0
    gt = gt + 10.0 / ft
    gt.backward()

    tol = 1e-6
    assert abs(g.data - gt.data.item()) < tol
    assert abs(a.grad - at.grad.item()) < tol
    assert abs(b.grad - bt.grad.item()) < tol


def _finite_diff(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_exp_log_tanh_grads():
    for x0 in (0.5, 1.7, -0.9):
        for name, fn in (
            ("exp", lambda v: v.exp() * 2.0),
            ("log", lambda v: (v * v + 0.5).log() * 3.0),
            ("tanh", lambda v: v.tanh() ** 2 + 1.0),
            ("mix", lambda v: ((v.exp() + 1.0).log() * v.tanh()).relu()),
        ):
            v = Value(x0)
            out = fn(v)
            out.backward()
            num = _finite_diff(lambda x: fn(Value(x)).data, x0)
            assert abs(v.grad - num) < 1e-4, (name, x0, v.grad, num)


def test_deep_graph_no_recursion_error():
    # 50k-node chain: the reference recursive topo sort would blow the stack here.
    x = Value(0.5)
    y = x
    for _ in range(50_000):
        y = y * 1.0001 + 0.001
    y.backward()
    # large intermediate values (~1.5e3) limit central-difference precision here,
    # so use a bigger eps and a looser tolerance than the other grad checks
    num = _finite_diff(lambda t: _chain_value(t, 50_000).data, 0.5, eps=1e-3)
    assert abs(x.grad - num) < 1e-4


def _chain_value(x0, n):
    y = Value(x0)
    for _ in range(n):
        y = y * 1.0001 + 0.001
    return y


if __name__ == "__main__":
    test_torch_parity_basic()
    test_torch_parity_more_ops()
    test_exp_log_tanh_grads()
    test_deep_graph_no_recursion_error()
    print("all engine tests passed")
