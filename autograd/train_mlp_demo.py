"""Train a small MLP on a synthetic spiral dataset to demo the engine end to end."""

import numpy as np

from autograd import MLP, Value


def make_spiral(n_per_class=100, noise=0.15):
    pts, labels = [], []
    for k in range(2):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = np.linspace(k * 3.14, k * 3.14 + 2.5, n_per_class) + np.random.randn(n_per_class) * noise
        pts.append(np.c_[r * np.sin(t), r * np.cos(t)])
        labels.append(np.full(n_per_class, k))
    return np.vstack(pts), np.concatenate(labels)


def main():
    np.random.seed(42)
    X, y = make_spiral()
    model = MLP(2, [16, 16, 1])
    print(model)

    lr = 0.5
    for step in range(200):
        # hinge-like loss on batch (labels mapped to ±1)
        scores = [model([Value(float(a)) for a in row]) for row in X]
        targets = [1.0 if yi == 1 else -1.0 for yi in y]
        losses = [(1 + -ti * si).relu() for ti, si in zip(targets, scores)]
        data_loss = sum(losses) / len(losses)
        reg = 1e-4 * sum(p * p for p in model.parameters())
        loss = data_loss + reg
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        acc = float(np.mean(np.array([s.data > 0 for s in scores]) == (y == 1)))
        if step % 40 == 0 or step == 199:
            print(f"step {step:3d}  loss {loss.data:.4f}  acc {acc:.2%}")
    assert acc > 0.9, "expected >90% accuracy on the toy spiral"
    print("training demo passed")


if __name__ == "__main__":
    main()
