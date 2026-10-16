"""Finite-difference check of every layer's backward pass (float64, central differences).
Reports the maximum relative error over sampled parameter and input coordinates for each layer type."""
import numpy as np
import nn

nn.F32 = np.float64  # run the check in double precision


def rel(a, b):
    return abs(a - b) / max(abs(a) + abs(b), 1e-8)


def check(name, model, x, y, n_coord=40, h=1e-5, seed=0):
    rng = np.random.default_rng(seed)

    def loss():
        l, _ = nn.softmax_ce(model.fwd(x.copy(), train=True), y)
        return l

    for p in model.params: p.grad[...] = 0
    out = model.fwd(x.copy(), train=True); _, d = nn.softmax_ce(out, y); dx = model.bwd(d)
    errs = []
    for p in model.params:
        g = p.grad.copy()
        for _ in range(min(n_coord, p.data.size)):
            i = tuple(rng.integers(0, s) for s in p.data.shape); old = p.data[i]
            p.data[i] = old + h; lp = loss(); p.data[i] = old - h; lm = loss(); p.data[i] = old
            errs.append(rel(g[i], (lp - lm) / (2 * h)))
    if dx is not None:
        for _ in range(n_coord):
            i = tuple(rng.integers(0, s) for s in x.shape); old = x[i]
            x[i] = old + h; lp = loss(); x[i] = old - h; lm = loss(); x[i] = old
            errs.append(rel(dx[i], (lp - lm) / (2 * h)))
    m = max(errs); print(f"{name:<38s} max rel. error {m:.1e}"); return m


if __name__ == "__main__":
    r = np.random.default_rng(1)
    y = r.integers(0, 10, 4)
    res = {}
    res["Linear + ReLU (MLP)"] = check("Linear + ReLU (MLP)", nn.Seq([nn.Linear(12, 16, r), nn.ReLU(), nn.Linear(16, 10, r)]),
                                       r.normal(size=(4, 12)), y)
    res["Conv 3x3 s1 + bias, MaxPool, Flatten"] = check(
        "Conv 3x3 s1 + bias, MaxPool, Flatten",
        nn.Seq([nn.Conv(3, 4, r), nn.ReLU(), nn.MaxPool2(), nn.Flatten(), nn.Linear(4 * 4 * 4, 10, r)]), r.normal(size=(4, 8, 8, 3)), y)
    res["Conv 3x3 s2 / 1x1 s2, BatchNorm"] = check(
        "Conv 3x3 s2 / 1x1 s2, BatchNorm",
        nn.Seq([nn.Conv(3, 4, r, 3, 2, bias=False), nn.BN(4), nn.ReLU(), nn.Conv(4, 4, r, 1, 2, bias=False), nn.BN(4),
                nn.GAP(), nn.Linear(4, 10, r)]), r.normal(size=(4, 8, 8, 3)), y)
    res["Residual block + GAP (ResNet)"] = check("Residual block + GAP (ResNet)", nn.resnet(r, widths=(4, 8, 8)),
                                                 r.normal(size=(4, 8, 8, 3)), y)
    worst = max(res.values()); print(f"worst case: {worst:.1e}")
