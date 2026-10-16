"""Small-scale, fully reproducible sensitivity study of SADAM's initialization scales.

Pure NumPy (the cloud workspace has no deep-learning framework).  Dataset: the UCI
handwritten-digits set bundled with scikit-learn (1,797 8x8 images, 10 classes).
Model: MLP 64-128-64-10 (ReLU, He init).  Paired design: for a given seed the
network weights and the mini-batch order are identical across all optimizer
configurations, so differences are attributable to the optimizer state only.
"""
import numpy as np, json, sys, itertools
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
Ytr = np.eye(10)[ytr]
SIZES = [64, 128, 64, 10]
EPOCHS, BS, LR = int(sys.argv[2]) if len(sys.argv) > 2 else 40, 32, 1e-3
B1, B2, EPS = 0.9, 0.999, 1e-8
THRESH = 0.05


def init_params(rng):
    P = []
    for a, b in zip(SIZES[:-1], SIZES[1:]):
        P += [rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)]
    return P


def forward(P, x):
    hs = [x]
    h = x
    for i in range(0, len(P) - 2, 2):
        h = np.maximum(h @ P[i] + P[i + 1], 0); hs.append(h)
    logits = h @ P[-2] + P[-1]
    return hs, logits


def loss_grad(P, x, Y):
    hs, z = forward(P, x)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    loss = -np.mean(np.sum(Y * np.log(p + 1e-12), 1))
    d = (p - Y) / len(x)
    G = [None] * len(P)
    for li in range(len(P) // 2 - 1, -1, -1):
        G[2 * li] = hs[li].T @ d; G[2 * li + 1] = d.sum(0)
        if li > 0:
            d = (d @ P[2 * li].T) * (hs[li] > 0)
    return loss, G


def accuracy(P, x, yy):
    return float(np.mean(forward(P, x)[1].argmax(1) == yy))


def full_loss(P):
    return loss_grad(P, Xtr, Ytr)[0]


def run(seed, sm, sv, opt_seed_offset=1000, fan_scale=None):
    rng_w = np.random.default_rng(seed)
    P = init_params(rng_w)
    rng_o = np.random.default_rng(seed + opt_seed_offset)  # optimizer-state RNG
    M = [rng_o.normal(0, sm, p.shape) if sm > 0 else np.zeros_like(p) for p in P]
    if fan_scale is not None:  # Abuduweili & Liu (2025) v0,rnd: fan-scaled chi-square on weight matrices only
        V = [rng_o.normal(0, np.sqrt(fan_scale / sum(p.shape)), p.shape) ** 2 if p.ndim == 2 else np.zeros_like(p)
             for p in P]
    else:
        V = [rng_o.normal(0, sv, p.shape) ** 2 if sv > 0 else np.zeros_like(p) for p in P]
    rng_b = np.random.default_rng(seed + 2000)  # batch order (shared across configs)
    t = 0; hist = []; first_steps = []; eff = []
    for ep in range(EPOCHS):
        perm = rng_b.permutation(len(Xtr))
        for k in range(0, len(perm), BS):
            idx = perm[k:k + BS]
            _, G = loss_grad(P, Xtr[idx], Ytr[idx]); t += 1
            ratios = []
            for i in range(len(P)):
                M[i] = B1 * M[i] + (1 - B1) * G[i]
                V[i] = B2 * V[i] + (1 - B2) * G[i] ** 2
                mh = M[i] / (1 - B1 ** t); vh = V[i] / (1 - B2 ** t)
                step = LR * mh / (np.sqrt(vh) + EPS)
                P[i] -= step
                ratios.append(np.abs(step).ravel() / LR)
            if t <= 400 and (t <= 20 or t % 20 == 0):
                r = np.concatenate(ratios)
                eff.append([t, float(np.median(r)), float(np.mean(r > 1)), float(r.max())])
        hist.append([full_loss(P), accuracy(P, Xte, yte)])
    hist = np.array(hist)
    hit = np.where(hist[:, 0] < THRESH)[0]
    return dict(final_acc=float(hist[-1, 1]), final_loss=float(hist[-1, 0]),
                epochs_to_thresh=int(hit[0] + 1) if len(hit) else None,
                loss_curve=hist[:, 0].tolist(), acc_curve=hist[:, 1].tolist(), step_stats=eff)


if __name__ == "__main__":
    grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    seeds = range(int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    res = {}
    for sm, sv in itertools.product(grid, grid):
        key = f"{sm:g}_{sv:g}"
        res[key] = [run(s, sm, sv) for s in seeds]
        a = [r["final_acc"] for r in res[key]]
        e = [r["epochs_to_thresh"] or EPOCHS + 1 for r in res[key]]
        print(key, f"acc {np.mean(a)*100:.2f}±{np.std(a, ddof=1)*100:.2f}  ep2thr {np.mean(e):.1f}", flush=True)
    json.dump(res, open(sys.argv[1], "w"))
