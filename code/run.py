"""Train one configuration with the NumPy implementation and write a JSON record
(aggregated by npanalyze.py)."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import argparse, json, os, time
import numpy as np
import nn, data
from optim import Opt


def augment(xb, rng):
    """Random crop (pad 4) + horizontal flip, NHWC."""
    N = len(xb); xp = np.pad(xb, ((0, 0), (4, 4), (4, 4), (0, 0)))
    ox = rng.integers(0, 9, N); oy = rng.integers(0, 9, N); fl = rng.random(N) < 0.5
    out = np.empty_like(xb)
    for i in range(N):
        im = xp[i, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32]
        out[i] = im[:, ::-1] if fl[i] else im
    return out


def evaluate(model, X, Y, bs=1000):
    correct = 0; loss = 0.0
    for k in range(0, len(X), bs):
        out = model.fwd(X[k:k + bs], train=False)
        l, _ = nn.softmax_ce(out.copy(), Y[k:k + bs]); loss += l * len(out)
        correct += int((out.argmax(1) == Y[k:k + bs]).sum())
    return loss / len(X), correct / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset"); ap.add_argument("--model"); ap.add_argument("--opt")
    ap.add_argument("--seed", type=int); ap.add_argument("--epochs", type=int); ap.add_argument("--bs", type=int)
    ap.add_argument("--sigma_m", type=float, default=0.01); ap.add_argument("--sigma_v", type=float, default=0.01)
    ap.add_argument("--augment", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--log_steps", type=int, default=500); ap.add_argument("--out")
    a = ap.parse_args()
    name = a.tag or a.opt
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{a.dataset}_{a.model}_{name}_sm{a.sigma_m:g}_sv{a.sigma_v:g}_seed{a.seed}.json")
    if os.path.exists(fn): return
    Xtr, Ytr, Xte, Yte = data.mnist() if a.dataset == "mnist" else data.cifar()
    rng_w = np.random.default_rng(a.seed)
    model = {"mlp": nn.mlp, "smallcnn": nn.small_cnn, "resnet8": nn.resnet}[a.model](rng_w)
    opt_name = "adam" if name == "adam_warmup_alias" else a.opt
    opt = Opt(opt_name, model.params, lr=1e-3, seed=a.seed, sigma_m=a.sigma_m, sigma_v=a.sigma_v, log_steps=a.log_steps)
    rng_b = np.random.default_rng(20_000 + a.seed); rng_a = np.random.default_rng(30_000 + a.seed)
    rec = dict(args=dict(vars(a), opt=a.opt, tag=a.tag), device="cpu-numpy", epochs=[], first_iters=[])
    t_train = 0.0; it = 0
    for ep in range(1, a.epochs + 1):
        perm = rng_b.permutation(len(Xtr)); run_loss = 0.0; n = 0; t0 = time.perf_counter(); bad = False
        for k in range(0, len(perm), a.bs):
            idx = perm[k:k + a.bs]; xb = Xtr[idx]
            if a.augment: xb = augment(xb, rng_a)
            for p in model.params: p.grad[...] = 0
            out = model.fwd(xb, train=True); loss, d = nn.softmax_ce(out, Ytr[idx]); model.bwd(d)
            if not np.isfinite(loss): bad = True; break
            opt.step(); it += 1
            run_loss += loss * len(idx); n += len(idx)
            if it <= 200: rec["first_iters"].append(loss)
        t_train += time.perf_counter() - t0
        if bad: rec["diverged_at_epoch"] = ep; break
        te_loss, te_acc = evaluate(model, Xte, Yte)
        rec["epochs"].append(dict(epoch=ep, train_loss=run_loss / n, test_loss=te_loss, test_acc=te_acc, train_time_s=t_train))
        print(f"[{name} s{a.seed}] ep {ep} train {run_loss/n:.4f} acc {te_acc*100:.2f} t {t_train:.0f}s", flush=True)
    rec["step_log"] = opt.step_log
    json.dump(rec, open(fn, "w"))


if __name__ == "__main__":
    main()
