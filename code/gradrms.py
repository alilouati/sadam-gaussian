"""Median per-coordinate gradient RMS during the first epoch (Adam trajectory), used for the warm-up horizon t*."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import numpy as np, json, sys, nn, data
from optim import Opt
from run import augment
out = {}
for ds, model, bs, aug in [("mnist", "mlp", 256, 0), ("cifar10", "smallcnn", 256, 0), ("cifar10", "resnet8", 128, 1)]:
    Xtr, Ytr, _, _ = data.mnist() if ds == "mnist" else data.cifar()
    m = {"mlp": nn.mlp, "smallcnn": nn.small_cnn, "resnet8": nn.resnet}[model](np.random.default_rng(0))
    opt = Opt("adam", m.params); rng = np.random.default_rng(20000); rng_a = np.random.default_rng(30000)
    perm = rng.permutation(len(Xtr)); G2 = [np.zeros_like(p.data, dtype=np.float64) for p in m.params]; n = 0
    for k in range(0, len(perm), bs):
        idx = perm[k:k + bs]; xb = Xtr[idx]
        if aug: xb = augment(xb, rng_a)
        for p in m.params: p.grad[...] = 0
        o = m.fwd(xb); _, d = nn.softmax_ce(o, Ytr[idx]); m.bwd(d)
        for g, p in zip(G2, m.params): g += p.grad.astype(np.float64) ** 2
        opt.step(); n += 1
    rms = np.concatenate([np.sqrt(g / n).ravel() for g in G2])
    out[model] = dict(median=float(np.median(rms)), q25=float(np.quantile(rms, .25)), q75=float(np.quantile(rms, .75)),
                      iters_per_epoch=n, tstar_sv001=float(np.log(1 + 1e-4 / np.median(rms) ** 2) / -np.log(0.999)))
    print(model, out[model], flush=True)
json.dump(out, open(_os.path.join(ROOT, "results", "gradrms.json"), "w"), indent=1)
