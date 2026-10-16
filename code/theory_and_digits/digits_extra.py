"""Extra digits diagnostics: gradient RMS (to test the warm-up-horizon formula), the fan-scaled v0 baseline
of Abuduweili & Liu (2025), and paired statistics of SADAM(0.01,0.01) vs Adam."""
import numpy as np, json, sys
import digits_sensitivity as d
from scipy import stats

out = {}
# 1) per-coordinate gradient RMS during the first epoch at the initial weights (Adam trajectory)
rms = []
for seed in range(5):
    P = d.init_params(np.random.default_rng(seed)); rng_b = np.random.default_rng(seed + 2000)
    perm = rng_b.permutation(len(d.Xtr)); G2 = None; n = 0
    for k in range(0, len(perm), d.BS):
        idx = perm[k:k + d.BS]; _, G = d.loss_grad(P, d.Xtr[idx], d.Ytr[idx])
        g2 = np.concatenate([g.ravel() ** 2 for g in G]); G2 = g2 if G2 is None else G2 + g2; n += 1
    rms.append(np.sqrt(G2 / n))
rms = np.concatenate(rms)
out["grad_rms_median"] = float(np.median(rms)); out["grad_rms_q25"] = float(np.quantile(rms, .25)); out["grad_rms_q75"] = float(np.quantile(rms, .75))
b2 = d.B2
for sv in [1e-4, 1e-3, 1e-2, 1e-1]:
    out[f"tstar_sv{sv:g}"] = float(np.log(1 + sv**2 / out["grad_rms_median"]**2) / (-np.log(b2)))
out["iters_per_epoch"] = int(np.ceil(len(d.Xtr) / d.BS))
# 2) fan-scaled v0 baseline, 20 seeds
v0 = [d.run(s, 0.0, 0.0, fan_scale=100.0) for s in range(20)]
out["v0rnd_acc"] = [r["final_acc"] for r in v0]; out["v0rnd_ep"] = [r["epochs_to_thresh"] for r in v0]
json.dump(out, open(sys.argv[1], "w"), indent=1)
print({k: v for k, v in out.items() if not isinstance(v, list)})
a = np.array(out["v0rnd_acc"]) * 100; e = np.array([x or 41 for x in out["v0rnd_ep"]])
print(f"v0rnd acc {a.mean():.2f}±{a.std(ddof=1):.2f} ep {e.mean():.1f}")
