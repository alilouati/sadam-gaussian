import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
D = _os.path.join(ROOT, "results", "digits")
_os.makedirs(_os.path.join(ROOT, "outputs", "tables"), exist_ok=True); _os.makedirs(_os.path.join(ROOT, "outputs", "figs"), exist_ok=True)
import json, numpy as np, itertools
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open(_os.path.join(D, "digits_results.json"))); X = json.load(open(_os.path.join(D, "digits_extra.json")))
G = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]; lab = ["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
key = lambda a, b: f"{a:g}_{b:g}"
A = np.zeros((5, 5)); S = np.zeros((5, 5)); E = np.zeros((5, 5)); NR = np.zeros((5, 5), int)
for i, sm in enumerate(G):
    for j, sv in enumerate(G):
        rs = R[key(sm, sv)]; a = np.array([r["final_acc"] for r in rs]) * 100
        e = np.array([r["epochs_to_thresh"] if r["epochs_to_thresh"] else np.nan for r in rs], float)
        A[i, j], S[i, j] = a.mean(), a.std(ddof=1); E[i, j] = np.nanmean(e) if np.isfinite(e).any() else np.nan
        NR[i, j] = np.isnan(e).sum()
# table
with open(_os.path.join(ROOT, "outputs", "tables", "tab_digits.tex"), "w") as f:
    for i, sm in enumerate(G):
        cells = []
        for j in range(5):
            c = f"{A[i,j]:.2f}$\\pm${S[i,j]:.2f} ({'--' if np.isnan(E[i,j]) else f'{E[i,j]:.1f}'}"
            c += (f"$^{{{20-NR[i,j]}/20}}$" if 0 < NR[i, j] < 20 else "") + ")"
            cells.append(c)
        f.write(f"{lab[i]} & " + " & ".join(cells) + " \\\\\n")
# paired tests vs Adam
base = np.array([r["final_acc"] for r in R["0_0"]]) * 100
be = np.array([r["epochs_to_thresh"] for r in R["0_0"]], float)
out = {}
for k in ["0.01_0.01", "0_0.01", "0.0001_0.0001", "0.001_0.001", "0.01_0.001"]:
    a = np.array([r["final_acc"] for r in R[k]]) * 100; d = a - base
    e = np.array([r["epochs_to_thresh"] or 41 for r in R[k]], float)
    out[k] = dict(diff=d.mean(), sd=d.std(ddof=1), dz=d.mean() / d.std(ddof=1), p_t=stats.ttest_rel(a, base).pvalue,
                  p_w=stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0, ep=e.mean(), ep_diff=(e - be).mean(),
                  p_ep=stats.wilcoxon(e - be).pvalue if np.any(e != be) else 1.0)
    print(k, {kk: round(float(v), 4) for kk, v in out[k].items()})
va = np.array(X["v0rnd_acc"]) * 100; d = va - base
print("v0rnd", va.mean(), va.std(ddof=1), d.mean(), stats.ttest_rel(va, base).pvalue)
json.dump({k: {kk: float(v) for kk, v in vv.items()} for kk2, vv in [(0, 0)] for k, vv in out.items()}, open(_os.path.join(D, "digits_stats.json"), "w"), indent=1)
# figure 1: heatmaps
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for a_, M, ttl, cm in ((ax[0], A, "Final test accuracy (%)", "viridis"), (ax[1], E, "Epochs to training loss < 0.05", "magma_r")):
    Mm = np.where(np.isnan(M), 41, M)
    im = a_.imshow(Mm, origin="lower", cmap=cm, vmin=(90 if a_ is ax[0] else None), vmax=(97.5 if a_ is ax[0] else 41))
    a_.set_xticks(range(5)); a_.set_xticklabels(lab, fontsize=7); a_.set_yticks(range(5)); a_.set_yticklabels(lab, fontsize=7)
    a_.set_xlabel(r"$\sigma_v$"); a_.set_ylabel(r"$\sigma_m$"); a_.set_title(ttl, fontsize=8)
    for i, j in itertools.product(range(5), range(5)):
        txt = (f"{M[i,j]:.1f}" if not np.isnan(M[i, j]) else ">40")
        a_.text(j, i, txt, ha="center", va="center", fontsize=6, color="w" if (a_ is ax[0] and M[i,j] < 95) or (a_ is ax[1] and Mm[i,j] > 20) else "k")
    fig.colorbar(im, ax=a_, fraction=0.046)
fig.tight_layout(); plt.close(fig)  # final figure: code/replot.py
# figure 2: step diagnostics on the diagonal + sigma_v = 0 row
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for (sm, sv), ls in zip([(0, 0), (1e-3, 1e-3), (1e-2, 1e-2), (1e-1, 1e-1), (1e-2, 0)], ["k-", "C0-", "C1-", "C2-", "C3--"]):
    st = np.array([r["step_stats"] for r in R[key(sm, sv)]])  # seeds x points x [t, med, frac>1, max]
    t = st[0, :, 0]
    name = "Adam" if sm == 0 and sv == 0 else rf"$\sigma_m$={sm:g}, $\sigma_v$={sv:g}"
    ax[0].plot(t, st[:, :, 1].mean(0), ls, label=name, lw=1.2)
    ax[1].plot(t, st[:, :, 2].mean(0) * 100, ls, lw=1.2)
for a_ in ax: a_.set_xscale("log"); a_.set_xlabel("Iteration $t$")
ax[0].set_yscale("log"); ax[0].set_ylabel(r"median $|\Delta\theta_t|/\alpha$"); ax[1].set_ylabel(r"coordinates with $|\Delta\theta_t|>\alpha$ (%)")
ax[0].legend(fontsize=6); fig.tight_layout(); plt.close(fig)  # final figure: code/replot.py
for (sm, sv) in [(0,0),(1e-2,1e-2),(1e-1,1e-1),(1e-2,0)]:
    st = np.array([r["step_stats"] for r in R[key(sm, sv)]])
    print(sm, sv, "t=1 med %.3f frac>1 %.3f max %.1f" % tuple(st[:,0,1:].mean(0)), " t=400 med %.4f" % st[:,-1,1].mean(), "maxmax", st[:,0,3].max())
print("A", np.round(A,2)); print("E", np.round(E,1)); print(NR)
