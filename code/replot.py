"""Publication-quality re-rendering of all data figures (no overlapping legends, consistent styles)."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import json, glob, itertools, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mfig
_orig_savefig = _mfig.Figure.savefig


def _savefig_pdf_and_png(self, fname, *a, **k):
    """Save the PDF used in the paper and a PNG preview next to it (for notebooks and the README)."""
    _orig_savefig(self, fname, *a, **k)
    if str(fname).endswith(".pdf"):
        _orig_savefig(self, str(fname)[:-4] + ".png", dpi=130)


_mfig.Figure.savefig = _savefig_pdf_and_png
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import npanalyze as A

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
FG = _os.path.join(ROOT, "outputs", "figs")
# Okabe-Ito based, colour-blind safe; line style adds a second cue
STY = {"adam": ("#0072B2", "-", 1.3), "nadam": ("#E69F00", "--", 1.0), "aadam": ("#56B4E9", "-.", 1.0),
       "adamw": ("#009E73", "--", 1.0), "radam": ("#CC79A7", ":", 1.4), "adabelief": ("#D55E00", "-.", 1.0),
       "adam_warmup": ("#882255", "--", 1.3), "adam_v0rnd": ("#999999", ":", 1.4), "sadamw": ("#8C6D00", "-", 1.0),
       "sadam": ("#000000", "-", 2.2)}
LBL = {"adam": "Adam", "nadam": "NAdam", "aadam": "AAdam", "adamw": "AdamW", "radam": "RAdam", "adabelief": "AdaBelief",
       "adam_warmup": "Adam + warm-up", "adam_v0rnd": r"Adam-$v_0$rnd", "sadamw": "SADAM-W", "sadam": "SADAM"}
THR = {"E1": 0.05, "E2": 0.5, "E3": 1.0}
TITLE = {"E1": "MNIST / MLP", "E2": "CIFAR-10 / small CNN", "E3": "CIFAR-10 / ResNet-8"}


def curves(e):
    recs = A.load(e); by = {}
    for r in recs: by.setdefault(r["name"], []).append(r)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    handles = []
    for n in A.ORDER:
        if n not in by: continue
        L = np.array([[x["train_loss"] for x in r["epochs"]] for r in by[n]])
        Acc = np.array([[x["test_acc"] * 100 for x in r["epochs"]] for r in by[n]])
        x = np.arange(1, L.shape[1] + 1); c, ls, lw = STY[n]
        h, = ax[0].plot(x, L.mean(0), ls, color=c, lw=lw, label=LBL[n], zorder=3 if n == "sadam" else 2)
        ax[1].plot(x, Acc.mean(0), ls, color=c, lw=lw, zorder=3 if n == "sadam" else 2)
        if n in ("sadam", "adam"):
            for a_, Y in ((ax[0], L), (ax[1], Acc)):
                a_.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), color=c, alpha=0.12, lw=0)
        handles.append(h)
    ax[0].set_yscale("log"); ax[0].axhline(THR[e], ls=(0, (1, 2)), c="0.4", lw=0.8)
    ax[0].text(ax[0].get_xlim()[1], THR[e], " threshold", va="bottom", ha="right", fontsize=6.5, color="0.35")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Training loss (log scale)")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Test accuracy (%)")
    allacc = np.concatenate([[x["test_acc"] * 100 for x in r["epochs"][len(r["epochs"]) // 4:]] for r in recs])
    ax[1].set_ylim(np.percentile(allacc, 1) - 0.3, allacc.max() + 0.4)
    for a_ in ax: a_.grid(alpha=0.25, lw=0.5)
    fig.legend(handles=handles, loc="lower center", ncol=5 if len(handles) > 6 else len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.2)
    fig.suptitle(TITLE[e], fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.13 if len(handles) > 6 else 0.09, 1, 0.97))
    fig.savefig(f"{FG}/fig_{e}_curves.pdf"); plt.close(fig)


DIAG = [((0, 0), "Adam ($\\sigma_m=\\sigma_v=0$)", "#0072B2", "-"), ((1e-4, 1e-4), r"$\sigma_m=\sigma_v=10^{-4}$", "#009E73", "-"),
        ((1e-3, 1e-3), r"$\sigma_m=\sigma_v=10^{-3}$", "#E69F00", "-"), ((1e-2, 1e-2), r"$\sigma_m=\sigma_v=10^{-2}$ (default)", "#000000", "-"),
        ((1e-1, 1e-1), r"$\sigma_m=\sigma_v=10^{-1}$", "#CC79A7", "-"), ((1e-2, 0), r"$\sigma_m=10^{-2},\ \sigma_v=0$ (unstable)", "#D55E00", "--")]


def steps(recs, tag, title, both=True):
    fig, ax = plt.subplots(1, 2 if both else 1, figsize=(7.0, 2.8) if both else (3.5, 3.1))
    ax = np.atleast_1d(ax); hs = []
    for (sm, sv), lab, c, ls in DIAG:
        rs = [r for r in recs if np.isclose(r["args"]["sigma_m"], sm) and np.isclose(r["args"]["sigma_v"], sv) and r.get("step_log")]
        if not rs: continue
        Lm = np.array([[x[1] for x in r["step_log"]] for r in rs]); Fr = np.array([[x[2] for x in r["step_log"]] for r in rs])
        t = np.arange(1, Lm.shape[1] + 1); lw = 1.8 if sm == 1e-2 and sv == 1e-2 else 1.2
        if both: ax[0].plot(t, Lm.mean(0), ls, color=c, lw=lw)
        h, = ax[-1].plot(t, Fr.mean(0) * 100, ls, color=c, lw=lw, label=lab); hs.append(h)
    for a_ in ax: a_.set_xscale("log"); a_.set_xlabel("Iteration $t$"); a_.grid(alpha=0.25, lw=0.5)
    if both: ax[0].set_yscale("log"); ax[0].set_ylabel(r"Median $|\Delta\theta_t|/\alpha$")
    ax[-1].set_ylabel(r"Coordinates with $|\Delta\theta_t|>\alpha$ (%)")
    fig.legend(handles=hs, loc="lower center", ncol=3 if both else 2, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=6.8 if both else 6.3)
    fig.suptitle(title, fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0.2 if both else 0.25, 1, 0.97))
    fig.savefig(f"{FG}/fig_steps_{tag}.pdf"); plt.close(fig)


def heat(M, E, title_e, fname, epmax, Kfail=None):
    lab = ["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for a_, X, ttl, cm, fmt in ((ax[0], M, "Final test accuracy (%)", "viridis", "{:.1f}"), (ax[1], E, title_e, "magma_r", "{:.1f}")):
        Xm = np.where(np.isnan(X), epmax + 1, X)
        im = a_.imshow(Xm, origin="lower", cmap=cm, aspect="auto")
        a_.set_xticks(range(5)); a_.set_xticklabels(lab); a_.set_yticks(range(5)); a_.set_yticklabels(lab)
        a_.set_xlabel(r"$\sigma_v$"); a_.set_ylabel(r"$\sigma_m$"); a_.set_title(ttl, fontsize=8.5)
        for s in a_.spines.values(): s.set_visible(False)
        for i, j in itertools.product(range(5), range(5)):
            txt = fmt.format(X[i, j]) if np.isfinite(X[i, j]) else f">{epmax}"
            a_.text(j, i, txt, ha="center", va="center", fontsize=7.2,
                    color="w" if sum(im.cmap(im.norm(Xm[i, j]))[:3]) < 1.5 else "k")
        cb = fig.colorbar(im, ax=a_, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)


if __name__ == "__main__":
    for e in ["E1", "E2", "E3"]: curves(e)
    r5 = A.load("E5"); mn = [r for r in r5 if r["args"]["dataset"] == "mnist"]; cf = [r for r in r5 if r["args"]["dataset"] == "cifar10"]
    steps(mn, "mnist", "MNIST / MLP", both=True)
    e2 = A.load("E2")
    cf_all = cf + [dict(r, args=dict(r["args"], sigma_m=0.0, sigma_v=0.0)) for r in e2 if r["name"] == "adam"] + [r for r in e2 if r["name"] == "sadam"]
    steps(cf_all, "cifar", "CIFAR-10 / small CNN", both=False)
    M, S, E, NR = A.sens_mnist(mn)  # also rewrites the table (unchanged)
    heat(M, E, "Epochs to training loss < 0.05", f"{FG}/fig_sens_mnist.pdf", 30)
    # digits study figures
    R = json.load(open(_os.path.join(ROOT, "results", "digits", "digits_results.json"))); G = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    DM = np.zeros((5, 5)); DE = np.zeros((5, 5))
    for i, sm in enumerate(G):
        for j, sv in enumerate(G):
            rs = R[f"{sm:g}_{sv:g}"]; DM[i, j] = np.mean([r["final_acc"] for r in rs]) * 100
            ep = np.array([r["epochs_to_thresh"] if r["epochs_to_thresh"] else np.nan for r in rs], float)
            DE[i, j] = np.nanmean(ep) if np.isfinite(ep).any() else np.nan
    heat(DM, DE, "Epochs to training loss < 0.05", f"{FG}/fig_digits_sens.pdf", 40)
    drecs = []
    for i, sm in enumerate(G):
        for j, sv in enumerate(G):
            for r in R[f"{sm:g}_{sv:g}"]:
                drecs.append(dict(args=dict(sigma_m=sm, sigma_v=sv), step_log=[(x[0], x[1], x[2], x[3]) for x in r["step_stats"]]))
    # digits step stats are logged at irregular iterations; plot against the logged t
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8)); hs = []
    for (sm, sv), lab, c, ls in DIAG:
        rs = [r for r in drecs if np.isclose(r["args"]["sigma_m"], sm) and np.isclose(r["args"]["sigma_v"], sv)]
        if not rs: continue
        st = np.array([r["step_log"] for r in rs]); t = st[0, :, 0]; lw = 1.8 if sm == 1e-2 and sv == 1e-2 else 1.2
        ax[0].plot(t, st[:, :, 1].mean(0), ls, color=c, lw=lw); h, = ax[1].plot(t, st[:, :, 2].mean(0) * 100, ls, color=c, lw=lw, label=lab); hs.append(h)
    for a_ in ax: a_.set_xscale("log"); a_.set_xlabel("Iteration $t$"); a_.grid(alpha=0.25, lw=0.5)
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"Median $|\Delta\theta_t|/\alpha$"); ax[1].set_ylabel(r"Coordinates with $|\Delta\theta_t|>\alpha$ (%)")
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
    fig.suptitle("UCI digits / MLP", fontsize=9, y=0.99); fig.tight_layout(rect=(0, 0.2, 1, 0.97))
    fig.savefig(f"{FG}/fig_digits_steps.pdf"); plt.close(fig)
    print("done")
