"""Aggregate the NumPy runs (results/) into the LaTeX tables and figures of the manuscript."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import glob, json, os, itertools, csv, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = _os.path.join(ROOT, "results"); T = _os.path.join(ROOT, "outputs", "tables"); FG = _os.path.join(ROOT, "outputs", "figs")
os.makedirs(T, exist_ok=True); os.makedirs(FG, exist_ok=True)
NAMES = {"adam": "Adam", "nadam": "NAdam", "aadam": "AAdam", "adamw": "AdamW", "radam": "RAdam", "adabelief": "AdaBelief",
         "adam_warmup": "Adam+warm-up", "adam_v0rnd": "Adam-$v_0$rnd", "sadamw": "SADAM-W", "sadam": "SADAM"}
ORDER = ["adam", "nadam", "aadam", "adamw", "radam", "adabelief", "adam_warmup", "adam_v0rnd", "sadamw", "sadam"]
THR = {"E1": 0.05, "E2": 0.5, "E3": float(os.environ.get("E3THR", "1.0"))}
TITLE = {"E1": "MNIST / MLP", "E2": "CIFAR-10 / CNN", "E3": "CIFAR-10 / ResNet-8"}
G = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
SUMMARY = {}


def load(sub):
    out = []
    for f in glob.glob(f"{R}/{sub}/*.json"):
        r = json.load(open(f)); a = r["args"]; r["name"] = a["tag"] or a["opt"]
        r["diverged"] = "diverged_at_epoch" in r
        r["acc"] = r["epochs"][-1]["test_acc"] * 100 if r["epochs"] else np.nan
        out.append(r)
    return out


def thr(r, th, upto=None):
    for e in r["epochs"][:upto]:
        if e["train_loss"] < th: return e["epoch"], e["train_time_s"]
    return np.nan, np.nan


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj


def fp(p): return "--" if not np.isfinite(p) else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")


def main_table(e, recs):
    by = {}
    for r in recs: by.setdefault(r["name"], []).append(r)
    rows = []
    for n in ORDER:
        if n not in by: continue
        rs = sorted(by[n], key=lambda r: r["args"]["seed"]); acc = np.array([r["acc"] for r in rs]); k = len(acc)
        ep, tm = map(lambda v: np.array(v, float), zip(*[thr(r, THR[e]) for r in rs])); reach = int(np.isfinite(ep).sum())
        sd = acc.std(ddof=1) if k > 1 else np.nan; ci = stats.t.ppf(.975, k - 1) * sd / np.sqrt(k) if k > 1 else np.nan
        rows.append((n, k, acc.mean(), sd, ci, np.nanmean(ep) if reach else np.nan, np.nanmean(tm) if reach else np.nan, reach))
        SUMMARY[(e, n)] = dict(k=k, acc=acc.mean(), sd=sd, ci=ci, ep=rows[-1][5], tm=rows[-1][6], reach=reach,
                               final_loss=np.mean([r["epochs"][-1]["train_loss"] for r in rs]))
    best = max(r[2] for r in rows)
    with open(f"{T}/tab_{e}.tex", "w") as f:
        for n, k, m, s, h, ep, tm, rc in rows:
            mm = f"\\textbf{{{m:.2f}}}" if abs(m - best) < 1e-9 else f"{m:.2f}"
            es = "--" if np.isnan(ep) else f"{ep:.1f}" + ("" if rc == k else f"$^{{{rc}/{k}}}$")
            ts = "--" if np.isnan(tm) else (f"{tm:.0f}" if tm < 600 else f"{tm:.0f}")
            f.write(f"{NAMES[n]} & {mm} $\\pm$ {s:.2f} & $\\pm${h:.2f} & {es} & {ts} \\\\\n")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    cmap = plt.get_cmap("tab10")
    for ci_, n in enumerate(ORDER):
        if n not in by: continue
        L = np.array([[x["train_loss"] for x in r["epochs"]] for r in by[n] if not r["diverged"]])
        A = np.array([[x["test_acc"] * 100 for x in r["epochs"]] for r in by[n] if not r["diverged"]])
        x = np.arange(1, L.shape[1] + 1); lw = 2.2 if n == "sadam" else 1.0; ls = "-" if n in ("sadam", "adam") else "--"
        col = "k" if n == "sadam" else cmap(ci_)
        for a_, Y in ((ax[0], L), (ax[1], A)):
            a_.plot(x, Y.mean(0), ls, lw=lw, color=col, label=NAMES[n].replace("$", "").replace("_0", "0"))
            if n in ("sadam", "adam"): a_.fill_between(x, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0), alpha=.15, color=col)
    ax[0].set_yscale("log"); ax[0].axhline(THR[e], ls=":", c="gray", lw=.8)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Training loss"); ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Test accuracy (%)")
    lo = np.percentile([x["test_acc"] * 100 for r in recs for x in r["epochs"][len(r["epochs"]) // 3:]], 2)
    ax[1].set_ylim(lo, None); ax[1].legend(fontsize=5.5, ncol=2, loc="lower right")
    fig.suptitle(TITLE[e], fontsize=9); fig.tight_layout(); fig.savefig(f"{FG}/fig_{e}_curves.pdf"); plt.close(fig)
    return by


def stats_table(allby):
    exps = [e for e in ["E1", "E2", "E3"] if e in allby]; cells = {}; rowsout = []
    for e in exps:
        by = allby[e]; s = {r["args"]["seed"]: r["acc"] for r in by["sadam"]}; tmp = []
        for n in ORDER:
            if n == "sadam" or n not in by: continue
            b = {r["args"]["seed"]: r["acc"] for r in by[n]}; cm = sorted(set(s) & set(b))
            d = np.array([s[k] - b[k] for k in cm])
            pt = stats.ttest_rel([s[k] for k in cm], [b[k] for k in cm]).pvalue if d.std() > 0 else 1.0
            try: pw = stats.wilcoxon(d).pvalue
            except ValueError: pw = np.nan
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            tmp.append([n, d.mean(), dz, pt, pw, len(cm)])
        adj = holm([t[3] for t in tmp])
        for t, pa in zip(tmp, adj):
            cells[(e, t[0])] = (t[1], t[2], pa, t[4]); rowsout.append([e, t[0], t[5], t[1], t[2], t[3], pa, t[4]])
    with open(f"{T}/tab_stats.tex", "w") as f:
        for n in ORDER:
            if n == "sadam": continue
            line = [NAMES[n]]
            for e in exps:
                if (e, n) in cells:
                    dm, dz, pa, pw = cells[(e, n)]; line += [f"{dm:+.2f}", "--" if not np.isfinite(dz) else f"{dz:+.2f}", fp(pa)]
                else: line += ["--", "--", "--"]
            f.write(" & ".join(line) + " \\\\\n")
    with open(f"{R}/stats_full.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["exp", "baseline", "n_pairs", "mean_diff_pp", "d_z", "p_paired_t", "p_holm", "p_wilcoxon"]); w.writerows(rowsout)
    return cells


def sens_mnist(recs):
    M = np.full((5, 5), np.nan); S = M.copy(); E = M.copy(); NR = np.zeros((5, 5), int); K = np.zeros((5, 5), int)
    for i, sm in enumerate(G):
        for j, sv in enumerate(G):
            rs = [r for r in recs if np.isclose(r["args"]["sigma_m"], sm) and np.isclose(r["args"]["sigma_v"], sv)]
            if not rs: continue
            a = np.array([r["acc"] for r in rs]); M[i, j] = a.mean(); S[i, j] = a.std(ddof=1); K[i, j] = len(rs)
            ep = np.array([thr(r, THR["E1"])[0] for r in rs], float); NR[i, j] = np.isnan(ep).sum()
            E[i, j] = np.nanmean(ep) if np.isfinite(ep).any() else np.nan
    lab = ["0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"]
    with open(f"{T}/tab_sens_mnist.tex", "w") as f:
        for i in range(5):
            cells = []
            for j in range(5):
                c = f"{M[i,j]:.2f}$\\pm${S[i,j]:.2f} ({'--' if np.isnan(E[i,j]) else f'{E[i,j]:.1f}'}"
                c += (f"$^{{{K[i,j]-NR[i,j]}/{K[i,j]}}}$" if 0 < NR[i, j] < K[i, j] else "") + ")"
                cells.append(c)
            f.write(f"{lab[i]} & " + " & ".join(cells) + " \\\\\n")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for a_, Mx, ttl, cm in ((ax[0], M, "Final test accuracy (%)", "viridis"), (ax[1], E, "Epochs to training loss < 0.05", "magma_r")):
        Mm = np.where(np.isnan(Mx), 31, Mx)
        im = a_.imshow(Mm, origin="lower", cmap=cm, vmin=(np.nanmin(M) if a_ is ax[0] else None))
        a_.set_xticks(range(5)); a_.set_xticklabels(lab, fontsize=7); a_.set_yticks(range(5)); a_.set_yticklabels(lab, fontsize=7)
        a_.set_xlabel(r"$\sigma_v$"); a_.set_ylabel(r"$\sigma_m$"); a_.set_title(ttl, fontsize=8)
        for i, j in itertools.product(range(5), range(5)):
            a_.text(j, i, f"{Mx[i,j]:.1f}" if np.isfinite(Mx[i, j]) else ">30", ha="center", va="center", fontsize=6,
                    color="w" if sum(im.cmap(im.norm(Mm[i, j]))[:3]) < 1.5 else "k")
        fig.colorbar(im, ax=a_, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{FG}/fig_sens_mnist.pdf"); plt.close(fig)
    return M, S, E, NR


def steps_fig(recs, tag, title, frac_only=False):
    if frac_only:
        fig, a1 = plt.subplots(1, 1, figsize=(3.5, 2.5)); ax = [plt.figure().add_subplot(), a1]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for sm, sv, st in [(0, 0, "k-"), (1e-4, 1e-4, "C0-"), (1e-3, 1e-3, "C1-"), (1e-2, 1e-2, "C2-"), (1e-1, 1e-1, "C3-"), (1e-2, 0, "C4--")]:
        rs = [r for r in recs if np.isclose(r["args"]["sigma_m"], sm) and np.isclose(r["args"]["sigma_v"], sv) and r.get("step_log")]
        if not rs: continue
        L = np.array([[x[1] for x in r["step_log"]] for r in rs]); F_ = np.array([[x[2] for x in r["step_log"]] for r in rs])
        t = np.arange(1, L.shape[1] + 1); lab = "Adam" if sm == 0 and sv == 0 else rf"$\sigma_m$={sm:g}, $\sigma_v$={sv:g}"
        ax[0].plot(t, L.mean(0), st, label=lab, lw=1.1); ax[1].plot(t, F_.mean(0) * 100, st, lw=1.1)
    for a_ in ax: a_.set_xscale("log"); a_.set_xlabel("Iteration $t$")
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"median $|\Delta\theta_t|/\alpha$"); ax[1].set_ylabel(r"coords. with $|\Delta\theta_t|>\alpha$ (%)")
    (ax[1] if frac_only else ax[0]).legend(*ax[0].get_legend_handles_labels(), fontsize=5.5) if frac_only else ax[0].legend(fontsize=5.5)
    fig.suptitle(title, fontsize=9); fig.tight_layout(); fig.savefig(f"{FG}/fig_steps_{tag}.pdf"); plt.close(fig)


def sens_cifar(recs5, e2):
    """CIFAR CNN subset: 2 seeds, 15 epochs; Adam and default SADAM taken from E2 (seeds 0-1) truncated at epoch 15."""
    rows = []
    def summ(rs, label):
        a = np.array([r["epochs"][14]["test_acc"] * 100 if len(r["epochs"]) >= 15 else np.nan for r in rs])
        ep = np.array([thr(r, 0.5, 15)[0] for r in rs], float)
        rows.append((label, np.nanmean(a), np.nanstd(a, ddof=1) if len(a) > 1 else np.nan,
                     np.nanmean(ep) if np.isfinite(ep).any() else np.nan, int(np.isfinite(ep).sum()), len(rs), sum(r["diverged"] for r in rs)))
    e2s = lambda n: [r for r in e2 if r["name"] == n and r["args"]["seed"] in (0, 1)]
    summ(e2s("adam"), "0 & 0 (Adam)")
    for sm, sv in [(1e-4, 1e-4), (1e-3, 1e-3), ("E2", None), (1e-1, 1e-1), (0.0, 1e-2), (1e-2, 0.0)]:
        if sm == "E2": summ(e2s("sadam"), "$10^{-2}$ & $10^{-2}$ (default)"); continue
        rs = [r for r in recs5 if np.isclose(r["args"]["sigma_m"], sm) and np.isclose(r["args"]["sigma_v"], sv)]
        if rs: summ(rs, f"{'0' if sm == 0 else f'$10^{{{int(np.log10(sm))}}}$'} & {'0' if sv == 0 else f'$10^{{{int(np.log10(sv))}}}$'}")
    with open(f"{T}/tab_sens_cifar.tex", "w") as f:
        for lab, m, s, ep, rc, k, dv in rows:
            es = "--" if np.isnan(ep) else f"{ep:.1f}" + ("" if rc == k else f"$^{{{rc}/{k}}}$")
            f.write(f"{lab} & {m:.2f} $\\pm$ {s:.2f}{'$^\\dagger$' if dv else ''} & {es} \\\\\n")
    return rows


if __name__ == "__main__":
    allby = {}
    for e in ["E1", "E2", "E3"]:
        rs = load(e)
        if rs: allby[e] = main_table(e, rs)
    cells = stats_table(allby) if allby else {}
    r5 = load("E5")
    mn = [r for r in r5 if r["args"]["dataset"] == "mnist"]; cf = [r for r in r5 if r["args"]["dataset"] == "cifar10"]
    if mn: sens_mnist(mn); steps_fig(mn, "mnist", "MNIST / MLP")
    if cf and "E2" in allby:
        e2 = load("E2"); sens_cifar(cf, e2)
        steps_fig(cf + [dict(r, args=dict(r["args"], sigma_m=0.0, sigma_v=0.0)) for r in e2 if r["name"] == "adam"] +
                  [r for r in e2 if r["name"] == "sadam"], "cifar", "CIFAR-10 / CNN", frac_only=True)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in SUMMARY.items()}, open(f"{R}/summary.json", "w"), indent=1, default=float)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in cells.items()}, open(f"{R}/stats.json", "w"), indent=1, default=float)
    for k, v in SUMMARY.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
    for k, v in cells.items(): print(k, [round(float(x), 4) for x in v])
