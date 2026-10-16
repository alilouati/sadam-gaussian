"""Numerical checks of the analytical results added to the revised SADAM manuscript.

(A) Proposition (update bound):  |m_t|/sqrt(v_t) <= sqrt(gamma^t rho^2 + (1-b1)^2 (1-gamma^t)/((1-b2)(1-gamma)))
(B) Lemma (log-sum bound):       sum_t g_t^2/(eps+v_t) <= (1/(1-b2)) [ln((eps+v_T)/(eps+v_0)) - T ln b2]
(C) Bias of bias-corrected moments under Gaussian initialization (Monte Carlo vs closed form)
(D) Effective first-step statistics and the chi-square(1) mass near zero
"""
import numpy as np, json, sys

rng = np.random.default_rng(0)
b1, b2, eps = 0.9, 0.999, 1e-8
gamma = b1**2 / b2
out = {}

# ---------- (A) update-ratio bound ----------
worst = 0.0
for trial in range(2000):
    T = 300
    scale = 10 ** rng.uniform(-5, 0)
    g = rng.standard_normal(T) * scale * (10 ** rng.uniform(-2, 1, size=T)) + rng.uniform(-1, 1) * scale
    sm, sv = 10 ** rng.uniform(-4, 0), 10 ** rng.uniform(-4, 0)
    zm, zv = rng.normal(0, sm), rng.normal(0, sv)
    m, v = zm, zv ** 2
    rho = abs(zm / zv)
    for t in range(1, T + 1):
        m = b1 * m + (1 - b1) * g[t - 1]
        v = b2 * v + (1 - b2) * g[t - 1] ** 2
        bound = np.sqrt(gamma**t * rho**2 + (1 - b1) ** 2 * (1 - gamma**t) / ((1 - b2) * (1 - gamma)))
        worst = max(worst, abs(m) / np.sqrt(v) / bound)
out["A_max_ratio_over_bound"] = worst  # must be <= 1

# ---------- (B) log-sum lemma ----------
worstB = -np.inf
for trial in range(2000):
    T = 500
    g = rng.standard_normal(T) * 10 ** rng.uniform(-4, 0)
    v0 = rng.normal(0, 10 ** rng.uniform(-4, -1)) ** 2
    e = 10 ** rng.uniform(-10, -4)
    v, s = v0, 0.0
    for t in range(T):
        v = b2 * v + (1 - b2) * g[t] ** 2
        s += g[t] ** 2 / (e + v)
    rhs = (np.log((e + v) / (e + v0)) - T * np.log(b2)) / (1 - b2)
    worstB = max(worstB, s - rhs)
out["B_max_lhs_minus_rhs"] = worstB  # must be <= 0

# ---------- (C) bias of corrected moments ----------
N = 400000
sm = sv = 0.01
mu, s = 1e-3, 3e-3          # stationary gradient mean / std (illustrative)
Eg2 = mu**2 + s**2
zm = rng.normal(0, sm, N); zv = rng.normal(0, sv, N)
m, v = zm.copy(), zv**2
rows = []
for t in range(1, 3001):
    g = rng.normal(mu, s, N)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g**2
    if t in (1, 2, 5, 10, 50, 100, 500, 1000, 3000):
        mh = m / (1 - b1**t); vh = v / (1 - b2**t)
        rows.append(dict(t=t,
                         E_mhat_mc=float(mh.mean()), E_mhat_theory=mu,
                         Var_mhat_extra_theory=float((b1**t / (1 - b1**t))**2 * sm**2),
                         E_vhat_mc=float(vh.mean()), E_vhat_theory=float(Eg2 + b2**t * sv**2 / (1 - b2**t))))
out["C_bias_table"] = rows

# ---------- (D) chi-square(1) mass and first SADAM step ----------
z = rng.standard_normal(10**7)
out["D_P(v0<0.01*sv^2)"] = float(np.mean(z**2 < 0.01))
out["D_P(v0<0.1*sv^2)"] = float(np.mean(z**2 < 0.1))
out["D_median_v0_over_sv2"] = float(np.median(z**2))
# first-step magnitude in units of alpha, for gradient scale |g| = G
for G in (1e-4, 1e-3, 1e-2):
    for sig in (1e-3, 1e-2, 1e-1):
        zm = rng.normal(0, sig, 10**6); zv = rng.normal(0, sig, 10**6)
        g = rng.normal(0, G, 10**6)
        m1 = (b1 * zm + (1 - b1) * g) / (1 - b1)
        v1 = (b2 * zv**2 + (1 - b2) * g**2) / (1 - b2)
        step = np.abs(m1) / (np.sqrt(v1) + eps)
        out[f"D_step1_G{G}_sig{sig}"] = dict(median=float(np.median(step)), p99=float(np.quantile(step, 0.99)),
                                              frac_gt_1=float(np.mean(step > 1)), frac_gt_5=float(np.mean(step > 5)))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if not k.startswith("C")}, indent=1))
for r in rows: print(r)
