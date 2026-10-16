"""NumPy re-implementations of the optimizers, following torch.optim semantics
(Adam, AdamW, NAdam with momentum decay 0.004, RAdam) and the reference descriptions of
AAdam (Tato & Nkambou, TF code), AdaBelief (Zhuang et al., Alg. 2), Adam-v0rnd (Abuduweili & Liu, 2025),
Adam + linear warm-up, SADAM and SADAM-W."""
import math
import numpy as np

F32 = np.float32
B1, B2, EPS = 0.9, 0.999, 1e-8


class Opt:
    def __init__(self, name, params, lr=1e-3, seed=0, sigma_m=0.01, sigma_v=0.01, wd=1e-2, warmup=2000,
                 log_steps=0):
        self.name, self.P, self.lr, self.t = name, params, lr, 0
        self.wd, self.warmup = wd, warmup
        rng = np.random.default_rng(10_000 + seed)
        z = lambda p: np.zeros_like(p.data)
        self.m = [z(p) for p in params]; self.v = [z(p) for p in params]
        if name in ("sadam", "sadamw"):
            self.m = [(rng.normal(0, sigma_m, p.data.shape) if sigma_m > 0 else np.zeros(p.data.shape)).astype(F32) for p in params]
            self.v = [((rng.normal(0, sigma_v, p.data.shape) if sigma_v > 0 else np.zeros(p.data.shape)) ** 2).astype(F32) for p in params]
        if name == "adam_v0rnd":
            self.v = [((rng.normal(0, math.sqrt(100.0 / (p.fan_in + p.fan_out)), p.data.shape)) ** 2).astype(F32)
                      if p.kind == "weight" else np.zeros_like(p.data) for p in params]
        if name == "aadam":
            self.d = [z(p) for p in params]; self.pg = [z(p) for p in params]
        if name == "nadam":
            self.mu_prod = 1.0
        self.log_steps = log_steps; self.step_log = []

    def step(self):
        self.t += 1; t = self.t; n = self.name; lr = self.lr
        if n == "adam_warmup": lr = lr * min(1.0, t / self.warmup)
        bc1, bc2 = 1 - B1 ** t, 1 - B2 ** t
        log = self.log_steps and t <= self.log_steps; ratios = []
        if n == "nadam":
            mu = B1 * (1 - 0.5 * 0.96 ** (t * 0.004)); mu_n = B1 * (1 - 0.5 * 0.96 ** ((t + 1) * 0.004))
            self.mu_prod *= mu
        for i, p in enumerate(self.P):
            g = p.grad; m, v = self.m[i], self.v[i]
            if n in ("adamw", "sadamw"): p.data *= (1 - lr * self.wd)
            if n == "adabelief":
                m *= B1; m += (1 - B1) * g
                diff = g - m; v *= B2; v += (1 - B2) * diff * diff; v += 1e-16
                upd = (m / bc1) / (np.sqrt(v / bc2) + 1e-16)
            else:
                m *= B1; m += (1 - B1) * g
                v *= B2; v += (1 - B2) * g * g
                if n in ("adam", "adamw", "sadam", "sadamw", "adam_warmup", "adam_v0rnd"):
                    upd = (m / bc1) / (np.sqrt(v / bc2) + EPS)
                elif n == "nadam":
                    den = np.sqrt(v / bc2) + EPS
                    upd = ((1 - mu) / (1 - self.mu_prod)) * g / den + (mu_n / (1 - self.mu_prod * mu_n)) * m / den
                elif n == "radam":
                    rinf = 2 / (1 - B2) - 1; rt = rinf - 2 * t * B2 ** t / bc2
                    if rt > 5:
                        rect = math.sqrt((rt - 4) * (rt - 2) * rinf / ((rinf - 4) * (rinf - 2) * rt))
                        upd = (m / bc1) * rect * (math.sqrt(bc2) / (np.sqrt(v) + EPS))
                    else:
                        upd = m / bc1
                elif n == "aadam":
                    a = math.sqrt(bc2) / bc1; u = m * a / (np.sqrt(v) + EPS)
                    flip = (np.sign(self.pg[i]) * np.sign(g)) < 0
                    dab = np.abs(self.d[i])
                    upd = np.where(flip, np.sign(u) * (B1 * np.abs(u) - (1 - B1) * dab), np.sign(u) * (B1 * np.abs(u) + (1 - B1) * dab))
                    self.d[i] = upd.astype(F32); self.pg[i] = g.copy()
                else:
                    raise ValueError(n)
            if log: ratios.append(np.abs(upd).ravel())
            p.data -= (lr * upd).astype(F32)
        if log:
            r = np.concatenate(ratios)
            self.step_log.append((t, float(np.median(r)), float((r > 1).mean()), float(r.max())))
