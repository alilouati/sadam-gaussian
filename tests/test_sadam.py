"""Tests for the SADAM code base. Run from the repository root with:  pytest -q"""
import filecmp
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)

import nn  # noqa: E402
from optim import Opt, B1, B2  # noqa: E402

OPTIMIZERS = ["adam", "nadam", "aadam", "adamw", "radam", "adabelief", "adam_warmup", "adam_v0rnd", "sadam", "sadamw"]


def _param(shape, seed=0):
    p = nn.Param(np.random.default_rng(seed).normal(size=shape), "weight", shape[0], shape[1])
    return p


# ---------------------------------------------------------------- initialization (Section III)
def test_sadam_initial_states_follow_gaussian_scales():
    p = _param((400, 500))
    o = Opt("sadam", [p], seed=1, sigma_m=0.01, sigma_v=0.02)
    m0, v0 = o.m[0], o.v[0]
    assert abs(m0.mean()) < 2e-4
    assert m0.std() == pytest.approx(0.01, rel=0.02)
    assert (v0 >= 0).all()                                   # second moment is non-negative
    assert v0.mean() == pytest.approx(0.02 ** 2, rel=0.03)   # E[z_v^2] = sigma_v^2


def test_adam_starts_from_zero_states():
    o = Opt("adam", [_param((10, 10))])
    assert not o.m[0].any() and not o.v[0].any()


def test_sigma_zero_recovers_adam_exactly():
    rng = np.random.default_rng(0)
    pa, ps = _param((20, 30)), _param((20, 30))
    oa, os_ = Opt("adam", [pa]), Opt("sadam", [ps], sigma_m=0.0, sigma_v=0.0)
    for _ in range(50):
        g = rng.normal(size=(20, 30)).astype(np.float32)
        pa.grad[...] = g; ps.grad[...] = g; oa.step(); os_.step()
    np.testing.assert_array_equal(pa.data, ps.data)


# ---------------------------------------------------------------- Lemma 1: closed form after bias correction
def test_bias_corrected_initial_state_weights_at_t1():
    """With a zero gradient, one step gives m_hat_1 = c1(1) m0 and v_hat_1 = c2(1) v0, c1(1)=9, c2(1)=999."""
    p = _param((50, 50))
    o = Opt("sadam", [p], seed=3)
    m0, v0 = o.m[0].astype(np.float64).copy(), o.v[0].astype(np.float64).copy()
    p.grad[...] = 0
    o.step()
    np.testing.assert_allclose(o.m[0] / (1 - B1), 9 * m0, rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(o.v[0] / (1 - B2), 999 * v0, rtol=1e-4, atol=1e-14)


# ---------------------------------------------------------------- backward passes
def test_gradient_check_all_layers():
    out = subprocess.run([sys.executable, "gradcheck.py"], cwd=CODE, capture_output=True, text=True, check=True).stdout
    worst = float(out.strip().splitlines()[-1].split()[-1])
    assert worst < 1e-6, out


# ---------------------------------------------------------------- every optimizer trains a small network
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_optimizer_reduces_loss(name):
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 3, (10, 20))
    y = rng.integers(0, 10, 512)
    x = (centers[y] + rng.normal(size=(512, 20))).astype(np.float32)
    model = nn.Seq([nn.Linear(20, 32, rng), nn.ReLU(), nn.Linear(32, 10, rng)])
    opt = Opt(name, model.params, lr=1e-2, warmup=20)
    first = None
    for it in range(150):
        idx = rng.integers(0, 512, 64)
        for p in model.params: p.grad[...] = 0
        loss, d = nn.softmax_ce(model.fwd(x[idx]), y[idx]); model.bwd(d); opt.step()
        first = loss if first is None else first
    final, _ = nn.softmax_ce(model.fwd(x), y)
    assert np.isfinite(final) and final < 0.5 * first


# ---------------------------------------------------------------- the paper's tables are reproduced from the logs
def test_tables_regenerated_from_logs_match_paper():
    for script in ("npanalyze.py", os.path.join("theory_and_digits", "digits_report.py")):
        r = subprocess.run([sys.executable, script], cwd=CODE, capture_output=True, text=True)
        assert r.returncode == 0, f"{script} failed:\n{r.stderr[-3000:]}"
    gen = os.path.join(ROOT, "outputs", "tables")
    for f in sorted(os.listdir(os.path.join(ROOT, "tables"))):
        assert filecmp.cmp(os.path.join(gen, f), os.path.join(ROOT, "tables", f), shallow=False), f
