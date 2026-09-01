"""Tests for the WIS decomposition.

The decomposition is only trustworthy if three things hold exactly: WIS really is a mean of pinball
losses, the isotonic recalibration really minimizes pinball loss, and the identity
score = MCB - DSC + UNC closes. All three are asserted, and the binned PAV is checked against the
exact version rather than assumed equivalent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from decompose import (  # noqa: E402
    QUANTILES,
    decompose_quantile,
    decompose_wis,
    isotonic_quantile,
    pinball,
)
from score import wis  # noqa: E402


# ------------------------------------------------------------ the WIS identity
def test_wis_equals_twice_mean_pinball():
    """The identity the whole module rests on, against the independent wis() implementation."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        m, y = rng.uniform(0, 1), rng.uniform(0, 1)
        hw = np.sort(rng.uniform(0, 0.4, 3))
        q = {0.50: m, 0.25: m - hw[0], 0.75: m + hw[0], 0.10: m - hw[1], 0.90: m + hw[1],
             0.025: m - hw[2], 0.975: m + hw[2]}
        iv = {50: (q[0.25], q[0.75]), 80: (q[0.10], q[0.90]), 95: (q[0.025], q[0.975])}
        pb = np.mean([pinball(t, q[t], y) for t, _ in QUANTILES])
        assert wis(y, m, iv) == pytest.approx(2 * pb, abs=1e-12)


# -------------------------------------------------------------- pinball loss
def test_pinball_is_zero_at_a_perfect_forecast():
    assert pinball(0.5, 0.3, 0.3) == pytest.approx(0.0)


def test_pinball_is_asymmetric_in_the_right_direction():
    """A high tau should punish under-prediction more than over-prediction."""
    under = pinball(0.9, 0.4, 0.6)      # forecast below truth
    over = pinball(0.9, 0.6, 0.4)       # forecast above truth by the same amount
    assert under > over


def test_pinball_median_is_half_absolute_error():
    assert pinball(0.5, 0.2, 0.7) == pytest.approx(0.5 * 0.5)


def test_pinball_minimised_at_the_true_quantile():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 4000)
    for tau in (0.1, 0.5, 0.9):
        best = np.quantile(y, tau)
        loss_at_best = pinball(tau, best, y).mean()
        for delta in (-0.3, -0.1, 0.1, 0.3):
            assert pinball(tau, best + delta, y).mean() >= loss_at_best - 1e-9


# ---------------------------------------------------------- isotonic quantile
def test_isotonic_output_is_monotone_in_the_prediction():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 500)
    y = x + rng.normal(0, 0.2, 500)
    f = isotonic_quantile(x, y, 0.5, bins=0)
    order = np.argsort(x)
    assert np.all(np.diff(f[order]) >= -1e-9)


def test_isotonic_recalibration_never_increases_the_loss():
    """The defining property: recalibration is an improvement, so MCB >= 0."""
    rng = np.random.default_rng(3)
    for tau in (0.1, 0.5, 0.9):
        x = rng.uniform(0, 1, 800)
        y = rng.uniform(0, 1, 800)          # deliberately unrelated: badly calibrated
        f = isotonic_quantile(x, y, tau, bins=50)
        assert pinball(tau, f, y).mean() <= pinball(tau, x, y).mean() + 1e-12


def test_isotonic_recovers_a_constant_when_x_is_uninformative():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 600)
    y = rng.normal(5.0, 0.1, 600)
    f = isotonic_quantile(x, y, 0.5, bins=30)
    assert f.std() < 0.1
    assert f.mean() == pytest.approx(5.0, abs=0.1)


def test_binned_pav_matches_exact_pav_closely():
    """The documented approximation must actually be close, not merely assumed to be."""
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 400)
    y = x + rng.normal(0, 0.15, 400)
    exact = isotonic_quantile(x, y, 0.5, bins=0)
    binned = isotonic_quantile(x, y, 0.5, bins=100)
    le, lb = pinball(0.5, exact, y).mean(), pinball(0.5, binned, y).mean()
    assert abs(lb - le) / le < 0.10, f"binned loss {lb:.5f} vs exact {le:.5f}"


def test_isotonic_handles_empty_input():
    assert len(isotonic_quantile([], [], 0.5)) == 0


# ------------------------------------------------------------- the identity
def test_decomposition_identity_holds_for_one_quantile():
    rng = np.random.default_rng(6)
    q = rng.uniform(0, 1, 1000)
    y = np.clip(q + rng.normal(0, 0.2, 1000), 0, 1)
    for tau in (0.025, 0.5, 0.975):
        r = decompose_quantile(q, y, tau)
        assert r["score"] == pytest.approx(r["mcb"] - r["dsc"] + r["unc"], abs=1e-12)


def test_mcb_is_non_negative():
    """Miscalibration cannot be negative: recalibration cannot hurt in-sample."""
    rng = np.random.default_rng(7)
    q = rng.uniform(0, 1, 800)
    y = rng.uniform(0, 1, 800)
    for tau, _ in QUANTILES:
        assert decompose_quantile(q, y, tau)["mcb"] >= -1e-12


def test_dsc_is_zero_for_an_uninformative_forecast():
    """A forecast unrelated to the outcome has no discrimination."""
    rng = np.random.default_rng(8)
    q = rng.uniform(0, 1, 3000)
    y = rng.normal(0.5, 0.1, 3000)
    r = decompose_quantile(q, y, 0.5, bins=30)
    assert abs(r["dsc"]) < 0.02


def test_dsc_is_large_for_an_informative_forecast():
    rng = np.random.default_rng(9)
    y = rng.uniform(0, 1, 3000)
    q = y + rng.normal(0, 0.01, 3000)        # nearly perfect
    r = decompose_quantile(q, y, 0.5, bins=50)
    assert r["dsc"] > 0.5 * r["unc"]


def test_decompose_wis_closes_the_identity_and_returns_all_levels():
    rng = np.random.default_rng(10)
    n = 2000
    m = rng.uniform(0.1, 0.9, n)
    y = np.clip(m + rng.normal(0, 0.1, n), 0, 1)
    df = pd.DataFrame({"median": m, "truth": y})
    for tau, col in QUANTILES:
        if col != "median":
            df[col] = m + (tau - 0.5) * 0.4
    agg, per_q = decompose_wis(df)
    assert len(per_q) == len(QUANTILES)
    assert agg["score"] == pytest.approx(agg["mcb"] - agg["dsc"] + agg["unc"], abs=1e-12)
    assert abs(agg["identity_residual"]) < 1e-12


# ------------------------------------------------------------ real-data guards
def _real(label="gisaid"):
    p = f"results/decomposition_{label}.json"
    if not os.path.exists(p):
        pytest.skip("decomposition not computed")
    import json
    with open(p) as fh:
        return json.load(fh)


@pytest.mark.parametrize("label", ["gisaid", "open", "flu"])
def test_real_identity_closes(label):
    o = _real(label)["overall"]
    assert abs(o["identity_residual"]) < 1e-12
    assert o["score"] == pytest.approx(o["mcb"] - o["dsc"] + o["unc"], abs=1e-12)


def test_real_miscalibration_is_a_large_share_of_the_score():
    """F12's headline. If this drops below ~a quarter, the finding must be rewritten."""
    o = _real()["overall"]
    assert o["mcb"] / o["score"] > 0.25


def test_real_discrimination_is_positive_everywhere():
    """The forecasts do carry information: the paper is not a debunking."""
    for label in ("gisaid", "open", "flu"):
        assert _real(label)["overall"]["dsc"] > 0


def test_real_mcb_share_rises_with_horizon():
    """The robust cross-dataset pattern behind F12."""
    for label in ("gisaid", "open", "flu"):
        h = {r["hbin"]: r for r in _real(label)["by_horizon"]}
        assert h["22-30d"]["mcb_share"] > h["1-7d"]["mcb_share"], label
