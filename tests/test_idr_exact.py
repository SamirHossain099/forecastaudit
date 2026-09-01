"""Tests for exact partial-order isotonic quantile regression.

`test_pav_matches_lp_on_a_total_order` is the one that earned its keep: it failed on first run and
exposed that np.quantile's default linear interpolation is not a pinball-loss minimizer, which made
the PAV fit suboptimal by 21.7% of the loss at tau = 0.025. Without an independent exact solver to
check against, that bug was invisible.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from idr_decompose import isotonic_quantile_ordered  # noqa: E402
from idr_exact import (  # noqa: E402
    comparable_pairs,
    isotonic_quantile_poset,
    mcb_exact,
    mcb_linear,
)


def pinball_mean(q, y, tau):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.where(d >= 0, tau * d, (tau - 1) * d).mean())


def _total_order(n=100, seed=0):
    """Intervals increasing in BOTH bounds — the partial order collapses to a total one."""
    rng = np.random.default_rng(seed)
    lo = np.sort(rng.uniform(0, 1, n))
    return lo, lo + 0.05, lo + rng.normal(0, 0.2, n)


def _partial_order(n=120, seed=0):
    """Nested intervals — genuinely incomparable pairs."""
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0, 1, n)
    hi = lo + rng.uniform(0.01, 0.4, n)
    return lo, hi, (lo + hi) / 2 + rng.normal(0, 0.15, n)


# --------------------------------------------------- the check that found the bug
@pytest.mark.parametrize("tau", [0.025, 0.25, 0.5, 0.75, 0.975])
def test_pav_matches_lp_on_a_total_order(tau):
    """On a total order PAV is provably exact, so it must attain the LP optimum."""
    lo, hi, y = _total_order(seed=int(tau * 1000))
    lp = isotonic_quantile_poset(lo, hi, y, tau)
    pav = isotonic_quantile_ordered(np.argsort(lo), y, tau, bins=0)
    assert pinball_mean(pav, y, tau) == pytest.approx(pinball_mean(lp, y, tau), abs=1e-9)


def test_linear_interpolation_would_fail_this():
    """Pin the root cause: the default interpolation is not a pinball minimizer."""
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 9)
    tau = 0.975
    linear = float(np.quantile(y, tau))
    order_stat = float(np.quantile(y, tau, method="inverted_cdf"))
    assert pinball_mean(order_stat, y, tau) <= pinball_mean(linear, y, tau) + 1e-12
    grid = np.linspace(y.min(), y.max(), 2001)
    best = min(pinball_mean(g, y, tau) for g in grid)
    assert pinball_mean(order_stat, y, tau) == pytest.approx(best, abs=1e-6)


# ------------------------------------------------------------------ LP correctness
def test_lp_respects_every_order_constraint():
    lo, hi, y = _partial_order(n=90, seed=1)
    q = isotonic_quantile_poset(lo, hi, y, 0.5)
    pairs = comparable_pairs(lo, hi)
    assert (q[pairs[:, 0]] <= q[pairs[:, 1]] + 1e-7).all()


@pytest.mark.parametrize("tau", [0.025, 0.5, 0.975])
def test_lp_is_no_worse_than_the_linear_extension(tau):
    """The whole lower-bound argument: more constraints cannot help."""
    lo, hi, y = _partial_order(seed=2)
    lp = isotonic_quantile_poset(lo, hi, y, tau)
    pav = isotonic_quantile_ordered(np.lexsort((hi, lo)), y, tau, bins=0)
    assert pinball_mean(lp, y, tau) <= pinball_mean(pav, y, tau) + 1e-9


def test_lp_interpolates_exactly_when_the_order_is_an_antichain():
    """Fully nested intervals are pairwise incomparable, so nothing constrains the fit.

    An unconstrained isotonic regression reproduces the data exactly and attains zero loss. (An
    earlier version of this test wrongly expected a constant fit; the constant is what you get
    under a TOTAL order with a decreasing signal, not under an antichain.)
    """
    n = 60
    hw = np.arange(1, n + 1, dtype=float)
    lo, hi = -hw, hw
    rng = np.random.default_rng(4)
    y = rng.normal(2.0, 0.5, n)
    assert len(comparable_pairs(lo, hi)) == 0
    q = isotonic_quantile_poset(lo, hi, y, 0.5)
    assert pinball_mean(q, y, 0.5) == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(q, y, atol=1e-6)


def test_comparable_pairs_excludes_self_and_is_correct():
    lo = np.array([0.0, 1.0, 0.5])
    hi = np.array([1.0, 2.0, 0.4])
    pairs = {tuple(p) for p in comparable_pairs(lo, hi)}
    assert (0, 1) in pairs
    assert (2, 1) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs   # nested: incomparable
    assert not any(a == b for a, b in pairs)


# ------------------------------------------------------------------ MCB ordering
@pytest.mark.parametrize("level", [50, 80, 95])
def test_mcb_linear_never_exceeds_mcb_exact(level):
    """The guarantee the paper relies on, at every interval level."""
    lo, hi, y = _partial_order(n=140, seed=5)
    assert mcb_linear(lo, hi, y, level) <= mcb_exact(lo, hi, y, level) + 1e-9


# ------------------------------------------------------------ real-data guards
def _real():
    p = "results/idr_exact_gisaid.json"
    if not os.path.exists(p):
        pytest.skip("exact-IDR validation not run")
    with open(p) as fh:
        return json.load(fh)


def test_real_validation_has_no_ordering_violations():
    """F19's headline. Any violation means a bug, not noise."""
    assert _real()["total_violations"] == 0


def test_real_gap_is_small_and_tightest_at_the_95_level():
    r = _real()
    assert r["mean_rel_gap"] < 0.10, "shortcut should be within ~10% of exact"
    by = r["by_level"]
    assert by["95"]["rel_gap"] < by["50"]["rel_gap"], "tightest where the tail claim lives"


def test_real_exact_always_at_least_linear_per_level():
    for lvl, v in _real()["by_level"].items():
        assert v["mcb_exact"] >= v["mcb_linear"], f"level {lvl} violates the bound"
