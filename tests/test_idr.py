"""Tests for the interval-score decomposition and its applicability diagnostic."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from comparability import comparable_fraction, zero_arm_fraction  # noqa: E402
from idr_decompose import (  # noqa: E402
    decompose_interval,
    interval_score,
    isotonic_quantile_ordered,
    linear_extension,
)
from score import wis  # noqa: E402


# --------------------------------------------------------------- interval score
def test_covered_point_costs_only_the_width():
    assert interval_score(0.2, 0.8, 0.5, 0.05) == pytest.approx(0.6)


def test_interval_score_penalises_both_directions_equally():
    a = 0.1
    below = interval_score(0.4, 0.6, 0.3, a)
    above = interval_score(0.4, 0.6, 0.7, a)
    assert below == pytest.approx(above)
    assert below > interval_score(0.4, 0.6, 0.5, a)


def test_penalty_grows_as_alpha_shrinks():
    assert interval_score(0.4, 0.6, 0.9, 0.05) > interval_score(0.4, 0.6, 0.9, 0.5)


def test_interval_score_agrees_with_the_independent_wis_implementation():
    """Cross-check against score.wis for K=1, using the standard weighting."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = rng.uniform(0.2, 0.8)
        h = rng.uniform(0.01, 0.2)
        y = rng.uniform(0, 1)
        lo, hi, a = m - h, m + h, 0.05
        expect = (0.5 * abs(y - m) + (a / 2) * interval_score(lo, hi, y, a)) / 1.5
        assert wis(y, m, {95: (lo, hi)}) == pytest.approx(expect, abs=1e-12)


# ------------------------------------------------------------ linear extensions
@pytest.mark.parametrize("mode", ["lo_then_hi", "hi_then_lo", "midpoint"])
def test_linear_extension_respects_the_partial_order(mode):
    """If i precedes j componentwise, it must precede j in the extension too.

    This is what licenses the lower-bound claim: a linear extension only ever ADDS constraints.
    """
    rng = np.random.default_rng(1)
    lo = rng.uniform(0, 1, 200)
    hi = lo + rng.uniform(0.01, 0.4, 200)
    order = linear_extension(lo, hi, mode)
    pos = np.empty(len(lo), int)
    pos[order] = np.arange(len(lo))
    for i in range(0, 200, 7):
        for j in range(0, 200, 11):
            if lo[i] <= lo[j] and hi[i] <= hi[j] and (lo[i] < lo[j] or hi[i] < hi[j]):
                assert pos[i] <= pos[j], f"{mode} violates the partial order"


def test_unknown_extension_raises():
    with pytest.raises(ValueError):
        linear_extension(np.zeros(3), np.ones(3), "nonsense")


# ------------------------------------------------------------------- isotonic
def test_ordered_isotonic_is_monotone_along_the_order():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 1, 400)
    order = np.argsort(rng.uniform(0, 1, 400))
    f = isotonic_quantile_ordered(order, y, 0.5, bins=0)
    assert np.all(np.diff(f[order]) >= -1e-9)


def test_ordered_isotonic_recovers_a_monotone_signal():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 1, 600))
    y = x + rng.normal(0, 0.05, 600)
    f = isotonic_quantile_ordered(np.arange(600), y, 0.5, bins=60)
    assert np.corrcoef(f, x)[0, 1] > 0.95


# ---------------------------------------------------------------- decomposition
def _frame(n=3000, seed=0, half_width=0.01):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.05, 0.95, n)
    y = np.clip(m + rng.normal(0, 0.15, n), 0, 1)
    return pd.DataFrame({"median": m, "truth": y,
                         "lo_95": m - half_width, "hi_95": m + half_width})


def test_identity_closes():
    df = _frame()
    r = decompose_interval(df.lo_95, df.hi_95, df.truth, 95)
    assert abs(r["identity_residual"]) < 1e-9
    assert r["score"] == pytest.approx(r["mcb"] - r["dsc"] + r["unc"], abs=1e-9)


def test_mcb_is_large_for_an_overconfident_forecast():
    df = _frame(half_width=0.01)
    r = decompose_interval(df.lo_95, df.hi_95, df.truth, 95)
    assert r["mcb"] / r["score"] > 0.3


def test_mcb_is_small_for_a_well_specified_forecast():
    rng = np.random.default_rng(7)
    n = 4000
    m = rng.uniform(0.05, 0.95, n)
    y = np.clip(m + rng.normal(0, 0.05, n), 0, 1)
    r = decompose_interval(m - 1.96 * 0.05, m + 1.96 * 0.05, y, 95)
    assert r["mcb"] / r["score"] < 0.25


def test_mcb_is_non_negative():
    for seed in range(4):
        df = _frame(seed=seed)
        r = decompose_interval(df.lo_95, df.hi_95, df.truth, 95)
        assert r["mcb"] >= -1e-9


def test_too_few_points_returns_none():
    assert decompose_interval([0.1] * 10, [0.2] * 10, [0.15] * 10, 95) is None


# ------------------------------------------------------------- comparability
def test_comparable_fraction_is_one_for_a_totally_ordered_family():
    lo = np.arange(300, dtype=float)
    hi = lo + 1.0
    r = comparable_fraction(lo, hi, np.random.default_rng(0), n_sample=300, repeats=1)
    assert r["comparable"] == pytest.approx(1.0)


def test_comparable_fraction_is_low_for_a_nested_family():
    """Nested intervals are pairwise INcomparable: the pathology Remark 11 warns about."""
    hw = np.arange(1, 301, dtype=float)
    r = comparable_fraction(-hw, hw, np.random.default_rng(0), n_sample=300, repeats=1)
    assert r["comparable"] < 0.05


def test_zero_arm_fraction_detects_median_on_endpoint():
    df = pd.DataFrame({"median": [0.0, 0.5], "lo_95": [0.0, 0.4], "hi_95": [0.3, 0.6]})
    z = zero_arm_fraction(df, 95)
    assert z["zero_lower"] == pytest.approx(0.5)
    assert z["zero_either"] == pytest.approx(0.5)


# ------------------------------------------------------------- real-data guards
def _real(label="gisaid"):
    import json
    for cand in (f"results/idr_decomposition_{label}_full.json",
                 f"results/idr_decomposition_{label}.json"):
        if os.path.exists(cand):
            with open(cand) as fh:
                return json.load(fh)
    pytest.skip("IDR decomposition not computed")


@pytest.mark.parametrize("label", ["gisaid", "open", "flu"])
def test_real_two_conditioning_schemes_agree(label):
    """F13's headline. If these diverge, the weaker-notion objection returns."""
    r = _real(label)
    mi, mq = r["interval"]["mcb"], r["quantile_wise"]["mcb"]
    assert abs(mi - mq) / mq < 0.05, f"{label}: MCB differs by {(mi - mq) / mq:.1%}"


def test_real_mcb_share_rises_with_interval_level():
    per = {r["level"]: r["mcb"] / r["score"] for r in _real()["per_level"]}
    assert per[50] < per[80] < per[95]


def test_real_comparability_meets_remark_11():
    p = "results/comparability_gisaid.json"
    if not os.path.exists(p):
        pytest.skip("comparability not computed")
    import json
    with open(p) as fh:
        c = json.load(fh)
    for lvl in ("50", "80", "95"):
        assert c["levels"][lvl]["pooled_comparable"] > 0.8
        assert c["levels"][lvl]["isotonic_rank_corr"] > 0.5
