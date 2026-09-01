"""Tests for the regression-to-the-mean correction.

F16 retracted a headline claim because binning by the forecast and then measuring that forecast's
error inflated the gradient 2.78-fold. These tests exist so the RTM-free conditioning cannot quietly
stop being the default, and so the artifact ships the corrected diagnostic rather than the
convenient one.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from heterosked import LABS, overconfidence_table, spread  # noqa: E402


def _synthetic(n=40000, seed=0, true_gradient=False):
    """Homoscedastic by construction unless `true_gradient` is set.

    The forecast median is a noisy view of the outcome, which is exactly the setup that produces
    regression to the mean when you bin by the forecast.
    """
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0, 1, n)                 # pre-forecast instrument
    # NO clipping: clipping at [0, 1] induces genuine boundary heteroscedasticity, which would
    # confound the very thing these tests isolate. Values may stray slightly outside the unit
    # interval; the binning edges (-0.001, 1.001) absorb that.
    truth = baseline + rng.normal(0, 0.06, n)
    median = baseline + rng.normal(0, 0.06, n)
    # A genuine gradient means NARROWER intervals where we want overconfidence to be worse,
    # so that |error| / half-width rises there.
    half = np.where(baseline < 0.05, 0.02, 0.08) if true_gradient else np.full(n, 0.05)
    return pd.DataFrame({
        "median": median, "truth": truth, "naive": baseline,
        "abs_error": np.abs(truth - median),
        "width_95": 2 * half, "cov_95": (np.abs(truth - median) <= half).astype(int),
        "wis": np.abs(truth - median),
    })


def test_table_covers_the_expected_bins():
    g = overconfidence_table(_synthetic(4000), "median", "x")
    assert set(g.index) <= set(LABS)
    assert (g.n > 0).all()


def test_ratio_is_error_over_half_width():
    df = _synthetic(4000)
    g = overconfidence_table(df, "naive", "x")
    b = g.index[0]
    assert g.loc[b, "ratio"] == pytest.approx(g.loc[b, "abs_err"] / (g.loc[b, "width"] / 2))


def test_rtm_is_induced_by_conditioning_on_the_forecast():
    """The core demonstration: homoscedastic data, yet binning by the forecast shows a gradient."""
    df = _synthetic(40000)
    s_fc = spread(overconfidence_table(df, "median", "a"))
    s_bl = spread(overconfidence_table(df, "naive", "c"))
    assert s_fc["ratio"] > s_bl["ratio"], "forecast-conditioning should look more heteroscedastic"
    assert s_bl["ratio"] < 1.6, f"instrument should be near-flat on flat data, got {s_bl['ratio']}"


def test_instrument_recovers_a_genuine_gradient():
    """The correction must not be so blunt that it erases real heteroscedasticity."""
    df = _synthetic(40000, true_gradient=True)
    g = overconfidence_table(df, "naive", "c")
    assert g.loc["<1%", "ratio"] > 1.5 * g.loc["35-65%", "ratio"]


def test_spread_needs_enough_bins():
    s = spread(pd.DataFrame({"ratio": [1.0, 2.0]}))
    assert np.isnan(s["ratio"])


# ------------------------------------------------------------ real-data guards
def _real(label="gisaid"):
    for cand in (f"results/heterosked_{label}_full.json",
                 f"results/heterosked_{label}.json"):
        if os.path.exists(cand):
            with open(cand) as fh:
                return json.load(fh)
    pytest.skip("heteroscedasticity check not computed")


def test_real_rtm_inflates_the_forecast_conditioned_gradient():
    """F16's headline. If this stops holding, the retraction should be revisited."""
    c = _real()["conditionings"]
    a = c["(a) forecast level"]["spread"]["ratio"]
    inst = c["(c) baseline level, pre-forecast"]["spread"]["ratio"]
    assert a > inst, "forecast-level conditioning should overstate the gradient"
    assert a / inst > 2.0, f"expected ~2.8x inflation on gisaid, got {a / inst:.2f}x"


def test_real_low_frequency_is_not_the_worst_bin_under_the_instrument():
    """The retracted claim, pinned so the draft cannot silently restore it."""
    c = _real()["conditionings"]["(c) baseline level, pre-forecast"]
    assert c["worst_bin"] != "<1%", "on gisaid the <1% claim does not survive RTM correction"


def test_real_heteroscedasticity_still_exists():
    """The recalibration's motivation must survive the correction."""
    for label in ("gisaid", "open"):
        c = _real(label)["conditionings"].get("(c) baseline level, pre-forecast")
        if not c:
            continue
        assert c["spread"]["ratio"] > 1.5, f"{label}: some heteroscedasticity must remain"


def test_real_open_branch_still_shows_the_low_frequency_effect():
    """Reported honestly as branch-dependent, so both directions are pinned."""
    c = _real("open")["conditionings"].get("(c) baseline level, pre-forecast")
    if not c:
        pytest.skip("open instrument unavailable")
    assert c["worst_bin"] == "<1%"
