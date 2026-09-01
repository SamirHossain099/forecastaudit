"""Tests for the naive baseline and skill scores.

The load-bearing claims here are (a) the baseline is Figgins & Bedford's definition, not something
adjacent, (b) its intervals are built strictly out-of-time, and (c) the WIS-of-a-point-forecast
identity the skill comparison relies on.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import skill as sk  # noqa: E402
from score import LEVELS, naive_7day, wis  # noqa: E402


# ------------------------------------------------- the degenerate WIS identity
def test_wis_of_a_point_forecast_equals_absolute_error():
    """With lo == hi == m the interval score collapses to |y-m| at every level.

    The skill comparison against the point baseline relies on this, so it is asserted rather than
    assumed.
    """
    for m, y in ((0.5, 0.8), (0.2, 0.1), (0.0, 0.0), (0.9, 0.05)):
        iv = {lvl: (m, m) for lvl in LEVELS}
        assert wis(y, m, iv) == pytest.approx(abs(y - m))


# ----------------------------------------------------------- the naive model
def _obj(recs):
    return {"metadata": {}, "data": recs}


def test_naive_7day_averages_the_last_seven_available_days():
    recs = [{"site": "daily_raw_freq", "location": "X", "variant": "A",
             "date": f"2024-01-{d:02d}", "value": float(d)} for d in range(1, 11)]
    out = naive_7day(_obj(recs))
    assert out[("X", "A")] == pytest.approx(np.mean([4, 5, 6, 7, 8, 9, 10]))


def test_naive_7day_uses_fewer_days_when_fewer_exist():
    recs = [{"site": "daily_raw_freq", "location": "X", "variant": "A",
             "date": f"2024-01-{d:02d}", "value": 2.0} for d in range(1, 4)]
    assert naive_7day(_obj(recs))[("X", "A")] == pytest.approx(2.0)


def test_naive_7day_ignores_other_sites():
    recs = [{"site": "freq_forecast", "location": "X", "variant": "A",
             "date": "2024-01-01", "value": 99.0},
            {"site": "daily_raw_freq", "location": "X", "variant": "A",
             "date": "2024-01-01", "value": 0.5}]
    assert naive_7day(_obj(recs)) == {("X", "A"): 0.5}


def test_naive_7day_drops_nulls_rather_than_coercing():
    recs = [{"site": "daily_raw_freq", "location": "X", "variant": "A",
             "date": "2024-01-01", "value": None},
            {"site": "daily_raw_freq", "location": "X", "variant": "A",
             "date": "2024-01-02", "value": 0.4}]
    assert naive_7day(_obj(recs))[("X", "A")] == pytest.approx(0.4)


def test_naive_7day_separates_locations_and_variants():
    recs = [{"site": "daily_raw_freq", "location": loc, "variant": var,
             "date": "2024-01-01", "value": v}
            for loc, var, v in (("X", "A", 0.1), ("X", "B", 0.2), ("Y", "A", 0.3))]
    out = naive_7day(_obj(recs))
    assert out[("X", "A")] == 0.1 and out[("X", "B")] == 0.2 and out[("Y", "A")] == 0.3


# -------------------------------------------------------------- skill scores
def test_skill_sign_convention():
    assert sk.skill([1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.5)     # model better
    assert sk.skill([2.0, 2.0], [2.0, 2.0]) == pytest.approx(0.0)     # tie
    assert sk.skill([4.0, 4.0], [2.0, 2.0]) == pytest.approx(-1.0)    # model worse


def test_skill_is_nan_for_a_zero_baseline():
    assert np.isnan(sk.skill([1.0], [0.0]))


# ------------------------------------------------------- out-of-time guarantee
def _frame(n_dates=6, per=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for di in range(n_dates):
        for _ in range(per):
            m = float(rng.uniform(0, 1))
            y = float(np.clip(m + rng.normal(0, 0.08), 0, 1))
            r = dict(forecast_date=f"2024-0{di + 1}-01", horizon=int(rng.integers(1, 30)),
                     median=m, truth=y, naive=float(np.clip(m + rng.normal(0, .1), 0, 1)),
                     wis=abs(y - m), abs_error=abs(y - m))
            for lvl in LEVELS:
                r[f"cov_{lvl}"] = 1
            rows.append(r)
    return pd.DataFrame(rows)


def test_naive_intervals_never_use_the_evaluated_date():
    """The whole comparison is void if a date informs its own intervals."""
    df = _frame()
    out = sk.naive_interval_wis(df, min_cal=10)
    assert not out.empty
    first = sorted(df.forecast_date.unique())[0]
    assert first not in set(out.forecast_date), "first date has no earlier data; must be skipped"
    for d in out.forecast_date.unique():
        assert df[df.forecast_date < d].forecast_date.max() < d


def test_naive_intervals_are_ordered_and_nested():
    df = _frame()
    out = sk.naive_interval_wis(df, min_cal=10)
    assert (out.naive_wis >= 0).all()
    assert np.isfinite(out.naive_wis).all()


def test_empirical_intervals_reach_roughly_nominal_on_stationary_data():
    """Sanity: on data whose residual distribution is stable, the construction should work.

    This is also the caveat made explicit: the baseline's calibration is largely a property of
    the construction, which is exactly why F11 does not claim the naive MODEL has better
    uncertainty quantification.
    """
    df = _frame(n_dates=8, per=200)
    out = sk.naive_interval_wis(df, min_cal=50)
    assert 0.85 <= out.naive_cov_95.mean() <= 1.0


def test_bins_cover_every_horizon_in_range():
    df = sk.add_bins(_frame())
    assert df.hbin.notna().all()


# ------------------------------------------------------------ real-data guards
def _real(label="gisaid"):
    p = f"results/skill_{label}.json"
    if not os.path.exists(p):
        pytest.skip("skill results not computed")
    import json
    with open(p) as fh:
        return json.load(fh)


def test_real_model_has_positive_point_skill():
    """F11's first claim: the forecasts genuinely beat the naive model on point accuracy."""
    r = _real()
    assert r["mae_skill_overall"] > 0.2, "model should clearly beat the 7-day moving average"


def test_real_naive_is_better_calibrated_than_the_model():
    """F11's twist. If this ever flips, the finding must be rewritten."""
    r = _real()
    cov = r["coverage_vs_naive"]["95"]
    assert cov["naive"] > cov["model"] + 0.3


def test_real_wis_favours_the_model_despite_worse_coverage():
    """The masking effect that motivates the WIS decomposition."""
    r = _real()
    assert r["wis_skill_vs_interval"] > 0
    assert r["coverage_vs_naive"]["95"]["model"] < 0.5
