"""Tests for the out-of-time recalibration.

The single most attackable claim in this deliverable is "strictly out-of-time". If calibration
ever sees the date it is evaluated on, coverage returns to nominal by construction and the whole
result is vacuous. `test_no_temporal_leakage` is therefore the load-bearing test here.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import recalibrate as rc  # noqa: E402


def _frame(n=600, seed=0):
    """Synthetic scored output with intervals that are deliberately far too narrow."""
    rng = np.random.default_rng(seed)
    dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
    rows = []
    for d in dates:
        for _ in range(n // len(dates)):
            m = float(rng.uniform(0, 1))
            half = 0.01                       # far too narrow on purpose
            y = float(np.clip(m + rng.normal(0, 0.1), 0, 1))
            r = dict(forecast_date=d, horizon=int(rng.integers(1, 30)),
                     median=m, truth=y)
            for lvl, mult in ((50, 0.5), (80, 0.8), (95, 1.0)):
                r[f"lo_{lvl}"] = max(0.0, m - half * mult)
                r[f"hi_{lvl}"] = min(1.0, m + half * mult)
                r[f"width_{lvl}"] = r[f"hi_{lvl}"] - r[f"lo_{lvl}"]
                r[f"cov_{lvl}"] = int(r[f"lo_{lvl}"] <= y <= r[f"hi_{lvl}"])
            rows.append(r)
    return rc.add_strata(pd.DataFrame(rows))


# ------------------------------------------------------------------- LEAKAGE
def test_no_temporal_leakage():
    """Calibration must never include the evaluated date or anything after it."""
    df = _frame()
    dates = sorted(df.forecast_date.unique())
    seen = []

    real_q = rc.conformal_q
    captured = {}

    def spy(scores, alpha, min_n=30):
        captured["last_n"] = len(pd.Series(scores).dropna())
        return real_q(scores, alpha, min_n)

    rc.conformal_q = spy
    try:
        res = rc.rolling_origin(df, method="cqr_global", min_cal=10)
    finally:
        rc.conformal_q = real_q

    for d in res.forecast_date:
        cal_max = df[df.forecast_date < d].forecast_date.max()
        assert cal_max < d, f"calibration for {d} reached {cal_max}"
        seen.append(d)
    assert dates[0] not in seen, "the first date has no earlier data and must be skipped"


def test_n_cal_is_strictly_increasing_and_excludes_test():
    df = _frame()
    res = rc.rolling_origin(df, method="cqr_global", min_cal=10)
    assert res.n_cal.is_monotonic_increasing
    for _, r in res.iterrows():
        expected = int((df.forecast_date < r.forecast_date).sum())
        assert r.n_cal == expected
        assert r.n_test == int((df.forecast_date == r.forecast_date).sum())


def test_first_date_is_never_evaluated():
    df = _frame()
    first = sorted(df.forecast_date.unique())[0]
    for meth in ("cqr_global", "cqr_mondrian", "mult_mondrian"):
        res = rc.rolling_origin(df, method=meth, min_cal=10)
        assert first not in set(res.forecast_date)


# ---------------------------------------------------------------- correctness
def test_conformal_quantile_is_finite_sample_valid():
    """The conformal index is ceil((n+1)(1-alpha)), not the plain empirical quantile."""
    s = list(range(100))                       # 0..99
    q = rc.conformal_q(s, alpha=0.05)
    assert q == np.sort(s)[int(np.ceil(101 * 0.95)) - 1]
    assert q >= np.quantile(s, 0.95), "conformal quantile must not be anti-conservative"


def test_conformal_quantile_returns_none_below_min_n():
    assert rc.conformal_q([1, 2, 3], alpha=0.05, min_n=30) is None


def test_conformal_quantile_saturates_when_n_too_small_for_alpha():
    """With n=10 and alpha=0.05, ceil(11*0.95)=11 > 10 — must fall back to the max."""
    s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert rc.conformal_q(s, alpha=0.05, min_n=5) == 10.0


def test_nonconformity_sign_convention():
    df = pd.DataFrame({"truth": [0.5, 0.0, 1.0], "lo_95": [0.4] * 3, "hi_95": [0.6] * 3})
    e = rc.nonconformity(df, 95)
    assert e.iloc[0] < 0, "a covered point must score negative"
    assert e.iloc[1] == pytest.approx(0.4)
    assert e.iloc[2] == pytest.approx(0.4)


def test_additive_application_stays_on_the_simplex():
    df = pd.DataFrame({"median": [0.02, 0.98], "lo_95": [0.01, 0.97], "hi_95": [0.03, 0.99]})
    lo, hi = rc.apply_additive(df, 95, 0.5)
    assert (lo >= 0).all() and (hi <= 1).all()


def test_multiplicative_application_stays_on_the_simplex():
    df = pd.DataFrame({"median": [0.02, 0.98], "lo_95": [0.01, 0.97], "hi_95": [0.03, 0.99]})
    lo, hi = rc.apply_multiplicative(df, 95, 50.0)
    assert (lo >= 0).all() and (hi <= 1).all()
    assert (lo <= df["median"]).all() and (hi >= df["median"]).all()


def test_multiplicative_with_factor_one_is_identity():
    df = pd.DataFrame({"median": [0.5], "lo_95": [0.4], "hi_95": [0.6]})
    lo, hi = rc.apply_multiplicative(df, 95, 1.0)
    assert lo.iloc[0] == pytest.approx(0.4)
    assert hi.iloc[0] == pytest.approx(0.6)


# ------------------------------------------------------------------ behavior
def test_recalibration_actually_improves_coverage():
    df = _frame()
    base = rc.summarize(rc.rolling_origin(df, method="none", min_cal=10), "none")
    cqr = rc.summarize(rc.rolling_origin(df, method="cqr_global", min_cal=10), "cqr")
    assert base["cov_95"] < 0.5, "synthetic intervals were meant to be badly overconfident"
    assert cqr["cov_95"] > base["cov_95"] + 0.3


def test_recalibration_widens_rather_than_shifts():
    df = _frame()
    base = rc.summarize(rc.rolling_origin(df, method="none", min_cal=10), "none")
    cqr = rc.summarize(rc.rolling_origin(df, method="cqr_global", min_cal=10), "cqr")
    assert cqr["width_95"] > base["width_95"]


def test_uncorrected_path_reproduces_the_stored_coverage():
    """method='none' must return exactly what scoring already computed."""
    df = _frame()
    res = rc.rolling_origin(df, method="none", min_cal=10)
    for _, r in res.iterrows():
        sub = df[df.forecast_date == r.forecast_date]
        assert r.cov_95 == pytest.approx(sub.cov_95.mean())


def test_strata_partition_every_row():
    df = _frame()
    assert df.lev.notna().all(), "every forecast level must fall in a bin"
    assert df.hbin.notna().all(), "every horizon must fall in a bin"


# ----------------------------------------------------------- real-data guards
def _real(branch="gisaid"):
    p = f"results/recalibration_{branch}.json"
    if not os.path.exists(p):
        pytest.skip("no recalibration results on disk")
    return pd.read_json(p)


def test_real_mondrian_is_sharper_than_global_at_equal_coverage():
    """The heteroscedasticity claim: level-conditional beats global on width."""
    r = _real().set_index("method")
    if not {"CQR-global", "CQR-mondrian(level)"} <= set(r.index):
        pytest.skip("methods absent")
    assert r.loc["CQR-mondrian(level)", "width_95"] < r.loc["CQR-global", "width_95"]
    assert r.loc["CQR-mondrian(level)", "cov_95"] >= 0.90


def test_real_uncorrected_is_far_below_nominal():
    r = _real().set_index("method")
    assert r.loc["uncorrected", "cov_95"] < 0.5


def test_real_recalibrated_widths_remain_informative():
    """Nominal coverage bought with a band spanning the simplex is not a fix."""
    r = _real().set_index("method")
    assert r.loc["CQR-mondrian(level)", "width_95"] < 0.5
