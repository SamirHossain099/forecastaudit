"""Tests for the scoring layer.

Every test here exists because something in this project was wrong in a way that produced a
plausible number rather than an error. Plausible-but-wrong is the failure mode that matters.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import score  # noqa: E402
from score import extract_forecast, extract_truth, is_catchall, wis  # noqa: E402


# --------------------------------------------------------------------------- WIS
def test_wis_zero_for_perfect_forecast():
    assert wis(0.5, 0.5, {50: (0.5, 0.5), 80: (0.5, 0.5), 95: (0.5, 0.5)}) == 0.0


def test_wis_matches_hand_computation():
    iv = {50: (0.4, 0.6), 80: (0.3, 0.7), 95: (0.2, 0.8)}
    expected = (0.5 * 0.0 + (0.5 / 2) * 0.2 + (0.2 / 2) * 0.4 + (0.05 / 2) * 0.6) / 3.5
    assert wis(0.5, 0.5, iv) == pytest.approx(expected)


def test_wis_penalises_exceedance_monotonically():
    iv = {50: (0.4, 0.6), 80: (0.3, 0.7), 95: (0.2, 0.8)}
    inside, edge, outside = wis(0.5, 0.5, iv), wis(0.8, 0.5, iv), wis(1.0, 0.5, iv)
    assert inside < edge < outside


def test_wis_symmetric_in_direction():
    iv = {50: (0.4, 0.6), 80: (0.3, 0.7), 95: (0.2, 0.8)}
    assert wis(0.5 - 0.3, 0.5, iv) == pytest.approx(wis(0.5 + 0.3, 0.5, iv))


def test_wis_rewards_sharpness_when_both_are_correct():
    """Two forecasts that both cover: the narrower one must score better."""
    wide = {50: (0.2, 0.8), 80: (0.1, 0.9), 95: (0.0, 1.0)}
    tight = {50: (0.45, 0.55), 80: (0.4, 0.6), 95: (0.35, 0.65)}
    assert wis(0.5, 0.5, tight) < wis(0.5, 0.5, wide)


# ------------------------------------------------------------------- catch-all
@pytest.mark.parametrize("v", ["other", "Other", " OTHER ", "others", "residual",
                              "unassigned", "Unassigned", " UNASSIGNED "])
def test_catchall_detected(v):
    assert is_catchall(v)


def test_covid_clades_scored_contain_no_catchall_under_the_wider_rule():
    """Extending is_catchall for flu must not retroactively invalidate F7/F8."""
    p = "results/scores_matched_lag14_gisaid.csv"
    if not os.path.exists(p):
        pytest.skip("no scored results on disk")
    assert not pd.read_csv(p).variant.map(is_catchall).any()


@pytest.mark.parametrize("v", ["23B", "JN.1", "25C", "XBB.1.5", "24A", "J.2.2",
                              "K:145N", "K"])
def test_named_clades_are_not_catchall(v):
    assert not is_catchall(v)


# ------------------------------------------------------------------ extraction
def _obj(recs):
    return {"metadata": {"variants": ["A", "B"]}, "data": recs}


def test_extract_forecast_keeps_only_forecast_site():
    o = _obj([
        {"site": "freq_forecast", "location": "X", "variant": "A", "date": "2024-01-01",
         "value": 0.3, "ps": "median"},
        {"site": "freq", "location": "X", "variant": "A", "date": "2024-01-01",
         "value": 0.9, "ps": "median"},
    ])
    f = extract_forecast(o)
    assert f[("X", "A", "2024-01-01")] == {"median": 0.3}


def test_extract_drops_nulls_rather_than_coercing():
    """A null value must vanish, never become 0.0 — that would fake a zero frequency."""
    o = _obj([
        {"site": "weekly_raw_freq", "location": "X", "variant": "A", "date": "2024-01-01",
         "value": None},
        {"site": "weekly_raw_freq", "location": "X", "variant": "B", "date": "2024-01-01",
         "value": 0.25},
    ])
    t = extract_truth(o, "weekly_raw_freq")
    assert ("X", "A", "2024-01-01") not in t
    assert t[("X", "B", "2024-01-01")] == 0.25


def test_extract_drops_missing_dates():
    o = _obj([{"site": "freq_forecast", "location": "X", "variant": "A", "date": None,
               "value": 0.3, "ps": "median"}])
    assert extract_forecast(o) == {}


# ----------------------------------------------------------------- cache bound
def test_cache_is_bounded_in_bytes_not_entries():
    """Regression: a 400-ENTRY cap held ~10 GB because snapshots span 0.3-25 MB."""
    assert score._CACHE_BYTE_LIMIT <= 1024 ** 3, "cache limit must stay under 1 GB"
    st = score.cache_stats()
    assert set(st) == {"entries", "mb", "limit_mb"}
    assert st["mb"] <= st["limit_mb"]


def test_cache_stats_reports_the_live_total():
    score._SNAPSHOT_CACHE.clear()
    score._CACHE_SIZES.clear()
    score._CACHE_BYTES = 0
    assert score.cache_stats()["entries"] == 0
    assert score.cache_stats()["mb"] == 0.0


# ------------------------------------------------- the asymmetry regression
def test_hdi_bounds_are_not_reconstructible_from_width():
    """Regression for the check that reported 95% coverage as 0.327 instead of 0.392.

    HDIs are asymmetric about the median. Reconstructing them as median +/- width/2 moves the
    band and silently changes coverage, so any analysis MUST use the stored lo_/hi_ columns.
    """
    lo, med, hi = 0.0, 0.02, 0.30          # median near the lower endpoint, as in real data
    width = hi - lo
    truth = 0.25
    assert lo <= truth <= hi                                  # truly covered
    assert not (med - width / 2 <= truth <= med + width / 2)  # symmetric reconstruction says no


def test_scored_output_carries_raw_bounds_when_present():
    """If results exist, coverage must equal the fraction inside the stored bounds."""
    p = "results/scores_matched_lag14_gisaid.csv"
    if not os.path.exists(p):
        pytest.skip("no scored results on disk")
    df = pd.read_csv(p)
    for lvl in (50, 80, 95):
        assert f"lo_{lvl}" in df and f"hi_{lvl}" in df, "raw bounds must be persisted"
        inside = ((df[f"lo_{lvl}"] <= df.truth) & (df.truth <= df[f"hi_{lvl}"])).mean()
        assert inside == pytest.approx(df[f"cov_{lvl}"].mean(), abs=1e-9), (
            f"cov_{lvl} disagrees with the stored bounds")


def test_intervals_are_nested_and_ordered():
    p = "results/scores_matched_lag14_gisaid.csv"
    if not os.path.exists(p):
        pytest.skip("no scored results on disk")
    df = pd.read_csv(p)
    assert (df.lo_95 <= df.lo_80 + 1e-12).all(), "95% lower must not exceed 80% lower"
    assert (df.hi_95 >= df.hi_80 - 1e-12).all(), "95% upper must not fall below 80% upper"
    assert (df.width_95 >= df.width_50 - 1e-12).all()


def test_horizons_are_strictly_out_of_sample():
    p = "results/scores_matched_lag14_gisaid.csv"
    if not os.path.exists(p):
        pytest.skip("no scored results on disk")
    df = pd.read_csv(p)
    assert (df.horizon > 0).all(), "a scored point at horizon <= 0 is not a forecast"


def test_no_catchall_survived_into_the_scores():
    p = "results/scores_matched_lag14_gisaid.csv"
    if not os.path.exists(p):
        pytest.skip("no scored results on disk")
    df = pd.read_csv(p)
    assert not df.variant.map(is_catchall).any(), "the residual category must be excluded"


def test_frequencies_are_in_range():
    p = "results/scores_matched_lag14_gisaid.csv"
    if not os.path.exists(p):
        pytest.skip("no scored results on disk")
    df = pd.read_csv(p)
    for c in ("truth", "median"):
        assert df[c].between(-1e-9, 1 + 1e-9).all(), f"{c} outside [0,1]"


# --------------------------------------------------------------- WIS vs coverage
def test_wis_and_coverage_agree_in_direction():
    """Dates with worse coverage must, on average, have worse WIS. A violation would mean the
    two metrics are not measuring the same forecasts."""
    p = "results/scores_matched_lag14_gisaid.csv"
    if not os.path.exists(p):
        pytest.skip("no scored results on disk")
    df = pd.read_csv(p)
    g = df.groupby("forecast_date").agg(cov=("cov_95", "mean"), wis=("wis", "mean"))
    # NB: g.cov is DataFrame.cov, the method — column access must be bracketed here.
    rho = np.corrcoef(g["cov"].rank(), g["wis"].rank())[0, 1]
    assert rho < -0.3, f"coverage and WIS should be negatively ranked, got rho={rho:.3f}"
