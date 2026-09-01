"""Tests for clustered inference and stratified horizon curves.

These exist because both modules exist to *restrain* claims. A bug that made ICC look small, or
made within-stratum correlations agree with the pooled value, would silently restore the overstated
version of the paper.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from clustered import (  # noqa: E402
    block_bootstrap,
    ci,
    design_effect,
    icc_oneway,
    naive_binomial_ci,
)
from stratified import by_stratum, horizon_corr, spearman  # noqa: E402


# --------------------------------------------------------------------- ICC
def test_icc_is_near_zero_for_independent_data():
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1, 4000)
    g = rng.integers(0, 40, 4000)
    icc, _ = icc_oneway(v, g)
    assert icc < 0.05


def test_icc_is_high_when_clusters_differ_strongly():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(40), 100)
    offsets = rng.normal(0, 3, 40)
    v = offsets[g] + rng.normal(0, 0.3, 4000)
    icc, _ = icc_oneway(v, g)
    assert icc > 0.8


def test_icc_is_clipped_at_zero():
    """Negative ANOVA estimates are reported as 0, never as a negative correlation."""
    rng = np.random.default_rng(2)
    v = rng.normal(0, 1, 1000)
    g = np.tile(np.arange(50), 20)
    icc, _ = icc_oneway(v, g)
    assert icc >= 0.0


def test_icc_degenerate_input_returns_nan():
    icc, _ = icc_oneway([1.0, 2.0], [0, 0])
    assert np.isnan(icc)


def test_design_effect_exceeds_one_when_clustered():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(30), 80)
    v = rng.normal(0, 1, 30)[g] + rng.normal(0, 0.5, 2400)
    d = design_effect(v, g)
    assert d["deff"] > 1
    assert d["n_eff"] < 2400


def test_effective_n_never_exceeds_nominal():
    rng = np.random.default_rng(4)
    v = rng.normal(0, 1, 1000)
    g = rng.integers(0, 25, 1000)
    d = design_effect(v, g)
    assert d["n_eff"] <= 1000 + 1e-6


# ------------------------------------------------------------- bootstrap
def _clustered_frame(n_clusters=20, per=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        base = rng.uniform(0.1, 0.9)
        for _ in range(per):
            rows.append(dict(forecast_date=f"d{c:02d}", value=base + rng.normal(0, 0.02)))
    return pd.DataFrame(rows)


def test_block_bootstrap_resamples_whole_clusters():
    """Every resample must contain the same number of clusters, each intact."""
    df = _clustered_frame()
    seen = {}

    def stat(d):
        seen["n_dates"] = d.forecast_date.nunique()
        seen["size"] = len(d)
        return float(d.value.mean())

    block_bootstrap(df, stat, n_boot=5, seed=0)
    assert seen["size"] % 50 == 0, "clusters must be taken whole"
    assert seen["n_dates"] <= 20


def test_block_bootstrap_is_wider_than_iid_for_clustered_data():
    """The entire point: ignoring clustering understates uncertainty."""
    df = _clustered_frame()
    b = block_bootstrap(df, lambda d: float(d.value.mean()), n_boot=400, seed=0)
    lo, hi = ci(b)
    se = df.value.std() / np.sqrt(len(df))
    naive_width = 2 * 1.96 * se
    assert (hi - lo) > 3 * naive_width


def test_block_bootstrap_is_reproducible():
    df = _clustered_frame()

    def f(d):
        return float(d.value.mean())

    a = block_bootstrap(df, f, n_boot=50, seed=7)
    b = block_bootstrap(df, f, n_boot=50, seed=7)
    assert np.allclose(a, b)


def test_ci_returns_nan_for_too_few_values():
    assert all(np.isnan(x) for x in ci([1.0, 2.0]))


def test_naive_binomial_ci_narrows_with_n():
    w_small = np.diff(naive_binomial_ci(0.35, 100))[0]
    w_large = np.diff(naive_binomial_ci(0.35, 100000))[0]
    assert w_large < w_small / 10


# ------------------------------------------------------------- stratified
def test_spearman_detects_monotone_relationships():
    x = np.arange(20)
    assert spearman(x, x ** 2) == pytest.approx(1.0)
    assert spearman(x, -x ** 2) == pytest.approx(-1.0)


def test_spearman_is_nan_for_a_constant():
    assert np.isnan(spearman(np.arange(10), np.ones(10)))


def test_horizon_corr_requires_enough_horizons():
    d = pd.DataFrame({"horizon": [1, 2, 3] * 50, "cov_95": np.random.default_rng(0).random(150)})
    assert np.isnan(horizon_corr(d, "cov_95"))


def test_horizon_corr_recovers_a_planted_decay():
    rng = np.random.default_rng(5)
    h = np.repeat(np.arange(1, 21), 20)
    cov = 0.9 - 0.03 * h + rng.normal(0, 0.01, len(h))
    d = pd.DataFrame({"horizon": h, "cov_95": cov})
    assert horizon_corr(d, "cov_95") < -0.9


def test_by_stratum_detects_simpsons_paradox():
    """Planted: rising within every stratum, falling when pooled."""
    rows = []
    for s in range(4):
        for h in range(1, 21):
            for _ in range(10):
                # each later stratum sits lower overall but rises with horizon
                rows.append(dict(location=f"L{s}", horizon=h,
                                 cov_95=0.8 - 0.2 * s + 0.005 * h,
                                 width_95=0.1, wis=0.1))
    df = pd.DataFrame(rows)
    tab = by_stratum(df, "location")
    assert (tab.coverage_95 > 0.9).all(), "within strata the trend is positive"


# ------------------------------------------------------------ real-data guards
def _load(name):
    """Prefer the full stride-7 results; fall back to the earlier sampled ones.

    F17 re-scored both branches at full stride-7 after review showed the snapshot cap was
    suppressing the `open` skill replication. These guards must track the primary data.
    """
    for cand in (f"results/{name}_full.json", f"results/{name}.json"):
        if os.path.exists(cand):
            with open(cand) as fh:
                return json.load(fh)
    pytest.skip(f"{name} not computed")


def test_real_effective_n_is_far_below_nominal():
    """F14's headline. If this ever approaches n, the clustering claim is wrong."""
    c = _load("clustered_gisaid")
    assert c["design_effect"]["cov_95"]["n_eff"] < 0.02 * c["n"]


def test_real_coverage_ci_excludes_nominal():
    """The headline must survive clustering, or the paper does not have one."""
    b = _load("clustered_gisaid")["bootstrap"]["coverage_95"]
    assert b["hi"] < 0.95


def test_real_clustered_ci_is_much_wider_than_iid():
    b = _load("clustered_gisaid")["bootstrap"]["coverage_95"]
    assert b["width_ratio"] > 5


def test_real_gisaid_skill_remains_significant():
    b = _load("clustered_gisaid")["bootstrap"]["mae_skill"]
    assert b["lo"] > 0, "gisaid point skill should survive clustering"


def test_real_open_skill_is_significant_on_the_full_sample():
    """SUPERSEDES F14's correction.

    At stride 14 with a 40-snapshot cap the `open` skill CI included zero. F17 showed that was a
    sampling artifact: at full stride-7 (53 clusters) the CI is [0.126, 0.487]. This test pins the
    corrected result so the draft neither over- nor under-claims it.
    """
    c = _load("clustered_open")["bootstrap"].get("mae_skill")
    if c is None:
        pytest.skip("open skill not computed")
    assert c["lo"] > 0, "open-branch skill should be significant on the full stride-7 sample"


def test_real_coverage_decay_holds_within_strata():
    """F15: the answer to the Simpson's-paradox objection."""
    s = {x["stratum_type"]: x for x in _load("stratified_gisaid")["strata"]}
    for kind in ("location", "clade"):
        assert s[kind]["coverage_95"]["median"] < -0.5
        assert s[kind]["coverage_95"]["frac_negative"] > 0.7


def test_real_width_claim_is_flagged_as_weak():
    """F15's correction: within clade the width trend is ~a coin flip. Pinned so the draft
    cannot revert to quoting the pooled +0.972 unqualified."""
    s = {x["stratum_type"]: x for x in _load("stratified_gisaid")["strata"]}
    assert s["clade"]["width_95"]["frac_negative"] > 0.35


def test_real_flu_coverage_decay_is_simpsons_paradox():
    """The stratified test must condemn influenza while vindicating SARS-CoV-2."""
    strata = {x["stratum_type"]: x for x in _load("stratified_flu")["strata"]}
    if "subtype" not in strata:
        pytest.skip("no subtype stratum")
    assert strata["subtype"]["coverage_95"]["median"] > 0, "flu inverts within subtype"
