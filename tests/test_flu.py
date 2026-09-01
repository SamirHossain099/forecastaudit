"""Tests for the influenza replication path (F9).

The flu archive differs from ncov in path layout, filename infix, site names and catch-all
vocabulary. Each of those was a chance to silently score nothing, or to score the wrong series.
"""
import datetime as dt
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import flu  # noqa: E402
from score import is_catchall  # noqa: E402


def test_partition_lifetime_on_a_constant_partition():
    idx = {"2026-01-01": frozenset("AB"), "2026-02-01": frozenset("AB"),
           "2026-03-01": frozenset("AB")}
    life = flu.partition_lifetime(idx)
    assert life["n_partitions"] == 1
    assert life["median_lifetime_d"] == 59.0          # 2026-01-01 -> 2026-03-01
    assert life["frac_ge_30d"] == 1.0


def test_partition_lifetime_counts_each_change():
    idx = {"2026-01-01": frozenset("A"), "2026-01-08": frozenset("AB"),
           "2026-01-15": frozenset("ABC")}
    life = flu.partition_lifetime(idx)
    assert life["n_partitions"] == 3
    assert life["median_lifetime_d"] == 7.0


def test_partition_lifetime_handles_a_single_snapshot():
    life = flu.partition_lifetime({"2026-01-01": frozenset("A")})
    assert life["n_partitions"] == 1
    assert life["median_lifetime_d"] == 0.0


def test_partition_lifetime_empty():
    assert flu.partition_lifetime({}) == {}


def test_unassigned_is_treated_as_a_catchall():
    """The flu-specific trap: a second residual category under a different name."""
    assert is_catchall("unassigned")
    assert not is_catchall("K:145N")
    assert not is_catchall("J.2.4.2")


def test_flu_filename_stem_parses_after_stripping_the_MLR_infix():
    """ncov files are `{date}_results.json`; flu files are `{date}_MLR_results.json`."""
    base = "2026-08-27_MLR_results.json"
    stem = base[: -len("_results.json")].replace("_MLR", "")
    assert dt.date.fromisoformat(stem) == dt.date(2026, 8, 27)


def test_flu_constants_are_consistent():
    assert flu.PREFIX.endswith("forecasts-flu/")
    assert set(flu.SUBTYPES) == {"h3n2", "h1n1pdm", "vic"}


# ------------------------------------------------------------- real-data guards
def _scores():
    p = "results/scores_flu.csv"
    if not os.path.exists(p):
        pytest.skip("no flu scores on disk")
    return pd.read_csv(p)


def test_flu_scores_carry_raw_bounds_and_agree_with_coverage():
    df = _scores()
    for lvl in (50, 80, 95):
        inside = ((df[f"lo_{lvl}"] <= df.truth) & (df.truth <= df[f"hi_{lvl}"])).mean()
        assert inside == pytest.approx(df[f"cov_{lvl}"].mean(), abs=1e-9)


def test_flu_excludes_both_catchalls():
    df = _scores()
    assert not df.variant.map(is_catchall).any()
    assert "unassigned" not in set(df.variant)
    assert "other" not in set(df.variant)


def test_flu_horizons_are_out_of_sample():
    assert (_scores().horizon > 0).all()


def test_flu_covers_all_three_subtypes():
    assert set(_scores().subtype) == {"h3n2", "h1n1pdm", "vic"}


def test_flu_undercovers_in_every_subtype():
    """F9's headline. If this ever fails, the replication claim must be rewritten."""
    df = _scores()
    for sub, g in df.groupby("subtype"):
        assert g.cov_95.mean() < 0.95, f"{sub} no longer under-covers at the 95% level"


def test_flu_frequencies_are_in_range():
    df = _scores()
    for c in ("truth", "median"):
        assert df[c].between(-1e-9, 1 + 1e-9).all()
