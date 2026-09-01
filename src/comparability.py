"""Is the IDR route of Allen, Burnello & Ziegel (2025) even available on this archive?

Their Remark 11 states three requirements for the sample decomposition terms to be practically
meaningful:

  1. "IDR typically requires an (effective) sample of size of at least n = 500 to provide
     reasonable estimates of the decomposition terms."
  2. "There is an (approximately) isotonic relationship between the interval forecasts and the
     observations."
  3. "The interval forecasts are (mostly) comparable, in that [l_i,u_i] <= [l_j,u_j] or
     [l_i,u_i] >= [l_j,u_j] holds for a large proportion of the data. Incomparable pairs of
     forecasts reduce the effective sample size used to train IDR."

Two forecasts are COMPARABLE when one interval is entirely shifted relative to the other --
l_i <= l_j and u_i <= u_j, or both >=. They are INCOMPARABLE (nested) when one contains the other.
Nesting is the pathology: their own analysis notes CQR "often produces nested interval forecasts,
reducing the effective sample size to fit IDR".

We have specific reason to worry. ~10% of these HDIs have an arm of zero length (the median sits on
an endpoint, typically at near-zero frequencies), which mechanically produces nesting.

WHY THIS MATTERS FOR THE PAPER. Our current decomposition conditions on one scalar quantile at a
time, which is *quantile-wise* calibration -- the notion Allen et al. explicitly decline at their
equation (8) as "strictly weaker than isotonic calibration". If the IDR route is available we should
either use it or show the two agree. If the comparability requirement fails on this data, that is a
principled, checkable reason to report the per-quantile version instead, and it must be stated
rather than left for a referee to notice.

Pairwise comparability is O(n^2), so it is estimated on random subsamples with a reported standard
error across repeats, and computed both pooled and within forecast date (points sharing a date share
a model fit, so the within-date structure is what IDR would actually exploit).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LEVELS = (50, 80, 95)


def comparable_fraction(lo, hi, rng, n_sample=4000, repeats=5):
    """Fraction of pairs that are comparable, estimated by subsampling.

    Comparable: (lo_i <= lo_j and hi_i <= hi_j) or (lo_i >= lo_j and hi_i >= hi_j).
    Ties in both coordinates count as comparable (the ordering is non-strict).
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    lo, hi = lo[ok], hi[ok]
    n = len(lo)
    if n < 3:
        return dict(n=n, comparable=float("nan"), se=float("nan"))
    fracs = []
    for _ in range(repeats):
        idx = rng.choice(n, size=min(n_sample, n), replace=False)
        a, b = lo[idx], hi[idx]
        m = len(idx)
        # chunked to bound memory: m x m boolean arrays at m=4000 are 16 MB each
        comp = 0
        total = 0
        step = 1000
        for s in range(0, m, step):
            al = a[s:s + step, None] - a[None, :]
            bh = b[s:s + step, None] - b[None, :]
            same = ((al <= 0) & (bh <= 0)) | ((al >= 0) & (bh >= 0))
            # exclude the diagonal block's self-pairs
            block = np.ones(same.shape, bool)
            rows = np.arange(s, min(s + step, m))
            block[np.arange(len(rows)), rows] = False
            comp += int((same & block).sum())
            total += int(block.sum())
        fracs.append(comp / total)
    return dict(n=int(n), comparable=float(np.mean(fracs)), se=float(np.std(fracs)))


def zero_arm_fraction(df, lvl):
    """Points where the median sits on an interval endpoint — a nesting generator."""
    lo, hi, m = df[f"lo_{lvl}"], df[f"hi_{lvl}"], df["median"]
    lower_arm = (m - lo).abs()
    upper_arm = (hi - m).abs()
    return dict(zero_lower=float((lower_arm < 1e-12).mean()),
                zero_upper=float((upper_arm < 1e-12).mean()),
                zero_either=float(((lower_arm < 1e-12) | (upper_arm < 1e-12)).mean()),
                zero_width=float(((hi - lo).abs() < 1e-12).mean()))


def isotonic_strength(df, lvl):
    """Requirement 2: do wider/higher intervals actually go with larger outcomes?

    Reported as the rank correlation between the interval midpoint and the observation. A weak or
    negative value means conditional-calibration assessment is, in their words, "somewhat
    unnecessary, since the interval forecasts are clearly not informative".
    """
    mid = (df[f"lo_{lvl}"] + df[f"hi_{lvl}"]) / 2
    y = df.truth
    ok = np.isfinite(mid) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    r = pd.Series(mid[ok]).rank()
    s = pd.Series(y[ok]).rank()
    return float(np.corrcoef(r, s)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_matched_lag14_gisaid_dense.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--n-sample", type=int, default=4000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/comparability_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = pd.read_csv(a.infile)
    rng = np.random.default_rng(0)
    print(f"{a.label}: {len(df):,} points, {df.forecast_date.nunique()} forecast dates")
    print("Allen, Burnello & Ziegel (2025) Remark 11 requirements\n")

    res = {"label": a.label, "n": int(len(df)),
           "n_forecast_dates": int(df.forecast_date.nunique()), "levels": {}}

    print("=== req 3: are the interval forecasts (mostly) COMPARABLE? ===")
    print(f"  {'level':>6s} {'pooled':>16s} {'within-date (median)':>22s} {'zero-length arm':>16s}")
    for lvl in LEVELS:
        pooled = comparable_fraction(df[f"lo_{lvl}"], df[f"hi_{lvl}"], rng,
                                     a.n_sample, a.repeats)
        within = []
        for _, g in df.groupby("forecast_date"):
            if len(g) < 50:
                continue
            within.append(comparable_fraction(g[f"lo_{lvl}"], g[f"hi_{lvl}"], rng,
                                              min(a.n_sample, len(g)), 1)["comparable"])
        z = zero_arm_fraction(df, lvl)
        iso = isotonic_strength(df, lvl)
        wmed = float(np.nanmedian(within)) if within else float("nan")
        print(f"  {lvl:>5d}% {pooled['comparable']:>10.1%} +-{pooled['se']:.3f} "
              f"{wmed:>21.1%} {z['zero_either']:>15.1%}")
        res["levels"][str(lvl)] = dict(pooled_comparable=pooled["comparable"],
                                       pooled_se=pooled["se"],
                                       within_date_median=wmed,
                                       within_date_min=float(np.nanmin(within)) if within else None,
                                       within_date_max=float(np.nanmax(within)) if within else None,
                                       isotonic_rank_corr=iso, **z)

    print("\n=== req 2: is the forecast-outcome relationship (approximately) isotonic? ===")
    for lvl in LEVELS:
        print(f"  {lvl:>3d}%: rank corr(interval midpoint, outcome) = "
              f"{res['levels'][str(lvl)]['isotonic_rank_corr']:+.3f}")

    print("\n=== req 1: effective sample size ===")
    print(f"  nominal n                         : {len(df):,}")
    print(f"  forecast dates (clustering unit)  : {df.forecast_date.nunique()}")
    for lvl in LEVELS:
        c = res["levels"][str(lvl)]["pooled_comparable"]
        print(f"  {lvl}% level: comparable fraction {c:.1%} "
              f"-> IDR sees roughly {c * len(df):,.0f} usable pairs' worth of ordering")
    print("  (their threshold: effective n >= 500)")

    print("\n=== verdict ===")
    worst = min(res["levels"][str(v)]["pooled_comparable"] for v in LEVELS)
    if worst >= 0.80:
        print(f"  Comparability is high (min {worst:.1%}). The IDR route IS available; implement")
        print("  it and show the two decompositions agree.")
    elif worst >= 0.5:
        print(f"  Comparability is moderate (min {worst:.1%}). IDR is usable but the effective")
        print("  sample is materially reduced; report the fraction alongside any IDR result.")
    else:
        print(f"  Comparability is LOW (min {worst:.1%}). Nested intervals dominate, which is")
        print("  exactly the pathology their Remark 11 warns about. This is a principled,")
        print("  citable reason to report the per-quantile decomposition instead -- state it.")

    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
