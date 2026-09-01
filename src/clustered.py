"""Clustered inference: the effective sample size is 48, not 105,349.

Every scored point carries a forecast date, and points sharing a date share a model fit, an
information set and a variant partition. They are nowhere near independent. Any interval computed as
if n = 105,349 is far too narrow, and forecast-evaluation referees are alert to exactly this.

The headline itself is safe -- 0.354 against a nominal 0.95 cannot be rescued by any variance
correction -- but the *inferential* statements built on the pooled count are not. This module
replaces them.

Two things are computed:

  DESIGN EFFECT   The intra-cluster correlation of each per-point outcome, and the implied effective
                  sample size n_eff = n / (1 + (m_bar - 1) * ICC). This is the number to quote in the
                  paper, not the raw count.

  BLOCK BOOTSTRAP Resample whole FORECAST DATES with replacement, recompute the statistic, and take
                  percentile intervals. This respects the clustering without assuming a variance
                  model. Reported alongside the naive i.i.d. interval so the understatement is
                  visible rather than merely asserted.

ICC is estimated by one-way ANOVA on the per-point outcome with forecast date as the grouping
factor, which is the standard estimator for unequal cluster sizes.
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


def icc_oneway(values, groups):
    """One-way ANOVA intra-cluster correlation, for unequal cluster sizes."""
    v = np.asarray(values, float)
    g = np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    keys, inv = np.unique(g, return_inverse=True)
    k = len(keys)
    n = len(v)
    if k < 2 or n <= k:
        return float("nan"), float("nan")
    counts = np.bincount(inv)
    means = np.bincount(inv, weights=v) / counts
    grand = v.mean()
    ms_between = float((counts * (means - grand) ** 2).sum() / (k - 1))
    ms_within = float(((v - means[inv]) ** 2).sum() / (n - k))
    # m0: the "average" cluster size for unequal designs
    m0 = (n - (counts ** 2).sum() / n) / (k - 1)
    if m0 <= 0 or ms_between + (m0 - 1) * ms_within == 0:
        return float("nan"), float(m0)
    icc = (ms_between - ms_within) / (ms_between + (m0 - 1) * ms_within)
    return float(max(icc, 0.0)), float(m0)


def design_effect(values, groups):
    icc, m0 = icc_oneway(values, groups)
    if not np.isfinite(icc):
        return dict(icc=np.nan, m0=m0, deff=np.nan, n_eff=np.nan)
    n = int(np.isfinite(np.asarray(values, float)).sum())
    deff = 1.0 + (m0 - 1.0) * icc
    return dict(icc=icc, m0=m0, deff=float(deff), n_eff=float(n / deff) if deff > 0 else np.nan)


def block_bootstrap(df, stat_fn, n_boot=1000, seed=0, group="forecast_date"):
    """Resample whole clusters with replacement; return the distribution of stat_fn."""
    rng = np.random.default_rng(seed)
    keys = df[group].unique()
    idx_by_key = {k: np.flatnonzero((df[group] == k).to_numpy()) for k in keys}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([idx_by_key[k] for k in draw])
        try:
            out.append(stat_fn(df.iloc[idx]))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, float)


def ci(vals, level=95):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return (np.nan, np.nan)
    a = (100 - level) / 2
    return (float(np.percentile(v, a)), float(np.percentile(v, 100 - a)))


def naive_binomial_ci(p, n, level=95):
    """Wilson interval, assuming independence: the thing we are replacing."""
    z = {90: 1.645, 95: 1.96, 99: 2.576}[level]
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


STATS = {
    "coverage_50": lambda d: float(d.cov_50.mean()),
    "coverage_80": lambda d: float(d.cov_80.mean()),
    "coverage_95": lambda d: float(d.cov_95.mean()),
    "mean_wis": lambda d: float(d.wis.mean()),
    "mean_abs_error": lambda d: float(d.abs_error.mean()),
}


def mae_skill(d):
    if "naive" not in d.columns:
        return np.nan
    ok = np.isfinite(d.naive)
    if ok.sum() < 10:
        return np.nan
    m = d.abs_error[ok].mean()
    nv = (d.naive[ok] - d.truth[ok]).abs().mean()
    return float(1 - m / nv) if nv > 0 else np.nan


def mcb_share(d, bins=100):
    from decompose import decompose_wis
    agg, _ = decompose_wis(d, bins=bins)
    return float(agg["mcb"] / agg["score"]) if agg and agg["score"] else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_matched_lag14_gisaid_dense.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-decomp", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/clustered_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = pd.read_csv(a.infile)
    n = len(df)
    k = df.forecast_date.nunique()
    print(f"{a.label}: {n:,} points in {k} forecast-date clusters "
          f"(mean {n / k:.0f} per cluster)\n")

    print("=== design effect: how much independent information is really here? ===")
    print(f"  {'outcome':>16s} {'ICC':>8s} {'deff':>8s} {'n_eff':>10s} {'n_eff/n':>9s}")
    deff = {}
    for name, col in (("cov_50", "cov_50"), ("cov_80", "cov_80"), ("cov_95", "cov_95"),
                      ("wis", "wis"), ("abs_error", "abs_error")):
        d = design_effect(df[col], df.forecast_date)
        deff[name] = d
        print(f"  {name:>16s} {d['icc']:>8.3f} {d['deff']:>8.1f} {d['n_eff']:>10,.0f} "
              f"{d['n_eff'] / n:>8.2%}")
    print(f"\n  clusters (forecast dates): {k}   <- the honest denominator for inference")

    print(f"\n=== block bootstrap over forecast dates ({a.n_boot} resamples) ===")
    print(f"  {'statistic':>16s} {'estimate':>10s} {'clustered 95% CI':>24s} "
          f"{'naive i.i.d. CI':>22s} {'width ratio':>12s}")
    res = {"n": n, "n_clusters": k, "design_effect": deff, "bootstrap": {}}
    for name, fn in STATS.items():
        est = fn(df)
        b = block_bootstrap(df, fn, n_boot=a.n_boot)
        lo, hi = ci(b)
        if name.startswith("coverage"):
            nlo, nhi = naive_binomial_ci(est, n)
        else:
            col = "wis" if name == "mean_wis" else "abs_error"
            se = df[col].std() / np.sqrt(n)
            nlo, nhi = est - 1.96 * se, est + 1.96 * se
        ratio = (hi - lo) / (nhi - nlo) if nhi > nlo else np.nan
        res["bootstrap"][name] = dict(estimate=est, lo=lo, hi=hi,
                                      naive_lo=nlo, naive_hi=nhi, width_ratio=ratio)
        print(f"  {name:>16s} {est:>10.4f} {f'[{lo:.4f}, {hi:.4f}]':>24s} "
              f"{f'[{nlo:.4f}, {nhi:.4f}]':>22s} {ratio:>11.1f}x")

    if "naive" in df.columns:
        b = block_bootstrap(df, mae_skill, n_boot=a.n_boot)
        est = mae_skill(df)
        lo, hi = ci(b)
        res["bootstrap"]["mae_skill"] = dict(estimate=est, lo=lo, hi=hi)
        print(f"  {'mae_skill':>16s} {est:>10.4f} {f'[{lo:.4f}, {hi:.4f}]':>24s}")
        print(f"    -> skill is {'CONFIRMED positive' if lo > 0 else 'NOT significant'} "
              f"under clustering")

    print(f"\n=== MCB share, block bootstrapped ({a.n_boot_decomp} resamples) ===")
    est = mcb_share(df)
    b = block_bootstrap(df, mcb_share, n_boot=a.n_boot_decomp, seed=1)
    lo, hi = ci(b)
    res["bootstrap"]["mcb_share"] = dict(estimate=est, lo=lo, hi=hi)
    print(f"  MCB share {est:.3f}   clustered 95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n=== verdict ===")
    c95 = res["bootstrap"]["coverage_95"]
    print(f"  95% coverage {c95['estimate']:.3f}, clustered CI "
          f"[{c95['lo']:.3f}, {c95['hi']:.3f}]: nominal 0.950 is "
          f"{'OUTSIDE' if c95['hi'] < 0.95 else 'inside'} the interval.")
    print(f"  The clustered interval is {c95['width_ratio']:.0f}x wider than the i.i.d. one.")
    print(f"  Effective sample size for coverage: ~{deff['cov_95']['n_eff']:,.0f}, "
          f"not {n:,}.")

    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
