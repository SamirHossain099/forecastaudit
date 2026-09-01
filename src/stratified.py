"""Is the pooled horizon effect real, or Simpson's paradox?

We refuse to quote the pooled influenza rho(coverage, horizon) = -0.824, because within subtypes it
is +0.23, +0.88 and +0.05 -- the pooled value is an artifact of composition. A referee will
immediately ask why the pooled SARS-CoV-2 rho = -0.984, computed across 31 locations and 30 clades,
is not the same artifact. That question has to be answered before submission, not during review.

This computes the horizon relationships WITHIN strata:

  by location   31 countries, each with its own sequencing intensity and epidemic timing
  by clade      30 variants, each with its own frequency trajectory
  by forecast date  48 model fits, each with a single information set

and compares the distribution of within-stratum correlations against the pooled value. If the effect
holds within strata the pooled number is a summary, not an artifact, and that is a strength. If it
does not, we need to know first.

Reported per stratum: Spearman correlation of horizon against coverage, interval width, and WIS,
each computed on the stratum's own horizon means so a stratum with more points does not dominate.
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

TARGETS = [("cov_95", "coverage_95"), ("width_95", "width_95"), ("wis", "wis")]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    ra = pd.Series(a[ok]).rank().values
    rb = pd.Series(b[ok]).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def horizon_corr(d, col, min_horizons=8, min_n=100):
    """Spearman(horizon, col) computed on this stratum's per-horizon means."""
    if len(d) < min_n:
        return np.nan
    g = d.groupby("horizon")[col].mean()
    if len(g) < min_horizons:
        return np.nan
    return spearman(g.index.values, g.values)


def by_stratum(df, key, min_n=100):
    rows = []
    for name, g in df.groupby(key):
        rec = {"stratum": str(name), "n": len(g), "n_horizons": g.horizon.nunique()}
        for col, label in TARGETS:
            rec[label] = horizon_corr(g, col, min_n=min_n)
        if np.isfinite(rec["coverage_95"]):
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(tab, label):
    out = {"stratum_type": label, "n_strata": int(len(tab))}
    for _, name in TARGETS:
        v = tab[name].dropna()
        if v.empty:
            continue
        out[name] = dict(median=float(v.median()), q25=float(v.quantile(.25)),
                         q75=float(v.quantile(.75)), min=float(v.min()), max=float(v.max()),
                         frac_negative=float((v < 0).mean()), n=int(len(v)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_matched_lag14_gisaid_dense.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--min-n", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/stratified_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = pd.read_csv(a.infile)
    print(f"{a.label}: {len(df):,} points\n")

    pooled = {}
    print("=== the pooled values currently quoted in the draft ===")
    for col, name in TARGETS:
        g = df.groupby("horizon")[col].mean()
        pooled[name] = spearman(g.index.values, g.values)
        print(f"  rho(horizon, {name:>12s}) = {pooled[name]:+.3f}")

    keys = [("location", "location"), ("variant", "clade"), ("forecast_date", "forecast date")]
    if "subtype" in df.columns:
        keys.append(("subtype", "subtype"))

    res = {"label": a.label, "pooled": pooled, "strata": []}
    for key, label in keys:
        if key not in df.columns:
            continue
        tab = by_stratum(df, key, min_n=a.min_n)
        if tab.empty:
            print(f"\n  (no {label} stratum has enough data)")
            continue
        s = summarize(tab, label)
        res["strata"].append(s)
        print(f"\n=== within {label} ({len(tab)} strata with >= {a.min_n} points) ===")
        print(f"  {'quantity':>12s} {'pooled':>8s} {'median':>8s} {'IQR':>18s} "
              f"{'range':>18s} {'% negative':>11s}")
        for _, name in TARGETS:
            if name not in s:
                continue
            v = s[name]
            iqr = "[{:+.2f}, {:+.2f}]".format(v["q25"], v["q75"])
            rng = "[{:+.2f}, {:+.2f}]".format(v["min"], v["max"])
            print(f"  {name:>12s} {pooled[name]:>+8.3f} {v['median']:>+8.3f} "
                  f"{iqr:>18s} {rng:>18s} {v['frac_negative']:>10.0%}")
        tab.to_csv(out.replace(".json", f"_{key}.csv"), index=False)

    print("\n=== verdict on the Simpson's-paradox objection ===")
    for s in res["strata"]:
        c = s.get("coverage_95")
        if not c:
            continue
        agrees = (c["median"] < 0) == (pooled["coverage_95"] < 0)
        print(f"  by {s['stratum_type']:>13s}: median {c['median']:+.3f}, "
              f"{c['frac_negative']:.0%} negative -> "
              f"{'AGREES with pooled' if agrees else 'CONTRADICTS pooled'}")

    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
