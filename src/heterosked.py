"""Is the heteroscedasticity result an artifact of conditioning on the forecast?

F7 reports mean |error| divided by half-width, binned by FORECAST LEVEL, and finds overconfidence
6.5x worse below 1% frequency than at mid-range. Conditioning on the predictor and then measuring
its error invites regression to the mean: extreme predictions are followed by less extreme outcomes
for purely statistical reasons, which inflates |error| in the extreme bins whether or not the model
is genuinely worse there.

Three conditionings are computed, and the pattern is only trustworthy if it survives all three:

  (a) FORECAST level     bin by the predicted median. The original. RTM inflates the tails.
  (b) OUTCOME level      bin by the realised frequency. The mirror image -- RTM now inflates the
                         tails in the opposite direction, so (a) and (b) bracket the truth rather
                         than either being correct alone.
  (c) BASELINE level     bin by the 7-day moving average of observed frequency AS OF THE FORECAST
                         DATE. This is the instrument that settles it: it is fixed before the
                         forecast is evaluated, is not a function of either the forecast error or
                         the outcome being predicted, and therefore carries no RTM in either
                         direction.

(c) is the one to report in the paper, with (a) and (b) as the bracket.
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

BINS = [-.001, .01, .05, .15, .35, .65, 1.001]
LABS = ["<1%", "1-5%", "5-15%", "15-35%", "35-65%", ">65%"]


def overconfidence_table(df, by, label):
    """mean |error| / half-width_95, and coverage, within bins of `by`."""
    d = df.copy()
    d["bin"] = pd.cut(d[by], BINS, labels=LABS)
    g = d.groupby("bin", observed=True).agg(
        n=("wis", "size"), abs_err=("abs_error", "mean"),
        width=("width_95", "mean"), cov=("cov_95", "mean"))
    g["ratio"] = g.abs_err / (g.width / 2)
    g["conditioning"] = label
    return g


def spread(g):
    """How strong is the gradient across bins?"""
    r = g.ratio.dropna()
    if len(r) < 3:
        return dict(lo=np.nan, hi=np.nan, ratio=np.nan)
    return dict(lo=float(r.min()), hi=float(r.max()), ratio=float(r.max() / r.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_matched_lag14_gisaid_dense.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/heterosked_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = pd.read_csv(a.infile)
    conds = [("median", "(a) forecast level"), ("truth", "(b) outcome level")]
    if "naive" in df.columns:
        conds.append(("naive", "(c) baseline level, pre-forecast"))
    print(f"{a.label}: {len(df):,} points\n")
    print("mean |error| / half-width of the 95% HDI, by frequency bin\n")
    print(f"  {'bin':>8s} " + " ".join(f"{lab.split(')')[0] + ')':>10s}" for _, lab in conds))

    tables = {}
    for col, lab in conds:
        sub = df[np.isfinite(df[col])] if col == "naive" else df
        tables[lab] = overconfidence_table(sub, col, lab)

    for b in LABS:
        row = []
        for _, lab in conds:
            g = tables[lab]
            row.append(f"{g.loc[b, 'ratio']:10.2f}" if b in g.index else f"{'-':>10s}")
        print(f"  {b:>8s} " + " ".join(row))

    print()
    res = {"label": a.label, "conditionings": {}}
    for _, lab in conds:
        g = tables[lab]
        s = spread(g)
        res["conditionings"][lab] = dict(
            spread=s, table={str(k): dict(n=int(v["n"]), ratio=float(v["ratio"]),
                                          cov=float(v["cov"]))
                             for k, v in g.iterrows()})
        print(f"  {lab:34s} range {s['lo']:.2f}-{s['hi']:.2f}  spread {s['ratio']:.1f}x")

    print("\n=== does the low-frequency stratum remain the worst? ===")
    for _, lab in conds:
        g = tables[lab]
        if "<1%" not in g.index:
            continue
        worst = g.ratio.idxmax()
        lowest_is_worst = worst == "<1%"
        print(f"  {lab:34s} worst bin = {str(worst):>7s}  "
              f"{'YES' if lowest_is_worst else 'no'}")
        res["conditionings"][lab]["worst_bin"] = str(worst)

    print("\n=== verdict ===")
    key = "(c) baseline level, pre-forecast"
    if key in tables:
        g = tables[key]
        s = spread(g)
        rtm_free_worst = g.ratio.idxmax()
        print(f"  Under the RTM-free conditioning (c), the spread is {s['ratio']:.1f}x and the")
        print(f"  worst-calibrated stratum is {rtm_free_worst}.")
        a_spread = spread(tables["(a) forecast level"])["ratio"]
        print(f"  Conditioning on the forecast gives {a_spread:.1f}x, so RTM "
              f"{'inflates' if a_spread > s['ratio'] else 'does not inflate'} the original figure "
              f"by a factor of {a_spread / s['ratio']:.2f}.")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
