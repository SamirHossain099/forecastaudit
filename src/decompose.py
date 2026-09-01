"""Decomposing the weighted interval score into miscalibration, discrimination and uncertainty.

F11 produced the motivation by measurement rather than by citation: WIS says the shipped model
beats a naive baseline by 24.8%, while coverage says its 95% intervals contain the truth 36% of the
time against the naive baseline's 94%. Both are true. WIS aggregates point accuracy, sharpness and
calibration into one number, and here a large point-accuracy advantage masks a large calibration
failure. A reviewer reading only WIS would conclude the forecasts are fine.

The fix is the Murphy/CORP-style decomposition

    mean score  =  MCB  -  DSC  +  UNC

  MCB  miscalibration -- how much the score would improve if the forecasts were recalibrated
       without changing their ranking. Pure calibration failure.
  DSC  discrimination -- how much better the recalibrated forecast is than a constant
       climatological forecast. This is the genuine information content.
  UNC  uncertainty -- the score of the climatological forecast. A property of the data, not of the
       forecaster, and the reason raw scores are incomparable across settings.

## Why this can be done exactly for WIS

WIS is exactly an average of pinball losses. With K interval levels and median m,

    WIS = 1/(K + 1/2) * [ (1/2)|y - m| + sum_k (alpha_k/2) * IS_{alpha_k} ]

and since  IS_alpha = (2/alpha) * [ QL_{alpha/2}(l, y) + QL_{1-alpha/2}(u, y) ]  and
QL_{1/2}(m, y) = (1/2)|y - m|, every weight cancels:

    WIS = 1/(K + 1/2) * sum over the 2K+1 quantile levels of QL_tau(q_tau, y)
        = 2 * mean_tau QL_tau(q_tau, y)

So decomposing WIS reduces to decomposing a pinball loss at each of the 7 levels
(0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975) and averaging. The identity is asserted numerically in
the tests against the independent `score.wis` implementation, not taken on faith.

## The recalibration used

CORP (Dimitriadis, Gneiting & Jordan, PNAS 2021) uses isotonic regression under the relevant loss,
which for pinball loss means pool-adjacent-violators where a merged block takes the **tau-quantile**
of its pooled observations, not their mean. That is implemented here.

Two deliberate departures, both documented rather than hidden:

  BINNED   PAV runs on B quantile bins of the predictions rather than on all n points. Exact PAV
           on 100k points with repeated list merges is O(n^2) in the worst case. Binning is
           standard practice and the bin count is a reported parameter; `--bins 0` runs it exact
           for anyone who wants to check, and a test confirms the two agree on small inputs.

  IN-SAMPLE  The recalibration is fitted on the same data it scores, which is what the CORP
           decomposition specifies -- MCB is defined as the *attainable* improvement, an upper
           bound on what recalibration can buy. It is NOT a prediction of out-of-sample gain.
           F8's conformal recalibration is the out-of-time counterpart, and the two answer
           different questions. Conflating them would overstate the fix.
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

# The 2K+1 quantile levels implied by 50/80/95 central intervals plus the median,
# mapped to the columns the scorer persists.
QUANTILES = [(0.025, "lo_95"), (0.10, "lo_80"), (0.25, "lo_50"), (0.50, "median"),
             (0.75, "hi_50"), (0.90, "hi_80"), (0.975, "hi_95")]


def pinball(tau, q, y):
    """QL_tau(q, y) = (y - q) * tau if y >= q else (q - y) * (1 - tau)."""
    d = np.asarray(y, float) - np.asarray(q, float)
    return np.where(d >= 0, tau * d, (tau - 1.0) * d)


def isotonic_quantile(x, y, tau, bins=100):
    """Isotonic regression of y on x under pinball loss (PAV with quantile pooling).

    Returns the fitted conditional tau-quantile for each input point, in the original order.
    """

    # method="inverted_cdf" is REQUIRED, not cosmetic. np.quantile's default linear
    # interpolation returns a value between order statistics that does NOT minimize the
    # empirical pinball loss at extreme tau: measured against an exact LP solution it was
    # suboptimal by 21.7% at tau=0.025 and 13.8% at tau=0.975 (exactly 0 at the median).
    # Since MCB = score - recalibrated_score, a suboptimal fit UNDERSTATES MCB, and it does
    # so worst in the tails -- exactly where this paper makes its tail claim.
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n == 0:
        return np.array([])
    order = np.argsort(x, kind="mergesort")
    ys = y[order]

    if bins and n > bins:
        # group into `bins` contiguous chunks of the sorted predictions
        edges = np.linspace(0, n, bins + 1).astype(int)
        groups = [ys[edges[i]:edges[i + 1]] for i in range(bins) if edges[i + 1] > edges[i]]
    else:
        groups = [np.array([v]) for v in ys]

    # PAV: merge adjacent blocks whose fitted quantiles violate monotonicity.
    blocks = []                       # each entry is the pooled observation array
    for g in groups:
        blocks.append(g)
        while len(blocks) >= 2 and (np.quantile(blocks[-2], tau, method="inverted_cdf")
                                    > np.quantile(blocks[-1], tau, method="inverted_cdf")):
            merged = np.concatenate([blocks[-2], blocks[-1]])
            blocks = blocks[:-2] + [merged]

    fitted_sorted = np.empty(n)
    pos = 0
    for b in blocks:
        v = float(np.quantile(b, tau, method="inverted_cdf"))
        fitted_sorted[pos:pos + len(b)] = v
        pos += len(b)

    out = np.empty(n)
    out[order] = fitted_sorted
    return out


def decompose_quantile(q, y, tau, bins=100):
    """MCB / DSC / UNC for one quantile level. Returns a dict; score = MCB - DSC + UNC."""
    q = np.asarray(q, float)
    y = np.asarray(y, float)
    ok = np.isfinite(q) & np.isfinite(y)
    q, y = q[ok], y[ok]
    if len(y) < 10:
        return None
    s = pinball(tau, q, y).mean()
    recal = isotonic_quantile(q, y, tau, bins=bins)
    s_recal = pinball(tau, recal, y).mean()
    clim = float(np.quantile(y, tau, method="inverted_cdf"))                 # the constant climatological forecast
    unc = pinball(tau, np.full_like(y, clim), y).mean()
    return dict(tau=tau, n=int(len(y)), score=float(s),
                mcb=float(s - s_recal), dsc=float(unc - s_recal), unc=float(unc),
                score_recalibrated=float(s_recal), climatology=clim)


def decompose_wis(df, bins=100, prefix=""):
    """Decompose WIS as 2 x the mean pinball loss across the 2K+1 levels."""
    rows = []
    for tau, col in QUANTILES:
        c = f"{prefix}{col}"
        if c not in df.columns:
            continue
        r = decompose_quantile(df[c], df.truth, tau, bins=bins)
        if r:
            r["column"] = c
            rows.append(r)
    if not rows:
        return None, pd.DataFrame()
    q = pd.DataFrame(rows)
    # WIS = 2 * mean over levels
    agg = {k: float(2 * q[k].mean()) for k in ("score", "mcb", "dsc", "unc",
                                               "score_recalibrated")}
    agg["n_levels"] = len(q)
    agg["identity_residual"] = agg["score"] - (agg["mcb"] - agg["dsc"] + agg["unc"])
    return agg, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_matched_lag14_gisaid_dense.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/decomposition_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = pd.read_csv(a.infile)
    need = [c for _, c in QUANTILES if c not in df.columns]
    if need:
        print(f"missing quantile columns {need}: re-run score_matched.py")
        return 1
    print(f"{a.label}: {len(df):,} points, {len(QUANTILES)} quantile levels, {a.bins} PAV bins")

    agg, per_q = decompose_wis(df, bins=a.bins)
    print("\n=== WIS decomposition:  score = MCB - DSC + UNC ===")
    print(f"  WIS (mean)            {agg['score']:.4f}")
    print(f"  MCB  miscalibration   {agg['mcb']:.4f}   ({agg['mcb'] / agg['score']:.1%} of score)")
    print(f"  DSC  discrimination   {agg['dsc']:.4f}")
    print(f"  UNC  uncertainty      {agg['unc']:.4f}")
    print(f"  identity residual     {agg['identity_residual']:.2e}  (must be ~0)")
    print(f"\n  WIS after ideal recalibration: {agg['score_recalibrated']:.4f} "
          f"({1 - agg['score_recalibrated'] / agg['score']:.1%} better)")

    print("\n=== per quantile level ===")
    print(per_q[["tau", "column", "n", "score", "mcb", "dsc", "unc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== the point of the exercise: MCB vs DSC by horizon ===")
    df2 = df.copy()
    df2["hbin"] = pd.cut(df2.horizon, [0, 7, 14, 21, 30],
                         labels=["1-7d", "8-14d", "15-21d", "22-30d"])
    hrows = []
    for hb, g in df2.groupby("hbin", observed=True):
        ag, _ = decompose_wis(g, bins=a.bins)
        if ag:
            ag["hbin"] = str(hb)
            ag["n"] = len(g)
            ag["mcb_share"] = ag["mcb"] / ag["score"]
            hrows.append(ag)
    h = pd.DataFrame(hrows)
    if not h.empty:
        print(h[["hbin", "n", "score", "mcb", "dsc", "unc", "mcb_share"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    result = dict(overall=agg, per_quantile=per_q.to_dict("records"),
                  by_horizon=hrows, bins=a.bins, n=int(len(df)))
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
