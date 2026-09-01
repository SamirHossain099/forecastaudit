"""The interval-score decomposition of Allen, Burnello & Ziegel (2025), applied to WIS.

Their paper decomposes the interval score at a single level alpha by conditioning on the PAIR of
bounds (L, U) under the componentwise partial order, estimating the conditional quantiles with
isotonic distributional regression:

    E[IS_alpha([L,U],Y)] = UNC - DSC_I + MCB_I

They explicitly decline the weaker alternative -- conditioning on each bound separately -- noting at
their equation (8) that quantile calibration of the bounds "is strictly weaker than isotonic
calibration, [so] we do not consider it in the following". Their conclusion then states that "this
weighted interval score could similarly be decomposed using the methods proposed herein", and leaves
it undone. As of 2026-08-30 nobody has taken it up (see LITERATURE.md).

`decompose.py` implements the weaker, quantile-wise version. This module implements theirs, so the
two can be compared on the same data rather than one being asserted adequate.

## How the isotonic fit is done here, and why it is conservative

Fitting IDR under a genuine partial order requires solving a constrained problem over lower/upper
sets. Instead we fit isotonic quantile regression along a LINEAR EXTENSION of the componentwise
order -- sorting by (L, U) lexicographically, by (U, L), or by midpoint. Each is a valid linear
extension: if L_i <= L_j and U_i <= U_j then i precedes j under all three.

A linear extension imposes MORE constraints than the partial order, so its optimal recalibrated
score is >= the true IDR optimum. Therefore

    MCB_I(linear extension)  <=  MCB_I(IDR)

**The approximation under-states miscalibration.** Every number this module reports is a LOWER
BOUND on the isotonic miscalibration, which is the safe direction for a paper whose claim is that
miscalibration is large. We tighten the bound by maximising over the three extensions.

This is only sensible because the order is nearly total on this data: 96.0-98.9% of forecast pairs
are comparable on the SARS-CoV-2 branches, 84.7-94.7% on influenza (`comparability.py`), so the
partial order and its linear extensions nearly coincide.
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

from decompose import decompose_wis  # noqa: E402

LEVELS = (50, 80, 95)
EXTENSIONS = ("lo_then_hi", "hi_then_lo", "midpoint")


def interval_score(lo, hi, y, alpha):
    """IS_alpha([l,u],y) = (u-l) + (2/a)(l-y)1{y<l} + (2/a)(y-u)1{y>u}."""
    lo, hi, y = np.asarray(lo, float), np.asarray(hi, float), np.asarray(y, float)
    return ((hi - lo)
            + (2.0 / alpha) * np.maximum(lo - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - hi, 0.0))


def linear_extension(lo, hi, mode="lo_then_hi"):
    """A valid linear extension of the componentwise partial order on (L, U)."""
    if mode == "lo_then_hi":
        keys = (hi, lo)
    elif mode == "hi_then_lo":
        keys = (lo, hi)
    elif mode == "midpoint":
        keys = (hi, (lo + hi) / 2.0)
    else:
        raise ValueError(mode)
    return np.lexsort(keys)


def isotonic_quantile_ordered(order, y, tau, bins=200):
    """Isotonic quantile regression along a given ordering (PAV with quantile pooling)."""

    # method="inverted_cdf" is REQUIRED, not cosmetic. np.quantile's default linear
    # interpolation returns a value between order statistics that does NOT minimize the
    # empirical pinball loss at extreme tau: measured against an exact LP solution it was
    # suboptimal by 21.7% at tau=0.025 and 13.8% at tau=0.975 (exactly 0 at the median).
    # Since MCB = score - recalibrated_score, a suboptimal fit UNDERSTATES MCB, and it does
    # so worst in the tails -- exactly where this paper makes its tail claim.
    ys = np.asarray(y, float)[order]
    n = len(ys)
    if n == 0:
        return np.array([])
    if bins and n > bins:
        edges = np.linspace(0, n, bins + 1).astype(int)
        groups = [ys[edges[i]:edges[i + 1]] for i in range(bins) if edges[i + 1] > edges[i]]
    else:
        groups = [np.array([v]) for v in ys]

    blocks = []
    for g in groups:
        blocks.append(g)
        while len(blocks) >= 2 and (np.quantile(blocks[-2], tau, method="inverted_cdf")
                                   > np.quantile(blocks[-1], tau, method="inverted_cdf")):
            blocks = blocks[:-2] + [np.concatenate([blocks[-2], blocks[-1]])]

    fitted_sorted = np.empty(n)
    pos = 0
    for b in blocks:
        v = float(np.quantile(b, tau, method="inverted_cdf"))
        fitted_sorted[pos:pos + len(b)] = v
        pos += len(b)
    out = np.empty(n)
    out[order] = fitted_sorted
    return out


def decompose_interval(lo, hi, y, level, bins=200, extensions=EXTENSIONS):
    """Allen-Burnello-Ziegel decomposition of IS at one level. MCB_I is a lower bound."""
    alpha = 1.0 - level / 100.0
    lo, hi, y = np.asarray(lo, float), np.asarray(hi, float), np.asarray(y, float)
    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    lo, hi, y = lo[ok], hi[ok], y[ok]
    if len(y) < 50:
        return None

    score = interval_score(lo, hi, y, alpha).mean()
    # UNC: the constant climatological interval from the marginal quantiles of y
    ql, qu = np.quantile(y, alpha / 2), np.quantile(y, 1 - alpha / 2)
    unc = interval_score(np.full_like(y, ql), np.full_like(y, qu), y, alpha).mean()

    best = None
    for mode in extensions:
        order = linear_extension(lo, hi, mode)
        rl = isotonic_quantile_ordered(order, y, alpha / 2, bins=bins)
        ru = isotonic_quantile_ordered(order, y, 1 - alpha / 2, bins=bins)
        lo_r, hi_r = np.minimum(rl, ru), np.maximum(rl, ru)
        s_recal = interval_score(lo_r, hi_r, y, alpha).mean()
        rec = dict(extension=mode, score_recalibrated=float(s_recal),
                   mcb=float(score - s_recal), dsc=float(unc - s_recal))
        if best is None or rec["mcb"] > best["mcb"]:
            best = rec
    return dict(level=level, n=int(len(y)), score=float(score), unc=float(unc),
                mcb=best["mcb"], dsc=best["dsc"],
                score_recalibrated=best["score_recalibrated"],
                extension=best["extension"],
                identity_residual=float(score - (best["mcb"] - best["dsc"] + unc)))


def decompose_wis_interval(df, bins=200):
    """WIS decomposition built from per-level interval-score decompositions.

    WIS = 1/(K+1/2) * [ (1/2)|y-m| + sum_k (a_k/2) IS_{a_k} ], so each component of WIS is the same
    weighted combination of the corresponding per-level components plus the median term. The median
    term is NOT part of any interval, so it is decomposed as a quantile score -- their framing has
    no place for it, which is itself worth stating.
    """
    per = []
    for lvl in LEVELS:
        r = decompose_interval(df[f"lo_{lvl}"], df[f"hi_{lvl}"], df.truth, lvl, bins=bins)
        if r:
            per.append(r)
    if not per:
        return None, pd.DataFrame()
    K = len(per)
    w = {r["level"]: (1.0 - r["level"] / 100.0) / 2.0 for r in per}       # alpha_k / 2
    norm = K + 0.5

    # median term, as a quantile score at tau = 0.5, weight 1/2
    med = df["median"].to_numpy(float)
    y = df.truth.to_numpy(float)
    ok = np.isfinite(med) & np.isfinite(y)
    from decompose import decompose_quantile
    dq = decompose_quantile(med[ok], y[ok], 0.5, bins=bins)

    agg = {}
    for key in ("score", "mcb", "dsc", "unc"):
        s = sum(w[r["level"]] * r[key] for r in per)
        agg[key] = float((s + 0.5 * (2.0 * dq[key])) / norm)
    agg["n_levels"] = K
    agg["identity_residual"] = agg["score"] - (agg["mcb"] - agg["dsc"] + agg["unc"])
    agg["median_term"] = {k: float(dq[k]) for k in ("score", "mcb", "dsc", "unc")}
    return agg, pd.DataFrame(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_matched_lag14_gisaid_dense.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/idr_decomposition_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = pd.read_csv(a.infile)
    print(f"{a.label}: {len(df):,} points\n")

    print("=== per-level interval-score decomposition (Allen-Burnello-Ziegel) ===")
    print("    conditioning on the PAIR (L,U); MCB is a lower bound (see module docstring)")
    agg_i, per = decompose_wis_interval(df, bins=a.bins)
    per["mcb_share"] = per.mcb / per.score
    print(per[["level", "n", "score", "mcb", "dsc", "unc", "mcb_share", "extension"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== the comparison that matters: interval vs quantile-wise conditioning ===")
    agg_q, _ = decompose_wis(df, bins=a.bins)
    rows = []
    for name, ag in (("interval (L,U)  [theirs]", agg_i), ("quantile-wise   [ours]", agg_q)):
        rows.append(dict(conditioning=name, WIS=ag["score"], MCB=ag["mcb"], DSC=ag["dsc"],
                         UNC=ag["unc"], mcb_share=ag["mcb"] / ag["score"],
                         residual=ag["identity_residual"]))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    dm = agg_i["mcb"] - agg_q["mcb"]
    print(f"\n  MCB difference (interval - quantile-wise): {dm:+.4f} "
          f"({dm / agg_q['mcb']:+.1%} of the quantile-wise value)")
    print("  Theory: conditioning on the pair is a STRONGER calibration notion, so MCB_interval")
    print("  should be >= MCB_quantile-wise. Our MCB_interval is additionally a lower bound, so")
    print("  a smaller value does not contradict the theory -- it bounds the gap.")

    print(f"\n  median term (not part of any interval, so absent from their framing): "
          f"MCB {agg_i['median_term']['mcb']:.4f} of score "
          f"{agg_i['median_term']['score']:.4f} "
          f"({agg_i['median_term']['mcb'] / agg_i['median_term']['score']:.1%})")

    res = dict(interval=agg_i, quantile_wise=agg_q,
               per_level=per.to_dict("records"), bins=a.bins, n=int(len(df)))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
