"""Exact isotonic quantile regression under the 2-D partial order, and validation of the shortcut.

`idr_decompose.py` fits along LINEAR EXTENSIONS of the componentwise order on (L, U) rather than
under the order itself. That is provably conservative -- a linear extension adds constraints, so the
recalibrated score can only be worse and MCB can only be understated -- but "provably in the safe
direction" is not the same as "close". This module measures the gap.

## The exact problem

Isotonic quantile regression under a partial order minimizes a separable convex objective subject to
order constraints:

    minimize  sum_i pinball_tau(q_i, y_i)      subject to   q_i <= q_j  whenever  i <= j

1-D pool-adjacent-violators does not generalize to a partial order, but the problem is a linear
program. Introducing u_i >= tau (y_i - q_i) and u_i >= (tau - 1)(y_i - q_i) linearizes the pinball
loss exactly, giving 2n variables, 2n epigraph rows and one row per comparable pair. HiGHS solves
this in seconds at the subsample sizes used here.

## Why subsampling is the right vehicle

Allen, Burnello & Ziegel note in the same Remark 11 that IDR "can also be computationally expensive
if it is fit to large datasets, in which case it may be useful to combine IDR with subsample
aggregation". The comparability graph on 117,438 points has ~10^10 pairs; on 600 points it has
~1.7x10^5. We therefore draw B subsamples, fit BOTH estimators on each, and compare them
point-for-point on identical data. That isolates the estimator difference from any sampling
difference.

## What a pass looks like

Theory says MCB(linear extension) <= MCB(exact) on every subsample, with no exceptions -- a single
violation would indicate a bug rather than noise. The quantity of interest is the size of the gap.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402

from idr_decompose import (  # noqa: E402
    EXTENSIONS,
    interval_score,
    isotonic_quantile_ordered,
    linear_extension,
)

LEVELS = (50, 80, 95)


def comparable_pairs(lo, hi):
    """Index pairs (i, j) with lo_i <= lo_j and hi_i <= hi_j, i != j.

    Returned as the constraint set q_i <= q_j. Redundant (transitively implied) pairs are kept: they
    do not change the optimum, and at these sizes the solver handles them comfortably.
    """
    le_lo = lo[:, None] <= lo[None, :]
    le_hi = hi[:, None] <= hi[None, :]
    m = le_lo & le_hi
    np.fill_diagonal(m, False)
    return np.argwhere(m)


def isotonic_quantile_poset(lo, hi, y, tau, pairs=None):
    """Exact isotonic quantile regression under the componentwise order, by LP.

    Variables are [q (n), u (n)]. Objective minimizes sum u_i, with
        u_i >= tau      * (y_i - q_i)        ->   -tau*q_i      - u_i <= -tau*y_i
        u_i >= (tau - 1)* (y_i - q_i)        ->  (1-tau)*q_i    - u_i <= (1-tau)*y_i
    plus q_i - q_j <= 0 for each comparable pair.
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    y = np.asarray(y, float)
    n = len(y)
    if pairs is None:
        pairs = comparable_pairs(lo, hi)

    rows, cols, vals, rhs = [], [], [], []
    r = 0
    idx = np.arange(n)
    # epigraph row 1
    rows += list(r + idx)
    cols += list(idx)
    vals += [-tau] * n
    rows += list(r + idx)
    cols += list(n + idx)
    vals += [-1.0] * n
    rhs += list(-tau * y)
    r += n
    # epigraph row 2
    rows += list(r + idx)
    cols += list(idx)
    vals += [1.0 - tau] * n
    rows += list(r + idx)
    cols += list(n + idx)
    vals += [-1.0] * n
    rhs += list((1.0 - tau) * y)
    r += n
    # order constraints
    if len(pairs):
        pr = r + np.arange(len(pairs))
        rows += list(pr)
        cols += list(pairs[:, 0])
        vals += [1.0] * len(pairs)
        rows += list(pr)
        cols += list(pairs[:, 1])
        vals += [-1.0] * len(pairs)
        rhs += [0.0] * len(pairs)
        r += len(pairs)

    A = coo_matrix((vals, (rows, cols)), shape=(r, 2 * n)).tocsr()
    c = np.concatenate([np.zeros(n), np.ones(n)])
    lo_b = min(y.min(), 0.0) - 1.0
    hi_b = max(y.max(), 1.0) + 1.0
    bounds = [(lo_b, hi_b)] * n + [(0, None)] * n
    res = linprog(c, A_ub=A, b_ub=np.asarray(rhs), bounds=bounds, method="highs")
    if not res.success:
        return None
    return res.x[:n]


def mcb_exact(lo, hi, y, level, pairs=None):
    """MCB for one interval level under the exact partial-order fit."""
    alpha = 1.0 - level / 100.0
    score = interval_score(lo, hi, y, alpha).mean()
    rl = isotonic_quantile_poset(lo, hi, y, alpha / 2, pairs)
    ru = isotonic_quantile_poset(lo, hi, y, 1 - alpha / 2, pairs)
    if rl is None or ru is None:
        return None
    lo_r, hi_r = np.minimum(rl, ru), np.maximum(rl, ru)
    return float(score - interval_score(lo_r, hi_r, y, alpha).mean())


def mcb_linear(lo, hi, y, level):
    """MCB under the linear-extension shortcut, best of the three extensions."""
    alpha = 1.0 - level / 100.0
    score = interval_score(lo, hi, y, alpha).mean()
    best = -np.inf
    for mode in EXTENSIONS:
        order = linear_extension(lo, hi, mode)
        rl = isotonic_quantile_ordered(order, y, alpha / 2, bins=0)
        ru = isotonic_quantile_ordered(order, y, 1 - alpha / 2, bins=0)
        s = interval_score(np.minimum(rl, ru), np.maximum(rl, ru), y, alpha).mean()
        best = max(best, score - s)
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_full_gisaid.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--n-sub", type=int, default=600, help="subsample size")
    ap.add_argument("--n-boot", type=int, default=40, help="number of subsamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/idr_exact_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = pd.read_csv(a.infile)
    rng = np.random.default_rng(a.seed)
    print(f"{a.label}: {len(df):,} points; validating on {a.n_boot} subsamples of "
          f"{a.n_sub}\n")

    rows = []
    t0 = time.time()
    for b in range(a.n_boot):
        idx = rng.choice(len(df), size=a.n_sub, replace=False)
        sub = df.iloc[idx]
        y = sub.truth.to_numpy(float)
        for lvl in LEVELS:
            lo = sub[f"lo_{lvl}"].to_numpy(float)
            hi = sub[f"hi_{lvl}"].to_numpy(float)
            ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
            if ok.sum() < 50:
                continue
            l2, h2, y2 = lo[ok], hi[ok], y[ok]
            pairs = comparable_pairs(l2, h2)
            me = mcb_exact(l2, h2, y2, lvl, pairs)
            ml = mcb_linear(l2, h2, y2, lvl)
            if me is None:
                continue
            rows.append(dict(rep=b, level=lvl, n=int(ok.sum()), n_pairs=int(len(pairs)),
                             mcb_exact=me, mcb_linear=ml,
                             gap=me - ml, rel_gap=(me - ml) / me if me else np.nan))
        if (b + 1) % 10 == 0:
            print(f"    {b + 1}/{a.n_boot} subsamples  ({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        print("no results")
        return 1
    r = pd.DataFrame(rows)
    r.to_csv(out.replace(".json", ".csv"), index=False)

    print("\n=== exact partial-order IDR vs the linear-extension shortcut ===")
    print(f"  {'level':>6s} {'reps':>5s} {'MCB exact':>11s} {'MCB linear':>11s} "
          f"{'gap':>10s} {'rel gap':>9s} {'violations':>11s}")
    summ = {}
    for lvl, g in r.groupby("level"):
        viol = int((g.mcb_linear > g.mcb_exact + 1e-9).sum())
        summ[str(lvl)] = dict(n_reps=int(len(g)), mcb_exact=float(g.mcb_exact.mean()),
                              mcb_linear=float(g.mcb_linear.mean()),
                              gap=float(g.gap.mean()), rel_gap=float(g.rel_gap.mean()),
                              violations=viol)
        print(f"  {lvl:>5d}% {len(g):>5d} {g.mcb_exact.mean():>11.5f} "
              f"{g.mcb_linear.mean():>11.5f} {g.gap.mean():>10.5f} "
              f"{g.rel_gap.mean():>8.1%} {viol:>11d}")

    tot_viol = int((r.mcb_linear > r.mcb_exact + 1e-9).sum())
    print(f"\n  total ordering violations: {tot_viol}/{len(r)} "
          f"(theory says 0: the shortcut can never exceed the exact fit)")
    print(f"  mean relative gap across all levels: {r.rel_gap.mean():.1%}")
    print(f"  median pairs per subsample: {r.n_pairs.median():,.0f}")
    print(f"  runtime: {time.time() - t0:.0f}s")

    res = dict(label=a.label, n_sub=a.n_sub, n_boot=a.n_boot, by_level=summ,
               total_violations=tot_viol, n_fits=int(len(r)),
               mean_rel_gap=float(r.rel_gap.mean()),
               runtime_s=round(time.time() - t0, 1))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
