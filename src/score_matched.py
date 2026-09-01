"""Partition-matched scoring: the only well-defined way to score this archive.

Naive `(location, variant, date)` matching gives ~18% coverage for a nominal 95% interval. That
number is an artifact: Nextstrain clades are hierarchical and the partition is redefined over
time, so the same label denotes different quantities in different snapshots (see
`score.build_settled_truth_partition_matched` for the worked example).

Here a forecast is scored only against truth from snapshots using the IDENTICAL variant set.
That is restrictive, and quantifying how restrictive is itself a result: it says how much of a
3.7-year archive is actually auditable.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

from backfill import available_snapshots  # noqa: E402
from score import (  # noqa: E402
    LEVELS,
    extract_forecast,
    extract_truth,
    is_catchall,
    load_snapshot,
    naive_7day,
    partition_index,
    summarize,
    wis,
)


def score_matched(branch, scheme, region, model="mlr", site="freq",
                  stride=14, max_snapshots=40, min_lag_days=60, verbose=True):
    snaps = available_snapshots(branch, scheme, region, model)
    if not snaps:
        return pd.DataFrame(), {}
    dates = sorted(snaps)
    use = dates[:-1][::stride][-max_snapshots:]

    # ONE pass over the archive to learn every snapshot's partition, then reuse it.
    if verbose:
        print(f"  building partition index over {len(dates)} snapshots...")
    pidx = partition_index(snaps, verbose=verbose)

    rows, audit = [], []
    for i, fd in enumerate(use, 1):
        try:
            vset = pidx.get(fd)
            if not vset:
                continue
            f_dd = dt.date.fromisoformat(fd)
            used = [s for s, v in pidx.items() if v == vset
                    and min_lag_days <= (dt.date.fromisoformat(s) - f_dd).days <= 400]
            truth = {}
            for s in sorted(used, reverse=True):
                s_d = dt.date.fromisoformat(s)
                for k, v in extract_truth(load_snapshot(snaps[s][0]), site).items():
                    if is_catchall(k[1]) or k in truth:
                        continue
                    try:
                        tgt = dt.date.fromisoformat(k[2])
                    except ValueError:
                        continue
                    if (s_d - tgt).days >= min_lag_days:
                        truth[k] = v
            snap_fd = load_snapshot(snaps[fd][0])
            fc = extract_forecast(snap_fd)
            # The naive comparator, computed from the SAME snapshot the forecast was published in.
            naive = naive_7day(snap_fd)
        except Exception as e:
            print(f"    {fd}: SKIP {type(e).__name__}: {e}")
            continue
        audit.append(dict(forecast_date=fd, partition_size=len(vset),
                          matching_snapshots=len(used), truth_points=len(truth)))
        if not truth:
            continue
        f_d = dt.date.fromisoformat(fd)
        n = 0
        for (loc, var, d), ps in fc.items():
            if is_catchall(var):
                continue
            y = truth.get((loc, var, d))
            if y is None or "median" not in ps:
                continue
            try:
                horizon = (dt.date.fromisoformat(d) - f_d).days
            except ValueError:
                continue
            if horizon <= 0:
                continue
            iv = {}
            for lvl in LEVELS:
                lo, hi = ps.get(f"HDI_{lvl}_lower"), ps.get(f"HDI_{lvl}_upper")
                if lo is not None and hi is not None:
                    iv[lvl] = (min(lo, hi), max(lo, hi))
            if not iv:
                continue
            row = dict(branch=branch, scheme=scheme, region=region, forecast_date=fd,
                       target_date=d, horizon=horizon, location=loc, variant=var,
                       median=ps["median"], truth=y, abs_error=abs(ps["median"] - y),
                       wis=wis(y, ps["median"], iv), partition_size=len(vset),
                       naive=naive.get((loc, var), float("nan")))
            for lvl, (lo, hi) in iv.items():
                row[f"cov_{lvl}"] = int(lo <= y <= hi)
                row[f"width_{lvl}"] = hi - lo
                # Persist the RAW bounds. HDIs are NOT symmetric about the median, so any
                # downstream check that reconstructs them as median +/- width/2 is wrong --
                # doing exactly that understated 95% coverage as 0.327 instead of 0.392.
                row[f"lo_{lvl}"] = lo
                row[f"hi_{lvl}"] = hi
            rows.append(row)
            n += 1
        if verbose:
            print(f"    [{i}/{len(use)}] {fd}: +{n:,} scored (total {len(rows):,})", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(audit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="open")
    ap.add_argument("--scheme", default="nextstrain_clades")
    ap.add_argument("--region", default="global")
    ap.add_argument("--site", default="freq")
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--max-snapshots", type=int, default=40)
    ap.add_argument("--min-lag-days", type=int, default=60)
    ap.add_argument("--out", default="results/scores_matched.csv")
    a = ap.parse_args()

    print(f"partition-matched scoring: {a.branch}/{a.scheme}/{a.region} vs '{a.site}'")
    df, audit = score_matched(a.branch, a.scheme, a.region, site=a.site, stride=a.stride,
                              max_snapshots=a.max_snapshots, min_lag_days=a.min_lag_days)

    os.makedirs("results", exist_ok=True)
    if len(audit):
        audit.to_csv(a.out.replace(".csv", "_audit.csv"), index=False)
        print("\n=== how much of the archive is auditable at all? ===")
        print(f"  forecast dates examined      : {len(audit)}")
        print(f"  with >=1 matching-partition snapshot : "
              f"{int((audit.matching_snapshots > 0).sum())}")
        print(f"  with usable truth points     : {int((audit.truth_points > 0).sum())}")
        print(f"  median matching snapshots    : {audit.matching_snapshots.median():.0f}")

    if df.empty:
        print("\nNo partition-matched points. That is itself the result: the clade partition is")
        print("redefined faster than the settling lag, so this slice cannot be audited naively.")
        return 0

    df.to_csv(a.out, index=False)
    s = summarize(df)
    print(f"\nwrote {a.out}  ({len(df):,} points)\n")
    print("=== coverage vs nominal (partition-matched) ===")
    for lvl in LEVELS:
        k = f"coverage_{lvl}"
        if k in s:
            print(f"  {lvl}% HDI -> empirical {s[k]:.3f}   (nominal {lvl/100:.2f})")
    print(f"\n  mean WIS {s['mean_wis']:.4f}   mean |error| {s['mean_abs_error']:.4f}")
    print(f"  {s['n']:,} points over {s['n_forecast_dates']} forecast dates")
    json.dump(s, open(a.out.replace(".csv", "_summary.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
