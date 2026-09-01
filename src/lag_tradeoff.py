"""The settling-vs-stability trade-off (F6), recomputed on the full stride-7 sample.

The original version ran on a 37-date sample and reported ~57% auditable at a 14-day lag. Table 1b
now reports 60.9% / 62.9% at the same lag on the full stride-7 candidates. Those measure the same
quantity on different samples, so the smaller one is superseded rather than contradicted -- but two
different numbers for one quantity cannot both sit in the paper.

Auditable = the forecast date has at least one snapshot sharing its exact variant partition, at
least `lag` days later, carrying settled truth for a target date that old.
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
from score import partition_index  # noqa: E402

LAGS = [0, 7, 14, 21, 30, 45, 60, 90]


def tradeoff(branch, scheme="nextstrain_clades", region="global", stride=7, max_lag=400):
    snaps = available_snapshots(branch, scheme, region)
    if not snaps:
        return []
    dates = sorted(snaps)
    use = dates[:-1][::stride]
    print(f"  {branch}: {len(dates)} snapshots, {len(use)} stride-{stride} candidates")
    pidx = partition_index(snaps, verbose=True)
    rows = []
    for lag in LAGS:
        n_aud, matches = 0, []
        for fd in use:
            vset = pidx.get(fd)
            if not vset:
                continue
            f_d = dt.date.fromisoformat(fd)
            k = sum(1 for s, v in pidx.items() if v == vset
                    and lag <= (dt.date.fromisoformat(s) - f_d).days <= max_lag)
            matches.append(k)
            if k > 0:
                n_aud += 1
        rows.append(dict(branch=branch, min_lag_days=lag, n_dates=len(use),
                         n_auditable=n_aud, frac=n_aud / max(len(use), 1),
                         median_matches=float(pd.Series(matches).median()) if matches else 0.0))
        print(f"    lag {lag:3d} d: {n_aud:3d}/{len(use)} auditable ({n_aud / len(use):.1%})",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branches", nargs="*", default=["gisaid", "open"])
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--out", default="results/lag_tradeoff_full.csv")
    a = ap.parse_args()
    rows = []
    for br in a.branches:
        rows += tradeoff(br, stride=a.stride)
    if not rows:
        print("no data")
        return 1
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)
    print("\n=== auditable fraction by settling lag (full stride-7) ===")
    piv = df.pivot(index="min_lag_days", columns="branch", values="frac")
    print(piv.to_string(float_format=lambda v: f"{v:.1%}"))
    with open(a.out.replace(".csv", ".json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
