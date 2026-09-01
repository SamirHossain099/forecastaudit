"""Measure how much the observed truth is revised, across slices of the archive.

F1 established on ONE slice (open/clades/global) that observed frequencies are revised for 100%
of date-variant pairs within two months, by up to 0.53. One slice is an anecdote — in projects 01
and 04 roughly half the single-slice claims did not replicate.

This measures the revision profile across branches, lineage schemes and regions, and compares
`daily_raw_freq` against `weekly_raw_freq` so the scoring target can be chosen on evidence.

Output: for each slice and lag, the fraction of values revised and the magnitude.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from verify import BUCKET, PREFIX, list_all  # noqa: E402

SESSION = requests.Session()


def available_snapshots(branch, scheme, region, model="mlr"):
    pfx = f"{PREFIX}{branch}/{scheme}/{region}/{model}/"
    keys = list_all(pfx)
    out = {}
    for k, size in keys:
        base = k.rsplit("/", 1)[-1]
        if base.endswith("_results.json"):
            stem = base[: -len("_results.json")]
            # The archive contains alias pointers such as "latest_results.json" alongside dated
            # files. They are not snapshots and break date parsing, so drop anything that is not
            # an ISO date.
            try:
                dt.date.fromisoformat(stem)
            except ValueError:
                continue
            out[stem] = (k, size)
    return out


def load_truth(key, site):
    """Return {(location, variant, date): value} for the given raw-frequency site.

    `value` is sometimes null in the archive (the gisaid branch in particular), which crashed the
    first run with `unsupported operand type(s) for -: 'float' and 'NoneType'`. Nulls are dropped
    here rather than coerced, so a missing observation never masquerades as a zero frequency.
    """
    r = SESSION.get(BUCKET + key, timeout=180)
    r.raise_for_status()
    d = r.json()["data"]
    return {(x["location"], x["variant"], x["date"]): float(x["value"])
            for x in d
            if x.get("site") == site and x.get("date") is not None
            and isinstance(x.get("value"), (int, float))}


def revision_profile(branch, scheme, region, site, ref_date, lags_days, model="mlr"):
    snaps = available_snapshots(branch, scheme, region, model)
    if not snaps:
        print(f"  {branch}/{scheme}/{region}: no snapshots")
        return []
    # Use the NEAREST available snapshot, not an exact match. Requiring an exact date silently
    # dropped the entire gisaid branch and every usa region on the first run - 4 of 6 slices
    # vanished with no error, which is precisely the kind of quiet gap that fakes a replication.
    if ref_date not in snaps:
        nearest = min(snaps, key=lambda s: abs(
            (dt.date.fromisoformat(s) - dt.date.fromisoformat(ref_date)).days))
        off = (dt.date.fromisoformat(nearest) - dt.date.fromisoformat(ref_date)).days
        print(f"  {branch}/{scheme}/{region}: ref {ref_date} absent, using {nearest} ({off:+d}d)")
        ref_date = nearest
    ref = load_truth(snaps[ref_date][0], site)
    if not ref:
        print(f"  {branch}/{scheme}/{region} {site}: no '{site}' records in {ref_date}")
        return []
    ref_d = dt.date.fromisoformat(ref_date)
    rows = []
    for lag in lags_days:
        target = (ref_d + dt.timedelta(days=lag)).isoformat()
        # nearest available snapshot at or after target
        later = sorted(s for s in snaps if s >= target)
        if not later:
            continue
        s2 = later[0]
        try:
            cur = load_truth(snaps[s2][0], site)
        except Exception as e:
            rows.append(dict(branch=branch, scheme=scheme, region=region, site=site,
                             ref=ref_date, cmp=s2, lag_days=lag, error=str(e)))
            continue
        common = set(ref) & set(cur)
        if not common:
            continue
        diffs = np.array([abs(cur[k] - ref[k]) for k in common])
        rows.append(dict(
            branch=branch, scheme=scheme, region=region, site=site,
            ref=ref_date, cmp=s2, lag_days=lag, n_shared=len(common),
            frac_changed=float((diffs > 1e-9).mean()),
            frac_gt_05=float((diffs > 0.05).mean()),
            median_abs=float(np.median(diffs)), max_abs=float(diffs.max()),
        ))
        print(f"  {branch}/{scheme}/{region} {site:16s} lag={lag:3d}d n={len(common):5,} "
              f"changed={rows[-1]['frac_changed']:.1%} >0.05={rows[-1]['frac_gt_05']:.1%} "
              f"max={diffs.max():.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-date", default="2024-06-01")
    ap.add_argument("--lags", nargs="*", type=int, default=[14, 30, 60, 120])
    ap.add_argument("--branches", nargs="*", default=["open", "gisaid"])
    ap.add_argument("--schemes", nargs="*", default=["nextstrain_clades", "pango_lineages"])
    ap.add_argument("--regions", nargs="*", default=["global", "usa"])
    ap.add_argument("--sites", nargs="*", default=["daily_raw_freq", "weekly_raw_freq"])
    ap.add_argument("--out", default="results/backfill_profile.csv")
    a = ap.parse_args()

    rows = []
    for br, sch, reg, site in itertools.product(a.branches, a.schemes, a.regions, a.sites):
        try:
            rows += revision_profile(br, sch, reg, site, a.ref_date, a.lags)
        except Exception as e:
            print(f"  {br}/{sch}/{reg} {site}: SKIP {type(e).__name__}: {e}")
        if rows:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            pd.DataFrame(rows).to_csv(a.out, index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no data")
        return 1
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows)\n")

    ok = df[df.get("n_shared", pd.Series(dtype=float)).notna()]
    print("=== does the revision profile replicate across slices? ===")
    for site, g in ok.groupby("site"):
        print(f"\n{site}:")
        piv = g.pivot_table(index="lag_days", values=["frac_changed", "frac_gt_05", "max_abs"],
                            aggfunc=["median", "min", "max"])
        print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
        n_slices = g.groupby("lag_days").size()
        print(f"  slices per lag: {n_slices.to_dict()}")

    print("\n=== daily vs weekly: which is the more stable scoring target? ===")
    cmp = ok.pivot_table(index=["branch", "scheme", "region", "lag_days"],
                         columns="site", values="frac_gt_05")
    if {"daily_raw_freq", "weekly_raw_freq"} <= set(cmp.columns):
        cmp = cmp.dropna()
        wins = (cmp["weekly_raw_freq"] < cmp["daily_raw_freq"]).sum()
        print(f"  weekly more stable than daily in {wins}/{len(cmp)} slice-lags")
        print(cmp.to_string(float_format=lambda v: f"{v:.3f}"))
    json.dump(rows, open(a.out.replace(".csv", ".json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
