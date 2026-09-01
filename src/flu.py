"""Independent replication on a second pathogen: the forecasts-flu archive.

Why this exists. F7/F8 are established on SARS-CoV-2 clade forecasts. A reviewer's fair question is
whether the overconfidence is a property of *that* model on *that* pathogen, or of the modeling
approach. `files/workflows/forecasts-flu/` runs the same MLR machinery on influenza and was never
mentioned in the project brief; it is a genuinely independent target that costs no new methodology.

Differences from the ncov archive, all handled here:

  path      gisaid/{scheme}/{subtype}/{resolution}/  with files `{date}_MLR_results.json`
            (ncov: {branch}/{scheme}/{region}/mlr/ with `{date}_results.json`)
  truth     `raw_freq` / `smoothed_raw_freq`
            (ncov: `daily_raw_freq` / `weekly_raw_freq`)
  variants  include `unassigned` as a SECOND catch-all beside `other` -- see score.is_catchall
  bonus     `agg_counts` carries sequence counts, so sequencing volume is directly available
            rather than proxied

Scale: 779 objects / 0.35 GB, ~42 weekly snapshots per slice over 2025-12-23 -> 2026-08-27. Much
shorter than the 3.7-year ncov archive, so the partition-lifetime constraint from F4/F5 must be
re-measured here rather than assumed.
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

from score import (  # noqa: E402
    LEVELS,
    extract_forecast,
    extract_truth,
    is_catchall,
    load_snapshot,
    summarize,
    wis,
)
from verify import list_all  # noqa: E402

PREFIX = "files/workflows/forecasts-flu/"
SUBTYPES = ("h3n2", "h1n1pdm", "vic")
SCHEMES = ("emerging_haplotype", "aa_haplotype")
RESOLUTIONS = ("region", "country")


def flu_snapshots(scheme, subtype, resolution):
    """{date: (key, size)} for one flu slice. Filenames carry an `_MLR` infix."""
    pfx = f"{PREFIX}gisaid/{scheme}/{subtype}/{resolution}/"
    out = {}
    for k, size in list_all(pfx):
        base = k.rsplit("/", 1)[-1]
        if not base.endswith("_results.json"):
            continue
        stem = base[: -len("_results.json")].replace("_MLR", "")
        try:
            dt.date.fromisoformat(stem)
        except ValueError:
            continue                      # alias pointers such as latest_*, as in the ncov archive
        out[stem] = (k, size)
    return out


def partition_index(snaps, verbose=True):
    idx = {}
    for i, d in enumerate(sorted(snaps), 1):
        try:
            idx[d] = frozenset(load_snapshot(snaps[d][0]).get("metadata", {})
                               .get("variants", []))
        except Exception:
            continue
        if verbose and i % 10 == 0:
            print(f"    partition index {i}/{len(snaps)}", flush=True)
    return idx


def partition_lifetime(idx):
    """How long does a variant partition survive? The F4/F5 constraint, re-measured for flu."""
    dates = sorted(idx)
    if not dates:
        return {}
    runs, cur, start = [], idx[dates[0]], dates[0]
    for d in dates[1:]:
        if idx[d] != cur:
            runs.append((dt.date.fromisoformat(d) - dt.date.fromisoformat(start)).days)
            cur, start = idx[d], d
    runs.append((dt.date.fromisoformat(dates[-1]) - dt.date.fromisoformat(start)).days)
    s = pd.Series(runs)
    return dict(n_snapshots=len(dates), n_partitions=len(set(idx.values())),
                median_lifetime_d=float(s.median()), max_lifetime_d=float(s.max()),
                frac_ge_30d=float((s >= 30).mean()), frac_ge_60d=float((s >= 60).mean()),
                partition_sizes=f"{min(len(v) for v in idx.values())}-"
                                f"{max(len(v) for v in idx.values())}")


def score_slice(scheme, subtype, resolution, site="smoothed_raw_freq",
                min_lag_days=14, max_lag_days=400, stride=2, max_snapshots=40, verbose=True):
    snaps = flu_snapshots(scheme, subtype, resolution)
    if not snaps:
        return pd.DataFrame(), pd.DataFrame(), {}
    if verbose:
        print(f"  {scheme}/{subtype}/{resolution}: {len(snaps)} snapshots")
    pidx = partition_index(snaps, verbose=verbose)
    life = partition_lifetime(pidx)

    dates = sorted(snaps)
    use = dates[:-1][::stride][-max_snapshots:]
    rows, audit = [], []
    for i, fd in enumerate(use, 1):
        vset = pidx.get(fd)
        if not vset:
            continue
        f_d = dt.date.fromisoformat(fd)
        used = [s for s, v in pidx.items() if v == vset
                and min_lag_days <= (dt.date.fromisoformat(s) - f_d).days <= max_lag_days]
        truth = {}
        try:
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
            fc = extract_forecast(load_snapshot(snaps[fd][0]))
        except Exception as e:
            print(f"    {fd}: SKIP {type(e).__name__}: {e}")
            continue
        audit.append(dict(scheme=scheme, subtype=subtype, resolution=resolution,
                          forecast_date=fd, partition_size=len(vset),
                          matching_snapshots=len(used), truth_points=len(truth)))
        if not truth:
            continue
        n = 0
        for (loc, var, d), ps in fc.items():
            if is_catchall(var) or "median" not in ps:
                continue
            y = truth.get((loc, var, d))
            if y is None:
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
            row = dict(scheme=scheme, subtype=subtype, resolution=resolution,
                       forecast_date=fd, target_date=d, horizon=horizon, location=loc,
                       variant=var, median=ps["median"], truth=y,
                       abs_error=abs(ps["median"] - y), wis=wis(y, ps["median"], iv),
                       partition_size=len(vset))
            for lvl, (lo, hi) in iv.items():
                row[f"cov_{lvl}"] = int(lo <= y <= hi)
                row[f"width_{lvl}"] = hi - lo
                row[f"lo_{lvl}"] = lo
                row[f"hi_{lvl}"] = hi
            rows.append(row)
            n += 1
        if verbose:
            print(f"    [{i}/{len(use)}] {fd}: +{n:,} scored (total {len(rows):,})", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(audit), life


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="*", default=["emerging_haplotype"])
    ap.add_argument("--subtypes", nargs="*", default=list(SUBTYPES))
    ap.add_argument("--resolutions", nargs="*", default=["region"])
    ap.add_argument("--site", default="smoothed_raw_freq")
    ap.add_argument("--min-lag-days", type=int, default=14)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="results/scores_flu.csv")
    a = ap.parse_args()

    all_rows, all_audit, lifetimes = [], [], {}
    for sch in a.schemes:
        for sub in a.subtypes:
            for res in a.resolutions:
                df, aud, life = score_slice(sch, sub, res, site=a.site,
                                            min_lag_days=a.min_lag_days, stride=a.stride)
                if life:
                    lifetimes[f"{sch}/{sub}/{res}"] = life
                if len(df):
                    all_rows.append(df)
                if len(aud):
                    all_audit.append(aud)

    os.makedirs("results", exist_ok=True)
    print("\n=== partition lifetime on the flu archive (cf. F5: ncov clades 22-34 d) ===")
    if lifetimes:
        print(pd.DataFrame(lifetimes).T.to_string())
    json.dump(lifetimes, open("results/flu_partition_lifetime.json", "w"), indent=2)

    if not all_rows:
        print("\nNo partition-matched points on any flu slice. Given only ~42 weekly snapshots")
        print("over 8 months, that would itself be the finding: the archive is too young to audit.")
        return 0

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(a.out, index=False)
    if all_audit:
        pd.concat(all_audit, ignore_index=True).to_csv(a.out.replace(".csv", "_audit.csv"),
                                                       index=False)
    print(f"\nwrote {a.out}  ({len(df):,} points)")

    print("\n=== coverage vs nominal, per subtype ===")
    for sub, g in df.groupby("subtype"):
        s = summarize(g)
        line = "  ".join(f"{lvl}%={s.get(f'coverage_{lvl}', float('nan')):.3f}" for lvl in LEVELS)
        print(f"  {sub:10s} n={len(g):6,}  {line}   WIS={s['mean_wis']:.4f}")
    s = summarize(df)
    print(f"\n  ALL      n={len(df):,}  " +
          "  ".join(f"{lvl}%={s.get(f'coverage_{lvl}', float('nan')):.3f}" for lvl in LEVELS))
    print("  nominal          50%=0.500  80%=0.800  95%=0.950")
    json.dump(s, open(a.out.replace(".csv", "_summary.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
