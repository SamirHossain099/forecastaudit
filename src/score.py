"""Score the archived forecasts: the thing nobody has done.

Ground-truth convention (decided from evidence in FINDINGS.md, stated here because the whole
audit rests on it):

  TARGET SERIES : `weekly_raw_freq`. It is more stable than `daily_raw_freq` in 16/16
                  slice x lag comparisons, so scoring against it measures forecast skill rather
                  than the deposition process.
  TRUTH SOURCE  : the MOST RECENT snapshot available. That is the maximally-settled version of
                  each observation. Fixed-lag variants are reported as sensitivity, because
                  revision severity varies ~4x across slices and a single global lag would be
                  arbitrary.
  MATCHING      : a forecast made on date T for target date D is scored only if D > T (strictly
                  out-of-sample) and the settled truth has a value for (location, variant, D).

What is computed, per (forecast_date, horizon):
  coverage_50/80/95 : did the settled truth fall inside the stated HDI?
  wis               : weighted interval score (Bracher et al. decomposition-compatible form)
  abs_error         : |median forecast - truth|
  pit-ish           : where the truth sits relative to the interval, for calibration plots

Nothing here is Nextstrain-specific beyond the field names; the same code will score the
`forecasts-flu` archive.
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
from collections import defaultdict  # noqa: E402

import pandas as pd  # noqa: E402

from backfill import SESSION, available_snapshots  # noqa: E402
from verify import BUCKET  # noqa: E402

LEVELS = (50, 80, 95)


_SNAPSHOT_CACHE = {}
# Snapshots range from 0.3 MB (clades) to 25 MB (Pango). A 400-entry cache cost ~5 GB of RAM in
# practice, which is not acceptable on a machine that has already been overloaded once. Bound the
# cache by BYTES, not entries, so a Pango run cannot quietly consume the same budget as a clade
# run holding 80x more data.
_CACHE_BYTES = 0
_CACHE_BYTE_LIMIT = 512 * 1024 ** 2          # 512 MB
_CACHE_SIZES = {}


def load_snapshot(key, cache=True):
    """Fetch a snapshot, memoised.

    Partition-matched scoring asks "which snapshots share this forecast's variant set?" for every
    forecast date. Without a cache that is O(n_forecast_dates x n_snapshots) downloads,
    40 x 770 = ~31,000 requests, which is why the first attempt had to be killed. With the cache
    it is one download per snapshot.
    """
    global _CACHE_BYTES
    # Hard stop before the machine is squeezed. An earlier unbounded cache took free RAM from
    # 24.4 GB to 6.2 GB and the run had to be killed; never again silently.
    if cache and _CACHE_BYTES > 0:
        free = resources.available_ram_bytes()
        if free < 4 * 1024 ** 3:
            _SNAPSHOT_CACHE.clear()
            _CACHE_SIZES.clear()
            _CACHE_BYTES = 0
            print(f"  [cache cleared: only {free/1e9:.1f} GB RAM free]", flush=True)
    if cache and key in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[key]
    r = SESSION.get(BUCKET + key, timeout=180)
    r.raise_for_status()
    nbytes = len(r.content)
    obj = r.json()
    if cache:
        while _SNAPSHOT_CACHE and _CACHE_BYTES + nbytes > _CACHE_BYTE_LIMIT:
            old = next(iter(_SNAPSHOT_CACHE))          # FIFO eviction
            _SNAPSHOT_CACHE.pop(old)
            _CACHE_BYTES -= _CACHE_SIZES.pop(old, 0)
        _SNAPSHOT_CACHE[key] = obj
        _CACHE_SIZES[key] = nbytes
        _CACHE_BYTES += nbytes
    return obj


def cache_stats():
    return dict(entries=len(_SNAPSHOT_CACHE), mb=round(_CACHE_BYTES / 1e6, 1),
                limit_mb=round(_CACHE_BYTE_LIMIT / 1e6, 1))


def partition_index(snaps, stride=1, verbose=True):
    """{snapshot_date: frozenset(variants)}: one pass over the archive.

    Loading metadata still requires the whole object, so this is the expensive step; doing it
    once and reusing it is what makes partition matching tractable.
    """
    idx = {}
    dates = sorted(snaps)[::stride]
    for i, d in enumerate(dates, 1):
        try:
            idx[d] = frozenset(load_snapshot(snaps[d][0]).get("metadata", {}).get("variants", []))
        except Exception:
            continue
        if verbose and i % 50 == 0:
            print(f"    partition index {i}/{len(dates)}", flush=True)
    return idx


def extract_forecast(obj):
    """{(location, variant, date): {ps: value}} for site == freq_forecast."""
    out = defaultdict(dict)
    for x in obj["data"]:
        if x.get("site") != "freq_forecast" or x.get("date") is None:
            continue
        v = x.get("value")
        if not isinstance(v, (int, float)):
            continue
        out[(x["location"], x["variant"], x["date"])][x.get("ps")] = float(v)
    return out


def extract_truth(obj, site="weekly_raw_freq"):
    return {(x["location"], x["variant"], x["date"]): float(x["value"])
            for x in obj["data"]
            if x.get("site") == site and x.get("date") is not None
            and isinstance(x.get("value"), (int, float))}


def wis(truth, median, intervals):
    """Weighted interval score.

    WIS = (1/(K+1/2)) * [ |y-m|/2 + sum_k (alpha_k/2) * IS_alpha_k ]
    with IS_alpha = (u-l) + (2/alpha)(l-y)1{y<l} + (2/alpha)(y-u)1{y>u}.
    """
    terms = 0.5 * abs(truth - median)
    k = 0
    for lvl, (lo, hi) in intervals.items():
        alpha = 1.0 - lvl / 100.0
        is_a = (hi - lo)
        if truth < lo:
            is_a += (2.0 / alpha) * (lo - truth)
        elif truth > hi:
            is_a += (2.0 / alpha) * (truth - hi)
        terms += (alpha / 2.0) * is_a
        k += 1
    return terms / (k + 0.5)


def is_catchall(variant):
    """`other` is a residual category whose MEMBERSHIP changes between snapshots.

    2024-03-16 variants: ['23B', '23F', 'other', '23I']
    2026-08-23 variants: ['25B', '25I', 'other', '25C']

    So `other` means "everything except 23B/23F/23I" in one file and "everything except
    25B/25I/25C" in another. Scoring a forecast of one against an observation of the other
    compares different quantities that happen to share a label: the same label-semantics trap
    that invalidated a naive cross-cohort comparison in the B-ALL work.

    Named clades/lineages ('23B', 'JN.1', ...) do have stable meaning and are safe to score.

    `unassigned` is the same trap under a different name. It appears in the forecasts-flu archive
    (variants: [..., 'other', 'unassigned', 'K']) and means "sequenced but not matched to any
    named haplotype" -- a set whose membership changes every time a haplotype is designated.
    Verified absent from the scored SARS-CoV-2 clades, so adding it does not alter F7/F8.
    """
    return str(variant).strip().lower() in {"other", "others", "residual",
                                            "unassigned", "unassigned_"}


def naive_7day(obj, site="daily_raw_freq", window=7):
    """Figgins & Bedford's naive model, adopted VERBATIM for comparability.

    Their definition (PLOS Comput Biol 2024, 10.1371/journal.pcbi.1012443):

        "The naive model is implemented as a 7-day moving average on the retrospective raw
         frequencies using the most recent seven days for which sequencing data is available."

    Two properties worth stating because they matter for interpretation:
      - it is a POINT forecast, constant across horizon. A naive model has no opinion about how
        frequencies will move, only about where they are now.
      - it is computed from the snapshot AS PUBLISHED on the forecast date, so it sees exactly the
        data the real forecast saw. No hindsight leaks in.

    Returns {(location, variant): value}.
    """
    series = defaultdict(dict)
    for x in obj["data"]:
        if x.get("site") != site or x.get("date") is None:
            continue
        v = x.get("value")
        if not isinstance(v, (int, float)):
            continue
        series[(x["location"], x["variant"])][x["date"]] = float(v)
    out = {}
    for k, by_date in series.items():
        recent = sorted(by_date)[-window:]
        if recent:
            out[k] = sum(by_date[d] for d in recent) / len(recent)
    return out


def variant_set(obj):
    return frozenset(obj.get("metadata", {}).get("variants", []))


def build_settled_truth_partition_matched(snaps, forecast_date, site="freq",
                                          min_lag_days=60, max_lag_days=400, verbose=False):
    """Truth for ONE forecast date, taken only from snapshots using the SAME variant partition.

    Why this is necessary: and why the naive audit fails:

    Nextstrain clades are HIERARCHICAL. When a clade's descendants are later split out as their
    own clades, the parent's label narrows to mean only the residual. Concretely, a 2024-09-26
    forecast put UK clade `24C` at 0.967; the settled series has `24C` at 0.070 for a target date
    TWO DAYS later. That is not a forecast error, it is two different quantities sharing a label.
    The residual `other` category has the same problem in a more obvious form.

    Frequencies are a PARTITION summing to 1 (verified: sums are exactly 1.0000). A frequency is
    therefore only comparable between two snapshots that use the same partition.

    So: score a forecast only against truth drawn from snapshots whose `variants` set is
    identical to the forecast snapshot's. This is restrictive (it discards most of the archive)
    but it is the only comparison that is well defined.
    """
    f_d = dt.date.fromisoformat(forecast_date)
    fobj = load_snapshot(snaps[forecast_date][0])
    vset = variant_set(fobj)
    truth, used = {}, []
    for s in sorted(snaps, reverse=True):
        s_d = dt.date.fromisoformat(s)
        lag = (s_d - f_d).days
        if lag < min_lag_days or lag > max_lag_days:
            continue
        try:
            obj = load_snapshot(snaps[s][0])
        except Exception:
            continue
        if variant_set(obj) != vset:
            continue                       # different partition -> incomparable
        used.append(s)
        for k, v in extract_truth(obj, site).items():
            if is_catchall(k[1]) or k in truth:
                continue
            try:
                target = dt.date.fromisoformat(k[2])
            except ValueError:
                continue
            if (s_d - target).days >= min_lag_days:
                truth[k] = v
    if verbose:
        print(f"    {forecast_date}: partition |V|={len(vset)}, "
              f"{len(used)} matching snapshots, {len(truth):,} truth points")
    return truth, vset, used


def build_settled_truth(snaps, site="weekly_raw_freq", min_lag_days=60,
                        truth_stride=30, verbose=True):
    """Assemble the settled truth by unioning across snapshots.

    Taking truth from a single (latest) snapshot does NOT work: each file spans only ~180 days,
    so the newest snapshot has no value for older target dates. A first run scored just 120
    points on 1 forecast date for exactly this reason.

    Correct definition: for each (location, variant, date), take the value from the NEWEST
    snapshot published at least `min_lag_days` after that date: the most-settled observation
    that still satisfies the lag requirement. Walking newest -> oldest and never overwriting an
    existing key gives that in one pass.

    `min_lag_days=60` follows FINDINGS.md F1: by +60 d most slices have largely settled, while
    +120 d would discard the final months of the archive entirely.
    """
    dates = sorted(snaps)
    sampled = dates[::truth_stride]
    if dates[-1] not in sampled:
        sampled.append(dates[-1])
    truth = {}
    for s in sorted(sampled, reverse=True):          # newest first
        s_d = dt.date.fromisoformat(s)
        try:
            obj = load_snapshot(snaps[s][0])
        except Exception:
            continue
        for k, v in extract_truth(obj, site).items():
            if is_catchall(k[1]):
                continue                              # unstable membership; see is_catchall()
            if k in truth:
                continue                              # already have a more-settled value
            try:
                target = dt.date.fromisoformat(k[2])
            except ValueError:
                continue
            if (s_d - target).days >= min_lag_days:   # enforce the settling lag
                truth[k] = v
    if verbose:
        print(f"  settled truth: {len(truth):,} points from {len(sampled)} snapshots "
              f"(min lag {min_lag_days}d, stride {truth_stride})")
    return truth


def score_slice(branch, scheme, region, model="mlr", site="weekly_raw_freq",
                max_snapshots=None, stride=1, verbose=True,
                min_lag_days=60, truth_stride=30):
    snaps = available_snapshots(branch, scheme, region, model)
    if not snaps:
        return pd.DataFrame()
    dates = sorted(snaps)
    truth = build_settled_truth(snaps, site, min_lag_days, truth_stride, verbose)
    if not truth:
        return pd.DataFrame()

    use = dates[:-1][::stride]
    if max_snapshots:
        use = use[-max_snapshots:]
    rows = []
    for i, fd in enumerate(use, 1):
        try:
            fc = extract_forecast(load_snapshot(snaps[fd][0]))
        except Exception as e:
            print(f"    {fd}: SKIP {type(e).__name__}")
            continue
        f_d = dt.date.fromisoformat(fd)
        n_scored = 0
        for (loc, var, d), ps in fc.items():
            if is_catchall(var):
                continue
            y = truth.get((loc, var, d))
            if y is None or "median" not in ps:
                continue
            try:
                target = dt.date.fromisoformat(d)
            except ValueError:
                continue
            horizon = (target - f_d).days
            if horizon <= 0:
                continue          # strictly out-of-sample only
            intervals = {}
            for lvl in LEVELS:
                lo, hi = ps.get(f"HDI_{lvl}_lower"), ps.get(f"HDI_{lvl}_upper")
                if lo is not None and hi is not None:
                    intervals[lvl] = (min(lo, hi), max(lo, hi))
            if not intervals:
                continue
            row = dict(branch=branch, scheme=scheme, region=region,
                       forecast_date=fd, target_date=d, horizon=horizon,
                       location=loc, variant=var,
                       median=ps["median"], truth=y,
                       abs_error=abs(ps["median"] - y),
                       wis=wis(y, ps["median"], intervals))
            for lvl, (lo, hi) in intervals.items():
                row[f"cov_{lvl}"] = int(lo <= y <= hi)
                row[f"width_{lvl}"] = hi - lo
            rows.append(row)
            n_scored += 1
        if verbose and (i % 25 == 0 or i == len(use)):
            print(f"    [{i}/{len(use)}] {fd}: +{n_scored:,} scored "
                  f"(total {len(rows):,})", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    if df.empty:
        return {}
    out = {"n": int(len(df)), "n_forecast_dates": int(df.forecast_date.nunique())}
    for lvl in LEVELS:
        c = f"cov_{lvl}"
        if c in df:
            out[f"coverage_{lvl}"] = float(df[c].mean())
            out[f"nominal_{lvl}"] = lvl / 100.0
    out["mean_wis"] = float(df.wis.mean())
    out["mean_abs_error"] = float(df.abs_error.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="open")
    ap.add_argument("--scheme", default="nextstrain_clades")
    ap.add_argument("--region", default="global")
    ap.add_argument("--site", default="weekly_raw_freq")
    ap.add_argument("--stride", type=int, default=7, help="score every Nth snapshot")
    ap.add_argument("--max-snapshots", type=int, default=60)
    ap.add_argument("--min-lag-days", type=int, default=60)
    ap.add_argument("--truth-stride", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tag = f"{a.branch}_{a.scheme}_{a.region}"
    out = a.out or f"results/scores_{tag}.csv"
    print(f"scoring {a.branch}/{a.scheme}/{a.region} against '{a.site}' "
          f"(stride {a.stride}, last {a.max_snapshots} snapshots)")
    df = score_slice(a.branch, a.scheme, a.region, site=a.site,
                     stride=a.stride, max_snapshots=a.max_snapshots,
                     min_lag_days=a.min_lag_days, truth_stride=a.truth_stride)
    if df.empty:
        print("no scored points")
        return 1
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df):,} scored points)\n")

    s = summarize(df)
    print("=== HEADLINE: empirical coverage vs nominal ===")
    for lvl in LEVELS:
        k = f"coverage_{lvl}"
        if k in s:
            print(f"  {lvl}% HDI -> empirical {s[k]:.3f}   (nominal {lvl/100:.2f})  "
                  f"{'UNDER' if s[k] < lvl/100 - 0.02 else 'over' if s[k] > lvl/100 + 0.02 else 'ok'}")
    print(f"\n  mean WIS {s['mean_wis']:.4f}   mean |error| {s['mean_abs_error']:.4f}")
    print(f"  {s['n']:,} scored points over {s['n_forecast_dates']} forecast dates")

    print("\n=== coverage by horizon ===")
    df["hbin"] = pd.cut(df.horizon, [0, 7, 14, 21, 28, 60, 10 ** 6],
                        labels=["1-7d", "8-14d", "15-21d", "22-28d", "29-60d", "60d+"])
    cols = [f"cov_{lv}" for lv in LEVELS if f"cov_{lv}" in df]
    print(df.groupby("hbin", observed=True)[cols + ["wis"]].mean()
          .to_string(float_format=lambda v: f"{v:.3f}"))

    json.dump(s, open(out.replace(".csv", "_summary.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
