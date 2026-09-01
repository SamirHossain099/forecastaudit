"""Is partition-matching a biased filter?

Only ~58% of sampled forecast dates survive partition matching at a 14-day settling lag. If the
surviving dates differ systematically from the excluded ones, every result is conditioned on a
non-random subset and the audit describes a favorable slice rather than the archive.

Outcomes cannot be compared -- the excluded dates have no scorable outcome, which is precisely why
they are excluded. What CAN be compared is everything observable at or before the forecast date:

  partition size        number of variants the model was tracking
  calendar time         are exclusions clustered in particular periods?
  forecast breadth      locations and variants carried in the snapshot
  forecast sharpness    mean stated interval width
  forecast level        mean predicted frequency
  matching snapshots    zero by construction for excluded dates (reported, not tested)

Similar distributions do not prove absence of bias -- the selection operates on partition
STABILITY, which is not fully captured by any of these. But a large difference would be decisive
evidence of bias, and its absence is the strongest available reassurance.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backfill import available_snapshots  # noqa: E402
from score import extract_forecast, is_catchall, load_snapshot  # noqa: E402


def snapshot_features(snaps, date):
    """Everything observable in the snapshot published on `date`."""
    try:
        obj = load_snapshot(snaps[date][0])
    except Exception:
        return None
    fc = extract_forecast(obj)
    if not fc:
        return None
    locs, vars_, widths, meds = set(), set(), [], []
    for (loc, var, _), ps in fc.items():
        if is_catchall(var):
            continue
        locs.add(loc)
        vars_.add(var)
        if "median" in ps:
            meds.append(ps["median"])
        lo, hi = ps.get("HDI_95_lower"), ps.get("HDI_95_upper")
        if lo is not None and hi is not None:
            widths.append(abs(hi - lo))
    if not meds:
        return None
    return dict(forecast_date=date,
                n_locations=len(locs), n_variants=len(vars_),
                mean_width_95=float(np.mean(widths)) if widths else np.nan,
                mean_median=float(np.mean(meds)),
                frac_below_1pct=float(np.mean(np.asarray(meds) < 0.01)))


def welch(a, b):
    """Welch t statistic and a normal-approximation two-sided p, for a quick screen."""
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return np.nan, np.nan
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    from math import erfc
    return float(t), float(erfc(abs(t) / np.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/scores_matched_lag14_gisaid_dense_audit.csv")
    ap.add_argument("--branch", default="gisaid")
    ap.add_argument("--scheme", default="nextstrain_clades")
    ap.add_argument("--region", default="global")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/selection_{a.label}.json"
    if not os.path.exists(a.audit):
        print(f"missing {a.audit}")
        return 1

    aud = pd.read_csv(a.audit)
    aud["auditable"] = aud.truth_points > 0
    print(f"{a.label}: {len(aud)} sampled forecast dates, "
          f"{int(aud.auditable.sum())} auditable ({aud.auditable.mean():.1%})\n")

    snaps = available_snapshots(a.branch, a.scheme, a.region)
    feats = []
    for i, d in enumerate(aud.forecast_date, 1):
        f = snapshot_features(snaps, d)
        if f:
            feats.append(f)
        if i % 30 == 0:
            print(f"    features {i}/{len(aud)}", flush=True)
    F = pd.DataFrame(feats).merge(
        aud[["forecast_date", "partition_size", "auditable"]], on="forecast_date")
    F["days_from_start"] = [
        (dt.date.fromisoformat(x) - dt.date.fromisoformat(F.forecast_date.min())).days
        for x in F.forecast_date]

    cols = ["partition_size", "n_locations", "n_variants", "mean_width_95",
            "mean_median", "frac_below_1pct", "days_from_start"]
    print("\n=== observable characteristics: auditable vs excluded ===")
    print(f"  {'feature':>18s} {'auditable':>12s} {'excluded':>12s} {'diff':>9s} "
          f"{'t':>7s} {'p':>7s}")
    res = {"label": a.label, "n_sampled": int(len(aud)),
           "n_auditable": int(aud.auditable.sum()), "features": {}}
    for c in cols:
        x = F[F.auditable][c]
        y = F[~F.auditable][c]
        t, p = welch(x, y)
        res["features"][c] = dict(auditable=float(x.mean()), excluded=float(y.mean()),
                                  t=t, p=p)
        flag = " *" if np.isfinite(p) and p < 0.05 else ""
        print(f"  {c:>18s} {x.mean():>12.3f} {y.mean():>12.3f} "
              f"{x.mean() - y.mean():>9.3f} {t:>7.2f} {p:>7.3f}{flag}")

    print("\n=== is exclusion clustered in time? ===")
    F = F.sort_values("days_from_start")
    q = pd.qcut(F.days_from_start, 4, labels=["Q1 (early)", "Q2", "Q3", "Q4 (late)"])
    tab = F.groupby(q, observed=True).auditable.agg(["mean", "size"])
    print(tab.to_string(float_format=lambda v: f"{v:.2f}"))
    res["by_time_quartile"] = {str(k): dict(auditable_rate=float(v["mean"]), n=int(v["size"]))
                               for k, v in tab.iterrows()}

    print("\n=== verdict ===")
    sig = [c for c, v in res["features"].items()
           if np.isfinite(v["p"]) and v["p"] < 0.05]
    if not sig:
        print("  No observable pre-forecast characteristic differs at p < 0.05 between auditable")
        print("  and excluded dates. Selection is on partition STABILITY, which none of these")
        print("  measures directly, so this is reassurance rather than proof.")
    else:
        print(f"  Differs at p < 0.05: {', '.join(sig)}")
        print("  These must be reported; the audited subset is not observationally identical")
        print("  to the excluded one.")
    res["significant"] = sig
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
