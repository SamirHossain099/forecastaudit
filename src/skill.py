"""Skill scores against a naive baseline — closing the "0.0265 WIS compared to what?" objection.

An absolute WIS is uninterpretable. Forecast-evaluation practice is to report skill relative to a
naive reference, and a reviewer will ask for it immediately.

TWO baselines, because one is not enough and each answers a different question.

  naive-point     Figgins & Bedford's model verbatim: a 7-day moving average of raw frequencies
                  over the most recent seven days available in the snapshot the forecast was
                  published in. It is a POINT forecast, constant across horizon.
                  Compared on MEAN ABSOLUTE ERROR, which is what they report, so our numbers are
                  directly comparable to a published reference.

  naive-interval  The same point forecast, given uncertainty from the EMPIRICAL DISTRIBUTION OF ITS
                  OWN PAST ERRORS at the same horizon, using only strictly earlier forecast dates.
                  This exists because comparing a probabilistic forecast's WIS against a point
                  forecast's WIS is not a fair fight: a point forecast has zero interval width, so
                  its WIS degenerates to |y - m| and it is punished for having no opinion about
                  uncertainty rather than for being wrong. Giving the baseline honest, out-of-time
                  intervals makes the WIS comparison meaningful.

Note the degenerate identity used above: with lo = hi = m, the interval score reduces to
(2/alpha)|y-m| for every level, and WIS collapses exactly to |y - m|. That is asserted in the
tests rather than assumed here.

Skill is reported as 1 - score_model / score_baseline, so positive means the real forecast beats
the naive one and 0 means it does not.
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

from score import LEVELS, wis  # noqa: E402

HBINS = [0, 7, 14, 21, 30]
HLABS = ["1-7d", "8-14d", "15-21d", "22-30d"]


def add_bins(df):
    df = df.copy()
    df["hbin"] = pd.cut(df.horizon, HBINS, labels=HLABS)
    return df


def naive_interval_wis(df, min_cal=100):
    """WIS for the naive point forecast given out-of-time empirical intervals.

    For each forecast date, the residual distribution is estimated from STRICTLY EARLIER forecast
    dates at the same horizon bin. Nothing from the evaluated date informs its own intervals.
    """
    df = add_bins(df).sort_values("forecast_date")
    dates = sorted(df.forecast_date.unique())
    out = []
    for i, d in enumerate(dates):
        if i == 0:
            continue
        cal = df[df.forecast_date < d]
        tst = df[df.forecast_date == d]
        if len(cal) < min_cal:
            continue
        for hb, sub in tst.groupby("hbin", observed=True):
            res = (cal[cal.hbin == hb].truth - cal[cal.hbin == hb].naive).dropna()
            if len(res) < 30:
                res = (cal.truth - cal.naive).dropna()      # fall back to all horizons
            if len(res) < 30:
                continue
            for _, r in sub.iterrows():
                if not np.isfinite(r.naive):
                    continue
                iv = {}
                for lvl in LEVELS:
                    a = (1 - lvl / 100) / 2
                    lo = r.naive + np.quantile(res, a)
                    hi = r.naive + np.quantile(res, 1 - a)
                    iv[lvl] = (min(lo, hi), max(lo, hi))
                out.append(dict(forecast_date=d, horizon=r.horizon, hbin=hb,
                                truth=r.truth, naive=r.naive,
                                naive_wis=wis(r.truth, r.naive, iv),
                                model_wis=r.wis, model_abs=r.abs_error,
                                naive_abs=abs(r.naive - r.truth),
                                **{f"naive_cov_{lvl}": int(iv[lvl][0] <= r.truth <= iv[lvl][1])
                                   for lvl in LEVELS},
                                **{f"model_cov_{lvl}": int(r[f"cov_{lvl}"]) for lvl in LEVELS}))
    return pd.DataFrame(out)


def skill(model, baseline):
    m, b = np.nanmean(model), np.nanmean(baseline)
    return float(1 - m / b) if b and np.isfinite(b) and b > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/scores_matched_lag14_gisaid_dense.csv")
    ap.add_argument("--label", default="gisaid")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"results/skill_{a.label}.json"
    if not os.path.exists(a.infile):
        print(f"missing {a.infile}")
        return 1

    df = add_bins(pd.read_csv(a.infile))
    if "naive" not in df.columns:
        print("no `naive` column — re-run score_matched.py after the baseline was added")
        return 1
    ok = df[np.isfinite(df.naive)]
    print(f"{a.label}: {len(df):,} scored points, {len(ok):,} with a naive baseline "
          f"({len(ok) / len(df):.1%})")

    res = {"n": int(len(ok))}

    print("\n=== point accuracy vs the Figgins-Bedford naive model (their metric) ===")
    print(f"  {'horizon':>10s} {'n':>7s} {'model MAE':>10s} {'naive MAE':>10s} {'skill':>8s}")
    rows = []
    for hb, g in ok.groupby("hbin", observed=True):
        mm, nn = g.abs_error.mean(), (g.naive - g.truth).abs().mean()
        rows.append(dict(hbin=str(hb), n=len(g), model_mae=mm, naive_mae=nn,
                         mae_skill=1 - mm / nn if nn else np.nan))
        print(f"  {str(hb):>10s} {len(g):>7,} {mm:>10.4f} {nn:>10.4f} "
              f"{rows[-1]['mae_skill']:>+8.1%}")
    mm, nn = ok.abs_error.mean(), (ok.naive - ok.truth).abs().mean()
    overall = 1 - mm / nn if nn else np.nan
    print(f"  {'ALL':>10s} {len(ok):>7,} {mm:>10.4f} {nn:>10.4f} {overall:>+8.1%}")
    res["mae_by_horizon"] = rows
    res["mae_skill_overall"] = float(overall)
    res["model_mae"] = float(mm)
    res["naive_mae"] = float(nn)

    print("\n=== probabilistic: WIS vs a naive model given out-of-time empirical intervals ===")
    nb = naive_interval_wis(ok)
    if nb.empty:
        print("  (insufficient calibration data)")
    else:
        s_pt = skill(nb.model_wis, nb.naive_abs)       # naive as a point forecast (WIS == MAE)
        s_iv = skill(nb.model_wis, nb.naive_wis)       # naive with empirical intervals
        print(f"  n = {len(nb):,} points over {nb.forecast_date.nunique()} out-of-time dates")
        print(f"  model WIS                      {nb.model_wis.mean():.4f}")
        print(f"  naive WIS (point, == MAE)      {nb.naive_abs.mean():.4f}   "
              f"skill {s_pt:+.1%}")
        print(f"  naive WIS (empirical intervals){nb.naive_wis.mean():>10.4f}   "
              f"skill {s_iv:+.1%}")
        res.update(model_wis=float(nb.model_wis.mean()),
                   naive_wis_point=float(nb.naive_abs.mean()),
                   naive_wis_interval=float(nb.naive_wis.mean()),
                   wis_skill_vs_point=float(s_pt), wis_skill_vs_interval=float(s_iv),
                   n_out_of_time=int(len(nb)))

        print("\n=== and the twist: is the NAIVE baseline better calibrated? ===")
        print(f"  {'level':>7s} {'model':>8s} {'naive':>8s} {'nominal':>8s}")
        cov = {}
        for lvl in LEVELS:
            m_c = nb[f"model_cov_{lvl}"].mean()
            n_c = nb[f"naive_cov_{lvl}"].mean()
            cov[lvl] = dict(model=float(m_c), naive=float(n_c))
            print(f"  {lvl:>6d}% {m_c:>8.3f} {n_c:>8.3f} {lvl / 100:>8.2f}")
        res["coverage_vs_naive"] = cov

        print("\n=== WIS skill by horizon ===")
        for hb, g in nb.groupby("hbin", observed=True):
            print(f"  {str(hb):>8s} n={len(g):>6,}  vs point {skill(g.model_wis, g.naive_abs):+7.1%}"
                  f"   vs intervals {skill(g.model_wis, g.naive_wis):+7.1%}")
        nb.to_csv(out.replace(".json", ".csv"), index=False)

    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
