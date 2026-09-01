"""Deliverable 2 — how much of the coverage deficit is recoverable post hoc?

F7 established that the archived intervals are severely overconfident (95% HDI covers 0.392) and
that the failure is *width*, not bias (bias^2/MSE = 1.9%). It also showed the miss is strongly
heteroscedastic: |error|/half-width is 5.4x at forecast levels <1% but ~1.0x at 35-65%. So a single
global widening cannot work, and the recalibration must be conditional.

Three methods, in increasing adaptivity:

  CQR-global      Conformalized Quantile Regression (Romano, Patterson & Candes, NeurIPS 2019).
                  Nonconformity E_i = max(lo_i - y_i, y_i - hi_i); the interval becomes
                  [lo - Q, hi + Q] with Q the (1-alpha) empirical quantile of E on calibration
                  data. Distribution-free and finite-sample valid under exchangeability.
  CQR-mondrian    The same, but with a separate Q per forecast-level stratum (group-conditional
                  / Mondrian conformal). Targets the heteroscedasticity directly.
  MULT-mondrian   Multiplicative: scale each interval about its own median by a per-stratum
                  factor. Respects the [0,1] frequency support better than an additive shift,
                  which can push a band below 0.

EVALUATION IS STRICTLY OUT-OF-TIME. For each test forecast date, calibration uses only forecast
dates STRICTLY EARLIER (rolling origin). Nothing is ever fit and evaluated on the same date. That
is the whole point: an in-sample recalibration would report ~nominal coverage by construction and
would be worthless.

Reported jointly: coverage AND mean interval width. A method that reaches nominal coverage by
making every band span [0,1] has not fixed anything, and the width column is what exposes that.
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

LEVELS = (50, 80, 95)
LEVEL_BINS = [-.001, .01, .05, .15, .35, .65, 1.001]
LEVEL_LABS = ["<1%", "1-5%", "5-15%", "15-35%", "35-65%", ">65%"]
CLIP = (0.0, 1.0)          # frequencies live on the simplex


def add_strata(df):
    df = df.copy()
    df["lev"] = pd.cut(df["median"], LEVEL_BINS, labels=LEVEL_LABS)
    df["hbin"] = pd.cut(df.horizon, [0, 7, 14, 21, 10 ** 6],
                        labels=["1-7d", "8-14d", "15-21d", "22d+"])
    return df


def nonconformity(df, lvl):
    """E_i = max(lo - y, y - hi). Negative when the point is comfortably inside."""
    return np.maximum(df[f"lo_{lvl}"] - df.truth, df.truth - df[f"hi_{lvl}"])


def mult_nonconformity(df, lvl):
    """Factor by which the interval must be scaled about its median to cover the point."""
    m, lo, hi = df["median"], df[f"lo_{lvl}"], df[f"hi_{lvl}"]
    y = df.truth
    lower_arm = (m - lo).replace(0, np.nan)
    upper_arm = (hi - m).replace(0, np.nan)
    need = np.where(y < m, (m - y) / lower_arm, (y - m) / upper_arm)
    return pd.Series(need, index=df.index).replace([np.inf, -np.inf], np.nan)


def conformal_q(scores, alpha, min_n=30):
    """Finite-sample-valid conformal quantile: ceil((n+1)(1-alpha))/n."""
    s = pd.Series(scores).dropna()
    n = len(s)
    if n < min_n:
        return None
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(s.max())                 # cannot certify; widen to the observed extreme
    return float(np.sort(s.values)[k - 1])


def apply_additive(df, lvl, q):
    lo = (df[f"lo_{lvl}"] - q).clip(*CLIP)
    hi = (df[f"hi_{lvl}"] + q).clip(*CLIP)
    return lo, hi


def apply_multiplicative(df, lvl, k):
    m = df["median"]
    lo = (m - k * (m - df[f"lo_{lvl}"])).clip(*CLIP)
    hi = (m + k * (df[f"hi_{lvl}"] - m)).clip(*CLIP)
    return lo, hi


def _stratum_mask(cal, strata, key):
    vals = key if isinstance(key, tuple) else (key,)
    mask = np.ones(len(cal), bool)
    for col, val in zip(strata, vals):
        mask &= (cal[col] == val).values
    return mask


def _mondrian(cal, tst, lvl, alpha, strata, score_fn, apply_fn):
    lo, hi = tst[f"lo_{lvl}"].copy(), tst[f"hi_{lvl}"].copy()
    fallback = conformal_q(score_fn(cal, lvl), alpha)
    for key, sub in tst.groupby(list(strata), observed=True):
        cal_s = cal[_stratum_mask(cal, strata, key)]
        q = conformal_q(score_fn(cal_s, lvl), alpha)
        if q is None:
            q = fallback
        if q is None:
            continue
        a, b = apply_fn(sub, lvl, q)
        lo.loc[sub.index] = a
        hi.loc[sub.index] = b
    return lo, hi


def rolling_origin(df, method="cqr_mondrian", strata=("lev",), min_cal=200):
    """For each forecast date, calibrate on strictly earlier dates only."""
    dates = sorted(df.forecast_date.unique())
    out = []
    for i, d in enumerate(dates):
        if i == 0:
            continue                                   # nothing earlier to calibrate on
        cal = df[df.forecast_date < d]
        tst = df[df.forecast_date == d]
        if len(cal) < min_cal or tst.empty:
            continue
        row = {"forecast_date": d, "n_cal": len(cal), "n_test": len(tst)}
        for lvl in LEVELS:
            alpha = 1 - lvl / 100.0
            lo, hi = tst[f"lo_{lvl}"].copy(), tst[f"hi_{lvl}"].copy()
            if method == "cqr_global":
                q = conformal_q(nonconformity(cal, lvl), alpha)
                if q is not None:
                    lo, hi = apply_additive(tst, lvl, q)
            elif method == "cqr_mondrian":
                lo, hi = _mondrian(cal, tst, lvl, alpha, strata,
                                   nonconformity, apply_additive)
            elif method == "mult_mondrian":
                lo, hi = _mondrian(cal, tst, lvl, alpha, strata,
                                   mult_nonconformity, apply_multiplicative)
            elif method != "none":
                raise ValueError(method)
            row[f"cov_{lvl}"] = float(((lo <= tst.truth) & (tst.truth <= hi)).mean())
            row[f"width_{lvl}"] = float((hi - lo).mean())
        out.append(row)
    return pd.DataFrame(out)


def summarize(res, label):
    if res.empty:
        return None
    d = {"method": label, "n_dates": len(res), "n_test": int(res.n_test.sum())}
    for lvl in LEVELS:
        # weight by test-set size so a 75-point date does not count like a 5,000-point one
        w = res.n_test
        d[f"cov_{lvl}"] = float((res[f"cov_{lvl}"] * w).sum() / w.sum())
        d[f"width_{lvl}"] = float((res[f"width_{lvl}"] * w).sum() / w.sum())
    return d


RUNS = [("none", "uncorrected", ("lev",)),
        ("cqr_global", "CQR-global", ("lev",)),
        ("cqr_mondrian", "CQR-mondrian(level)", ("lev",)),
        ("cqr_mondrian", "CQR-mondrian(level x horizon)", ("lev", "hbin")),
        ("mult_mondrian", "MULT-mondrian(level)", ("lev",))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="gisaid")
    ap.add_argument("--infile", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    inp = a.infile or f"results/scores_matched_lag14_{a.branch}.csv"
    out = a.out or f"results/recalibration_{a.branch}.json"
    if not os.path.exists(inp):
        print(f"missing {inp} - run score_matched.py first")
        return 1

    df = add_strata(pd.read_csv(inp))
    print(f"{a.branch}: {len(df):,} points, {df.forecast_date.nunique()} forecast dates")
    print("rolling origin - each date calibrated on STRICTLY EARLIER dates only\n")

    rows = []
    for meth, label, strata in RUNS:
        res = rolling_origin(df, method=meth, strata=strata)
        s = summarize(res, label)
        if s:
            rows.append(s)
            slug = label.split("(")[0].strip().replace(" ", "_")
            suffix = "_h" if len(strata) > 1 else ""
            res.to_csv(f"results/recal_{a.branch}_{slug}{suffix}.csv", index=False)

    r = pd.DataFrame(rows)
    print("=== out-of-time coverage (weighted by test-set size) ===")
    print(r[["method", "n_dates", "n_test"] + [f"cov_{v}" for v in LEVELS]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== mean interval width - the cost of that coverage ===")
    print(r[["method"] + [f"width_{v}" for v in LEVELS]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  nominal: 0.500 / 0.800 / 0.950")
    print("  A method reaching nominal coverage with a width near 1.0 has not fixed anything.")

    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
