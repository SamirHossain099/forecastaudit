"""Figures for the audit.

Money plot (brief §5): x = forecast horizon, y = empirical coverage of the nominal 95% interval,
one line per branch, bold reference at 0.95, shaded Monte-Carlo band.

Five-second read: *the stated intervals are not the intervals you get.*
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BRANCHES = {"gisaid": "#c0392b", "open": "#2c6fbb"}
# The flu archive is a second pathogen (F9); it joins the recalibration panel only,
# because its horizon structure (to 98 d) is not comparable with ncov's 30 d.
RECAL_SETS = ("gisaid_full", "open_full", "flu")
LEVELS = (50, 80, 95)


def wilson(k, n, z=1.96):
    """Wilson interval — honest at the small n some horizon bins have."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(branch):
    """Prefer the full stride-7 sample (F17), then the earlier capped/dense ones."""
    for p in (f"results/scores_full_{branch}.csv",
              f"results/scores_matched_lag14_{branch}_dense.csv",
              f"results/scores_matched_lag14_{branch}.csv"):
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def fig_coverage_vs_horizon(dfs, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, lvl in zip(axes, LEVELS):
        for br, df in dfs.items():
            g = df.groupby("horizon")[f"cov_{lvl}"].agg(["mean", "sum", "size"])
            g = g[g["size"] >= 20]
            lohi = [wilson(k, n) for k, n in zip(g["sum"], g["size"])]
            lo = [x[0] for x in lohi]
            hi = [x[1] for x in lohi]
            ax.fill_between(g.index, lo, hi, color=BRANCHES[br], alpha=0.18, lw=0)
            ax.plot(g.index, g["mean"], color=BRANCHES[br], lw=2,
                    label=f"{br} (n={len(df):,})")
        ax.axhline(lvl / 100, color="k", lw=2.2, ls="--", zorder=5)
        ax.text(ax.get_xlim()[1], lvl / 100 + .02, f"nominal {lvl/100:.2f}",
                ha="right", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(f"{lvl}% HDI", fontsize=11)
        ax.set_xlabel("forecast horizon (days)")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle("Archived Nextstrain variant forecasts: stated vs achieved coverage "
                 "(partition-matched, 14-day settling lag)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_overconfidence_by_level(dfs, out):
    """Where the overconfidence lives: error magnitude relative to interval scale."""
    bins = [-.001, .01, .05, .15, .35, .65, 1.001]
    labs = ["<1%", "1-5%", "5-15%", "15-35%", "35-65%", ">65%"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.38
    for i, (br, df) in enumerate(dfs.items()):
        d = df.copy()
        d["lev"] = pd.cut(d["median"], bins, labels=labs)
        g = d.groupby("lev", observed=True).agg(
            ae=("abs_error", "mean"), w=("width_95", "mean"), cov=("cov_95", "mean"),
            n=("wis", "size"))
        x = np.arange(len(g))
        a1.bar(x + (i - .5) * width, g.ae / (g.w / 2), width, color=BRANCHES[br],
               alpha=.85, label=br)
        a2.bar(x + (i - .5) * width, g["cov"], width, color=BRANCHES[br], alpha=.85,
               label=br)
        for ax in (a1, a2):
            ax.set_xticks(x)
            ax.set_xticklabels(g.index, fontsize=9)
    a1.axhline(1.0, color="k", ls="--", lw=1.8)
    a1.text(0.02, 1.05, "error reaches band edge", transform=a1.get_yaxis_transform(),
            fontsize=8)
    a1.set_ylabel("mean |error| / half-width of 95% HDI")
    a1.set_xlabel("forecast level")
    a1.set_title("Overconfidence is worst at LOW frequencies", fontsize=11)
    a2.axhline(0.95, color="k", ls="--", lw=1.8)
    a2.set_ylabel("empirical coverage of 95% HDI")
    a2.set_xlabel("forecast level")
    a2.set_ylim(0, 1)
    a2.set_title("...and no stratum reaches nominal", fontsize=11)
    for a in (a1, a2):
        a.grid(alpha=.25, axis="y")
        a.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_recalibration(out):
    """Before/after: coverage recovered, and the width it cost."""
    import json
    files = {b: f"results/recalibration_{b}.json" for b in RECAL_SETS}
    files = {b: p for b, p in files.items() if os.path.exists(p)}
    if not files:
        print("  (no recalibration results yet)")
        return
    fig, axes = plt.subplots(2, len(files), figsize=(6.6 * len(files), 7.6), squeeze=False)
    for j, (br, path) in enumerate(files.items()):
        with open(path) as fh:
            r = pd.DataFrame(json.load(fh))
        br = br.replace("_dense", "").replace("_full", "")
        short = {"uncorrected": "as shipped", "CQR-global": "CQR global",
                 "CQR-mondrian(level)": "CQR by level",
                 "CQR-mondrian(level x horizon)": "CQR by level x horizon",
                 "MULT-mondrian(level)": "multiplicative by level"}
        r["lab"] = r.method.map(lambda m: short.get(m, m))
        x = np.arange(len(r))
        w = 0.26
        a = axes[0][j]
        for i, lvl in enumerate(LEVELS):
            a.bar(x + (i - 1) * w, r[f"cov_{lvl}"], w, label=f"{lvl}%",
                  color=plt.cm.viridis(i / 2.4), alpha=.9)
            a.axhline(lvl / 100, color=plt.cm.viridis(i / 2.4), ls="--", lw=1.6)
        a.set_xticks(x)
        a.set_xticklabels(r.lab, rotation=18, ha="right", fontsize=8)
        a.set_ylabel("out-of-time coverage")
        a.set_ylim(0, 1.05)
        a.set_title(f"{br}: coverage recovered (dashed = nominal)", fontsize=11)
        a.legend(fontsize=8, frameon=False, ncol=3)
        a.grid(alpha=.25, axis="y")
        b = axes[1][j]
        for i, lvl in enumerate(LEVELS):
            b.bar(x + (i - 1) * w, r[f"width_{lvl}"], w, label=f"{lvl}%",
                  color=plt.cm.viridis(i / 2.4), alpha=.9)
        b.set_xticks(x)
        b.set_xticklabels(r.lab, rotation=18, ha="right", fontsize=8)
        b.set_ylabel("mean interval width")
        b.set_title(f"{br}: the width it cost (1.0 = whole simplex)", fontsize=11)
        b.legend(fontsize=8, frameon=False, ncol=3)
        b.grid(alpha=.25, axis="y")
    fig.suptitle("Out-of-time conformal recalibration: the deficit is largely recoverable, "
                 "and level-conditioning buys sharpness", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_tradeoff(out):
    """Figure 1: the settling-vs-stability trade-off that sets the operating point."""
    if not os.path.exists("results/lag_tradeoff_full.csv"):
        print("  (no lag_tradeoff.csv)")
        return
    t = pd.read_csv("results/lag_tradeoff_full.csv")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for br, g in t.groupby("branch"):
        g = g.sort_values("min_lag_days")
        ax.plot(g.min_lag_days, g.frac, marker="o", lw=2.2,
                color=BRANCHES.get(br, "#555"), label=f"{br}: auditable fraction")
    ax.set_xlabel("minimum settling lag (days)")
    ax.set_ylabel("fraction of forecast dates that remain auditable")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.25)

    if os.path.exists("results/backfill_profile_all.csv"):
        b = pd.read_csv("results/backfill_profile_all.csv")
        b = b[(b.site == "weekly_raw_freq") & (b.scheme == "nextstrain_clades")
              & (b.region == "global") & (b.branch == "gisaid")]
        if len(b):
            ax2 = ax.twinx()
            b = b.sort_values("lag_days")
            ax2.plot(b.lag_days, b.frac_gt_05, marker="s", ls="--", lw=2.2, color="#c47f00",
                     label="observations still revised >0.05")
            ax2.set_ylabel("fraction of observations revised by >0.05", color="#c47f00")
            ax2.tick_params(axis="y", labelcolor="#c47f00")
            ax2.set_ylim(0, max(0.5, float(b.frac_gt_05.max()) * 1.15))
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="center right")
        else:
            ax.legend(fontsize=9, frameon=False)
    ax.axvspan(14, 30, color="#2e8b57", alpha=.10)
    ax.text(22, 0.93, "operating\nrange", ha="center", fontsize=9, color="#2e8b57")
    ax.set_title("Waiting for the truth to settle destroys the comparison faster than it\n"
                 "improves the truth", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


DATASET_COLOUR = {"gisaid": "#c0392b", "open": "#2c6fbb", "flu": "#8e44ad"}


def fig_decomposition(out):
    """Figure 3: MCB rises with horizon while DSC does not."""
    import json
    rows = []
    for lab in ("gisaid", "open", "flu"):
        p = f"results/decomposition_{lab}_full.json"
        if not os.path.exists(p):
            p = f"results/decomposition_{lab}.json"
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        for r in d.get("by_horizon", []):
            rows.append(dict(dataset=lab, hbin=r["hbin"], mcb=r["mcb"], dsc=r["dsc"],
                             score=r["score"], share=r["mcb_share"]))
    if not rows:
        print("  (no decomposition results)")
        return
    df = pd.DataFrame(rows)
    order = ["1-7d", "8-14d", "15-21d", "22-30d"]
    df["x"] = df.hbin.map({h: i for i, h in enumerate(order)})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for lab, g in df.groupby("dataset"):
        g = g.sort_values("x")
        c = DATASET_COLOUR[lab]
        a1.plot(g.x, g.mcb, marker="o", lw=2.2, color=c, label=f"{lab} MCB")
        a1.plot(g.x, g.dsc, marker="s", lw=2.0, ls="--", color=c, alpha=.6,
                label=f"{lab} DSC")
        a2.plot(g.x, g.share, marker="o", lw=2.2, color=c, label=lab)
    for ax in (a1, a2):
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_xlabel("forecast horizon")
        ax.grid(alpha=.25)
    a1.set_ylabel("score component")
    a1.set_title("Miscalibration grows with horizon;\ndiscrimination does not", fontsize=11)
    a1.legend(fontsize=7.5, frameon=False, ncol=3)
    a2.set_ylabel("MCB as a share of WIS")
    a2.set_ylim(0, 0.7)
    a2.set_title("Share of the score that is\nrecoverable miscalibration", fontsize=11)
    a2.legend(fontsize=9, frameon=False)
    fig.suptitle("Decomposing the weighted interval score by horizon", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def fig_rtm(out):
    """The regression-to-the-mean correction: three conditionings side by side."""
    import json
    labs = ["<1%", "1-5%", "5-15%", "15-35%", "35-65%", ">65%"]
    sets = [(lab, f"results/heterosked_{lab}_full.json") for lab in ("gisaid", "open")]
    sets = [(lab, p) for lab, p in sets if os.path.exists(p)]
    if not sets:
        print("  (no heterosked results)")
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(6.6 * len(sets), 4.7), squeeze=False)
    for j, (lab, p) in enumerate(sets):
        with open(p) as fh:
            d = json.load(fh)
        ax = axes[0][j]
        x = np.arange(len(labs))
        w = 0.27
        for i, (key, color, name) in enumerate([
                ("(a) forecast level", "#bbb", "by forecast level (RTM-inflated)"),
                ("(b) outcome level", "#888", "by outcome level"),
                ("(c) baseline level, pre-forecast", "#c0392b", "by pre-forecast baseline")]):
            cond = d["conditionings"].get(key)
            if not cond:
                continue
            v = [cond["table"].get(b, {}).get("ratio", np.nan) for b in labs]
            ax.bar(x + (i - 1) * w, v, w, color=color, alpha=.9, label=name)
        ax.axhline(1.0, color="k", ls="--", lw=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=9)
        ax.set_xlabel("frequency bin")
        ax.set_ylabel("mean |error| / half-width of 95% HDI")
        ax.set_title(f"{lab}", fontsize=11)
        ax.grid(alpha=.25, axis="y")
        if j == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Conditioning on the forecast overstates the heteroscedasticity; "
                 "a pre-forecast instrument does not", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dfs = {b: d for b in BRANCHES if (d := load(b)) is not None}
    if not dfs:
        print("no score files yet — run score_matched.py first")
        return 1
    for b, d in dfs.items():
        print(f"  {b}: {len(d):,} points, {d.forecast_date.nunique()} forecast dates")
    fig_tradeoff(f"{a.outdir}/fig1_settling_tradeoff.png")
    fig_coverage_vs_horizon(dfs, f"{a.outdir}/money_coverage_vs_horizon.png")
    fig_overconfidence_by_level(dfs, f"{a.outdir}/overconfidence_by_level.png")
    fig_decomposition(f"{a.outdir}/fig3_decomposition_by_horizon.png")
    fig_rtm(f"{a.outdir}/fig_rtm_correction.png")
    fig_recalibration(f"{a.outdir}/recalibration.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
