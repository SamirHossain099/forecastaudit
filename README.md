# forecastaudit

Scoring, decomposing and recalibrating **archived probabilistic forecasts**: forecasts that were
published and timestamped before their outcomes existed.

This is the code behind *Skillful but overconfident: scoring the archived real-time SARS-CoV-2
variant forecasts*. It is written to be reusable on any quantile-format forecast archive; nothing
in the decomposition or the recalibrator is pathogen-specific.

## Why it exists

Scoring an archive is not just applying a scoring rule. Two things have to happen first, and
`scoringutils` (which solves the scoring itself well, and which you should use if you already have
matched pairs) does neither:

1. **Backfill-aware truth.** Observed frequencies for a past date keep changing as sequences are
   deposited, so "the outcome" depends on when you look.
2. **Partition matching.** The categories these forecasts are defined over are redefined roughly
   every three to five weeks, faster than the observations settle. Matching a forecast to an
   outcome *by label* therefore compares different quantities, and produces coverage numbers that
   are artifacts. `score_matched` restricts comparison to snapshots sharing an identical category
   set.

And one thing has to happen after: an **MCB / DSC / UNC decomposition of the weighted interval
score**, separating how much of the score is recoverable miscalibration from genuine
discrimination. No existing tool computes this for the WIS.

## Install

```bash
pip install forecastaudit
```

From a clone, for development:

```bash
pip install -e ".[dev]"
pytest -q
```

`results/` ships the scored summaries and the small tables, which is enough for almost the whole
suite. Seven tests guard claims against the full gisaid pair table, which is 24 MB and is not
committed; they skip with `no scored results on disk` until you regenerate it with
`python src/score_matched.py`.

## The pieces

| Module | What it does |
|---|---|
| `score` | snapshot loader, WIS, interval scores |
| `backfill` | backfill-aware truth construction and the settling-lag profile |
| `score_matched` | partition-matched scoring; keeps the raw interval bounds, which matters because most intervals are asymmetric about the median |
| `decompose` | quantile-wise MCB / DSC / UNC decomposition of the WIS |
| `idr_decompose` | interval-conditioned decomposition, fitted along linear extensions of the componentwise partial order |
| `idr_exact` | the same problem solved exactly as a linear program, to bound the shortcut above |
| `recalibrate` | conformalized quantile regression applied out-of-time by rolling origin |
| `skill`, `clustered`, `stratified`, `heterosked`, `selection` | baselines and the robustness battery |

## The data

Public, no credentials, no account. The archive is at

    s3://nextstrain-data/files/workflows/forecasts-ncov/

with the influenza replication set at `files/workflows/forecasts-flu/`. Note the prefix depth:
`forecasts-ncov/` on its own returns zero objects.

One thing worth knowing before writing a loader. The server sets `Content-Encoding: gzip` and
`requests` decompresses transparently, so `r.json()` works directly. The files are not gzip
members despite the `.json` extension, and calling `gzip.decompress()` on the body raises.

## Reproducing the paper

Every figure and table is regenerated from `results/` by:

```bash
python src/figures.py
```

The pipeline that produces `results/` is `src/score_matched.py`. `RESULTS.md` records the
numbers it yields, and `results/README.md` explains the file layout, including the `_full`
suffix that distinguishes the full stride-7 run from an earlier capped one.

## A note on the estimator

Isotonic quantile regression under a partial order is not solvable by pool-adjacent-violators.
`idr_decompose` fits along linear extensions instead, which is provably conservative: more
constraints can only make the fit worse, so the reported miscalibration is a lower bound.
`idr_exact` measures how loose that bound is by solving the exact problem as a linear program on
matched subsamples. On the archive in the paper the shortcut understates miscalibration by 3.8% on
average and 2.0% at the 95% level, with zero ordering violations in 120 fits.

## License

MIT. See `LICENSE`.
