# results/

Machine-written outputs. Nothing here is edited by hand; every file is reproducible from `src/`.

## The `_full` suffix is the one thing to know

Two generations of runs live side by side:

| suffix | what it is | use it? |
|---|---|---|
| `*_full.*` | the **stride-7 run over every forecast date** (55 gisaid dates, 53 open) | **yes: every number in the paper comes from these** |
| no suffix | an earlier run under a sampling cap (21 open clusters) | only as a fallback |
| `*_dense.*` | the dense-lag variant of the matched scorer | figures only |

The distinction is not cosmetic. The sampling cap suppressed a real result: `open` MAE skill was
[-0.015, 0.516] under the cap and [0.126, 0.487] at full stride-7: insignificant to significant.
If a number disagrees with the draft, check you are not reading a non-`_full` file.

**Do not delete the non-`_full` JSONs.** They are load-bearing test fixtures.
`tests/test_robustness.py:179` tries `{name}_full.json` and falls back to `{name}.json`, but
several other tests read the non-suffixed name directly, and the `flu` branch has no `_full`
counterpart at all because the flu run was never capped.

## Branches

`gisaid` and `open` are the two Nextstrain clade branches; `flu` is the FluSight cross-check.

## What is here

- `scores_full_{branch}.csv`: the scored forecast/outcome pairs; input to the figures.
- `scores_matched_lag14_{branch}[_dense].csv`: partition-matched scorer output, raw `lo_*`/`hi_*`
  interval bounds preserved (85.2% of HDIs are asymmetric about the median, so reconstructing them
  as median +/- width/2 gets coverage wrong).
- `decomposition_*`, `idr_decomposition_*`, `idr_exact_gisaid.*`: WIS/interval-score decompositions
  and the exact partial-order LP validation.
- `clustered_*`, `stratified_*`, `heterosked_*`, `selection_*`, `comparability_*`: the robustness
  battery.
- `recal_*`, `recalibration_*`: out-of-time recalibration.
- `skill_*`, `lag_tradeoff*`, `backfill_profile*`, `archive_inventory.json`: baselines, the
  settling-vs-stability trade-off, and archive provenance.
- `*_summary.json` / `*_audit.csv`: per-run headline stats and row-level audit trails.

## superseded/

Outputs from 2026-08-29 exploratory runs that nothing reads: four orphaned score tables and their
summaries, plus `skill_gisaid.csv` (byte-identical to `skill_gisaid_full.csv`). Kept rather than
deleted so the exploration is still on the record. Safe to remove.
