# Results — Project 05: auditing the archived Nextstrain variant forecasts

**State as of 2026-08-30.** (F16 corrects the low-frequency claim in §2 — read it.) Both deliverables in the brief now exist and both replicate on an
independent pathogen, with a naive baseline and a WIS decomposition. 103 tests pass; ruff clean; every number below is reproducible from
`src/` with no credentials and no GPU.

Detailed evidence and the reasoning behind each choice live in `FINDINGS.md` (F1–F16). The prior
literature is checked in `LITERATURE.md`. This file is the consolidated read.

---

## 1. What was claimed, and what was actually possible

The brief proposed scoring 3.7 years of daily probabilistic variant-frequency forecasts that nobody
had ever scored. That framing turned out to be **half right**: the archive exists and is rich, but
it **cannot be scored by matching variant labels**, which is the likeliest reason it has sat
unevaluated (F4).

Nextstrain clade labels are hierarchical and the label *partition* is redefined over time. When a
clade's descendants are split out, the parent label narrows to the residual. A 2024-09-26 forecast
put UK clade `24C` at 0.967; the settled series has `24C` at 0.070 for a target date two days
later. That is not forecast error — it is two quantities sharing a name. The residual `other`
category is the same trap in more obvious form, and excluding it is **not** sufficient.

Two requirements are in direct conflict:

| requirement | measured |
|---|---|
| observations must settle before scoring | ~60 days (F1) |
| the clade partition survives | median 22–34 days (F4, F5) |

Pango lineages are far worse, not better: **every** sampled Pango snapshot has a unique partition
(87/87 and 97/97), median lifetime 7 days (F5). Granularity was the wrong instinct.

**Resolution (F6):** operate at a **14–30 day** settling lag inside a matched partition. At 14 days
57% of forecast dates remain auditable while only ~3% of observations have moved by more than 0.05.
Waiting the "safe" 60 days costs three-quarters of the auditable archive and buys *worse* settling,
because revision keeps accruing. This inverted the project's starting intuition.

---

## 2. The audit (F7)

**105,349 partition-matched forecast–observation pairs** on `gisaid/nextstrain_clades/global`
(48 forecast dates, 31 locations), independently replicated on `open` (6,454 points). A
half-size stride-14 run (47,245 points, 20 dates) agrees throughout and is kept as a consistency
check (F10).

| nominal | gisaid | open |
|---|---|---|
| 50% HDI | 0.176 | 0.122 |
| 80% HDI | 0.259 | 0.187 |
| **95% HDI** | **0.354** | **0.265** |
| mean WIS | 0.0319 | 0.0465 |

**A stated 95% interval contains the truth 35% of the time.** Figgins & Bedford conceded "coverage
under 50%" — from a *retrospective refit*. The forecasts that actually shipped are worse.

Uniformity is what makes this a result and not an anecdote:
- **0 of 48** forecast dates reach nominal 95%; **0 of 48** reach even 0.80.
- **25 of 25** locations with ≥500 points are below nominal (0.157–0.904).
- Both branches agree in direction at all three interval levels.

### It is not the matching artifact

F4 established the artifact signature: coverage and width both flat across horizon. Both
diagnostics come out the opposite way, on both branches:

| diagnostic | gisaid | open | artifact |
|---|---|---|---|
| ρ(width₉₅, horizon) | +0.972 | +0.958 | ≈ 0 |
| ρ(coverage₉₅, horizon) | **−0.984** | −0.824 | ≈ 0 |
| ρ(WIS, horizon) | **+1.000** | +1.000 | ≈ 0 |

Intervals widen with horizon, coverage decays, WIS degrades — monotonically over 30 horizons. A
genuine predictive distribution is being scored. It knows longer horizons are harder; it simply
does not widen nearly enough.

### It is a width failure, and it is heteroscedastic

- **bias² / MSE = 1.3%** (gisaid), 0.4% (open). Essentially all error is variance.
- It is **heteroscedastic, but by less than a naive analysis suggests (F16)**. Binned by forecast
  level, mean |error| / half-width₉₅ spans 5.3-fold and is worst below 1% — but conditioning on
  the predictor and then measuring its own error induces regression to the mean. Binned instead
  on the pre-forecast 7-day moving average, the spread falls to **1.9-fold** and the worst
  stratum moves to 1–5%. RTM inflates the apparent gradient **2.78-fold**.
- **The claim that the intervals are least trustworthy exactly where emerging variants live is
  RETRACTED** on the primary branch. It survives on `open` (spread 4.4-fold, <1% still worst) and
  never held on influenza. What survives everywhere: heteroscedasticity of roughly 2–4x, which is
  all the level-conditional recalibration needs.

### Backfill does not explain it

Granting **every** point an allowance δ toward its interval — a concession F1 measured for only ~3%
of values:

| δ | gisaid 50 / 80 / 95 |
|---|---|
| 0.00 | 0.176 / 0.259 / 0.354 |
| 0.01 | 0.503 / 0.555 / 0.610 |
| 0.02 | 0.619 / 0.667 / 0.716 |
| 0.05 | 0.793 / 0.822 / **0.847** |
| *nominal* | *0.500 / 0.800 / 0.950* |

At δ = 0.05 the 95% level still falls short **and** the 50% level over-covers by 29 points. **No δ
repairs 95% without breaking 50%**, which rules backfill out as the explanation. Note δ = 0.01
already lands the 50% level at almost exactly nominal (0.503) while 95% is still at 0.610 — the
levels cannot be fixed together by any uniform allowance, which is the point.

---

## 3. The fix (F8)

Conformalized quantile regression on the published intervals, **rolling origin — every forecast
date calibrated on strictly earlier dates only**, enforced by test rather than convention.

| method | gisaid 50 / 80 / 95 | width₉₅ |
|---|---|---|
| as shipped | 0.174 / 0.259 / 0.354 | 0.039 |
| CQR-global | 0.558 / 0.864 / 0.976 | 0.238 |
| **CQR-Mondrian (level)** | **0.552 / 0.837 / 0.967** | **0.186** |
| CQR-Mondrian (level × horizon) | 0.550 / 0.837 / 0.961 | 0.181 |
| MULT-Mondrian (level) | 0.508 / 0.769 / 0.904 | 0.284 |

*(47 out-of-time forecast dates, 103,459 held-out points.)*

**Nominal coverage is recovered almost exactly, out of sample in time, on both branches.** The
deficit is a calibration failure, not an information failure.

The interesting part is the width. **Level-conditional recalibration reaches nominal with intervals
22% narrower than global on gisaid and 34% narrower on open** — the direct payoff of the
heteroscedasticity result: a single global correction must over-widen the well-behaved strata to
rescue the badly-behaved ones. Adding a horizon axis on top buys little (0.186 → 0.181) but at this sample size it is both
narrower *and* closer to nominal, so the finer stratification does eventually earn its keep — a
mild revision of the smaller-sample conclusion that two strata suffice.

The multiplicative variant lost despite being wider, because scaling about the median cannot help
the ~10% of points where one HDI arm has zero length. Reported because it was the intuitive choice.

Final bands average 0.186 (gisaid) — about a sixth of the frequency range. Wide, but
decision-relevant. Compare the single global inflation factor F7 showed would be needed: 18–25×,
producing bands of ~0.73 that span most of the simplex and carry no information.

---

## 3b. Skill against a naive baseline (F11) — the forecasts work, which sharpens the problem

An absolute WIS is uninterpretable, so everything is scored against **Figgins & Bedford's naive
model adopted verbatim**: a 7-day moving average of raw frequencies over the most recent seven days
available in the snapshot the forecast was published in.

**The forecasts have real point skill**: **+32.6%** MAE improvement overall (gisaid), **+30.1%** at
22–30 days, replicated at **+28.6%** on `open`. Figgins & Bedford report 6.8–26.2% for MLR at +30 d
from retrospective refits, so the shipped forecasts sit at the top of that range. **This is not a
debunking — the models work.**

But give the naive model honest out-of-time uncertainty (empirical quantiles of its own past errors
at the same horizon, from strictly earlier dates only) and it is nearly calibrated:

| nominal | model | naive + empirical intervals |
|---|---|---|
| 50% | 0.177 | 0.533 |
| 80% | 0.264 | 0.775 |
| 95% | **0.362** | **0.936** |

The naive baseline's calibration is *partly by construction*, so the honest claim is not that a
moving average has better uncertainty quantification, but: **the information needed for calibrated
intervals is present and recoverable by an elementary procedure; the model's posterior intervals do
not recover it.**

**And the recalibrated model dominates both**: coverage 0.967 at width 0.186, versus the naive
baseline's 0.936 at width 0.427. Better calibrated *and* 2.3× sharper, keeping the point-accuracy
advantage.

### The problem this exposes

| | model | naive + intervals |
|---|---|---|
| WIS | 0.0301 | 0.0401 → model **+24.8%** better |
| coverage₉₅ | 0.362 | 0.936 → model far **worse** |

**A reviewer reading only WIS would conclude the forecasts are fine.** At 22–30 days on `open`, the
model's WIS advantage over a moving average collapses to **+0.6%**.

---

## 3c. WIS decomposition (F12) — half the score is miscalibration

WIS is *exactly* twice the mean pinball loss over the seven quantile levels (verified against the
independent implementation on 2,000 random cases, exact to 1e-12), so it decomposes cleanly as
**score = MCB − DSC + UNC** via isotonic regression under pinball loss.

| | gisaid | open | flu |
|---|---|---|---|
| WIS | 0.0319 | 0.0465 | 0.0690 |
| **MCB** miscalibration | 0.0150 | 0.0240 | 0.0203 |
| DSC discrimination | 0.0652 | 0.0736 | 0.0265 |
| UNC uncertainty | 0.0820 | 0.0960 | 0.0752 |
| **MCB share of score** | **47.1%** | **51.7%** | **29.4%** |

**Ideal recalibration would improve WIS by 47% / 52% / 29% with no change to the model.** DSC is
large and positive everywhere, so the forecasts genuinely discriminate.

**Miscalibration is concentrated in the tails.** MCB share by quantile (gisaid): 82% at τ=0.025,
20% at the median, 81% at τ=0.975 — a clean U-shape that makes F7's coverage deficit quantitative
and separates it from loss of information.

**The horizon result.** MCB's share rises monotonically with horizon in all three datasets
(gisaid 43.2%→54.1%). Attributing the WIS degradation from 1–7 d to 22–30 d: **75% of it is
miscalibration on gisaid**, where discrimination is essentially horizon-invariant (DSC −1.0% across
30 days). On `open` and flu, DSC does decline (−13.1%, −23.5%), so horizon-invariant discrimination
is a gisaid result. What replicates everywhere: **miscalibration is the largest single contributor
to degradation with horizon, and it grows while discrimination does not.**

Caveat: CORP recalibration is fitted in-sample, as the decomposition specifies. MCB is the
*attainable* improvement, an upper bound — not a prediction of out-of-sample gain. F8's conformal
recalibration is the out-of-time counterpart.

---

## 4. Independent replication on influenza (F9)

`forecasts-flu` — same MLR machinery, different pathogen, different lineage system. Not in the
brief; found during verification. **4,969 partition-matched points, three subtypes.**

| slice | 50% | 80% | 95% |
|---|---|---|---|
| h3n2 | 0.510 | 0.597 | 0.677 |
| h1n1pdm | 0.360 | 0.460 | 0.552 |
| vic | 0.159 | 0.263 | 0.376 |
| *nominal* | *0.500* | *0.800* | *0.950* |

Under-coverage in every subtype at every level. **The overconfidence is a property of the modeling
approach, not of SARS-CoV-2.** The fix transfers too (0.490 / 0.724 / 0.911) and the Mondrian
sharpness advantage replicates (11% narrower than global).

**Two mechanistic claims from F7 do not carry over**, and are written up as failures:

1. **Coverage does not decay with horizon on flu** — ρ = +0.23 (h3n2), +0.88 (h1n1pdm), +0.05
   (vic). Width still grows (ρ ≥ +0.75), so the artifact is still ruled out, but the pooled
   ρ = −0.824 is **Simpson's paradox** across subtypes and must not be quoted.
2. **Overconfidence is not worst at low frequency on flu** — it is U-shaped, and the <1% stratum is
   flu's *best*-covered (0.727) versus covid's worst.

Flu is also **less badly calibrated overall** (0.528 vs 0.392 at 95%), with a tail-underestimation
profile: h3n2's achieved/nominal is 1.02 / 0.75 / 0.71 across the three levels — the bulk is right,
the tails are thin. Covid fails at the 50% level too.

---

## 5. Figures

| file | content |
|---|---|
| `figures/fig1_settling_tradeoff.png` | the settling-vs-stability trade-off that sets the 14-day operating point |
| `figures/fig3_decomposition_by_horizon.png` | MCB rises with horizon, DSC does not, all three datasets |
| `figures/fig_rtm_correction.png` | the regression-to-the-mean correction: three conditionings side by side |
| `figures/money_coverage_vs_horizon.png` | **the money plot** — empirical coverage vs horizon at 50/80/95, both branches, Wilson bands, bold nominal references |
| `figures/overconfidence_by_level.png` | where the overconfidence concentrates, and that no stratum reaches nominal |
| `figures/recalibration.png` | coverage recovered and the width it cost, all methods, all three datasets |

---

## 6. Bugs found in my own analysis, and how they were caught

Listed because the project's stated value is leakage-controlled, honest evaluation, and because
each of these produced a *plausible number* rather than an error.

| bug | wrong answer it gave | how it was caught |
|---|---|---|
| Wrong S3 prefix in the brief | zero objects | verification before building |
| "Files are gzipped" (brief) | `gzip.decompress()` raises | verification |
| `latest_results.json` parsed as a date | killed every `usa` slice silently | slice count audit |
| Null `value` fields | killed the whole gisaid branch | slice count audit |
| My own "the brief's counts are wrong" | retracted — I had counted a different scope | re-checking scope, not just magnitude |
| Naive label matching | 95% coverage "0.135" | flat coverage/width vs horizon |
| **HDI reconstructed as median ± width/2** | 95% coverage 0.327 instead of 0.392 | self-consistency: inside-band fraction must equal `cov_95` |
| Unbounded snapshot cache | free RAM 24.4 → 6.2 GB | watching RAM during the run |
| `unassigned` not treated as a catch-all | would have contaminated flu | reading the flu variant list |
| Pooled ρ(coverage, horizon) on flu | "−0.824, replicates!" | per-subtype breakdown (Simpson's paradox) |

**85.2% of these HDIs are asymmetric about the median**, and at the 90th percentile one arm has
zero length. The scorer now persists raw `lo_*`/`hi_*` bounds and every downstream analysis uses
them.

---

## 7. Scope and limits, stated plainly

- **ncov:** `nextstrain_clades`, `global`, 48 forecast dates at stride 7 from a 90-snapshot cap
  (gisaid) and 21 at stride 14 (open), 14-day settling lag, horizons 1–30 d. Covers ~57% of
  forecast dates (F6).
- **Excluded, not omitted:** `usa` regions (discontinued Feb 2023) and both Pango slices (partition
  lifetime 7 d — unauditable by this method).
- **flu:** `emerging_haplotype`, `region`, three subtypes, 8-month archive. Corroboration, not an
  equal-weight second study. `aa_haplotype` and `country` remain unscored.
- Truth is `weekly_raw_freq` (ncov) / `smoothed_raw_freq` (flu), chosen on evidence: weekly beat
  daily in **16 of 16** slice × lag comparisons (F2).

---

## 8. Open routes

1. **Partition-invariant scoring via growth advantage (`ga`)** — still untested, still the most
   attractive: it would sidestep the settling-vs-stability trade-off rather than optimise within
   it. Requires defining a realised-`ga` truth series, which is real work.
2. **Fixed-taxonomy reconstruction** — map every label to a stable reference and re-aggregate. The
   only route to auditing the *whole* archive, and the most defensible. Substantial.
3. ~~Scale the current audit~~ — **done (F10)**: stride 7 / 90-snapshot cap gave 105,349 points
   over 48 dates and every finding strengthened. Further scaling would need the `open` branch and
   the flu `aa_haplotype` slices, not more of this one.
4. **`aa_haplotype` and `country` flu slices**, plus the `cases/` prefix, all unexamined.

---

## 9. Artifact status

`src/` is already close to the separable-module structure the brief calls for:

| module | role |
|---|---|
| `verify.py` | archive listing and structural verification |
| `backfill.py` | revision profiling — the backfill-aware truth builder |
| `score.py` | snapshot loader (byte-bounded cache), WIS, catch-all detection, partition index |
| `score_matched.py` | partition-matched scoring — the piece that makes the audit well-defined |
| `recalibrate.py` | out-of-time conformal recalibration — the most reusable component |
| `skill.py` | naive baselines and skill scores (F&B's naive model, plus an out-of-time interval version) |
| `decompose.py` | WIS decomposition into MCB / DSC / UNC |
| `flu.py` | second-pathogen replication |
| `figures.py` | the money plot and companions |
| `resources.py` | BLAS thread capping and RAM guards |

The recalibrator is the piece most likely to be reused across other hub forecasts, and it depends
on nothing pathogen-specific.
