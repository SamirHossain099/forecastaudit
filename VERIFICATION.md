# Verification — Project 05 (Nextstrain forecast audit)

**Run:** 2026-08-29. Method: S3 REST listing + direct GET, no credentials, no bulk download.
**Verdict: the project is viable and the archive is richer than the brief describes. TWO of the
brief's data facts are wrong — the S3 prefix and the gzip claim. A third "correction" of mine was
itself wrong and is retracted in §3 below.**

---

## ❌ Corrections to the brief

### 1. The S3 prefix does not exist
The brief gives `forecasts-ncov/`. That returns **zero objects**.

**Real prefix: `files/workflows/forecasts-ncov/`** — three levels deeper, under
`https://nextstrain-data.s3.amazonaws.com/`. Sub-prefixes: `open/`, `gisaid/`, `cases/`, `trial/`.

*(This is the second project in a row whose recorded S3 path was wrong. Verify paths first.)*

### 2. "Files are gzipped despite a `.json` extension" — WRONG, and following it would break
The gzip magic bytes are **absent** from the response body. The server sets
`Content-Encoding: gzip` and `requests` decompresses transparently. `r.json()` works directly.

Manually calling `gzip.decompress()` — which the brief instructs — **raises**. The "gotcha" is a
non-issue, and acting on it is the actual hazard.

### 3. ~~Snapshot counts and date range are off~~ — RETRACTED, the brief was right

An earlier version of this file claimed the brief's counts were wrong. **They are not.** The brief
gives 732 open-clade / 607 open-Pango / 680 GISAID-Pango from 2022-12-23. Per slice:

| slice | snapshots | range |
|---|---|---|
| open / clades / global | **731** | 2022-12-23 → 2026-08-23 |
| open / pango / global | **606** | 2023-04-25 → 2026-08-23 |
| gisaid / pango / global | **679** | 2023-04-25 → 2026-08-24 |
| gisaid / clades / global | 770 | 2022-12-23 → 2026-08-24 |

Accurate to ±1. My "correction" counted dates across *all* objects including `usa` and `trial`,
which answers a different question. Checking a claim means matching its **scope**, not just its
magnitude.

### 3b. A real structural fact the brief does omit

| slice | snapshots | range |
|---|---|---|
| open / clades / **usa** | **46** | 2022-12-23 → **2023-02-18** |
| gisaid / clades / **usa** | **39** | 2022-12-23 → **2023-02-11** |
| open + gisaid / pango / usa | **none** | — |

**USA regional forecasts were discontinued in February 2023**, and Pango-lineage forecasts only
begin **2023-04-25**. Coverage is not uniform across the archive, and any design assuming it is
will silently analyze a two-month window while believing it has 3.7 years.

Archive totals (all prefixes, for sizing only): **3,284 objects, 4.57 GB.**

---

## ✅ Confirmed — and the key claim holds

**The GISAID-derived branch is readable with no credential.** `HEAD` on a `gisaid/` object
returns **HTTP 200** with no auth sent. This is the workaround that makes the project possible
without touching GISAID's redistribution terms, and it is real.

**The archive is genuinely a daily, dated, prospective record**: 731-770 dated forecasts per
global slice over ~3.7 years (2022-12-23 to 2026-08-24).

---

## Data structure — better than expected

One file = one (branch, lineage-scheme, region, model, date). Top level: `metadata`, `data`.

**`metadata.ps` — the interval levels, exactly as needed:**
`median`, `HDI_50_lower/upper`, `HDI_80_lower/upper`, `HDI_95_lower/upper`.
The 50/80/95 HDIs the brief promised are present.

**`metadata.sites` — five series per file:**

| site | meaning | role in the audit |
|---|---|---|
| `freq_forecast` | **the forecast** (out-of-sample) | what gets scored |
| `freq` | fitted/smoothed frequency (in-sample) | not the target |
| `ga` | growth advantage | secondary endpoint |
| **`daily_raw_freq`** | **raw observed daily frequency** | **the ground truth** |
| `weekly_raw_freq` | raw observed weekly | smoothed truth alternative |

Also present: `forecast_dates` (the explicit horizon), `variants`, `location`, `updated`.

`data` is long-format: `{location, site, variant, date, value, ps}`. A 2024-06-01 global clade
file holds 3,088 records; a 2026-05-20 Pango file holds 208,475.

## 🔑 The finding that changes the method

**Observed frequencies (`daily_raw_freq`) are bundled inside every snapshot.**

That means the backfill problem the brief flagged as "the main technical trap" is not just
avoidable — it is **directly measurable**. The same calendar date's observed frequency can be
read from snapshot *T₁* and from a later snapshot *T₂*, and the difference *is* the backfill.

Two consequences:
1. **Ground truth can be defined rigorously** — take the observed frequency as reported at a
   fixed lag after the target date, and state the lag.
2. **Backfill magnitude becomes its own result.** Quantifying how much observed variant
   frequencies move after the fact is a contribution in itself, and it is required before any
   coverage number is trustworthy. No extra data is needed.

---

## Practical notes

- Total 4.57 GB; individual files 0.3–25 MB. Fully tractable; no bulk download needed to start.
- `cases/` and `trial/` prefixes exist and were not in the brief — `trial/` is sparse (10 dates)
  and should be excluded.
- Bonus: the bucket also holds **`files/workflows/forecasts-flu/`**, an equivalent influenza
  forecast archive the brief never mentions. A second, independent replication target — which,
  given how many single-source claims failed in projects 01 and 04, is worth knowing about now.

## Next

1. Build the snapshot loader (no manual gunzip — `r.json()` works).
2. **Define and document the ground-truth/backfill convention before scoring anything.** This is
   the single decision the whole paper rests on.
3. Score one location × one horizon end to end; sanity-check against Figgins & Bedford's admitted
   "coverage under 50%" before scaling.
