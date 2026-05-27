# Incytr pair-mode significance filter — wire-in + retroactive apply

**Date:** 2026-05-26
**Status:** PROPOSED — awaiting approval before implementation

## Motivation

The current pair-mode output emits **all** paths (`cutoff_SigProb=0`,
`cutoff_PDS=0` — parity override #5) → **54.7M rows/contrast** in the
`wide_nboot0` set. A collaborator clarified the analysis filters actually used:

> "cutoff_0.1" = `cutoff_SigProb = 0.1` of `Cal_SigProb`, keeps pathways with
> `SigWeight >= 0.10` in at least one condition.
> `cutoff_PDS = 0.2` of `Cal_PDS`, keeps pathways with `abs(final_score) >= 0.2`.

Column mapping in our schema: `SigWeight` → the per-condition `SigProb_<cond>`
columns; `final_score` → `PDS`. Incytr's actual operators (verified in
`~/Projects/work/incytr/R/analysis.R:222-228` and `R/evaluation.R:176-177`):
`SigProb > cutoff` (strict, OR across arms) and `abs(PDS) >= cutoff`.

**Production filter:**
```
(SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1) AND abs(PDS) >= 0.2
```
Trims 54.7M → ~20.9M rows/contrast (38%). The `SigProb>0.1` term does almost
all the cutting (`|PDS|>=0.2` alone keeps 81%).

## Two key facts that make this safe

### 1. The cutoffs are pure row subsets — no re-run needed
`Cal_SigProb`/`Cal_PDS` cutoffs only **drop rows**; they do not recompute
`SigProb` or `PDS` for survivors (both are per-path scores; corrections are
per-path additive constants — no cross-path normalization). Therefore filtering
the existing `wide_nboot0` parquets is **mathematically identical** to having
re-run the driver with these cutoffs, for the surviving rows. No ~10h re-run.

### 2. The filter preserves sce4 parity exactly
The bench verifier (`verify_sce4_parity.py:121-124`) **already** applies
`SigProb>0.1` (either arm), so the `573/600` and `599/600` numbers already
include that half. The only new constraint is `abs(PDS)>=0.2`. Tested:

| Pair | baseline recall | +`abs(PDS)>=0.2` | recall lost | non-App lost |
|---|---|---|---|---|
| Microglia→Cholinergic | 573/600 | **573/600** | 0 | 0 |
| Ndnf×Ndnf | 599/600 | **599/600** | 0 | 0 |

And **all 600/600 sce4 reference paths already satisfy both cutoffs** in each
pair. The Ndnf reference's **min `|PDS|` = 0.2122** — sitting on the 0.2
threshold — is direct evidence that `cutoff_PDS=0.2` is the exact filter sce4
used to build the Top300. The filter is not merely safe; it reproduces sce4's
own gating.

## The existing 0.30 cut (to be removed)

`build_unified_viewer.py:2113` pre-filters `|PDS| >= 0.30` in the pathway-shard
builder — a viewer-only "coarse no-signal" storage optimization. It never
touched the `wide_nboot0` parquets or the parity gate, so parity numbers never
involved it. The collaborator filter supersedes it (and is more principled:
`|PDS|>=0.2` is *more* permissive, but the `SigProb>0.1` gate is added). Per
anti-shim, the arbitrary 0.30 cut is removed, not kept alongside.

## Architecture

The six parity overrides stay **untouched** — `Cal_SigProb`/`Cal_PDS` keep
running at `cutoff=0` inside the driver. The filter is applied to the driver's
**output**, downstream — exactly the "emit all paths and filter downstream"
design override #5 already describes. We are implementing the downstream half
that was never concretely wired. One reusable filter, called by the runners and
once retroactively.

## Steps

1. **New script `alz/incytr_pair/filter_significant_paths.py`**
   - DuckDB-streamed COPY, atomic (`.tmp` + rename), `memory_limit` capped,
     spills to `~/.cache/duckdb`.
   - Auto-detects the two `SigProb_*` columns + `PDS`; applies the filter.
   - Operates on a single parquet or `--dir` of `*_incytr_output.parquet`.
   - Idempotent (re-running is a no-op — survivors still pass).
   - Reports before/after row counts per file.

2. **Wire into `alz/incytr_pair/run_pair_mode.sh`** — after the 9-contrast loop,
   filter each parquet in `OUTPUT_DIR`. Covers the full pipeline (its E2 step
   calls this runner).

3. **Wire into `bench/run_nboot0_w3.sh`** — after the loop, **before** the
   parity gate. Covers future nboot=0 runs.

4. **Retroactive apply (no re-run):** run the filter on
   `outputs/reports/incytr_pair_mode/wide_nboot0/*.parquet`, **in place**.
   - ⚠️ **Destructive:** overwrites the full 54.7M-row set (atomic, so no
     corruption risk). The full set is not retained — it was the flagged
     overkill, and is reproducible by re-run if ever needed.

5. **Re-verify parity:** `verify_sce4_parity --all-known-pairs --wide-dir
   wide_nboot0` → must still show 573/600, 599/600.

6. **Remove the `|PDS|>=0.30` cut** in `build_unified_viewer.py:2113`; fix the
   stale `|PDS|>=0.01` comment at 2096. Keep the `has_pvalue` cube fix and the
   reshape pvalue NULL-fill (from the nboot=0 viewer work).

7. **Build the viewer** from filtered `wide_nboot0`
   (`PAIR_MODE_INPUT_DIR=...wide_nboot0`, `PAIR_MODE_STRICT=1`). Confirm cubes
   non-zero, shards populate, `PAYLOAD.meta.generated_at` fresh. `wide/`
   (nboot=100) stays untouched for later comparison.

8. **Docs same pass:** `bench/bench.md` + CLAUDE.md document the downstream
   production filter (`SigProb>0.1 & |PDS|>=0.2`), that it is parity-preserving,
   and the sce4 ref `min|PDS|=0.21` evidence. Override #5 (`cutoff=0` inside the
   driver) is reaffirmed, not changed.

## Out of scope / unchanged
- The six sce4-parity call-site overrides (driver internals).
- `wide/` (nboot=100, May 15–16) — retained on disk for comparison.
- nboot decision (pvalue) — still pending with superiors; orthogonal.
