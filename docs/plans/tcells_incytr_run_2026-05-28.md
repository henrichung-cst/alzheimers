# T-cell pair-mode Incytr — execution plan (OOM-aware, resumable)

Date: 2026-05-28
Owner: hchung
Driver: `alz/incytr_pair/incytr_commandline.R` (env-parameterized)
Runner: `alz/incytr_pair/run_pair_mode_tcells.sh`
Pixi task: `pixi run tcells-incytr`

## Calibration (probe v4, single pair, donor2)

| metric | value |
|---|---|
| per-pair scoring (Cal_pairwise_grid + perm@nboot=100) | 14.2 s |
| per-pair RSS HWM | 5.06 GB |
| paths/pair | 244,278 |
| parquet/pair | 16 MB |
| total wall (incl. substrate build) | 44.6 s |

Box: 30 GB RAM, 16 cores, 8 GB swap. DuckDB spill at `~/.cache/duckdb`.

## Run shape

| cohort | states | pairs/contrast | contrasts | channels |
|---|---|---|---|---|
| donor1 | 14 | 196 | d13,d17,d20 vs d2 (3) | pr,py,ps |
| donor2 | 11 | 121 | d5,d7,d9,d11 vs d2 (4) | pr,py |

## Concurrency budget

Memory ceiling: keep peak resident under **18 GB** (leave 12 GB headroom for OS, R startup, parquet writes, the shared box).

- donor2 (2-channel, ~5 GB/pair HWM): **NPAIR_WORKERS=3** → ~15 GB peak. Safe.
- donor1 (3-channel, ps adds ~62k sites; estimate ~6.5 GB/pair HWM): **NPAIR_WORKERS=2** → ~13 GB peak. If first contrast HWM <5.5 GB, escalate to W=3 for contrasts 2–3.

`NPERM_WORKERS=1` throughout (permutation parallelism multiplies per-pair RSS — don't stack it with pair parallelism).

## Schedule (single sequential script, ~2 hr wall)

1. **donor2 first** (smaller, simpler, validates concurrency). 4 contrasts × ~10 min ≈ 40 min at W=3.
2. **donor1** after donor2 completes cleanly. 3 contrasts × ~17 min ≈ 50 min at W=2 (60 min budget incl. ps overhead).
3. **Significance filter** per donor after its contrasts finish (driver emits unfiltered; cutoff is `SigProb>0.1 OR ... AND |PDS|>=0.2`, applied in-place via `filter_significant_paths.py`).

## OOM defenses

1. **Pre-flight smoke** before each donor: `bash alz/incytr_pair/run_pair_mode_tcells.sh --smoke <donor>` (nboot=2, first later-day, single pair). Confirms substrate, gene list, DB selection, and assay handle load without spending hours.
2. **Resume on parquet exists** — already wired in `run_one()`. A crashed contrast resumes by re-running the same script; finished contrasts are skipped.
3. **Hard cap via systemd-run memory cgroup**:
   ```
   systemd-run --scope --user -p MemoryMax=22G -p MemoryHigh=18G \
     bash alz/incytr_pair/run_pair_mode_tcells.sh
   ```
   Triggers OOM-kill of the runner *only* (not the whole shell) if peak crosses 22 G. Survivors of any partial contrast are preserved as parquets; rerun picks up where it stopped.
4. **Conservative start** — first donor1 contrast at W=1, inspect log `rss=...MB hwm=...MB` line, then bump W for the next two contrasts. Don't precommit to W=2 if HWM is unknown for ps-bearing pairs.
5. **No background-process monitoring** (per global rule). Run as foreground `nohup ... &` with `tee` log, check on it explicitly at known checkpoints (after each donor).

## Progress tracking

The driver already logs one line per finished pair:
```
[pair-driver] pair K/N <S> -> <R> rows=... <wall>s rss=<cur>MB hwm=<peak>MB
```
Wrap each run with a heartbeat that surfaces this progress without tailing:

- **Per-contrast log**: `outputs/reports/incytr_pair_mode_tcells/pair_run.log` (already wired).
- **Per-contrast status file**: amend `run_one()` to write `OUT/.status_<c1>_<c2>.txt` with `pair K/N` updated each line (tail-friendly, single file per contrast).
- **Quick check** (during run): `grep -c "pair-driver] pair" pair_run.log` → instantaneous % done.
- **Post-contrast**: `ls -lh outputs/reports/incytr_pair_mode_tcells/<donor>/wide/` shows parquet sizes; sanity that they're 10–20 MB pre-filter.

## Verification gates (post-run)

1. **All contrasts present**: 3 donor1 + 4 donor2 = 7 `*_incytr_output.parquet` under `*/wide/`.
2. **Row counts pre-filter**: ≈ 121 × 244k ≈ 30 M / contrast (donor2); ≈ 196 × similar for donor1 (3-channel will be larger).
3. **Filter survival**: filter log prints `before -> after`. Expected drop 80–95% (mouse cohort drops to ~5–15% kept). If a donor1 contrast keeps <0.1% or >50%, inspect — that's anomalous.
4. **No NA `PDS` in survivors**: spot-check via DuckDB.
5. **|PDS| ranking sanity** (per the project invariant — pvalue is untrustworthy): top-10 |PDS| paths per contrast should not all share the same sender/receiver — that would suggest a substrate degeneracy.

## Abort / rollback

- If donor2 W=3 OOM-kills: fall back to W=1, re-launch (resumable). Do not silently skip pairs.
- If donor1 W=2 OOM-kills: drop to W=1; if still failing on a specific pair, log the pair and skip via a state-pair allowlist (do **not** lower nboot — that changes the science).
- Parquet partial-write: handled by driver's atomic write-then-rename (existing behavior); a crashed contrast leaves no half-written parquet for `run_one()` to skip.

## Out of scope

- sce4 parity — does **not** apply to T-cell data (independent cohort, human DB, distinct vocabulary).
- Donor1 ps channel for donor2 — donor2 has no IMAC; channels = `pr,py`.

> **Update 2026-06-04 — SiK kinase scoring enabled (this was wrongly scoped out above).**
> The original "out of scope" reasoning conflated SiK with kinase MEA. SiK is
> **scRNA-only**: `SiK_score` = mean Exclusiveness Index (`Cal_EI`) of a path's
> kinase nodes in the receiver cluster, computed from the Seurat expression
> matrix + idents (both donors have these in `incytr_obj.rds`). It needs **no
> bulk deconvolution and no kinase MEA**, so it never touched the bulk-only rule.
> The only missing piece was a **human** kinase-substrate library (the Song
> `kldata` is mouse-cased). `alz/integration/build_tcells_kldata.py` derives one
> from donor1's `ps/py_bulk_linear.csv` via the species-agnostic PSPA ranking
> (no homologene), written to `data/datasets/tcells/kinase/kldata_human.csv` and
> symlinked into each donor's `INPUTS_DIR` as `kldata.csv`. The runner now sets
> `USE_KLDATA="TRUE"`. donor2 reuses donor1's kldata (no IMAC motifs of its own)
> but computes its EI on its own scRNA. Verified: PDS folds SiK exactly
> (`multimodel_score ± 0.5·SiK`); viewer SiK column populated, 0 nulls.

## Files touched / created

- (new) `docs/plans/tcells_incytr_run_2026-05-28.md` — this plan.
- (edit) `alz/incytr_pair/run_pair_mode_tcells.sh` — add `.status_<c1>_<c2>.txt` heartbeat, accept `NPAIR_WORKERS` from env per-donor (already supported), keep resume + filter as-is.
- (no edit) driver, filter, build scripts — all parameterized and validated by probe v4.
