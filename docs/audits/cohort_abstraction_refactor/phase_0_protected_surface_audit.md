# Phase 0 — Protected Surface Audit

Date: 2026-06-17
Status: APPLIED 2026-06-17 — Decision 1 (de-protect 68 + add 18 → 165) committed to
`_build_protected_files`; Decision 2 (stop producing dead sidecars) deferred to
Phase 2/3. See `phase_0_decisions.md` → "protected-surface tightening".

## Why this audit

At the Phase-0 gate the reviewer asked, before ratifying the 215-file protected
list, whether those files are genuinely consumed downstream / real data outcomes
worth retaining, and whether any are duplicates, reshapes, or otherwise
unnecessary surface we could lightly clean out.

Five read-only auditors swept the repo (one per cohort + Incytr/viewer),
classifying every protected file by downstream consumer evidence. No file was
modified. Findings below are backed by `consumer file:line` citations in the
per-cohort auditor outputs; the cross-cutting conclusions are summarized here.

## The one structural pattern

Across all four cohorts the same shape recurred:

- **Primary-track audit sidecars** (`mea_global_shift`, `winsorized_sites`,
  `mea_substrate_sets`, `recurrence`) in their `""` / `_pY` track variants are
  **actively read** by the viewer audit drawer → KEEP.
- Their **secondary-track variants are written but never read**:
  - Mukesh/T-cell `_raw`-infix sidecars (the viewer's `_KINASE_AUDIT_FILES` /
    `_human_track_load` only wire the `""`/`_pY` suffixes; the raw-phospho track
    is shimmed out of the audit drawer),
  - Song `*_per_cluster` decomposition sidecars (no reader past their writer),
  - 5xFAD `celltype_mea/*_global_shift` / `*_winsorized_sites` (source-files
    list only, never parsed).
- **Wide NES/FDR matrices** are pivots of the MEA long table but are consumed
  *as primary inputs* by the viewer and cross-reference scripts → KEEP (not
  redundant).
- **Deterministic transforms / derived reshapes** (gzip of a JSON, a
  repartition-by-receiver cache, a payload already inlined in HTML) carry no
  independent parity signal → protect the source, de-protect the transform.

Three seeded "redundancy" hypotheses were **refuted with evidence** and stay
fully protected: `unified_attribution.csv` vs `_full` (summary carries the
`mechanism_annotation` column; `recover.py` requires `_full`), `stoichiometry_
matrix.csv` vs `mea_stoichiometry.csv` (wide input matrix vs long MEA output),
and `.build_cache/` (stores input signatures + small summary blocks, **not**
copies of canonical data).

## Recommendation — two independent decisions

### Decision 1 — Tighten the protected list now (Phase-0 scope, NO files touched)

This only changes which files carry a parity contract. Nothing is deleted; every
de-protected file stays on disk. Within Phase 0's read-only mandate.

**DE-PROTECT — 68 entries** (derived/transform, write-only orphan, or
secondary-track sidecar; keep on disk, drop from parity baseline):

| cohort | count | files |
| --- | --- | --- |
| Mukesh | 12 | `raw_phospho_normalized_all.csv`, `stoichiometry_matrix_all.csv` (write-only `_concat` outputs); `perdonor/{mea_global_shift,winsorized_sites,mea_substrate_sets,recurrence,recurrence_ctrl}_raw{,_pY}.csv` (10 `_raw` sidecars, never loaded) |
| T-cell | 12 | `donor1/mea/{mea_timecourse,mea_global_shift,winsorized_sites,mea_substrate_sets}_raw{,_pY}.csv` (8 raw-track sidecars); `donor1/mea/recurrence_{pY,raw,raw_pY}.csv` (audit loop wires only base `recurrence.csv`); `donor2/total_proteome_normalized.csv` (measurement-trace hardcodes donor1) |
| Song | 9 | `total_proteome_normalized_pY.csv` (duplicate of track-invariant ST total proteome); `decomposition/levy_t5/{mea_substrate_sets,mea_global_shift,winsorized_sites}_per_cluster{,_pY}.csv` (6 per-cluster sidecars, no reader); `coverage_report.csv`, `proportions_provenance.csv` (QC logs, no reader) |
| 5xFAD | 2 | `celltype_mea/fivexfad_celltype_mea_global_shift.csv`, `celltype_mea/fivexfad_celltype_winsorized_sites.csv` (source-files list only) |
| Incytr/viewer | 33 | `incytr_pair_mode/receiver_cache/receiver=*/data_0.parquet` (30 — lossless repartition of `wide/` via `pair_to_receiver_cache.py`); `unified_viewer.payload.json.gz` (deterministic gzip of `.json`); `tcell_viewer.payload.json` + `.payload.json.gz` (content inlined in `index.html`) |

**ADD — 18 entries** (5xFAD files actively read by the viewer but currently
**unprotected** — a real gap: undetected drift in these would corrupt the viewer
silently):

- `{cortex,hippocampus}_{st,py}_contrast_qc.csv` (4) — drives `contrast_status`/`n_wt`/`n_tg` on every MEA row (`build_unified_viewer.py:2836`)
- `{cortex,hippocampus}_{st,py}_raw_phospho_normalized.csv` (4) — required in detail-shard build (`:2574`)
- `{cortex,hippocampus}_{st,py}_stoichiometry_matrix.csv` (4) — required in detail-shard build (`:2576`) + MEA input (`fivexfad.py:534`)
- `{cortex,hippocampus}_{st,py}_matched_total_protein.csv` (4) — required in detail-shard build (`:2575`)
- `sample_manifest.csv` (1) — sample-counts table (`:2904`)
- `celltype_mea/fivexfad_snrna_pseudobulk_counts.csv` (1) — preferred cell-count source, read before fallback (`:1633`)

**Net protected count: 215 → 165** (−68 +18). The Song baseline `total_proteome_
normalized.csv` (ST) remains protected; only its pY duplicate is dropped.

### Decision 2 — Output-surface cleanup, DEFERRED to the producer refactor

The actual "stop producing dead output" is a **producer code change**, not a
baseline edit. It belongs to Phase 2/3 when we already have those writers open:

- **Stop writing** the never-read `_raw`/`_all`/`_per_cluster` audit sidecars at
  their sources — `mukesh_perdonor.py`, `tcells_perdonor.py`,
  `decomposition_mea/enrich_celltype.py`, `ingest/mukesh.py` (`_concat`). This is
  the anti-shim cleanup: a write with no reader is dead surface.
- **Do not delete the on-disk copies now** — most would simply regenerate on the
  next pipeline run. Remove them at the same commit that stops producing them, so
  they don't come back.
- **5xFAD exception:** its de-protected sidecars are **unrecoverable** (the cohort
  is on hold behind the `.sne` blocker). De-protect them but never delete, and do
  not touch the on-hold 5xFAD producer.

Tracking these as explicit cleanup-candidates here so they are not lost; they are
actioned in their owning phase, not now.

## What this does NOT change

- No producer code edited this phase.
- No canonical output deleted or regenerated.
- The baseline inventory generator and its determinism guarantee are unaffected;
  re-running it after the list edit reproduces the new (165-file) set
  deterministically.

## Audit provenance

Per-cohort auditor outputs (consumer citations) are the evidence of record for
each KEEP/DE-PROTECT call. The de-protect membership and add-file existence were
independently re-verified by the orchestrator against `protected_files.json` and
the on-disk tree (all 68 de-protect entries confirmed present + in-list; all 18
add candidates confirmed on-disk + not-yet-protected).
