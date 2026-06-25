# Theme A — T-Cell Analysis & Viewer Implementation Plan

Audit: `docs/plans/theme_a/audit.md` · Full audit: `docs/audits/tcell_theme_a_audit.md`
Design decisions below were resolved in a grilling session (recorded in "Resolved decisions").

## Scope

A1 (specificity/enrichment metrics + terminology), A2 (attribution NES deconvoluted), A5 (monotonic export run). A3 and A4 require no work — verified clean by audit.

---

## Resolved decisions (grilling)

1. **Full rename, not display-label-only.** The within-cohort metric is *enrichment* (concentration in a ProjecTILs T-state), end to end: `tcell_specificity.csv → tcell_enrichment.csv`, columns `tcell_specificity → tcell_enrichment`, `tcell_tier → tcell_enrichment_tier`, producer docstring, and JS labels. "Specificity" is reserved for the NSCLC reference breadth (below). Rationale: a data layer that says `specificity` under a UI that says "Enrichment" is the exact dual-representation trap the anti-shim / real-vocabulary rules forbid.

2. **T-cell-local vocabulary doc note (required).** Add a line to `docs/tcell_exhaustion_analysis_summary.md` stating that *specificity* and *enrichment* are defined **uniquely for the T-cell cohort** and diverge from the AD cohorts. Specifically: the within-cohort enrichment uses the same computational method the AD cohorts label "specificity" (WMB-tier share), and T-cell "specificity" (N-of-7 prevalence count) is a different *kind* of metric than human-AD specificity (a `log2(celltype_mean/brain_mean)` ratio with no detection gate). Without this note an agent will "harmonize" the names and reintroduce the collision.

3. **Specificity detection rule = `fraction ≥ 0.10`, pure prevalence.** No `mean_log2 > 1` floor. Literal to A1 ("at equal to or above 10% of those cells"). This is a fourth distinct detection rule in the repo (WMB: `mean>1 AND frac>0.10`; human AD: none/log-ratio; Song: animal-presence) — consistent with the established pattern that each cohort defines detection to fit its data. The audit panel's "expressed nowhere" flag keeps `binary_expressed` (different question: presence-anywhere vs. breadth); the doc note covers the divergence.

4. **Enrichment granularity = 14 native ProjecTILs states (base) + CD4/CD8/Treg grouping overlay.** Do not collapse to 3 — Tex/Tpex resolution is the point of an exhaustion cohort. CD4/CD8/Treg is a sort/group lens and an optional collapsed summary. Both within-cohort and NSCLC enrichment are computed as a per-kinase share across the same 14 canonical states (NSCLC states mapped via the existing `PROJECTILS_LABEL_MAP`, which already sends `CD4.Treg → Treg`, pulling Treg out as its own group). Each enrichment gets a tier binned at multiples of `1/N_states` uniform, for parity.

5. **A2 mirrors the bulk verdict exactly.** Replace the `tcell_concordance` "vs Bulk" column with the deconvoluted per-state NES (A2 says "replace," so no supplement). Use the **stoichiometry-normalized** projected-state NES (`mea_projected_state.csv`), looked up by the kinase row's own track (ST kinases → ST NES, pY → pY NES). The `_raw` form stays a disk-only diagnostic, exactly as the bulk `_raw` is. No mixing of normalization scales in one table.

6. **Replace the `nsclc_tier` "NSCLC TME" column.** The coarse T_NK fold-over-uniform was a stopgap for the broken strip. The verdict table's NSCLC columns become **Specificity** (N/7 + expressing types) and **NSCLC enrichment** (per-state share, CD4/CD8/Treg grouped). Drawer lineage strip renders the full 14-state breakdown. Drop `nsclc_tier` — keeping it is three NSCLC numbers where two answer everything.

---

## Critical findings that change the work (stale/orphan artifacts)

Two NSCLC reference files are not reproducible from current source — both must be fixed before the viewer can honestly depend on them:

- **`nsclc_kinase_expression.csv` is stale.** Current `nsclc_expression.py` emits `specificity_score`; the on-disk CSV has `concentration` (uncommitted intermediate version). The JS reads `specificity_score` → all NSCLC badges render 0.
- **`nsclc_kinase_specificity.csv` is an orphan.** It exists on disk (290 KB, carries `n_detected_coarse` — the column A1's "N of 7" maps to) but **no current source produces it**; `config.py` defines no constant for it, and its producer (`alz/cross_reference/specificity.py`) survives only as a stale `.pyc`. Wiring the viewer to it (as the audit suggested) would depend on an unreproducible file — a honesty-rule violation.

**Resolution:** make `nsclc_expression.py` the single canonical NSCLC producer. The coarse breadth (`N of 7`) is derivable from the per-native-type `binary_expressed`/`spec_group` data it already builds. Compute specificity (frac≥0.10 breadth) and NSCLC enrichment (14-state share) there, write one canonical set of columns, and **delete the orphan `nsclc_kinase_specificity.csv`**.

---

## Stage 0 — A5: run the monotonic export (standalone)

Independent of everything else.

1. `pixi run python -m alz.cohorts.tcells.monotonic_export --verify`.
2. Confirm `outputs/reports/kinase_attribution_tcells/monotonic/{monotonic_kinases.csv, monotonic_substrates.csv}` written and `--verify` passes.
3. Spot-check one `increase`-flagged kinase against its per-day NES.

No viewer wiring (A5 is a rarely-used export by design).

**Gate:** report counts (kinases up/down, substrates up/down).

---

## Stage 1 — A1 data: single canonical NSCLC producer

1. Extend `alz/reference/nsclc_expression.py` to compute, per kinase:
   - **Specificity:** count of the 7 coarse groups where `group_fraction ≥ 0.10` (pure prevalence; derive group_fraction from native-type cell-weighted aggregation), plus the list of expressing groups.
   - **NSCLC enrichment:** per-kinase share across the 14 canonical T-states (map NSCLC native labels via `PROJECTILS_LABEL_MAP`), with a tier at multiples of `1/14` uniform.
   - Keep `fraction_cells_expressing` per native type (A1's "fraction of cells per type") and the existing share/`specificity_score` (A1's "percent of expression in a type").
2. Regenerate, **streamed under a memory cap** (`systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0`) — the source is the 1.3B-nnz 10x CSC h5; confirm the script hyperslab/batch-reads and never full-loads.
3. Verify the regenerated `nsclc_kinase_expression.csv` has `specificity_score` (fixes the badge bug) and the new specificity/enrichment columns.
4. **Delete `nsclc_kinase_specificity.csv`** (orphan) and remove any viewer reference to it.

**Gate:** confirm on-disk columns match exactly what Stage 3 JS will read before any viewer edit.

---

## Stage 2 — A2 data: run projected-state MEA (feasible — `kinase_library 1.7.0` in pixi env)

1. Run `state_mea.py` for donor1 → `outputs/reports/kinase_attribution_tcells/donor1/state_mea/mea_projected_state{,_pY,_raw,_raw_pY}.csv`.
   - Confirm `ps_deconvoluted.csv` (ST) and `py_deconvoluted.csv` (pY) exist for donor1.
2. Sanity-check: per-state NES × {d13,d17,d20}, states matching the 14 ProjecTILs labels, both tracks present.

**Gate:** report the state × contrast NES table shape and representative values (new numbers — surface before wiring).

---

## Stage 3 — A1 payload + viewer

Depends on Stage 1.

Payload (`slices_kinase.py`, `slices_audit.py`):
1. Expose specificity (N/7 + expressing types) and NSCLC enrichment (14-state share + tier) per kinase. Source is the single canonical expression CSV — **no** second specificity-CSV registration.
2. Carry native-resolution NSCLC T-state rows (stop collapsing to one `T_NK` bucket).

`common.py`:
3. Add an `ENRICHMENT` label constant distinct from `SPECIFICITY`. Establish the contract: specificity = NSCLC type breadth; enrichment = ProjecTILs state concentration.

Rename (producer + consumer, one pass):
4. `tcell_within_cohort.py`: `tcell_specificity.csv → tcell_enrichment.csv`, columns + docstring. Grep for every consumer first (it's T-cell-namespaced; confirm no AD-viewer reader).

JS (`kinase_audit.js`, `kinase_explorer.js`):
5. `_renderNSCLCLineageStrip`: render 14-state breakdown grouped CD4/CD8/Treg (the `specificity_score` read now resolves post-Stage-1).
6. `ATTR_VERDICT_COLS`: relabel within-cohort columns to "Enrichment"/"Enrichment share"; add **Specificity** (N/7) and **NSCLC enrichment** columns; **remove `nsclc_tier`** "NSCLC TME".
7. Add CD4/CD8/Treg sort/group control.

**Anti-shim:** delete the T_NK-collapse path and `nsclc_tier`; do not leave them beside the replacements.

---

## Stage 4 — A2 payload + viewer

Depends on Stage 2.

1. `slices_kinase.py`: wire `mea_projected_state.csv` (+ `_pY`) into a `decomposition_index`-compatible structure (`{kinase_id[], contrast_id[], cell_type[], decomp_nes[], decomp_fdr[]}`), per track, selected by the kinase row's track. Exclude `_raw`.
2. `kinase_audit.js`: confirm the existing decomp-NES bar chart (`_renderDecompPanel`, reads `PAYLOAD.decomposition_index`) renders per-state NES.
3. `kinase_audit.js`: replace the `tcell_concordance` "vs Bulk" column with the deconvoluted per-state NES for the row's (kinase, state, contrast, track). States with no projected-state row render honest-empty.

---

## Stage 5 — rebuild + verify

1. `pixi run tcell-viewer` under a memory cap. Report exit code + payload size (raw + gzip).
2. Browser click-through (live browser):
   - NSCLC lineage strip non-empty, 14-state, grouped CD4/CD8/Treg.
   - **Specificity** (N/7) and **NSCLC enrichment** columns populated; `nsclc_tier` gone.
   - Within-cohort columns read "Enrichment"/"Enrichment share" (not "Specificity"/"Share").
   - CD4/CD8/Treg sort works.
   - Decomp-NES chart renders per-state NES; "vs Bulk" replaced by deconvoluted NES.
3. Hard-refresh (Ctrl+Shift+R); verify `PAYLOAD.meta.generated_at`.

**Visual changes are authoritative** — if a badge/strip/chart doesn't visibly change, it isn't done.

---

## Sequencing

```
Stage 0 (A5)  ─── standalone, any time
Stage 1 (NSCLC canonical producer) ──► Stage 3 (A1 payload+viewer) ──┐
Stage 2 (state_mea, both tracks)   ──► Stage 4 (A2 payload+viewer) ──┤
                                                                     ▼
                                                              Stage 5 (rebuild+verify)
```

Stages 1 and 2 are independent — parallelizable. Stage 3 (A1) and Stage 4 (A2) are independent — parallelizable, **except** the final `build_tcell_viewer` rebuild must be serialized (single payload writer). Stage 5 is the join.

## Out of scope / deferred

- A3, A4: no work (audit-clean).
- Surfacing A5 monotonic output in the viewer — export-only by design.
- Donor2 projected-state MEA — donor2 has no IMAC; kinase MEA / state MEA are donor1-only.
