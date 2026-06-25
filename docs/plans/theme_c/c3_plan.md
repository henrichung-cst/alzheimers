# Theme C3 — Disease-direction focus

**Contract:** `_contracts.md §C1` (consumer side). **Audit:** `c3_audit.md`. **Wave:** 2/3 (consumer; unified-viewer-only, no heavy compute). **Prereqs:** C1 (per-genotype `peak_NES_{g}`/`trajectory_{g}`/`songOverallPeak`), C2 (`MouseC1` display name). **Collision class:** unified viewer; shares `song.py` with C1 (C1 first) and `06_export_csv.js` with F1 (serialize).

## Decisions (locked, P3 grill 2026-06-25)
- **One new tab `diseasedirection`** (landscape group), **two stacked panels**: (top) **kinase directional-ranking**, (bottom) **candidate-biomarker table** keyed on substrate genes. Not a crosstable extension.
- **Early-change is site-level** (the literal TODO ask) but built as a **self-contained, deferrable stage** (Stage 4) — the rest of C3 has no dependency on it. The kinase-level `trajectory_{g}=="early"` flag (free, from C1) stays in the core top panel regardless. If Stage 4 is cut from the schedule, C3 still ships 1/2/3-kinase/4/5.
- **Secretability = HPA secretome, "secreted in human" semantics** (the intended filter — treatment target is human; a mouse-secretome flag would be true-but-unhelpful). Promote `hpa_secretome.tsv` to a tracked `data/` path; join on uppercased gene symbol; surface the **category string** ("Secreted to blood" / "Secreted in brain" / blank), **not a boolean**; annotation only, never drops rows; column labeled **"Secreted (human, HPA)"** so semantics are unambiguous.
- **Gene-list lookup = self-contained** textarea → filters the C3 table + shows an LFC column; `setWhitelist` Explorer handoff is a secondary action.
- **Ranking:** `songOverallPeak().nes` signed (F1) default; header-click ranks by a chosen genotype (App / Tau / ApTt).
- **F1 ordering:** C3 ships before F1 → C3 seeds `numCmp` into `06_export_csv.js`; F1 later *adopts* it. No inline shim.
- **C4 tie-in (veto on review):** biomarker panel carries a human-specificity column (`h_spec`, already in payload) — C4's guard bites exactly here.

## Stage 0 — Prereqs present (verify, don't build)
C1 merged (payload has `peak_NES_{App,Tau,ApTt}`, `trajectory_{g}`, `songOverallPeak` helper) and C2 merged (`COHORT_DISPLAY`/`COHORT_LABELS`, `MouseC1`). If C3 fans out before F1: confirm `numCmp` absent → C3 seeds it (Stage 1).

## Stage 1 — Secretome ingestion (build-time, bounded)
- Promote `hpa_secretome.tsv` from `outputs/reports/kinase_attribution_human/ctrl_audit/` to a tracked input (`data/external/hpa/hpa_secretome.tsv` or equiv); record provenance (HPA "Predicted secreted proteins") in a sidecar/manifest. **Do not commit the TSV** (data-file rule) — add to the runner's fetch/copy step + MANIFEST.
- In `build_unified_viewer.py`: load it (small read OK), build `{GENE_UPPER → Secretome location}`, attach `secretome_location` to each kinase/substrate gene in the Song payload (`song.py:_build_kinases_slice`). Blank where absent. **Shares `song.py` with C1 → apply on top of C1's merged diff.**
- If C3 ships before F1, add `numCmp(av,bv,dir)` (signed, null-last regardless of dir — exact body in `f1_plan.md` Stage 1) to `06_export_csv.js`.

## Stage 2 — Tab scaffold + top panel (kinase directional ranking)
- New `alz/viewer/template/js/tabs/kinase_disease_direction.js`; register in `TAB_MANIFEST` (`02_ui_chrome.js`), `body.html` (`<div id="tab-diseasedirection">`), `index.html.j2` raw include, `01_state.js` `TAB_GUIDE`.
- Table: one row per kinase; columns `MouseC1_App / _Tau / _ApTt` signed peak-NES via `_kxMedNesCell` (reuse), `trajectory_{g}` badge incl. the **kinase-level "early"** flag, `n_sig_{g}`. Default sort `songOverallPeak().nes` (signed, `numCmp`); header-click toggles per-genotype sort + asc/desc (negative tail). Reuse `_kxMouseGlyphCell` for an optional 3×3 expand.

## Stage 3 — Bottom panel (candidate-biomarker table)
- Row entity = substrate **gene** (the secreted protein), not kinase. Columns: gene, `Secreted (human, HPA)` category (Stage 1), an LFC column, **human specificity** (`h_spec`, C4 guard, veto on review).
- **Gene-list lookup:** textarea (one symbol/line) → parse, uppercase, filter the table to matched genes; show unmatched count honestly. LFC = kinase-level `top_celltype_1_song_lfc` for kinase rows; for non-kinase substrate rows use per-site `stoich_lfc` (from the Stage-4 shard if present, else omit the row's LFC and say so — no fabricated value).
- Secondary "open matched in Kinase Explorer" button → `setWhitelist(matchedKinaseIds)`.
- Sorting via `numCmp` (signed, F1).

## Stage 4 — Site-level early-change (SELF-CONTAINED, DEFERRABLE)
*Cut this stage without affecting Stages 1-3.* Recorded so the idea is not lost.
- New build step (`build_unified_viewer.py` or a small `alz/.../site_early_change.py`): read `site_level_ols.csv` (bounded, 10.7 MB — DuckDB or chunked pandas, not whole-file if it grows), classify each `site_id` per genotype: **early in g** iff `stoich_fdr_{g}_2mo < FDR_THRESH` AND not `< FDR_THRESH` at `{g}_4mo`/`{g}_6mo` (canonical attribution layer; `MEA_FDR_THRESH` from config). Emit a per-gene shard (`gene_symbol → early_sites[]` with genotype + 2mo LFC).
- Biomarker panel (Stage 3) gains an **"early sites"** sub-column listing the early phosphosites per gene, joined with the secretome flag → the actual "early + secretable" diagnostic-candidate view.

## Stage 5 — Runner + task
Add the secretome copy/fetch (Stage 1) and, if built, the site-early step (Stage 4) to the unified-viewer runner; `pixi run viewer` rebuilds. No new pixi task unless Stage 4 ships standalone.

## Verification
- **Browser (human, authoritative):** new Disease Direction tab present; top panel shows 3 `MouseC1_*` signed columns, default sort largest-|change| on top, header-click ranks per genotype + reaches negative tail; kinase-level "early" badge appears for a known 2mo-only kinase.
- Biomarker panel: secretome category shows for a known secreted gene (e.g. a "Secreted to blood" hit), **blank (not "no"/false) where HPA lacks the gene**; gene-list textarea filters to pasted symbols and reports unmatched count; LFC column populated, never fabricated.
- **Stage 4 (if built):** spot-check one site early in App but not Tau; the "early + secreted" intersection is non-empty and each entry traces to a real `site_level_ols.csv` row.
- `command grep -rn 'peak_NES\b' alz/viewer/template/js/tabs/kinase_disease_direction.js` → uses per-genotype/`songOverallPeak` forms only (no pooled remnant); no `Math.abs` in C3 sort keys (F1).
- Payload: `secretome_location` present on Song kinases; values ∈ HPA categories ∪ blank.

## Out of scope
The C1 split itself (consumed), C2 naming internals (consumed), the F1 sweep (C3 only seeds `numCmp`), tcell viewer, any MEA/pipeline rerun, protein-LFC layer for non-kinase non-site genes (not in payload — such rows show no LFC honestly), UniProt subcellular fetch (HPA chosen).
