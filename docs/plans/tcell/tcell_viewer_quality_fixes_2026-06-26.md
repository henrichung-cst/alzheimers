# T-cell viewer quality fixes (W1 gate, 2026-06-26)

Authoritative visual feedback: the T-cell viewer reads as shoddy. Six complaints, grounded in
code below. Three classes: **vocabulary unification** (JS only), **detection-gating recompute**
(Python + donor1 rebuild), and **one dead column** (hide / produce decision).

donor2 within-cohort is **correctly absent** (no IMAC/phospho for donor2 → no kinase MEA). Not a gap.
Within-cohort recompute stays donor1-only.

## Class 1 — vocabulary + a real bug (JS/template only, no recompute)

### 1a. Unify the "specificity" vocabulary (complaint #1)
Three columns currently collide on the word. Resolution — one concept, one label:
- `tcell_state_enrichment` (within-cohort, the kinase's headline axis) → **"Specificity"** everywhere
  (currently "State enrich." in explorer `body.html:75`, "Enrichment" in `attribution_manifest_tcell.js:70`).
- `nsclc_lineages_detected` → relabel **"Cross-lineage" → "NSCLC breadth"** (`body.html:77`,
  `kinase_explorer.js:536–557`). Drop the "cross-lineage" jargon entirely.
- `nsclc_specificity_count` → **"NSCLC spec (N/7)"** (`body.html:78`) — disambiguated from the within-cohort axis.

### 1b. "Δ vs d2" → "LFC" (complaint #6)
The value is log2FC (`tcell_within_cohort.py:302`). Match Song's "MouseC1 LFC" form → **"LFC"**.
Seven spots in `attribution_manifest_tcell.js`: comment 12, column label 81, explainer 204 + 210,
trace row 326, caption 333, trace header 335. (`body.html:265` "Day vs d2 baseline" refers to the
contrast, not the LFC column — leave it.)

### 1c. Fix the 0/7-renders-green bug (complaint #3, secondary)
`kinase_explorer.js:527`: `cls = cnt <= 1 ? "hi" : (cnt <= 2 ? "mid" : "lo")` →
`cnt === 0` must map to the lowest tier (muted/`lo`), not green `hi`. 0/7 = expressed above 10% in
no lineage = least specific.

## Class 2 — detection-gating recompute (Python; recompute donor1; rebuild viewer)

### 2a. Within-cohort detection floor 1% → 10% (complaints #2, #4)
`tcell_within_cohort.py:101` `DETECTION_FRAC_MIN = 0.01` → `0.10`. This is the value passed to
`specificity.compute(detection_frac_min=...)` at lines 238 + 261 and to the state-eligibility gate
at line 290 (`tcell_detected = tcell_fraction_expressing >= floor`). At 10%:
- state enrichment shows only where the kinase is in ≥10% of that state's cells (your "no enrichment
  over a <10% sample");
- the cell-type pill tier (`tcell_celltype_concentration_tier`) recomputes on the 10% basis, so the
  green "concentrated" pill (complaint #3a) collapses to `—` for the 1–9% trace cases.
Keep the specificity denominator over the curated state basis (effective-N convention) — gate
eligibility, do not shrink N. This is the `specificity.compute` `detection_frac_min` lever, not a
basis change. Prior 1% override is superseded per explicit instruction (10% matches the NSCLC spec
count + the Song viewer).

Recompute: `pixi run tcell-within-cohort` (donor1) under the memory cap, then `pixi run tcell-viewer`.

## Class 3 — produce the Decomp NES column (complaint #5)

`attribution_manifest_tcell.js:92` ships a "Decomp NES" column that was **always `—`**:
`PAYLOAD.decomposition_index` was absent because nothing wrote
`<donor>/state_mea/mea_projected_state.csv`. The MEA machinery already exists
(`run_projected_state_mea`); only a canonical entrypoint was missing. Decision: **produce it.**

- `state_mea.py` CLI reworked — `--runner-scratch-dir` → `--out-dir`, defaulting to the canonical
  `outputs/reports/kinase_attribution_tcells/<donor>/state_mea`. Scratch-only framing removed.
- New pixi task `tcell-state-mea` (donor1, ST track) writes the canonical tables.
- `_build_decomposition_index_from_state_mea` keyed on **`timepoint`** (`d13`), not raw `contrast`
  (`d13_vs_d2`) — `meta["contrasts"]` and the JS `CONTRASTS.indexOf(ctx.contrast)` both use bare
  days, so the raw `_vs_d2` form would drop every row.

Decomp NES = stoich track (the `mea_projected_state.csv` rows; raw is the sensitivity file). The
viewer's existing column scaffolding renders it once the canonical files land.

## Order of work
1. Class 1 (JS/template) — no recompute, immediately verifiable.
2. Class 2 — edit floor, recompute donor1, rebuild viewer.
3. Class 3 — per decision: hide now (A), or queue producer (B).
4. Browser-verify, then resume the W1 gate cleanup + tag.
