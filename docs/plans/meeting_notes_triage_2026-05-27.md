# Meeting-notes triage — 2026-05-27

Reformat of the 2026-05-27 meeting dump into actionable items, grouped by work
stream. Each item: **note** (what was said, lightly de-cryptified) → **reading**
(my interpretation) → **action** (proposed concrete change) → **status**.

Clarifications already obtained (2026-05-27):

- **First stream this session:** D (T-cell cohort).
- **"check incytr results against the existing ___":** against the **sce4
  reference** (already wired into `pixi run verify-incytr-sce4`).
- **T-cell quant type:** DIA label-free → reuse the Mukesh reshape path.
- **T-cell contrast "(after day 2) − (day 2 control)":** day-2 control is the
  **baseline t=0 reference**, subtracted from all later timepoints (per donor;
  timepoints are **not** comparable across donors).
- **T-cell data location:** a Google Drive folder → needs a `pixi run
  ingest-tcells` task. **Folder ID still required.**

Status legend: ☐ not started · ◐ in progress · ☑ done · ❓ needs clarification.

---

## Refresh — 2026-05-29 (high-recall re-pass)

Re-walked the raw note dump against this doc. Every line maps to an existing
item except one (now captured as **A15**). Status deltas and best-judgment
resolutions for the items the notes left under-specified (confirmed: do not
block on clarification — proceed on the reading below):

**Stream D progress.**
- D1 ☑ · D2 ☑ (unchanged).
- D3 ◐ — per-donor/per-timepoint contrast + kinase MEA on donor1 IMAC; folded
  into the Incytr run plan.
- D4 ◐ — resolved as **aggregate by ProjecTILs state, not Seurat cluster**
  (`docs/plans/tcells_percell_aggregation_2026-05-28.md`, approved/in-flight).
  Closes the D-stream "scRNA → spine" + "cluster spine" open questions.
- D5 ◐ — planned: `tcells_incytr_pair_2026-05-28.md` (design) +
  `tcells_incytr_run_2026-05-28.md` (OOM-aware execution, `pixi run tcells-incytr`).
- **D7. T-cell viewer** (new, not in original notes but now active): lift
  `alz/viewer/` verbatim and shape T-cell artifacts to its PAYLOAD/Store/
  TAB_MANIFEST contract — `docs/plans/tcell_viewer_lift_2026-05-29.md` +
  `tcell_viewer_payload_2026-05-29.md`. A payload + `index.html` are already
  generated under `outputs/reports/tcell_viewer/`. **All Stream A presentation
  changes below apply to BOTH `alz/viewer/` (mouse) and the lifted T-cell
  viewer** — shape the data, never fork the contract
  (`feedback_no_reimplementing_shared_viewer.md`).

**Stream A is still the un-started bulk** — these notes are mostly viewer/
crosstable presentation, and no A-item has landed. (`levers_AB_*` /
`optimization_levers_*` are Incytr-perf, unrelated to Stream A.)

**Resolutions for under-specified notes (stated assumptions, not blocking):**
- **A9 ranking metric** ("strongest evidence", flagged *extremely important*):
  primary sort = **magnitude of disease-vs-control NES change** (biggest movers
  first); secondary = consistency (bulk/deconv agreement + monotonicity). The
  "sorted visual of kinase up/down in disease" reads off this order.
- **A10 NES ratio**: show **median control NES** and a **disease ÷ control
  median-NES ratio** (activity fold) as adjacent columns.
- **A15 (was missing) — "Just condition, whatever is most up or down":**
  a timepoint-collapsed ranking — rank kinases purely by disease-condition
  direction/magnitude (most up / most down), no timepoint faceting. Consistent
  with A9/A12's "focus on disease direction." Lives next to A9 as the simplest
  ranked view.
- **C4 gene list**: the **early-change phosphosite genes** from C3 (flagged on
  CTRL-07/08/10). Two columns — **present-in-late-control (yes/no)** and the
  **LFC** — plus a cross-flag for which of those genes also move in AD.

---

## Stream D — T-cell cohort (ACTIVE)

This is the **T Cell Exhaustion** dataset (Donor 1 + Donor 2). Closest template
is the **Mukesh** human cohort (separate `data/datasets/<name>/`, dedicated
`alz/ingest/<name>.py` reshaper → Song-shaped artifacts, per-donor analysis).

Drive folder ID: `1YE_h1jIyBajtm6ArxJqevJ0rt0xLKQgX` (remote `gdrive_shared:`).

### Design facts (from notes, confirmed against the data)

| Aspect | donor1 | donor2 |
|---|---|---|
| Total proteome | ✅ | ✅ |
| pY (phosphotyrosine) | ✅ | ✅ |
| IMAC (global phospho) | ✅ | ❌ |
| scRNA | ✅ | ✅ |
| **Kinase MEA (bulk)** | ✅ | ❌ **skip** (no IMAC) |
| **Incytr (on deconvoluted)** | ✅ | ✅ |
| Timepoints (baseline = Day 2) | 2, 11, 13, 15, 17, 19, 20 · **2 reps** | 2, 5, 7, 9, 11 · **1 rep** |

Timepoint sets differ by donor *and* slightly by modality (e.g. donor1 pY skips
Day 11). Day 2 is the baseline in every track — the collaborator Log2FC files
carry no Day-2 FC column, confirming the "(later) − (Day 2)" contrast.

### Ingestion record (2026-05-27)

- **Representation = `ForPerseus`** for Total/pY (both donors): site-level
  (`PG.Genes`, `PTM.ModificationTitle`, `PTM.SiteAA`, `PTM.SiteLocation`,
  **`PTM.FlankingRegion`** = motif), genes pre-parsed, linear intensities,
  ~0.1–2.8 MB. Avoids the 737 MB "NotParsed" reports and the R header parser.
- **IMAC** (donor1 only) has no ForPerseus form → ingest the 18 May
  `Normalized_IMACSiteReporttsv.tsv` (137 MB, 113k site rows) as-is. It already
  carries `PG.Genes` + `PTM.FlankingRegion` + `SiteProbability`, so the R parser
  is **not** needed. IMAC timepoints: Day 2,13,15,17,19,20 (reps `_2`/`_3`, no
  Day 11) — align to Total by day, not by rep index.
- **Normalization basis = self-normalize.** All log2 + per-run median-centering
  happens in `alz/ingest/tcells.py` so both donors and the IMAC track share one
  basis (the collaborator's May "Normalized" reports are NOT used — they would
  mix vintages with the 10 Feb ForPerseus and need the R parser).
- **Contaminant filter:** pY/IMAC ForPerseus tables include Carbamidomethyl (C)
  and Oxidation (M) rows — filter to `PTM.ModificationTitle == "Phospho (STY)"`.
- **scRNA = Seurat `.rds`** (not h5ad): donor1 `Tcells.singlet.rds` (5.2 GB,
  27486×25678, 14 clusters, CITE-seq RNA/Protein/HTO, `sample_ID` days
  0/2/9/13/17/20), donor2 `Tcells_d2.singlet (1).rds` (4.9 GB). Decomposition (D4)
  needs an R-side export to h5ad / pseudobulk before the Python spine can consume
  it. (The earlier donor1 `singlets_0425.rds` was a mislabeled cell-line benchmark
  — replaced 2026-05-27.) **Memory:** a plain `readRDS` on these objects OOMs the
  shared box; D4 R steps must extract `meta.data`/counts without holding the full
  object (subset assays, `gc()`, or stream).
- **Out of scope:** KGG / AcK / MME enrichments, Flow cytometry.
- **Landing layout:** `data/datasets/tcells/{donor1,donor2}/{proteomics,scrna}/`,
  original vendor filenames preserved.
- **Tasks wired:** `pixi run ingest-tcells` (proteomics) and
  `pixi run ingest-tcells-scrna` (+10 GB scRNA) → `tcells` source in
  `conf/data_sources.yaml` (rclone-ingest engine; superseded the standalone
  `ingest_tcells.sh` on 2026-06-08).

Hard rules from the notes:

- **Kinase MEA on bulk only. NEVER kinase MEA on deconvoluted.** (Same
  architecture as mouse: MEA on bulk stoichiometry; Incytr on per-cluster
  decomposition.)
- **Incytr runs on the deconvoluted (per-cluster) substrate**, both donors.
- **Contrast of interest:** activation vs exhaustion = (later timepoint) −
  (day-2 baseline t=0), computed **per donor independently**.
- DIA label-free → **no IRS / no TMT plex** normalization; reuse Mukesh's DIA
  reshape, not `alz/bulk_mea/normalize.py`.

### Proposed actions

- ☑ **D1. Ingest task.** `ingest-tcells` (+ `ingest-tcells-scrna`) → `tcells`
  source in `conf/data_sources.yaml` → `data/datasets/tcells/`. Proteomics
  ForPerseus files landed; IMAC + scRNA pulling.
- ☑ **D2. Reshape module** `alz/ingest/tcells.py` (`pixi run tcells-reshape`).
  ForPerseus is already site-collapsed (gene/residue/location/flanking pre-parsed)
  → **none** of Mukesh's UniProt/peptide-mapping machinery is used. Emits
  Song-shaped `stoichiometry_matrix{,_pY}.csv`, `raw_phospho_normalized{,_pY}.csv`,
  `total_proteome_normalized.csv` under
  `outputs/reports/kinase_attribution_tcells/donor{1,2}/`, plus `sample_mapping.csv`
  + `ingest_manifest.json` under `data_ingest_tcells/donor{1,2}/`. IMAC → ST track
  (suffix `""`), pY → `_pY`. Three reconciliations + self-normalize:

  1. **Localization gate (IMAC).** `PTM.SiteProbability ≥ 0.75` (class-I). The
     18May "Normalized" IMAC report is **already pre-gated at exactly 0.75**
     upstream (min=0.750, 0 rows dropped here) — our gate confirms/records it.
     pY ForPerseus has no probability column.
  2. **Isoform collapse = canonical-except-isoform-specific** (mirrors Mukesh).
     The report lists each physical site once per isoform of its protein group
     (same measurement; position numbered per isoform; flanking can differ near
     splice junctions). Using `PTM.ProteinId`: keep every **canonical**-isoform
     site (canonical = first in `PG.ProteinGroups`); add back a non-canonical site
     only when its measurement (raw value vector) is absent from the canonical
     isoform (true isoform-specific). Confirmed: AAAS → S462/T57/Y452 (canonical
     `…437`), the `…438` copies S495/Y485 dropped. IMAC 98,344 rows → 62,807 sites,
     0 isoform-specific (first-in-group is the superset isoform here). Never merges
     distinct multi-phospho sites; residual same-value sites (2,201) are all
     same-gene multi-phospho peptides, 0 cross-gene. pY lacks `PTM.ProteinId`, so
     it collapses same-window copies by `(gene,residue,flanking)` instead (window
     separates multi-phospho); donor2 pY has no flanking → falls back to location.
  3. **Technical-replicate collapse.** Donor 1 is injected twice per timepoint
     (`_DIA`/`_DIA_2`, `_pY_1`/`_pY_2`, IMAC `_2`/`_3`), donor 2 once — *technical*
     re-injections (r1↔r2 correlation 0.96–0.99), so averaged to **one column per
     (donor, day)** after per-run median-centering. `sample_id = D{n}_d{day}`.

  Self-normalize: linear → log2 (empty/≤0 → NaN; ForPerseus `1` floor kept as
  log2=0 but counted) → per-run median-center → rep-average. Counts: donor1 Total
  8125 genes / pY 1180 sites / IMAC 62807 sites (7/6/6 samples); donor2 Total 7767
  / pY 514 (5 samples). **donor2 pY has no `PTM.FlankingRegion`** → motif empty for
  all 514 sites (kinase MEA skipped there anyway; Incytr uses the phospho
  substrate, not motif).
- ☐ **D3. Per-donor / per-timepoint contrast builder** (analog of
  `mukesh_perdonor.py`): per donor, delta vector = (timepoint) − (day-2
  baseline). Kinase MEA on **donor1 IMAC stoichiometry only**.
- ☐ **D4. scRNA decomposition spine**, per donor (analog of the Levy-t5
  decomposition_mea path) → per-cluster proportional substrate.
- ☐ **D5. Incytr pair-mode** on the per-donor deconvoluted substrate, both
  donors. Reuse `alz/incytr_pair/` with a T-cell cluster spine.
- ☐ **D6. Pixi bundle** `tcells = { depends-on = [...] }` + runner under
  `alz/runners/main/`.

### Open questions for Stream D

- ❓ **scRNA → spine:** `.rds` are Seurat objects. Convert to h5ad (SeuratDisk)
  and reuse the Levy-t5 decomposition path, or pseudobulk in R first? Donor1 is
  10x NSCLC, donor2 is a T-cell singlet object — different cluster vocabularies.
- ❓ **T-cell cluster spine:** reference taxonomy, or cluster each donor's scRNA
  de novo? (No T-cell analog of Levy-t5 exists in-repo.)
- ❓ **donor2 Incytr substrate** when IMAC is absent: pY-only phospho substrate,
  or total-proteome-derived?
- ☑ **Replicate handling (donor1):** the two columns per timepoint are
  *technical* re-injections, not biological replicates → averaged to one column
  per (donor, day) in D2 after per-run median-centering.
- ☑ **Isoform fanout:** resolved in D2 via canonical-except-isoform-specific
  (`PTM.ProteinId`); was an isoform-numbering effect, not localization ambiguity.
  Residual ambiguity was a non-issue — the IMAC report is pre-gated at
  `PTM.SiteProbability ≥ 0.75`.

---

## Stream A — Viewer / crosstable presentation

Self-contained (no new data). Touches the unified viewer, cross-reference
tables, and `alz/shared/map_kinases_to_genes.py`.

- ☐ **A1. Remove SEA-AD specificity** column from the crosstable/viewer.
- ☐ **A2. Remove the raw specificity number** — keep only the ranked / fold
  form (5×, 10× tiers).
- ☐ **A3. Degenerate-specificity filter — "AVOID AT ALL COSTS":** drop kinases
  specific to **exactly 1** cell type in mouse and **exactly 2** in human.
- ☐ **A4. Specificity correlation, mouse vs human**, focused on the **5× and
  10×** fold tiers.
- ☐ **A5. Double-check specificity in human data** (validation pass after A1–A4).
- ☐ **A6. Kinase→gene naming fix.** Cases like **ALK1 ≠ ACVRL1** are wrong.
  Audit `map_kinases_to_genes.py`.
- ☐ **A7. Crosstable restructure:** split it, make it **narrower / clearer**;
  separate **bulk / deconvolution / summary** metric blocks.
- ☐ **A8. Direction grids:** show **up/down in bulk AND in deconvoluted**, not
  just the agreement grid (the 3×3 grids). Three disease columns **App / Tau /
  ApTt**, each showing up-or-down.
- ☐ **A9. Ranking by magnitude — "extremely important":** rank kinases by
  biggest change; add a visual of kinase up/down in disease, sorted.
- ☐ **A10. Median NES (control) + ratio** column.
- ☐ **A11. "Monotonic down across everything"** flag — kinases monotone-down
  across all contrasts/timepoints.
- ☐ **A12. Focus framing:** up/down **during disease development**, "need to
  find things that relate to the disease."
- ☐ **A13. top 300 / bottom 300** most-differentiated sites per cell type.
- ☐ **A14. Spot-checks:** APOE snRNA transcript **should be up**; **PHKG1 →
  Astrocytes**; verify these resolve correctly.

---

## Stream B — Kinase → substrate → pathway

Method-heavy / novel. Defer until A and D are in flight.

- ☐ **B1. Substrate-activity consistency check:** for ranked kinases, pull
  **high-confidence substrates**, confirm the underlying phospho data agrees
  with the kinase call (substrate activity should track the enrichment).
- ☐ **B2. Insert a kinase node into the Incytr pathway** based on substrate
  phospho-activity, joined to **cell type + timepoint**.
- ☐ **B3. Motif reference:** in the measurement trace, reference the
  **summarized motif** (named motif class), not just the raw peptide sequence.
- ☐ **B4. Incytr vs sce4** cross-check (clarified target = sce4 reference).

---

## Stream C — Suspicious-sample early biomarkers

Already started: untracked `alz/cross_reference/ctrl_outlier_suspect_lfc_table.py`
builds suspect (CTRL-07/08/10) vs clean (01–04) vs AD per-site + by-gene LFC with
HPA secretome annotation. (See memory: CTRL-07/08/10 are genuinely AD-like.)

- ◐ **C1. Suspect-vs-AD LFC table** — implemented (per-site + by-gene, secretome
  column). Needs review/commit.
- ☐ **C2. "Add suspicious samples to AD":** variant grouping 07/08/10 into the
  AD arm (separate from the C1 contrast table).
- ☐ **C3. Early-change phosphosite labeling** + cleaved/secreted → CSF/blood
  framing → candidate early diagnostic markers.
- ☐ **C4. Late-control presence table:** out of a gene list, two columns —
  present-in-late (yes/no) and the LFC number.

---

## Suggested sequencing

1. **D1–D2** once the Drive folder ID lands (unblocks the whole T-cell cohort).
2. **A** in parallel — no data dependency, high-value presentation cleanup.
3. **C** — small, mostly done; review + commit C1, then C2–C4.
4. **B** last — depends on a settled kinase ranking (A9) and Incytr substrate.
