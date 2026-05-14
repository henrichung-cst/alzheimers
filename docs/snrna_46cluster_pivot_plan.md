# snRNA → 19-cluster spine pivot — implementation plan

**Status:** decisions locked 2026-05-13; ready for implementation.
**Goal:** replace the WMB-34-class cell-type spine with a 19-cluster
spine derived from the Levy 46-cluster taxonomy
(`data/incytr/v2_46clusters/provenance/kr_cluster_id_key.csv`).
**Net effect:** 34 → 19 cell types. 87.82% of cohort nuclei retained.
No `n/a` factorial contrasts in the primary deliverable.

---

## 1. Final spine (19 named, full-rank clusters)

Sorted by cell count (gate=20):

| # | Cluster name | Cells | Animals at gate=20 |
|---|---|---|---|
| 1 | `Erbb4-VIP-inhibitory-neurons` | 8,091 | 28 |
| 2 | `Astrocytes` | 6,954 | 28 |
| 3 | `Oligodendrocytes` | 6,394 | 28 |
| 4 | `Excitatory-Pyramidal-Satb2-Cux2` | 5,023 | 28 |
| 5 | `Striatal-medium-spiny-neuron` | 4,506 | 28 |
| 6 | `Excitatory-Rorb` | 3,582 | 28 |
| 7 | `Foxp2-Excitatory-Neurons-layers-6-and-2-3` | 2,774 | 28 |
| 8 | `Excitatory-Pyramidal` | 2,570 | 28 |
| 9 | `Microglia` | 2,287 | 28 |
| 10 | `glutamatergic-excitatory-neurons` | 1,707 | 24 |
| 11 | `OPC` | 1,636 | 28 |
| 12 | `Excitatory principal neurons in the hippocampal dentate gyrus` | 1,583 | 26 |
| 13 | `Erbb4-inhibitory-neurons` | 1,579 | 27 |
| 14 | `Excitatory-neurons` | 1,526 | 28 |
| 15 | `Endothelial-cell` | 1,485 | 28 |
| 16 | `VIP-positive-interneuron` | 1,386 | 26 |
| 17 | `Reln-neurons` | 1,100 | 24 |
| 18 | `Pericyte` | 1,014 | 25 |
| 19 | `Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic` | 751 | 20 |

**Total: 55,948 cells. All 9 factorial contrasts estimable for all 19.**

Excluded (27 of 46): 15 unnamed `cluster-NN`, 6 partial-rank named
(`Ptprz1-protoplasmic-astrocytes`, `Basal-Ganglia-GABAergic-Neurons`,
`Vascular-Leptomeningeal-Cells`, `Inhibitory-Neurons`, `Ependymal-cell`,
`GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4`), 3 severe
(`GABAergic inhibitory interneurons`,
`Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin`,
`Choroid-Plexus-Epithelial-Cells`), 3 fails-gate (rank-0 named).
Logged in `rejected_clusters.csv` for audit.

---

## 2. Decision summary

| Q | Topic | Decision |
|---|---|---|
| Q1 | Per-cell label source | Extract `seurat_clusters` column (0–109) from `incytr_obj.rds`, join via `kr_cluster_id_key.csv` |
| Q2 | WMB external evidence | Hand-curated cluster → WMB-class crosswalk (19 rows); WMB acts as coarse lineage gate |
| Q3 | SEA-AD concordance | Hand-curated cluster → SEA-AD-supertype bridge (19 rows; ~10–13 populated, MTG-only) |
| Q4 | Per-(animal,cluster) cells gate | `SONG_MIN_CELLS = 20` (was 50) |
| Q5 | Unnamed `cluster-NN` populations | Drop (3,486 cells / 5.47%) |
| Q6 | Factorial estimability | 19-cluster strict spine, all rank-10; rejected clusters logged but not in attribution |

---

## 3. New / changed data artifacts

| Path | Status | Contents |
|---|---|---|
| `data/incytr/v2_46clusters/barcode_to_cluster.csv` | NEW | (barcode, seurat_cluster_id) from `incytr_obj.rds` |
| `data/incytr/v2_46clusters/cluster_spine.csv` | NEW | 46 rows: cluster_name, in_spine (bool), exclusion_reason, n_cells |
| `data/incytr/v2_46clusters/rejected_clusters.csv` | NEW | 27 rows: cluster_name, reason, n_animals_g20, rank, missing_geno, missing_time |
| `data/incytr/v2_46clusters/cluster_to_wmb_class.csv` | NEW | 19 rows: cluster_name, wmb_class_label, notes |
| `data/external/sea_ad/cluster_to_seaad_supertype.csv` | NEW | ≤19 rows: cluster_name, seaad_supertype, weight, notes (many-to-many) |
| `data/external/allen_abc/wmb_class_manifest.csv` | RETIRED (or repurposed for WMB-class crosswalk only) |
| `data/external/sea_ad/seaad_subclass_to_wmb_class.csv` | RETIRED (replaced by direct cluster → supertype bridge) |
| `outputs/reports/snrna_integration/pseudobulk_cpm.csv` | Schema-stable; row count changes (19 cell types × animals) |
| `outputs/reports/snrna_integration/song_concordance.csv` | Schema-stable; row count changes |
| `outputs/reports/kinase_attribution/unified_attribution.csv` | Schema-stable; cell_type vocab changes to 19 names |

---

## 4. Implementation order

Each step ends with a verification check. Do not advance until the
check passes.

### Step 1 — Extract per-cell cluster labels

**File:** `alz/integration/extract_cluster_assignments.R` (new)

Read `data/incytr/v2_46clusters/incytr input/incytr_obj.rds`, emit
`data/incytr/v2_46clusters/barcode_to_cluster.csv` with columns
`barcode, seurat_cluster_id`. Use `obj@meta.data$seurat_clusters`
(verified: 110 unique values, 0–109; matches `kr_cluster_id_key.csv`
keys exactly).

**Verification:**
- 63,706 rows emitted (matches `obj@meta.data` nrow).
- All `seurat_cluster_id` values appear in `kr_cluster_id_key.csv$Cluster.ID`.
- Python audit script reports h5ad ∩ Seurat barcode intersection
  ≥99% (the two objects come from the same Song cohort; small
  differences acceptable if either side did extra QC).

### Step 2 — Build cluster manifest + rejection log

**Files:** `data/incytr/v2_46clusters/cluster_spine.csv`,
`rejected_clusters.csv` (both new, generated by an in-repo script
`alz/integration/build_cluster_spine.py`).

Logic: read `kr_cluster_id_key.csv`, group by `New_ID`, attach total
cell counts and per-(animal, cluster) gate=20 metrics, apply the
46 → 19 spine filter (full-rank + named + ≥20 animal coverage).
Emit both artifacts.

**Verification:**
- `cluster_spine.csv`: 46 rows, exactly 19 with `in_spine=True`.
- `rejected_clusters.csv`: 27 rows. Spot-check `Choroid-Plexus-Epithelial-Cells`
  shows `n_animals_g20=1, rank=1, reason=severe`.
- The 19 in-spine names exactly match §1 table above.

### Step 3 — Hand-curate crosswalks

**Files:**
- `data/incytr/v2_46clusters/cluster_to_wmb_class.csv` (Q2)
- `data/external/sea_ad/cluster_to_seaad_supertype.csv` (Q3)

Both are domain-curated. Suggested workflow: open
`data/incytr/v2_46clusters/incytr input/allmarkers.csv` (per-cluster
top markers) alongside SEA-AD `effect_sizes.h5ad` supertype markers
and Allen WMB class definitions. For each of the 19 cluster names,
assign:

- **WMB class** (1:1, every row populated). The 19 cluster names map
  unambiguously to coarse WMB lineages — astrocyte sub-populations
  collapse to `30 Astro-Epen`, microglia → `34 Immune`, etc.
- **SEA-AD supertype(s)** (1:N, many rows `n/a`). MTG-resident
  lineages (cortical excitatory, cortical GABAergic, glia, vascular,
  immune) get mappings; striatal, hippocampal-DG, and brainstem
  populations get `n/a`. Equal-weight averaging across mapped
  supertypes when multiple match.

**Owner:** flag for domain expert review before unblocking step 7.

**Verification:**
- WMB CSV: 19 rows, no `n/a` (every cluster has a WMB lineage parent).
- SEA-AD CSV: 19 distinct cluster names. ≈10–13 have ≥1 supertype
  mapping; remainder explicitly marked `n/a` with reason
  (`subcortical`, `hippocampal-DG`, etc).
- Both files round-trip-loadable via the new loaders in step 4.

### Step 4 — Update `alz/config.py`

**Changes:**
- Replace `WMB_CLASSES` (list of 34) with `CLUSTER_SPINE` (list of 19,
  loaded from `cluster_spine.csv` filtered to `in_spine=True`).
- Replace `load_song_to_wmb_class_map()` with
  `load_barcode_to_cluster_map()` (reads `barcode_to_cluster.csv` +
  `kr_cluster_id_key.csv`, returns `dict[barcode → cluster_name]`).
- Add `load_cluster_to_wmb_class_map()` (reads new Q2 crosswalk).
- Add `load_cluster_to_seaad_supertype_map()` (reads new Q3 bridge).
- Update `SONG_MIN_CELLS` to **20**. Drop the no-longer-used
  `SONG_MIN_SUBCLASS_PROB` constant (Seurat clusters have no
  probability gate; the gate becomes purely cell-count-based).
- Update `N_CELL_TYPES` derivation.
- Mark `WMB_SUBCLASS_TO_CLASS_FILE`, `SEAAD_SUBCLASS_TO_WMB_CLASS_FILE`,
  `WMB_CLASS_MANIFEST_FILE` deprecated; remove read sites after step 7
  validates.

**Verification:** `python -c "import alz.config; print(alz.config.CLUSTER_SPINE)"`
prints the 19 names from §1.

### Step 5 — Rewire `alz/snrna_integration.py`

**Changes:**
- `step_pseudobulk`: drop the `_build_class_name_to_label` helper and
  the `class_prob` filter. Use the new `barcode_to_cluster` map keyed
  on `adata.obs.index`. Filter to `cluster_name ∈ CLUSTER_SPINE`
  before pseudobulk. Apply `SONG_MIN_CELLS = 20` gate.
- `step_specificity`: no formula change. Output `cell_type` vocab is
  the 19-cluster spine.
- `step_concordance`: no formula change. Verify all 9 contrasts fit
  for all 19 clusters.

**Verification:** rerun `python alz/snrna_integration.py --run`.
Output `pseudobulk_cpm.csv` has exactly 19 unique `cell_type` values.
Cohort cell-count audit reproduces 55,948 ± small Seurat/h5ad
intersection delta.

### Step 6 — Update `alz/wmb_expression.py`

**Changes:** minimal. Continues to emit
`wmb_kinase_expression.csv` keyed by WMB class. Downstream
consumers (kinase_attribute) handle the cluster → WMB-class lookup.
Replace `WMB_CLASSES` reference with a fresh list of WMB classes
that any spine cluster maps to (probably ~10–15 of 34, drawn from
the new crosswalk).

**Verification:** output file row count drops proportionally; no
"Other" residual cells.

### Step 7 — Update `alz/kinase_attribute.py`

**Largest change.** Swap WMB-34 spine for 19-cluster strict spine
in the cross-join (`cell_type_df`). Replace:

- WMB-specificity merge — join via `cluster_to_wmb_class` crosswalk
  (cluster gets parent WMB class's expression score). Document in
  comment that this is lineage-level evidence, not cluster-level.
- SEA-AD merge — replace the 139-supertype → WMB-class rollup with
  direct cluster → supertype(s) lookup via Q3 bridge. Many-to-many
  collapses by mean (or weighted mean if weights are populated).
  Clusters with no supertype get `n/a` SEA-AD LFC.
- Song concordance merge — unchanged structurally; just uses the
  new cell_type vocab.

**Verification:** `unified_attribution.csv` has exactly
`n_kinases × 9 × 19` rows. No silent drops. `attribution_summary.json`
counts match.

### Step 8 — Update `alz/integration/export_factorial_inputs.py`

**Changes:**
- Replace `obs["subclass_name"]`-based lookup with the barcode-keyed
  `barcode_to_cluster` map from step 1. Filter to spine before writing
  the fixture (default behavior; add `--include-non-spine` flag if a
  future sweep needs the rejected clusters).
- Drop the `subclass_prob` filter.
- Manifest fields (lines 292–295): rename
  `wmb_class_map_source` → `cluster_spine_source`,
  `celltype_taxonomy` → `"Levy 19-cluster strict spine (filtered from v2_46clusters)"`.
- `SUBSET_NAME = "subset_immune_astro"`, `SUBSET_LABELS = ("Microglia", "Astrocytes")`.

**Verification:** `expression_metadata.csv` `labels` column contains
exactly the 19 spine names. `MANIFEST.json` records spine source +
git SHA + filter parameters.

### Step 9 — Validate downstream consumers

**Files to skim, edit minimally if at all:**
- `alz/attribution_recovery.py`: verify it consumes `unified_attribution.csv`
  without hard-coded cell-type-count assumptions; tier denominators
  may need refresh.
- `alz/build_unified_viewer.py`: line 618 `"celltypes": list(config.WMB_CLASSES)`
  becomes `list(config.CLUSTER_SPINE)`. Color palette needs 19 distinct
  colors (was 34); rebuild palette.
- `alz/supplementary/deconvolution_feasibility.py`: line 139 needs the
  WMB manifest replacement — either retire this supplementary or
  rekey it against the new manifest.

**Verification:** `pixi run live` completes end-to-end; viewer payload
loads without errors; all `cell_type` references are 19-cluster names.

### Step 10 — Re-run Incytr factorial

19² = 361 sender×receiver pairs (down from 34² = 1,156, ~3.2× cheaper).

```
pixi run export-factorial-inputs
pixi run incytr-factorial
```

**Verification:** `data/incytr_factorial_inputs/MANIFEST.json` records
the new spine; receiver_cache contains 361 pair parquets (or fewer if
PAIR_FILTER applied).

### Step 11 — Docs refresh

Update:
- `CLAUDE.md` — replace all "34 WMB classes" / `WMB_CLASSES`
  references; update Live Code section; add C-batch 2mo depth Gotcha.
- `docs/foundation/analysis_charter.md` §3 — cell-type spine.
- `docs/foundation/concordance.md` §4 — evidence sources at
  cluster resolution.
- `docs/foundation/live_pipeline_contract.md` — Stage 3 outputs.
- `docs/foundation/statistical_constraints.md` — DOF tables with 19
  clusters.
- `alz/integration/README.md` — sender/receiver count, fixture
  vocabulary.

---

## 5. Pre-flight checks (before step 1)

1. `incytr_obj.rds` is readable in the pixi env (verified 2026-05-13:
   `pixi run Rscript` with `SeuratObject` works).
2. `pixi.toml` includes `r-seuratobject` (verify; add if missing).
3. Barcode format in `incytr_obj.rds` (`04_01_11-0`) is comparable to
   barcode format in `170_gex_celltypes_00.h5ad` (audit needed).
4. SEA-AD `effect_sizes.h5ad` has accessible supertype marker
   metadata; if not, source from SEA-AD documentation.

---

## 6. Out-of-scope / follow-up workstreams

These are noted but **not** part of this pivot:

1. **WMB re-projection at cluster resolution.** Use scANVI or
   marker-based label transfer to project Allen WMB nuclei into the
   19-cluster space, giving cluster-level (not lineage-level) external
   specificity. Long-term answer for Q2; revisit when cluster-level
   attribution becomes a primary deliverable.
2. **C-batch 2mo animal depth.** The four animals
   `C198_ma_2mo_WTyp`, `C199_ma_2mo_AppP`, `C200_ma_2mo_Ttau`,
   `C202_ma_2mo_Ttau` are systematically lower-depth (14–17 of 31
   surviving clusters at gate=20 vs cohort median 21). Possible batch
   confound; investigate sequencing depth / QC stats. Add Gotcha to
   CLAUDE.md in step 11.
3. **Recovery of partial-rank clusters.** Six named clusters
   (`Ptprz1-protoplasmic-astrocytes`, `Basal-Ganglia-GABAergic-Neurons`,
   `Vascular-Leptomeningeal-Cells`, `Inhibitory-Neurons`,
   `Ependymal-cell`, `GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4`)
   are biologically interesting but rank-deficient at gate=20.
   Possible recovery via (a) per-contrast manifest reporting (current
   Q6-Option-B), (b) hierarchical pooling with the parent WMB class,
   (c) cohort expansion. Decision deferred to a follow-up phase.
4. **Many-to-many SEA-AD weighting refinement.** Equal-weight average
   across mapped supertypes is the v1 rule. If interpretation suggests
   marker-overlap weighting matters, revisit.

---

## 7. Provenance

- Empirical audits (2026-05-13): `/tmp/cluster_estimability_audit.csv`,
  `/tmp/cluster46_animal_cell_counts.csv` — move into the repo at
  step 2 (under `outputs/reports/snrna_integration/audit_2026-05-13/`).
- Predecessor scoping doc: previous version of
  `docs/audits/snrna_46cluster_pivot_scope.md` (replaced by this
  plan).
- Source taxonomy: `data/incytr/v2_46clusters/provenance/kr_cluster_id_key.csv`
  (Levy/Yuyu, 2025-07-21 release).
