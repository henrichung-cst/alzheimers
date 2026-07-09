# T-cell per-cell relabel → re-run pair-mode Incytr

**Goal:** Replace the 14-state ProjecTILs labeling in the T-cell Incytr input chain
with the evidence-backed per-cell labels from
`outputs/reports/tcell_labeling/cells/`, then re-run pair-mode Incytr for both
donors.

**Trigger:** The evidence report showed that the native Seurat clusters mix
proliferation and inhibitory-marker states. A cluster-level exhausted label is
therefore invalid: donor1 clusters 4 and 6, for example, are overwhelmingly
dividing. CD8 state is now assigned per cell from raw marker detection plus
cell-cycle phase; CD4 state and contaminant identity retain their cluster context.

## Locked labeling decision

- `CD8 exhausted` (donor1): G1 and at least two detected inhibitory receptors
  among HAVCR2/TIM-3, LAG3, ENTPD1/CD39, and PDCD1/PD-1.
- `CD8 TPEX` (donor2): TCF7 detected with at least one inhibitory receptor; these
  precursors may divide.
- `CD8 TEX` (donor2): not TPEX, G1, and at least two inhibitory receptors.
- `CD8 cytotoxic`: all remaining CD8 cells, including dividing activated
  effectors.
- Proliferation is retained as a separate categorical field and is never
  regressed out.
- TOX remains supporting raw evidence. The internal receptor-count gate is not
  exported as an analysis score.

The generated artifact passed these invariants:

- no donor1 `CD8 exhausted` or donor2 `CD8 TEX` cell is dividing;
- donor1 has no TPEX/TEX split;
- every donor2 TPEX cell detects TCF7 and inhibitory evidence;
- every input cell is represented exactly once before contaminants are dropped.

## Traced input chain

`Type` enters at exactly two existing builders. Everything downstream consumes
whatever vocabulary those builders emit.

| Step | Script | State-keyed? | Change needed |
|--|--|--|--|
| Label evidence | `tcell_export_marker_cells.R` → `tcell_state_labels.py` | **yes** | already emits one row per barcode with label, sanitized `type`, proliferation, and raw log-normalized marker evidence |
| Extract | `alz/ingest/tcells_scrna_extract.R` | **yes** — currently maps `functional.cluster` | join `{donor}_state_labels.csv` by barcode; use `type`; drop blank type (contaminants) |
| Seurat input | `alz/incytr_pair/build_tcells_seurat.R` | **yes** — currently maps `functional.cluster` | same barcode join; set `obj$Type` from `type`; drop ProjecTILs dependency |
| Deconvolution | `alz/ingest/tcells_decompose.py` | no | docstring only; its alphanumeric state parser accepts the new type names |
| Marker gene.use | `alz/incytr_pair/build_tcells_input_gene_list.R` | no | none; regenerates from rebuilt `Type_condition` identities |
| Bulk export | `alz.cohorts.tcells.ingest` | no | none; bulk remains per-day and gene/site keyed |
| Grid + filter | `run_pair_mode_tcells.sh`, `filter_significant_paths.py` | no | none; the grid derives from the rebuilt object’s `Type` levels |

## Decision: relabel replaces ProjecTILs in the input chain

This is not a sensitivity toggle. No compatibility flag, alternate output tree,
or second labeling branch is added. Historical ProjecTILs-based Incytr outputs
remain on disk as provenance, while the code path that builds new inputs is
re-keyed to the per-cell evidence artifact.

`alz/ingest/tcells_projectils_map.R` and the report’s ProjecTILs comparison remain
available for audit; only the Incytr input chain drops ProjecTILs.

## Single source of truth

`alz/analysis/tcell_state_labels.py` owns the biological rules, cluster context,
and sanitized type vocabulary. It writes:

- `outputs/reports/tcell_labeling/cells/donor1_state_labels.csv`
- `outputs/reports/tcell_labeling/cells/donor2_state_labels.csv`
- `outputs/reports/tcell_labeling/cells/cluster_context.csv`

The donor files are keyed by unique barcode (one pooled, HTO-demultiplexed 10x run
per donor) and contain `label`, `type`, `Phase`, `proliferation`, raw log-normalized
HAVCR2/LAG3/ENTPD1/PDCD1/TOX expression, and the negative-evidence precursor/resting
panel TCF7/LEF1/SELL/CCR7/IL7R. Generated report data remain uncommitted under the
repository’s ignore policy.

The report, UMAP script, and future Incytr builders read these files; no
cluster→state dictionary is duplicated in consumers.

### Sanitized type names and observed counts

`type` is alphanumeric because the driver splits `Type_condition` on `_`.

| Report label | Incytr `type` | donor1 n | donor2 n |
|--|--|--:|--:|
| CD8 cytotoxic | `CD8Cytotoxic` | 7,165 | 3,993 |
| CD8 exhausted | `CD8Exhausted` | 444 | — |
| CD8 TPEX | `CD8Tpex` | — | 5,593 |
| CD8 TEX | `CD8Tex` | — | 723 |
| CD4 activated | `CD4Activated` | 1,858 | 2,490 |
| CD4 activated / stress | `CD4ActivatedStress` | — | 2,913 |
| CD4 naive | `CD4Naive` | 3,043 | — |
| CD4 proliferating | `CD4Proliferating` | 9,090 | 3,083 |
| CD4 resting | `CD4Resting` | 3,744 | 1,755 |
| contaminant | blank / dropped | 334 | 104 |

Resulting grids per contrast:

- donor1: 6 retained types → 36 sender×receiver pairs;
- donor2: 7 retained types → 49 sender×receiver pairs;
- union across separately analyzed donors: 9 type names.

Day contrasts (`d<later>_vs_d2`) are unchanged.

## Phases (gated — stop for approval at each ▸)

**✓ Phase 1 — lock labels and report evidence (complete)**

- Generated raw per-cell marker exports under the 26 GB memory cap.
- Added `tcell_state_labels.py` and generated both donor artifacts.
- Migrated the evidence report and native UMAP plots to barcode-level labels.
- Rendered the report and verified label/accounting invariants.

**✓ Phase 2 — re-key the input builders (complete)**

- Added one shared R reader/validator for the canonical per-cell label artifacts;
  both raw-RDS consumers use it for complete barcode alignment and donor/day/cluster
  cross-checks.
- Replaced the `functional.cluster`/`LABEL_MAP` paths in `tcells_scrna_extract.R`
  and `build_tcells_seurat.R`; blank `type` rows now drop exactly the contaminant
  cells.
- Enforced sequential 26 GB memory-capped entrypoints for both raw-RDS builders.
- Regenerated `cell_counts.csv`, `aggexp_data.csv`, `pct_expressing.csv`, the
  scRNA `allmarkers.csv`, manifests/audits, and both `incytr_obj.rds` files.
- Rebuilt the root `allmarkers.csv` files from the new `Type_condition` identities;
  also repaired the pixi task's nested-`pixi` launcher failure.
- Verified donor1 = 25,344 retained / 334 contaminants / 6 types / 35 state-day
  groups; donor2 = 20,550 retained / 104 contaminants / 7 types / 34 state-day
  groups.
- `FindAllMarkers` emitted 33/35 donor1 and 32/34 donor2 `Type_condition`
  identities. The four omitted identities (`CD4Activated_d13`, `CD4Naive_d20`,
  `CD4Activated_d7`, `CD4Activated_d11`) each contain one cell and therefore have
  no positive one-vs-rest marker rows; all retained type names are represented.

**▸ Phase 3 — rebuild deconvolution inputs (not started)**

- Update only stale ProjecTILs wording in `tcells_decompose.py`.
- Run `pixi run tcells-decompose`.
- Verify manifest states, mass identity, nonnegative finite values, and
  `d<day>_<type>` column names.

**▸ Phase 4 — re-run pair-mode Incytr (not started; explicit go/no-go)**

- Smoke: `bash alz/incytr_pair/run_pair_mode_tcells.sh --smoke donor2` with nboot=2.
  Confirm 49 sender×receiver pairs and intact columns.
- Full: run both donors/all contrasts with nboot=100 in tmux with logging, then run
  `filter_significant_paths.py`.

**▸ Phase 5 — update provenance (not started)**

- Update `data/derived/tcells_incytr_inputs/INDEX.md` to describe per-cell evidence
  labels, the 6/7 type counts, and the removal of ProjecTILs from the input chain.

## Constraints carried

- Large `.rds` loads must use
  `systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G`.
- Use `pixi run` for Python, R, and Quarto.
- Do not commit generated data files.
- Do not regress out cell cycle.
- Stop at every ▸ phase boundary; Phase 2 requires explicit approval.
