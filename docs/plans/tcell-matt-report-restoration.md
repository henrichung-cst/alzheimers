# Matt T-cell method preservation and per-cell revision

## Purpose

Preserve Matt's reproducible marker analysis while preventing day-confounded
Seurat clusters from determining the experimental composition trajectory. The
current analysis assigns a cycle-independent marker state to each cell. Matt's
original report and AUROC reproduction remain historical method records.

## Current labeling contract

- Assignment unit: individual cell.
- Lineage: raw CD4/CD8 CITE-seq counts; native-cluster lineage is fallback only.
- State evidence: direct positive detection of non-cycle RNA programs, with
  expected-low modules used only to distinguish competing eligible states.
- Eligibility: every defining positive module must contain detected expression.
- Selection: the eligible state with the strongest weakest positive module wins;
  negative evidence breaks exact ties.
- TEX hierarchy: the aggregate HAVCR2/LAG3/ENTPD1/TOX/NR4A1 program must be above
  the donor-lineage mean and exceed both acute-activation and effector-function
  aggregates before other states are considered.
- Fallback: cells without a unique eligible state remain `CD4` or `CD8`; there is
  no `unclassified` label.
- Contaminants: explicit non-T clusters remain `contaminant`.
- Excluded: cycle genes, phase, cycle scores, and `% dividing` because proliferation
  was induced in silico.
- ProjecTILs: raw corroboration only; even confidence 1 never changes the label.

## State vocabulary

- `CD4`, `CD4 naive-like`, `CD4 resting/memory`, `CD4 activated/effector`,
  `CD4 cytotoxic`,
  `CD4 exhaustion-associated`
- `CD8`, `CD8 naive-like`, `CD8 resting/memory`, `CD8 activated/effector`,
  `CD8 cytotoxic effector`,
  `CD8 exhausted (TEX)`

The small provisional TPEX category is collapsed to `CD8` because its independent
reference corroboration is insufficient for a definitive precursor-exhausted call.

## Why per-cell assignment was restored

The native Seurat clusters are strongly associated with experimental day. Cluster
annotation made 59.6% of donor 1 day-2 cells and 69.2% of donor 2 day-2 cells
activated solely because those cells occupied day-specific activated clusters.
Several such clusters almost disappeared at later days. Per-cell marker assignment
retains within-cluster state heterogeneity and produces compositions directly tied
to the measured marker programs.

## Canonical artifacts

- Current report:
  `outputs/reports/tcell_labeling/tcell_state_labeling_evidence_percell.html`
- Labels:
  `outputs/reports/tcell_labeling/cells/{donor}_state_labels.csv`
- Evidence:
  `outputs/reports/tcell_labeling/percell_evidence/`
- Native UMAP comparison:
  `outputs/reports/tcell_labeling/umap/umap_percell_label_comparison.png`
- Preserved Matt report:
  `outputs/reports/tcell_labeling/tcell_state_labeling_evidence_matt.html`
- Pending compatible Incytr root:
  `outputs/reports/incytr_pair_mode_tcells_percell_posneg`

## Run order

```bash
pixi run tcells-label
pixi run tcells-scrna-extract
pixi run tcells-decompose
pixi run tcells-build-incytr-seurat
```

The viewer remains on the last complete historical root until the per-cell Incytr
root completes successfully.
