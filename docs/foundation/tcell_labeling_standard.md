# T-Cell Labeling Standard

The canonical way T-cells are assigned a state in this project: **per-cell marker
assignment**. One label per cell, derived independently from that cell's own
markers. Native Seurat clusters do not set state, and ProjecTILs does not set
state.

This is the **current default, not a lock** — re-running `pixi run tcells-label`
re-derives everything downstream, so a deliberate future re-labeling is a
supported operation.

Vocabulary glossary for the ProjecTILs reference nomenclature is a separate
document: [`tcell_reference.md`](tcell_reference.md). That is *not* the label set
used here — see "Two vocabularies" below.

---

## Why per-cell, not by-cluster

By-cluster annotation was evaluated and **rejected as day-confounded**: native
Seurat clusters track collection day, so cluster-occupancy labeling called 59.6%
of donor1 day-2 and 69.2% of donor2 day-2 cells "activated" on cluster membership
alone. Per-cell assignment removes that coupling — a cell is activated because its
own activation modules are detected and strong, not because of the cluster it
landed in.

Markers are deliberately **cycle-independent** (`per_cell_marker_genes()`), so
proliferation does not masquerade as activation.

---

## The pipeline

Producer: `alz/analysis/tcell_state_labels.py`, run via `pixi run tcells-label`
(→ `alz/analysis/run_tcell_labeling.sh`).

1. **Lineage from protein, not RNA.** Raw CITE-seq ADT counts decide CD4 vs CD8:
   whichever of `CD4_protein_umi` / `CD8_protein_umi` is larger *and* exceeds the
   `mouse_isotype_umi` background. Only when the antibody counts are inconclusive
   does it fall back to the donor's native-cluster lineage map (`CLUSTER_LINEAGE`).
   The call and its provenance are both exported (`lineage`, `lineage_source`).
   Designated contaminant clusters short-circuit to `contaminant`.

2. **Standardize within donor × lineage.** Marker expression is z-scored across
   the cells of that group. Zero-variance genes are held at 0 rather than dividing
   by zero.

3. **Exhaustion is adjudicated first, hierarchically.** The late-exhaustion
   aggregate (`HAVCR2, LAG3, ENTPD1, TOX, NR4A1`) must be positive **and** exceed
   both the acute-activation aggregate (`CD69, IL2RA, TNFRSF4, ICOS, CD40LG`) and
   the effector-function aggregate (`GZMB, PRF1, IFNG, TNF`). This ordering stops
   an activated effector from being read as exhausted merely for carrying some
   inhibitory receptors.

4. **Otherwise, detection gates eligibility.** A state is eligible only if **every**
   one of its defining positive modules is *directly detected* — raw count > 0 in
   at least one gene of each module. No imputation, no partial credit.

5. **Rank eligible states by their weakest positive module.** Each candidate scores
   the minimum of its standardized positive-module means; the maximum of those wins.
   Exact ties break on expected-low (negative) module evidence. Still tied, or no
   eligible state → the cell keeps the bare lineage label `CD4` / `CD8`.

6. **Collapse, then map to the Incytr type.** `CD8 precursor exhausted (TPEX)` is
   collapsed into `CD8` — TPEX is deliberately not an exported state. The
   human-readable `label` then maps to an alphanumeric `type` (Incytr cannot take
   cell-type names containing spaces or punctuation), which is why
   `CD8 exhausted (TEX)` → `CD8Exhausted`.

Internal signed module values are classification mechanics. They are **not**
exported as biological scores.

---

## The 12-value type vocabulary

Ten named states (5 per lineage) plus the two bare-lineage fallbacks:

| CD4 | CD8 |
|---|---|
| `CD4` (fallback) | `CD8` (fallback) |
| `CD4NaiveLike` | `CD8NaiveLike` |
| `CD4RestingMemory` | `CD8RestingMemory` |
| `CD4ActivatedEffector` | `CD8ActivatedEffector` |
| `CD4Cytotoxic` | `CD8CytotoxicEffector` |
| `CD4ExhaustionAssociated` | `CD8Exhausted` |

CD4's exhaustion state is named *exhaustion-associated* rather than *exhausted*:
the canonical exhaustion program is CD8 biology, and the CD4 call marks the same
module pattern without asserting the same terminal state.

### Two vocabularies — do not conflate

- **This 12-value `type` set** is what the pipeline assigns and what Incytr and
  the viewer consume. It is enforced by the validator.
- **`tcell_reference.md`** documents ProjecTILs nomenclature (`CM`, `EM`, `TEMRA`,
  `TPEX`, `TEX`, `MAIT`, `Tfh`, `Th17`, `Treg`, `CTL_*`). Those names appear only
  in the `projectils_*` evidence columns. **ProjecTILs is corroboration-only — it
  never sets a label.** `projectils_quality` records exact reference-neighborhood
  unanimity (confidence == 1.0) with no tuned cutoff.

---

## Enforcement

The standard is enforced in code, not by convention: **one producer, one
validator**. Every raw-RDS consumer routes through
`alz/ingest/tcells_state_labels.R :: load_tcell_state_labels()`, which hard-fails
— never warns, never coerces — on:

- missing required columns (`barcode, donor, seurat_cluster, day, lineage, label, type`)
- blank or duplicated barcodes, on either side
- **barcode-set drift** — any cell in the Seurat object without a label, or any
  label without a cell
- alignment failure after the `match()` reorder
- `donor` disagreeing with the requested donor
- `day` or `seurat_cluster` disagreeing with the Seurat metadata
- blank `label`
- blank `type` not corresponding **exactly** to the `contaminant` cells
- non-alphanumeric `type` (Incytr constraint)
- any `type` outside the 12-value vocabulary

The producer carries the mirrored guards: non-contaminant cells must have a type,
contaminants must not, and the donor's observed cluster roster must match
`CLUSTER_LINEAGE` exactly — so **re-clustering raises rather than silently
remapping lineage**.

There is no competing live labeling path.

---

## Artifacts and downstream

| What | Path |
|---|---|
| Per-cell labels | `outputs/reports/tcell_labeling/cells/{donor}_state_labels.csv` |
| Evidence report | `outputs/reports/tcell_labeling/tcell_state_labeling_evidence_percell.html` |
| Per-cell evidence tables | `outputs/reports/tcell_labeling/percell_evidence/` |
| Label-comparison UMAP | `outputs/reports/tcell_labeling/umap/umap_percell_label_comparison.png` |
| Incytr production root | `outputs/reports/incytr_pair_mode_tcells` |

`alz/build_tcell_viewer.py` **asserts** `incytr_pair_mode_tcells` is the resolver
default and fails the build otherwise, so the viewer cannot drift onto a stale
labeling root. Contaminant cells are dropped downstream (`keep = !contaminant`).

Donors are `donor1` and `donor2`. Note that within-cohort t-cell work carries a
separate 10% detection floor — see `specificity_confidence.md`.
