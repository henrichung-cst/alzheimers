# Unified Kinase-First View — Hurdles Assessment

Status: assessment of structural obstacles before designing the merged view.
Scope: combining bulk MEA, bulk-MEA reference attribution, and per-cell-type MEA into a single kinase-first viewer surface.

## What we are trying to combine

Three distinct evidence channels, all keyed by kinase:

| Channel | Source | Output grain | What it answers |
|---|---|---|---|
| (1) Bulk MEA | `kinase_attribution.py` → `mea_stoichiometry.csv` | kinase × contrast | Is this kinase modulated overall in disease? |
| (2) Bulk-MEA reference attribution | `kinase_attribution.py --attribute` → `unified_attribution.csv` | kinase × cell type × contrast (inferred) | Given (1) fires, which cell types likely host it, based on WMB specificity + SEA-AD LFC + Song LFC? |
| (3) Per-cell-type MEA | `alz/deconvolution/run_per_animal.py` → `kinase_enrichment_46clusters.csv` + `kinase_enrichment_wmb_rollup.csv` | kinase × cell type × contrast (measured via proportion proxy) | Is this kinase modulated within this cell type? |

Channels (2) and (3) both carry a cell-type axis but reach it by different routes:
- (2) is reference-driven: it accepts (1) as a real bulk hit and uses external reference data to point at which cell types likely host the activity.
- (3) is direct measurement: it splits the per-animal stoichiometry across clusters via Yuyu's proportion proxy, runs MEA inside each cluster, and reports a per-cell-type NES.

These are complementary, not competing estimates of the same thing.

## Hurdle 1 — Annotation-layer mapping problem (resolved CTM-native, 2026-05-04)

### Update — adopted resolution

The "Defensible options" table below is historical. The decomposition pipeline
now runs natively on the WMB-class spine: `alz/deconvolution/build_wmb_decomposition.py`
aggregates raw counts directly from the snRNA-seq h5ad on Allen CTM `class_name`,
applies the proportional formula, and emits `*_wmb_decomposition.csv` directly
on the WMB axis. There is no soft-mass projection step in the live path; the
46-cluster crosswalk + projection module are archived under
`alz/deconvolution/_archive/`. The question shifted from "how to project
46→27" to "why project at all" — re-aggregating on the CTM axis is mechanically
identical to Yuyu's proportional formula but removes a layer of interpretation.

Coverage with the live h5ad (after `class_prob ≥ 0.9` filter) is **24 / 34**
WMB classes; the 10 absent (15, 16, 21, 22, 23, 24, 25, 26, 27, 28) are
biological-sampling gaps. The original "27 / 34" figure was derived from a
cross-tab against barcodes that included nuclei outside the gated h5ad.



### Original framing (wrong)

The first pass of this assessment claimed the 12-of-34 WMB-class gap was a sampling problem — that the snRNA-seq simply did not capture nuclei from the missing 22 classes. That framing is incorrect.

### Corrected observation

The deconvolution pipeline and `snrna_integration.py` work from the **same nuclei** (Yuyu and Song are the same dataset). The two layers diverge at annotation:

| Annotation layer | Per-nucleus key | WMB-class reach |
|---|---|---|
| Yuyu's hand-curated 46-name labels (used by deconvolution proportion proxy) | `Idents` on `renamed_sobj.rds` | **12 / 34** + 15 unnamed `cluster-XX` collapsed to "Unclassified" |
| Allen Cell Type Mapper `class_name` (used by `snrna_integration.py`) | `class_name` on `170_gex_celltypes_00.h5ad` | **27 / 34** (CTM ceiling for this tissue) |

Same 63,695 nuclei. The 12-vs-27 delta is the lossy hand-annotation step. Re-mapping the 46 deconvolution clusters via Allen CTM (single auditable reference, applied per nucleus) recovers most of the gap.

### Root cause of the lossy step

`alz/deconvolution/yuyu_46_to_wmb_class.csv` is a 46 → WMB-class crosswalk built by hand:

- 31 named populations at uneven granularity ("Astrocytes", "VIP-positive-interneuron"), each force-mapped to one WMB class.
- 15 unnamed `cluster-XX` leftovers blanket-labeled "Unclassified" rather than mapped.

Several CTM-resolved classes (`05 OB-IMN GABA`, `08 CNU-MGE GABA`, `13 CNU-HYa Glut`, `32 OEC`) are absent from the hand crosswalk despite carrying real nucleus mass. The Allen CTM `class_name` per nucleus is the single point of reference that exposes them.

### Non-destructive remap (implemented)

`alz/deconvolution/extract_yuyu_obs.R` extracts per-nucleus metadata from `renamed_sobj.rds` (joined to the 170 h5ad on barcode → 63,695 nuclei). `alz/deconvolution/remap_clusters_via_ctm.py` builds two new files alongside the original (which is left untouched):

- `yuyu_46_to_wmb_class_v2.csv` — hard plurality assignment per cluster, with audit columns: `plurality_fraction`, `n_nuclei`, `second_class`, `second_fraction`, `wmb_class_v1_hand`, `agrees_with_v1`.
- `yuyu_46_to_wmb_class_soft.csv` — full mass matrix `cluster × WMB-class`, row-normalized to fractions over 27 CTM-reachable classes.

### Coverage tiers

| Approach | WMB-class coverage |
|---|---|
| v1 hand annotation | 12 / 34 |
| v2 CTM hard plurality | **16 / 34** |
| Soft mass projection (every CTM-non-zero class gets weighted signal) | **27 / 34** |
| True biological sampling gap (zero CTM nuclei) | 7 / 34: `15 HY Gnrh1 Glut`, `16 HY MM Glut`, `21 MB Dopa`, `22 MB-HB Sero`, `23 P Glut`, `25 Pineal Glut`, `28 CB Granule` |

### Audit notes

- 18 of 46 clusters keep their v1 WMB-class assignment under v2; 28 change. All major glia (Astrocytes, OPC, Oligo, Microglia, Endothelial, Pericyte, VLMC, Choroid, Ependymal) and the named cortical excitatory clusters are stable.
- Notable v1→v2 movements at high plurality: `Erbb4-VIP-inhibitory-neurons` (8,091 nuclei) `06 CTX-CGE GABA → 05 OB-IMN GABA` at 99.9% plurality; `Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic` `06 → 05` at 100%; `Ptprz1-protoplasmic-astrocytes` `30 Astro-Epen → 32 OEC` at 99.8%. These need a biology read before adoption — Allen CTM places them differently than the hand annotation, and the cross-tab is unambiguous, but downstream interpretation should be checked.
- Low-plurality / ambiguous clusters that warrant the soft mass matrix rather than hard plurality: `cluster-90` (28% IT-ET vs 23% DG-IMN), `glutamatergic-excitatory-neurons` (32% CNU-HYa GABA vs 20% TH Glut), `Inhibitory-Neurons` (31% CNU-MGE GABA vs 20% LSX GABA), `Glutamatergic-...-Cortical-layer-2-4-pyramidal-neurons` (54% CNU-HYa Glut vs 40% OB-CR Glut).

### Defensible options for the merged view

| Option | Coverage | Auditability | Cost |
|---|---|---|---|
| **A — Soft mass projection across 27 CTM-reachable classes; label remaining 7 as "outside snRNA-seq sampling"** | Per-cluster NES projected onto 27 WMB classes via the mass matrix; the 7 truly-absent classes shown explicitly empty. | Allen CTM per nucleus; mass matrix reproducible from the join. | Reader must understand soft-projection semantics; per-cluster NES is no longer 1:1 with one WMB class. |
| **B — Hard plurality (v2) across 16 reachable classes; label remaining 18 (7 absent + 11 reachable-only-via-soft) as not evaluated** | Direct per-cluster → WMB-class assignment; reconciliation panel populated for 16 classes. | Same Allen CTM source; plurality + plurality_fraction in audit columns. | Loses 11 classes' worth of mass that exists but never wins plurality. |
| C — Stay on v1 hand crosswalk (12 classes) | Status quo. | Hand-curated, not anchored to a published reference. | Strictly worse on coverage and auditability than A or B. |

### Recommendation

**Option A** (soft mass projection) for maximum coverage and auditability, with **Option B** (hard plurality v2) as a simpler fallback if downstream consumers cannot consume soft mass. In either case, the 7 truly-absent classes are labeled "outside snRNA-seq sampling" and the original `yuyu_46_to_wmb_class.csv` is left in place until adoption is signed off.

## Hurdle 2 — Bulk MEA and per-cell-type MEA answer different questions; they are not two estimates of one quantity

### Observation

Earlier framing called this a "two bulk MEAs disagreeing on 32% of hits" problem. That framing was wrong.

- Bulk MEA hits (channel 1, raw stoichiometry, one MEA across all sites): 1,196 (kinase, contrast) pairs at FDR<0.25, ST track.
- Per-cell-type MEA hits (channel 3, any cluster reaching FDR<0.25): 1,180 (kinase, contrast) pairs.
- Overlap: 808 (68%).

The two sides are **not running two versions of the same bulk MEA**. They are running:

- (1) one MEA on raw stoichiometry — answers "is this kinase enriched across the whole tissue?"
- (3) 46 separate MEAs on proportion-projected stoichiometry, one per cluster — answers "is this kinase enriched within this specific cell-type population?"

### Why hits differ

These are different statistical questions. The disagreements are biologically interpretable, not pipeline noise:

- **388 kinases hit bulk but not any cluster:** the bulk signal is broadly distributed across many cell types; no single cluster carries enough signal to reach significance on its own. The bulk hit is real; it is just not cell-type-localized.
- **372 kinases hit at least one cluster but not bulk:** the cell-type-concentrated signal is real but small relative to the total bulk; aggregation across all cells washes it out. The cluster-level hit is real; bulk MEA could not see it.
- **808 kinases hit both:** signal is both broadly present and cell-type-concentrated.

These three patterns are themselves a useful classification, not a noise floor.

### Why directional agreement is high where both fire

Among the 962 (kinase, contrast, WMB_class) rows in both sides: **94.4% sign agreement on NES**. When both pipelines have something to say about the same kinase × cell type, they almost always agree on direction. This is reassuring — it means we are not looking at two random projections, we are looking at two views of related biology that converge when both can speak.

### Confidence cross-tab on the overlap

Joined ST rows, UA `combined_confidence` × decon `confidence`:

| UA conf ↓ / decon → | Insufficient | Moderate | NotExpressed | Supported |
|---|---|---|---|---|
| **high** | 0 | 7 | 0 | 2 |
| **moderate** | 13 | 281 | 7 | 50 |
| **low** | 0 | 0 | 1 | 0 |
| **none** (no UA confidence) | 14 | 429 | 132 | 26 |

Notable cells:
- UA "high" + decon "Supported" — both top tiers — only 2 rows. The two pipelines rarely both reach their top tier on the same target.
- "none" UA × decon Moderate (429 rows) — rows where the proportion-proxy + cohort sign-concordance fires but the bulk-attribution side has no transcript or specificity backing. This is the cell most in need of explicit annotation in the merged view: the two pipelines disagree, and the reader needs to know which is firing alone.

### Implication for the merged view

The two cell-type-resolved channels (2 and 3) should be presented **side by side as complementary readouts of different questions**, not as two estimates of the same quantity:

- Channel (2) — reference-driven: "given the bulk hit is real, which cell types likely host it based on baseline expression and snRNA-seq differential expression?"
- Channel (3) — direct measurement: "what does MEA say when run inside each cell type via the proportion proxy?"

Disagreement is informative:
- Channel (2) firing alone: bulk activity is attributable via reference data, but we have no direct cell-type measurement (either the cell type is outside snRNA-seq sampling or the proportion-proxy MEA missed it).
- Channel (3) firing alone: cell-type-specific signal exists in the proportion-proxy projection but is not reflected in baseline expression or transcript change in that cell type — possibly post-translational, possibly proxy artifact.
- Both firing: triangulation across two structurally independent attributions to the same cell type.

## Cluster→WMB rollup loss (smaller issue, worth noting)

`alz/deconvolution/rollup_wmb.py` picks one cluster per (kinase, contrast, WMB_class, track) group by max|NES| (ties broken by lowest FDR), among bulk-sig clusters only.

- 1,664 rollup rows; 985 of these (59%) draw from groups with ≥2 bulk-sig clusters in the same WMB class.
- Of those 985, **23 (2.3%) have mixed NES signs across clusters in the same WMB class**.
- Of those 23 mixed-sign groups, **rollup picked the minority direction in 21/23** — max|NES| favors the loudest cluster, not the consensus.

This is small (1.4% of rollup rows), but the bias is in a predictable direction. Worth surfacing in the merged view: where rollup direction conflicts with the within-WMB-class majority, flag it.

## What this assessment changes about the proposed merged view

1. **Cell-type axis = full 34 WMB classes**, with 22 explicitly labeled "outside snRNA-seq sampling — proportion proxy cannot evaluate" rather than rendered as missing data.
2. **Channels (2) and (3) sit side by side**, framed as different questions, not two estimates. The reader uses agreement as triangulation across independent attributions.
3. **(1) Bulk MEA is the anchor**, shown above the cell-type panel as the primary measurement. Disagreement between (1) and (3)'s any-cluster behavior is itself informative ("broadly distributed", "cell-type-concentrated", "both") and should be summarized in the kinase header.
4. **The 429 decon-fires-alone rows need explicit per-row annotation** in the merged view, not silent presentation alongside corroborating rows.
5. **Rollup direction-loss flags** should surface where the picked cluster disagrees with within-WMB-class majority direction.

## Out of scope for this assessment

- Implementation plan for the viewer extension (separate document).
- Re-running Yuyu's clustering or the snRNA-seq pipeline to recover missing WMB classes.
- Changing the definitions of channels (1)–(3); the assessment takes them as given.
- Performance/payload-size implications of adding decon-side columns to `attribution_index`.
