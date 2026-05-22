# `alz/cross_reference/` — Mode 4: cross-cohort / cross-reference evidence

Holds the modules that compare per-cohort kinase MEA results against
external transcriptomic references — SEA-AD MTG, Allen HBCA,
WMB-10Xv3, Song snRNA-seq — to attribute or sanity-check findings at
cell-type resolution.

## File inventory

| File | Role |
|---|---|
| `seaad_human_agreement.py` | Cohort-level SEA-AD LFC per kinase for the human (NBB/Mukesh) cohort. Collapsed across the 139 MTG supertypes; supplies the viewer's human cell-type agreement panel. |
| `human_celltype_attribution.py` | Top-N specific cell types per kinase from SEA-AD MTG + Allen HBCA. Consumes specificity matrices from `alz/human_reference_expression.py`; emits `celltype_specificity.csv` for the viewer payload. |
| `evidence.py` | Shared evidence loaders for the mouse Stage 3 attribution: SEA-AD concordance (per-(kinase, contrast, cluster) weighted-mean LFC), WMB specificity, Song specificity, Song concordance. Imported by `alz/bulk_mea/attribute.py`. |

## Why these live together

All three operate on the same axis — *cross-reference comparison of a
kinase signal against a transcriptomic atlas*. The mouse-side functions
(in `evidence.py`) and the human-side scripts share crosswalks
(`cluster_to_seaad_supertype.csv`, etc.), pathway-strata mappings, and
the "weighted mean across many-to-many supertype links" idiom. Keeping
them in one module makes that vocabulary easy to find and re-use.
