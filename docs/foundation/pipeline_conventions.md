# Pipeline Conventions

Sources: `project_lfc_sign_convention.md`, `project_mechanism_after_attribute.md`, `project_direct_levy_t5_mapping.md`

## LFC / NES / sclog2FC / PDS sign convention

**Canonical: positive = up in disease, negative = up in WT.** No sign flips anywhere between raw OLS / Incytr output and the viewer payload.

| Column | Source | Direction |
|---|---|---|
| MEA OLS β | `alz/bulk_mea/enrich.py` (`_build_design_matrix`) | WT=0, disease=1 dummy → β = E[Y\|disease] − E[Y\|WT] |
| MEA `lfc` in `site_level_ols.csv` | `alz/bulk_mea/enrich.py` (`_run_ols_all_sites`) | same |
| MEA `NES` | `alz/bulk_mea/enrich.py` (`_run_mea`) | prerank descending on stoich β → +NES = more active in disease |
| Incytr `*_sclog2FC`, `*_pr_log2FC`, etc. | `incytr/R/math.R:53`, driver `incytr_commandline.R:107-108,199-200` | c1=disease, c2=WT → log2(disease/WT) |
| Incytr `PDS`, `TPDS`, `PPDS`, `PhPDS_*` | `incytr/R/evaluation.R:67-87,376-379` | logi-transformed aFC → + = pathway up in disease |

Note: stoichiometry = log2(phospho) − log2(protein), so positive stoich β = phosphosite more abundant relative to parent protein in disease — not about absolute phospho level.

When adding a new LFC-style column, default to this convention (condition1=disease, condition2=WT). If a sign flip is ever introduced, document it inline. Do not add a flag to flip convention; if the canonical direction changes, rewrite everything in one pass (anti-shim).

## Bulk pipeline order: attribute → mechanism → recover

`alz/bulk_mea/mechanism.py` must run **after** `attribute.py`. The dependency is counter-intuitive: `attribute.py` does NOT read any mechanism output; instead, `mechanism.py:166-171` reads `unified_attribution.csv` (attribute's output) and merges mechanism annotations back, guarded by `if os.path.exists(unified_path)`. If mechanism runs first on a clean build, the guard is False → merge silently skipped → attribute then overwrites with no mechanism columns.

Canonical order: `pixi run live` = `[ingest, normalize, enrich, attribute, mechanism, recover]`.

`run_all.sh` runs K-attr before K-mech. Mechanism's standalone outputs (`mea_raw_phospho.csv`, `mechanism_annotation.csv`) are produced in any order — only the merge into `unified_attribution.csv` is order-sensitive.

## Reference crosswalk to levy_t5: always direct

When joining any reference annotation (WMB 34-class specificity, SEA-AD supertype LFC, HBCA supercluster expression, etc.) onto the levy_t5 cell-type spine, build a **direct** crosswalk — one CSV mapping that reference vocabulary's cell-type → levy_t5 cluster. Do not chain through intermediate vocabularies (e.g. supertype → WMB-class → levy_t5): chained crosswalks compound mapping errors, lose resolution at each hop, and obscure which reference cell types actually map to which cluster.

Many-to-one mappings (multiple reference cells → one levy_t5 cluster) are expected — aggregate (mean/median/max) at write time.

The first artifact to build for any new attribution annotation is the direct crosswalk CSV.
