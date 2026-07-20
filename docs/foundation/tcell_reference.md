# Human T-Cell Nomenclature Reference

Glossary for the **ProjecTILs reference** nomenclature. These names appear only in
the `projectils_*` evidence columns — ProjecTILs is corroboration-only and never
sets a label. The vocabulary the pipeline actually assigns is the 12-value `type`
set in [`tcell_labeling_standard.md`](tcell_labeling_standard.md); the two are
different and must not be conflated.

## CD8+ T-Cells

* **NaiveLike**: T cells with naive-like phenotype
* **CM**: Central memory
* **EM**: Effector memory
* **TEMRA**: Effector memory re-expressing CD45RA
* **TPEX**: Precursor exhausted T cells
* **TEX**: Terminally exhausted T cells
* **MAIT**: Mucosal-associated invariant T cells

## CD4+ T-Cells

* **NaiveLike**: T cells with naive-like phenotype
* **Tfh**: T follicular helper cells
* **Th17**: Th17 helper cells
* **Treg**: T regulatory cells
* **CTL_EOMES**: Cytotoxic CD4 T cells expressing EOMES and GZMK
* **CTL_GNLY**: Cytotoxic CD4 T cells expressing GNLY
* **CTL_Exh**: Cytotoxic CD4 T cells with exhaustion phenotype
