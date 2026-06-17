# alz/reference/

Shared reference-atlas and per-cohort expression utilities. Outputs feed
Mode 1 (bulk MEA cell-type attribution) and Mode 4 (cross-reference
correlation).

| File | Purpose | CLI entry |
|------|---------|-----------|
| `atlas.py` | Download SEA-AD MTG Nebula h5ads + WMB-10Xv3 log2 matrices; gene-list helpers. | `python alz/reference/atlas.py --run` |
| `wmb_expression.py` | Per-class kinase/phosphatase expression on Allen WMB 10Xv3 (34 classes). | `python alz/reference/wmb_expression.py --run` |
| `human_expression.py` | Cross-cohort human expression aggregates (SEA-AD supertype, HBCA class). | `python alz/reference/human_expression.py` |
| `snrna_integration.py` | Song snRNA-seq integration (pseudobulk → specificity → concordance) on the Levy-t5 spine. | `python alz/reference/snrna_integration.py --run` |
| `snrna_proportions.py` | Per-(animal, cluster, gene) proportion weights from Song snRNA-seq (input to Mode 2 decomposition). | `python alz/reference/snrna_proportions.py --spine levy_t5` |
