# alz/shared/

Cross-mode configuration and small utilities shared by every analysis pipeline.

| File | Purpose |
| --- | --- |
| `config.py` | Single source of truth for paths, thresholds, WMB taxonomy, enrichment params, sample-filter parameter loader. Imported as `from alz.shared import config`. |
| `map_kinases_to_genes.py` | Regenerable kinase-abbreviation → gene-symbol cache (kinase_library → MyGene fallback → manual overrides sidecar). Run as `python alz/shared/map_kinases_to_genes.py`. Output: `data/derived/caches/kinase_to_gene_mapping.csv`. |

Anything that ends up imported by ≥2 of {`bulk_mea`, `decomposition_mea`, `incytr_pair`, `cross_reference`, `ctrl_outlier_audit`, `ingest`, `reference`, `viewer`} belongs here.
