# Never-read sidecar cleanup

Stop producing `_raw`/`_all`/`_per_cluster` audit sidecars at their writers (`mukesh_perdonor.py`,
`tcells_perdonor.py`, `enrich_celltype.py`, `mukesh.py` `_concat`) and delete the on-disk copies.

Deferred out of the cohort-namespace refactor.
