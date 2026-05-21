# Data layout consolidation — 2026-05-21

Fix all real issues surfaced by the input inventory audit. Two themes:
1. **Correctness bugs** — wrong/stale files being read silently.
2. **Layout drift** — derived bridges scattered across `data/external/`, `data/incytr_frozen/`, and `data/datasets/`, with no canonical home.

Per repo policy: research pivots replace, no back-compat shims. After this work, the prior paths must be removed from code (artifacts may remain on disk as historical record).

---

## Scope

Eight problems, grouped into three phases.

### Phase 1 — Correctness fixes (high priority, ship first)

These produce wrong outputs today.

#### 1.1 Repoint `config.KLDATA_FILE` to the canonical Song-built file

**Problem.** `alz/config.py:540` resolves to `data/datasets/song/kinase/kldata.csv` — a 47 MB generic pre-built file from Feb 2025. The canonical Song/Yuyu-derived artifact is `kldata_pspy.csv` (2.5 MB, built by `alz/integration/build_yuyu_kldata.py`, symlinked into `data/derived/incytr_inputs/kldata.csv` for the R driver).

Readers of `config.KLDATA_FILE` that silently consume the wrong file today:
- `alz/atlas_reference.py:96` (`get_all_kinase_genes`) — used by every downstream stage that needs the kinase universe.
- `alz/map_kinases_to_genes.py:81` — rebuilds the kinase→gene cache from the wrong universe.
- `alz/wmb_expression.py:18` (docstring reference; consumes via `get_all_kinase_genes`).

**Fix.**
- `alz/config.py`: `KLDATA_FILE = os.path.join(SONG_WORKSPACE_DIR, "kinase", "kldata_pspy.csv")`.
- Add a hardfail in `atlas_reference.get_all_kinase_genes` if `KLDATA_FILE` does not exist — surface the build dependency on `build_yuyu_kldata.py`.
- After repointing, **rerun**: `pixi run python alz/map_kinases_to_genes.py` to regenerate the mapping cache against the correct kinase universe, then `pixi run all` to propagate. The current `kinase_to_gene_mapping.csv` was built off the wrong universe and may have spurious entries / missing entries.

**Risk.** Mapping cache delta. Inspect `data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv` diff before and after; expect a smaller universe (Song-specific) and potentially renamed entries.

**Cleanup.** Delete `data/datasets/song/kinase/kldata.csv` (the stale 47 MB file). Per "no back-compat on research pivots", do not keep it behind a `--legacy-kldata` flag.

#### 1.2 Resolve `170_gex_celltypes_00.h5ad` duplication

**Problem.** Same h5ad lives at:
- `data/datasets/song/transcriptomics/170_gex_celltypes_00.h5ad` (canonical, read by `config.SONG_H5AD_FILE`)
- `data/gdrive_shared/yuyu01/transcriptomics/scanpy/170_gex_celltypes_00.h5ad` (old FUSE-mount copy)

**Fix.** Confirm md5 match, then delete the FUSE-mount copy. Update any code or docs that still reference the `gdrive_shared/` path (grep first).

#### 1.3 Lucie 5xFAD path drift

**Problem.** `alz/lucie_5xfad_manifest.py` reads `data/raw/external/lucie_proteomics/`, but files are at `data/lucie_proteomics/` (old FUSE-mount path). The rclone ingest task (`pixi run ingest-lucie-proteomics`) has apparently never been run since the FUSE migration.

**Fix.** Decide:
- (a) Run `pixi run ingest-lucie-proteomics` to populate `data/raw/external/lucie_proteomics/`, then delete the old FUSE copy at `data/lucie_proteomics/`.
- (b) If the rclone task is broken or the data hasn't changed, move the existing tree: `mv data/lucie_proteomics data/raw/external/lucie_proteomics`.

Recommend (a) so the ingest task is exercised. Either way, the end state is: files live under `data/raw/external/lucie_proteomics/`, nowhere else.

#### 1.4 Mukesh IMAC `.csv.xlsx` accidental duplicate

**Problem.** `data/datasets/mukesh/phospho/IMAC/` contains the same peptide report as both `.csv` and `.csv.xlsx`.

**Fix.** Confirm the `.csv` is the file `ingest_mukesh.py` reads, then delete the `.csv.xlsx`. Grep for the filename in `alz/` to be safe.

---

### Phase 2 — Bridge consolidation (structural cleanup)

#### 2.1 Establish `data/derived/bridges/` as the single home for curated crosswalks

**Problem.** Derived bridges are scattered:
- `data/external/sea_ad/`: `cluster_to_seaad_supertype.csv`, `expression_by_supertype.csv`, `seaad_subclass_to_wmb_class.csv`
- `data/external/allen_abc/`: `wmb_subclass_to_class.csv`, `cluster_to_wmb_class.csv`
- `data/external/allen_hbca/`: `cluster_to_hbca_supercluster.csv`

These are not downloads. Mixing them with raw atlas files obscures provenance and makes "what is reproducible from external sources alone?" hard to answer.

**Fix.** Create `data/derived/bridges/` and move:
| From | To |
|------|----|
| `data/external/sea_ad/cluster_to_seaad_supertype.csv` | `data/derived/bridges/cluster_to_seaad_supertype.csv` |
| `data/external/allen_abc/wmb_subclass_to_class.csv` | `data/derived/bridges/wmb_subclass_to_class.csv` |
| `data/external/allen_abc/cluster_to_wmb_class.csv` | `data/derived/bridges/cluster_to_wmb_class.csv` |
| `data/external/allen_hbca/cluster_to_hbca_supercluster.csv` | `data/derived/bridges/cluster_to_hbca_supercluster.csv` |

Update `alz/config.py` path constants and all readers (`kinase_attribute.py`, `pipelines/attribute/nodes.py`, `human_celltype_attribution.py`, `human_reference_expression.py`, `wmb_expression.py`, `atlas_reference.py`, `integration/config_integration.py`).

**Keep in `data/external/`:** raw downloads only — `effect_sizes*.h5ad`, the raw SEA-AD nuclei h5ad, the WMB expression matrices, `wmb_class_manifest.csv` (downloaded with the WMB pull).

**`expression_by_supertype.csv` — decide separately.** It's a derived aggregate over the raw SEA-AD nuclei h5ad. Two reasonable homes:
- `data/derived/aggregates/seaad/expression_by_supertype.csv` (treats it as a heavyweight derived dataset, not a bridge)
- `data/derived/bridges/` (lumps it in)

Recommend the `aggregates/` home — it's a 139-supertype × gene matrix, not a crosswalk.

#### 2.2 Move the MyGene API cache out of `data/datasets/song/`

**Problem.** `data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv` is an API cache, not Song-specific data. It will be regenerated in Phase 1.1 anyway.

**Fix.** Move to `data/derived/caches/kinase_to_gene_mapping.csv`. Update `config.KINASE_GENE_MAP_FILE` (and any callers in `map_kinases_to_genes.py`, `atlas_reference.py`).

#### 2.3 Delete orphan `seaad_subclass_to_wmb_class.csv`

**Problem.** Present at `data/external/sea_ad/seaad_subclass_to_wmb_class.csv`, no readers in `alz/`. Either superseded by `wmb_subclass_to_class.csv` or its consumer was removed.

**Fix.** Confirm zero references (`grep -rn "seaad_subclass_to_wmb_class" alz/ docs/`). Delete.

#### 2.4 Clean up stale `kldata.csv` copies

**Problem.** `kldata.csv` (or `kldata_pspy.csv`) exists in 8+ locations: `song/kinase/kldata.csv` (stale, removed in 1.1), `5xFAD/kinase/kldata_pspy.csv`, `seed_probe/`, `incytr_frozen/v1_8clusters/`, `incytr_frozen/v2_46clusters/`, `incytr_frozen/shared/kldata_5xad_fallback.csv`, `gdrive_shared/`.

**Fix.** Audit each:
- `data/datasets/song/kinase/kldata_pspy.csv` — **keep** (canonical).
- `data/derived/incytr_inputs/kldata.csv` — **keep** (symlink, used by R driver).
- `data/datasets/5xFAD/kinase/kldata_pspy.csv` — delete if `5xFAD` is not part of the live pipeline (CLAUDE.md only references it as a demo).
- `data/incytr_frozen/v1_8clusters/`, `v2_46clusters/`, `shared/`, `seed_probe/`, `gdrive_shared/` copies — frozen historical record; **keep** if referenced as provenance, **delete** otherwise. Default: keep `v2_46clusters/` (active spine provenance), delete the rest.

---

### Phase 3 — Documentation

#### 3.1 Update CLAUDE.md "Key Data Files" section

- Fix the stale claim that `song/proteomics/source/` median CSVs were deleted on 2026-05-07 (they're still present).
- Add `data/derived/bridges/` and `data/derived/aggregates/seaad/` to the inventory.
- Note new `KLDATA_FILE` target.
- Note `data/raw/external/lucie_proteomics/` as the canonical Lucie home.

#### 3.2 Add a top-level `data/README.md`

Single page describing the layout:
```
data/
├── datasets/              # raw collaborator drops (Song, Mukesh, 5xFAD)
├── external/              # raw downloads from public sources (SEA-AD, WMB, HBCA)
├── derived/
│   ├── bridges/           # curated crosswalks built once from raw sources
│   ├── aggregates/        # heavyweight derived matrices (SEA-AD supertype expression)
│   ├── caches/            # API caches (MyGene)
│   └── incytr_inputs/     # R driver inputs (symlinks + built tables)
├── incytr_frozen/         # run-once snRNA spine artifacts (levy_t5 + provenance)
└── raw/external/          # rclone ingest targets (Lucie 5xFAD)
```

State which tier each kind of artifact belongs to, so future additions go to the right place.

---

## Order of operations

1. **Phase 1.1 first** — it's a correctness bug. Land alone, regenerate mapping cache, run full pipeline, verify `kinase_to_gene_mapping.csv` delta is sensible, commit.
2. **Phase 1.2–1.4** — small independent cleanups, batch into one commit.
3. **Phase 2.1** — bridge moves + config updates + reader updates. Single commit. Run `pixi run all` end-to-end; nothing should change in outputs (same files, new paths).
4. **Phase 2.2–2.4** — smaller cleanups, batch.
5. **Phase 3** — docs.

Each phase ends with: `pixi run all` clean, `git grep` for any stale path string returns zero.

## Out of scope

- Reorganizing `outputs/reports/` (already coherent).
- Reorganizing `data/datasets/song/primary/` (raw, untouched).
- Lucie 5xFAD pipeline itself (only the path fix in 1.3).
- The CLAUDE.md `Layer-2 drive access` section (already accurate).

## Success criteria

- `grep -rn "kldata.csv" alz/` returns only references to the symlink target or `kldata_pspy.csv`.
- `data/external/` contains only raw downloads.
- `data/derived/bridges/` contains every curated crosswalk and nothing else.
- `pixi run all` runs clean end-to-end.
- `pixi run python alz/map_kinases_to_genes.py` regenerates the cache against the Song-specific kinase universe.
- CLAUDE.md and `data/README.md` describe the layout accurately.
