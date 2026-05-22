# Plan: Critical audit fixes (2026-05-22)

Punch list of the 5 critical issues from the 2026-05-22 cross-file audit. Items
ordered by blast radius — break-the-build first, stale-data next, contract gaps
last.

## C1 — Fix broken `import config` in `alz/shared/map_kinases_to_genes.py`

- **File**: `alz/shared/map_kinases_to_genes.py:32`
- **Symptom**: bare `import config` left over from the `fb7795d` reorg.
  `ImportError` from the repo root; `run_all.sh` step `K-map` and any caller
  in `alz/bulk_mea/attribute.py` will fail. The Kedro entry was masked because
  the cache happens to exist on disk.
- **Change**:
  - Replace `import config` with `from alz.shared import config` (parents-up
    `sys.path` shim is not needed — the package is importable from repo root).
  - Update the usage line in the module docstring (`alz/map_kinases_to_genes.py`
    → `alz/shared/map_kinases_to_genes.py`).
  - Sweep `alz/reference/atlas.py:87,95` for the same stale path string in its
    docstring + error message.
- **Verify**: `pixi run python alz/shared/map_kinases_to_genes.py --help` (or
  `python -c "from alz.shared import map_kinases_to_genes"`) imports cleanly.

## C2 — Drop hardcoded `"levy_t5"` in `alz/viewer/paths.py`

- **File**: `alz/viewer/paths.py:68`
- **Symptom**: `TRANSCRIPT_TRACE_PSEUDOBULK` interpolates the literal
  `"levy_t5"`, while the sibling `DECOMP_OLS_PARQUET` at line 29 already uses
  `config.CLUSTER_SPINE_NAME`. If `CLUSTER_SPINE_NAME` changes the transcript
  trace silently points at a non-existent path.
- **Change**: substitute `config.CLUSTER_SPINE_NAME` for the literal.
- **Verify**: `python -c "from alz.viewer import paths; print(paths.TRANSCRIPT_TRACE_PSEUDOBULK)"`
  yields the same path as before. Grep the repo for any other `"levy_t5"`
  literals in viewer / integration code as part of the same pass.

## C3 — Regenerate `cluster_to_seaad_supertype.csv`

- **File**: `data/derived/bridges/cluster_to_seaad_supertype.csv` (mtime
  2026-05-17) vs `alz/integration/build_seaad_bridge.py` (mtime 2026-05-22).
- **Symptom**: bridge generator was modified after the artifact was written;
  on-disk CSV may not reflect the current `SUBCLASS_MAP`. **Do not regenerate
  until C5 is resolved** — the current SUBCLASS_MAP only covers 19 of 31 spine
  clusters (see C5). Regenerating now would entrench the partial mapping.
- **Change**: deferred behind C5.

## C4 — Regenerate `data/datasets/song/kinase/kldata_pspy.csv`

- **File**: `data/datasets/song/kinase/kldata_pspy.csv` (mtime 2026-05-16) vs
  `alz/integration/build_yuyu_kldata.py` (mtime 2026-05-22).
- **Symptom**: the canonical kldata that the Incytr R driver reads (symlinked
  into `data/derived/incytr_inputs/kldata.csv`) predates the latest builder
  edits. Any Incytr pair-mode run is using stale kinase-substrate mappings.
- **Change**:
  - `pixi run python alz/integration/build_yuyu_kldata.py` (or whatever the
    repo task name is — `pixi task list | grep yuyu` to confirm).
  - Diff the new CSV against the old (`git diff --no-index` or `csvdiff`) and
    eyeball row/column counts; commit if non-trivial.
  - Ensure the `data/derived/incytr_inputs/kldata.csv` symlink still resolves
    after the rebuild.
- **Verify**: shasum the symlink target matches the freshly built file; mtime
  is now ahead of the builder.

## C5 — Extend `SUBCLASS_MAP` in `build_seaad_bridge.py` to 31 clusters

- **File**: `alz/integration/build_seaad_bridge.py:3, 23–43`
- **Symptom**: the Levy-t5 spine has 31 clusters
  (`data/incytr_frozen/v2_46clusters/spines/levy_t5/cluster_spine.csv`); the
  `SUBCLASS_MAP` only enumerates 19. The 12 missing clusters (e.g.
  `Ptprz1-protoplasmic-astrocytes`, `Basal-Ganglia-GABAergic-Neurons`,
  `Vascular-Leptomeningeal-Cells`, `Inhibitory-Neurons`, `Ependymal-cell`,
  `GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4`,
  `GABAergic inhibitory interneurons`,
  `Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin`,
  `Choroid-Plexus-Epithelial-Cells`,
  `Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons`,
  `GABAergic-inhibitory-interneurons-VIP-positive`, `Cholinergic-Neurons`)
  silently produce no SEA-AD evidence — violates the
  [[project_direct_levy_t5_mapping]] "no chained mappings / no silent drops"
  rule.
- **Change**:
  - Update module docstring "19 spine clusters" → "31 spine clusters".
  - Add an explicit entry (mapped or `None` with a `NA_REASONS` justification)
    for each of the 12 missing cluster names. Suggested first pass — confirm
    against SEA-AD MTG taxonomy before committing:
    - `Ptprz1-protoplasmic-astrocytes` → `["Astrocyte"]`
    - `Vascular-Leptomeningeal-Cells` → `["VLMC"]`
    - `Ependymal-cell` → `None`, reason `not-in-MTG-taxonomy`
    - `Choroid-Plexus-Epithelial-Cells` → `None`, reason `not-in-MTG-taxonomy`
    - `Basal-Ganglia-GABAergic-Neurons` → `None`, reason `subcortical-BG`
    - `Cholinergic-Neurons` → `None`, reason `subcortical-or-brainstem`
    - `Inhibitory-Neurons` → `None`, reason `too-generic`
    - `GABAergic inhibitory interneurons` → `None`, reason `too-generic`
    - `GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4` → `["Sncg", "Vip"]`
      (mirrors `Erbb4-inhibitory-neurons`)
    - `GABAergic-inhibitory-interneurons-VIP-positive` → `["Vip"]`
    - `Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin` → `["Lamp5"]`
      (or `None`, `not-confident`)
    - `Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons`
      → `["L2/3 IT", "L4 IT"]`
  - Add a coverage assert at the top of `main()`:
    `assert set(SUBCLASS_MAP) == set(load_cluster_spine()["cluster_name"])`
    so any future spine drift breaks loudly.
  - Regenerate the bridge CSV (closes C3).
- **Verify**:
  - Generator prints `distinct clusters: 31`.
  - Run a downstream consumer (`alz/bulk_mea/attribute.py` SEA-AD branch) and
    confirm no `KeyError` / silent dropped rows.

## Execution order

1. C1 (unblocks any downstream import).
2. C2 (one-line; bundle with C1 in the same commit if cohesive).
3. C5 → C3 (mapping fix must land before the artifact is rebuilt).
4. C4 (independent; can run in parallel with C5/C3).

Commit per step with `fix:` prefix per the repo convention. No PRs / pushes
without explicit go-ahead.
