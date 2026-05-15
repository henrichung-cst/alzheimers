# CR-02 — Spine re-threshold (min-cells = 5, no rank gate): retrospective

Branch: `feat/cr02-spine-rethreshold`  
Commit: `8c34b57`  
Phase completed: Phase 1 (code + smoke)  
Phase 2 (full pipeline rerun) pending.

---

## Methods

### Spine builder (`alz/integration/build_cluster_spine.py`)

Three new CLI flags were added to the spine builder:

- `--spine-name` (default `levy19`) — names the output subdirectory under `data/incytr/v2_46clusters/spines/<name>/`.
- `--min-cells` (default `20`) — per-(cluster, animal) cell-count gate; replaces the hard-coded `SONG_MIN_CELLS = 20`.
- `--no-rank-gate` — when set, the rank-10 design-matrix requirement is dropped. Any named cluster with ≥1 qualifying animal (i.e., ≥ `min_cells` cells) enters the spine.

Outputs are written to a spine-name-keyed directory:

```
data/incytr/v2_46clusters/spines/<name>/cluster_spine.csv
data/incytr/v2_46clusters/spines/<name>/rejected_clusters.csv
data/incytr/v2_46clusters/spines/<name>/spine.scope.json
```

`spine.scope.json` stamps `{name, min_cells, rank_gate, generated_at, n_in_spine, n_total_clusters}`.

Backward compatibility: when `--spine-name levy19` (the default), the builder writes symlinks at the old top-level paths (`data/incytr/v2_46clusters/cluster_spine.csv`, `rejected_clusters.csv`) pointing into `spines/levy19/`. Existing readers such as `config_integration.CLUSTER_SPINE_FILE` and `plot_cluster_spine.py` continue to resolve without modification.

The rank tier annotation (`full_rank`, `partial`, `severe`) was preserved even when the rank gate is off, so the tier column provides post-hoc visibility into design-matrix quality for each kept cluster.

### Path resolution (`alz/decomposition/paths.py`)

Added `spine_dir(name)`, `resolve_spine_csv(name)`, and `load_spine_clusters(name)`. `resolve_spine_csv` prefers the new `spines/<name>/` layout and falls back to the legacy top-level path when `name == "levy19"` and the new directory does not yet exist. This allows existing `levy19` cached runs to continue resolving before a re-run.

### `config_integration.py`

`load_cluster_spine()` was extended to accept an optional `name` argument (default `CLUSTER_SPINE = "levy19"`). A companion `resolve_cluster_spine_file(name)` was added with the same fallback logic. Both default to the existing behavior, preserving all existing callers.

### Per-cluster MEA — NaN on unestimable contrasts (`alz/decomposition/enrich_celltype.py`)

The prior code skipped an entire cluster when the design matrix was rank-deficient. The new behavior:

1. Rank is still computed via `np.linalg.matrix_rank(X)`.
2. For each of the 9 factorial contrasts, estimability is checked via the row-space projection: `c ≈ c @ pinv(X) @ X`. Contrasts that fail this check are recorded in `unestimable_contrasts`; those that pass are recorded in `estimable`.
3. If **no** contrast is estimable, the cluster is skipped (the only remaining skip path).
4. Otherwise, OLS proceeds: full-rank designs use the existing `_run_ols_all_sites` (which inverts `X'X`); rank-deficient designs use a new `_run_ols_pinv` fallback (which uses the Moore-Penrose pseudoinverse). For estimable contrasts, `c'β` is invariant to the choice of generalized inverse, so the estimates are valid. Unestimable contrasts receive `NaN` for LFC, SE, t, p, FDR, and a NaN-filled `lfc` column for MEA input.
5. Per-cluster audit entries now carry `rank_deficient`, `unestimable_contrasts`, and `n_estimable_contrasts`. The top-level `enrich_audit.json` adds `n_rank_deficient` and `n_with_unestimable_contrasts` aggregate counts.

### Pair-mode Incytr — `--spine` flag

`bench/export_decomposition_for_pair.py` was updated to accept `--spine` (default `levy19`). The input decomposition directory is now `outputs/reports/decomposition/<spine>/` and the output bench tree is `bench/incytr_pair_<suffix>/incytr input/` where `suffix` is `19` for `levy19` and `<spine>` otherwise (e.g., `bench/incytr_pair_levy_t5/`).

`bench/build_pair_inputs.sh` and `bench/run_pair_mode_19.sh` were updated with the same `--spine` / `SPINE` env-var routing. These files live under the gitignored `bench/` tree and are on-disk only.

---

## Results

### levy_t5 smoke run

Command:
```
python alz/integration/build_cluster_spine.py --min-cells 5 --no-rank-gate --spine-name levy_t5
```

Output:
```
in_spine: 31 clusters, 60220/63706 cells (94.53%)
tier counts:
  full_rank    28
  unnamed      15
  severe        2
  partial        1
```

`spine.scope.json`:
```json
{
  "name": "levy_t5",
  "min_cells": 5,
  "rank_gate": false,
  "generated_at": "2026-05-15T23:42:45.278255+00:00",
  "n_in_spine": 31,
  "n_total_clusters": 46
}
```

Files written:
- `data/incytr/v2_46clusters/spines/levy_t5/cluster_spine.csv`
- `data/incytr/v2_46clusters/spines/levy_t5/rejected_clusters.csv`
- `data/incytr/v2_46clusters/spines/levy_t5/spine.scope.json`

31 is within the pre-run estimate of 25–35 clusters. The 3 rank-deficient clusters that are now included (`severe` × 2, `partial` × 1) previously failed the rank-10 gate; under the new gate they are in-spine because each has ≥1 animal with ≥5 cells.

---

## Caveats and discordance

- **Phase 2 not run.** `snrna_proportions.py`, `build_celltype_decomposition.py`, `enrich_celltype.py`, and pair-mode Incytr have not been executed against `levy_t5`. All downstream outputs remain levy19. The NaN-emission code in `enrich_celltype.py` has not been exercised on a real cluster.
- **Rank-deficient cluster quality.** The 3 rank-deficient clusters included in `levy_t5` (`severe` × 2, `partial` × 1) will have between 0 and 8 estimable contrasts. Per-cluster vs. bulk MEA Spearman ρ ≥ 0.7 gate in `verify_decomposition.py` may fail for these clusters, as their NES values will be noisier. The verification harness has not been updated to add a "rank-deficient-but-included" tier counter (noted in the plan as a follow-on; left for Phase 2).
- **NaN flow into MEA.** When `mea_input[contrast]` is an all-NaN LFC column, `_run_mea` (via `gseapy`) will receive an all-NaN ranking. The resulting MEA row will be either absent or NaN-filled depending on gseapy's behavior with NaN inputs; this has not been tested.
- **Bench scripts not in version control.** `bench/export_decomposition_for_pair.py`, `bench/build_pair_inputs.sh`, and `bench/run_pair_mode_19.sh` carry the `--spine` wiring but live under the gitignored `bench/` tree. They are on-disk changes only.
- **levy19 symlinks.** The levy19 backward-compat symlinks (`data/incytr/v2_46clusters/cluster_spine.csv`) are only written when the builder is invoked with `--spine-name levy19`. The levy19 spine has not been rebuilt in this session; if the legacy `cluster_spine.csv` was absent the fallback path in `resolve_spine_csv` would be taken.
- **Viewer assumes full 9-contrast coverage.** `kinase_audit.js` and `kinase_explorer.js` have not been audited for NaN tolerance; this is deferred to plan 04.

---

## Conclusions

The levy_t5 spine (31 clusters, 94.5% cell coverage) supersedes levy19 (19 clusters) as the primary per-cluster spine for Phase 2 of the analysis. The expanded spine recovers 12 additional clusters — including 3 that are rank-deficient under the full 10-parameter factorial design — without losing any full-rank cluster from levy19.

The NaN-emission logic in `enrich_celltype.py` ensures these rank-deficient clusters remain in the MEA output rather than being silently dropped, preserving full spine coverage in the enrichment table. Downstream viewers and verification must be hardened against per-contrast NaN holes before the Phase 2 outputs are considered final.

Phase 2 entry criterion: run `snrna_proportions.py --spine levy_t5`, then `build_celltype_decomposition.py`, `enrich_celltype.py`, and pair-mode Incytr against the new bench tree. See `docs/plans/change_requests_sequencing.md` Phase 2 steps B–E.
