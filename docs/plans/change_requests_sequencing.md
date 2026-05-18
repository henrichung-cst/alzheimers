# Change requests — sequencing

## Model

Write all the code first, run the heavy pipeline once at the end.

The plan docs are numbered in the order they should be implemented:

1. **`change_request_01_human_control_mea.md`** — LOO control MEA on
   the human (Mukesh) tab. Smallest, self-contained, no new compute
   beyond a re-run of `ingest_mukesh_perdonor.py`. Validates the
   human-side payload pattern before changes 03 and 04 also touch it.
2. **`change_request_02_spine_rethreshold.md`** — relax the spine gate
   (min-cells = 5, no rank gate), wire pair-mode Incytr as the new
   per-cluster front door. The longest-running change; needs to land
   before 04 because the viewer consumes its outputs.
3. **`change_request_03_human_celltype_specificity.md`** — pull SEA-AD
   MTG expression + Allen HBCA references, add the human cell-type
   specificity module + viewer sub-panel. Independent of 01, 02, 04
   code-wise; shares the human payload block with 01 so it lands
   after 01 to keep the merge clean.
4. **`change_request_04_incytr_viewer.md`** — multi-contrast filter,
   temporal detail bar plot, trajectory tags on the Incytr pathways
   tab. Depends on 02's pair-mode Incytr output for the final payload
   build; JS can be developed against a mock first.

## Execution

Two phases, in this order:

### Phase 1 — write all the code (no heavy reruns)

Implement 01 → 02 → 03 → 04 in order. After each, run the *cheap*
verification step for that change:

- 01: `python alz/ingest_mukesh_perdonor.py --track both` (~minutes).
  Confirms CTRL LOO columns appear in `kinase_donor_nes*.csv`.
- 02: `python alz/integration/build_cluster_spine.py --min-cells 5
  --no-rank-gate --spine-name levy_t5` (seconds). Confirms the new
  spine file. Do *not* trigger downstream decomposition/enrich/Incytr
  yet — those are the heavy reruns saved for phase 2.
- 03: module-level smoke. Don't pull the multi-GB references yet;
  unit-test the specificity recipe against a small fake h5ad.
- 04: viewer JS against a mocked `trajectory_index` built from
  whatever Incytr cache currently exists. `pixi run viewer` and
  eyeball the tab.

By end of phase 1 the repo is *code-complete* but the live outputs
are still from the old pipeline.

### Phase 2 — one clean run

Sequential, because each step's outputs feed the next:

```
A. python alz/integration/build_cluster_spine.py --min-cells 5 \
       --no-rank-gate --spine-name levy_t5
B. python alz/snrna_proportions.py --spine levy_t5
C. python alz/decomposition/build_celltype_decomposition.py \
       --spine levy_t5 --track both
D. python alz/decomposition/enrich_celltype.py --spine levy_t5 --track st
   python alz/decomposition/enrich_celltype.py --spine levy_t5 --track py
E. pair-mode Incytr export + 9-contrast run against levy_t5
F. python alz/ingest_mukesh.py --reshape          # if any human-side change touched ingest
   python alz/ingest_mukesh_perdonor.py --track both
G. atlas downloads (SEA-AD MTG expression, Allen HBCA) — start during
   phase 1 if disk and bandwidth allow; otherwise here. Independent of
   A–F, so it can run in parallel with the rest of phase 2.
H. python alz/human_reference_expression.py --ref both
   python alz/human_celltype_attribution.py
I. python alz/build_unified_viewer.py
```

Wrap A–I as a single shell script (e.g.,
`alz/runners/main/run_pair_mode_pipeline.sh`) so phase 2 is "one button."

## Merge order

01 and 03 both extend `PAYLOAD.human`. If they merge to the same
branch, 01 lands first (it just adds CTRL donor columns to existing
blocks), then 03 (adds a new `celltype_specificity` top-level key).
No field collisions.

02 and 04 land together (or 04 immediately after 02). 04's Python
build only makes sense once 02's pair-mode Incytr exists.

## Where parallel still applies

The only meaningful parallelism inside phase 2 is the atlas downloads
(step G), which are I/O-bound and have no compute dependency on A–F.
Everything else is a sequential chain.

## Long-running job

Phase 2 is unattended. Single shell script, log to a timestamped
build log, write a `done.json` sentinel at the end.
