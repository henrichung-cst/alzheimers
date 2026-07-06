# Incytr regeneration (KsG + PTM + backbone)

Operational runbook for regenerating pair-mode Incytr with the three new feature
layers:

- **KsG** — kinase-substrate gene admission to `gene.use` (admission-only; the
  `|PDS| >= 0.2` floor governs, no cap/imputation). Toggle is data-presence:
  `KSG_MEA_FILE` set ⇒ active, unset ⇒ byte-identical to pre-KsG.
- **PTM** — acetylation (Ack) + ubiquitination (KGG) tracks. **5xFAD only**
  (the other cohorts have no Ack/KGG assay).
- **Backbone** — reduced-grain pathway tables (R-EM, L-R-EM, R-EM-T).

**Song is done.** Its KsG production output (`wide/` + `backbone/`) is canonical
in `outputs/reports/incytr_pair_mode/` and is **never swapped** — viewer
development reads it directly. This runbook covers the two remaining
un-regenerated cohorts: **5xFAD** and **t-cell**.

The pair-mode engine, scoring, and sce4-parity invariants are unchanged — these
layers widen `gene.use` and emit additional grains. The engine reproduces sce4
**with KsG OFF**; that regression check (`pixi run verify-incytr-sce4`) is a
standalone KsG-OFF test, **never run on the production burn** (KsG widens
`gene.use` beyond sce4's frozen node sets by design, so it would always diverge).

## Scripts in this folder

| Script | Role |
|---|---|
| `prep_incytr_night.sh` | Empty 5xFAD + t-cell production so the runners regenerate KsG fresh. Song is untouched. Run **once** before launching. |
| `launch_incytr_tmux.sh` | Stage a `tmux` session (`incytr`) with a combined run window plus monitor. The command is typed, **not executed** — operator presses Enter once. |
| `run_backbone_overnight_all.sh` | Combined sequential runner — 5xFAD pair-mode/post-processing first, then t-cell. 5xFAD bridge is deferred so it cannot block t-cell. |
| `run_backbone_overnight_5xfad.sh` | **5xFAD** — pair-mode (PTM-inclusive `pr,ps,py,Ack,KGG` → `wide/` + `backbone/`) → filter → bridge per tissue unless `SKIP_BRIDGE=yes`. No viewer (Phase 2). |
| `run_backbone_overnight_tcells.sh` | **T-cell** — pair-mode (donor2 then donor1; donor1 KsG-ON, donor2 KsG-OFF). No viewer/bridge. |

The runners call the leaf scripts that stay in `alz/incytr_pair/`
(`run_pair_mode*.sh`, `filter_significant_paths.py`), which are shared with the
verify/non-regeneration workflows.

## Workflow

Regeneration runs **overnight only** (working-hours runs are not permitted — the
box is shared and pair-mode peaks ~13–15 GB per cohort).

From the repo root, in a tmux session:

```bash
bash alz/incytr_pair/regeneration/prep_incytr_night.sh    # empty 5xFAD + t-cell prod
bash alz/incytr_pair/regeneration/launch_incytr_tmux.sh   # stage the session
tmux attach -t incytr
#   all -> Enter   (5xFAD pair-mode/post-processing, then t-cell)
```

The combined runner calls 5xFAD with `SKIP_BRIDGE=yes`; bridge is an explicit
post-processing step and should not gate the t-cell run:

```bash
pixi run kinase-incytr-bridge -- --cohort fivexfad --tissue cortex
pixi run kinase-incytr-bridge -- --cohort fivexfad --tissue hippocampus
```

`prep_incytr_night.sh` **empties** 5xFAD/t-cell production — empty production is
what makes the runners' skip-on-existing guard fall through and regenerate KsG
fresh.

## Gating + memory

- **One cohort at a time.** The combined runner is sequential; do not run another
  pair-mode job concurrently on the shared box. The 5xFAD bridge may be run after
  cohort jobs complete.
- **Review the combined log and per-cohort logs** after the run completes.
- Every compute step runs under `systemd-run --user --scope -p MemoryMax=24G
  -p MemorySwapMax=0` with `NPAIR_WORKERS=1`. Override `MEM_PAIR`/`MEM_PY`/
  `FULL_NBOOT`/`NPAIR_WORKERS` via env before calling a runner.

## Output locations

| Cohort | Production root |
|---|---|
| 5xFAD | `outputs/reports/incytr_pair_mode_5xfad/<tissue>/wide/` (PTM-inclusive) |
| T-cell | `outputs/reports/incytr_pair_mode_tcells/<donor>/wide/` |

Logs: `overnight_all_<timestamp>.log` under
`outputs/reports/incytr_pair_mode_regeneration/`, plus
`overnight_<timestamp>.log` under each cohort's reports root.
