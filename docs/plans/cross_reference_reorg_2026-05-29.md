# `alz/cross_reference/` audit + reorg — 2026-05-29

Goal: (1) audit `alz/cross_reference/` for dead code, (2) separate the CTRL-outlier
investigation into its own package, (3) leave `cross_reference/` as a clean Mode-4
attribution package and document the new subdir with its own README.

All 11 files are git-tracked. This is a structural reorg via `git mv` + reference
fixups — no analysis logic changes.

---

## 1. Liveness audit

### 1a. True cross-reference (Mode 4) — STAY in `alz/cross_reference/`

| File | External consumer | Notes |
|---|---|---|
| `evidence.py` | `alz/bulk_mea/attribute.py` imports all 4 of `compute_sea_ad_concordance`, `prepare_song_concordance`, `prepare_song_specificity`, `prepare_wmb_specificity` | mouse Stage 3 attribution loaders |
| `human_celltype_attribution.py` | `alz/build_unified_viewer.py` (`build_celltype_specificity_payload`); `run_pair_mode_pipeline.sh --force` | emits `celltype_specificity.csv` for viewer payload |
| `seaad_human_agreement.py` | pixi `human-seaad`; `run_all.sh` (H-seaad); `run_mukesh_perdonor.sh` | human SEA-AD agreement panel |

### 1b. CTRL-outlier investigation — MOVE OUT

| File | External consumer | Notes |
|---|---|---|
| `ctrl_outlier_audit.py` | none | Phases A–C (sample structure, artifact controls, site attribution) |
| `ctrl_outlier_audit_kinases.py` | none | Phase D (per-kinase leading-edge proof) |
| `ctrl_outlier_audit_report_figs.py` | none | meeting figure (raw-phospho heatmap) |
| `ctrl_outlier_suspect_lfc_table.py` | none (outputs referenced in `ctrl_audit/INDEX.md`) | per-site suspect-vs-AD LFC table |
| `human_group_mea_reanalysis.py` | pixi `mea-suspect-reanalysis` | 4 labeled CTRL-contamination contrasts |
| `human_group_mea.py` | **only** `human_group_mea_reanalysis.py` (imports `AD_LIKE_CTRL`) | see §1c |

### 1c. Orphan-module finding — `human_group_mea.py`

Docstring positions it as the **production** human AD-vs-clean-control metric and says
`human_group_mea_clean_ctrl.csv` "remains canonical." Reality:

- No pixi task, no runner invokes it.
- No code/doc/viewer reads `human_group_mea_clean_ctrl.csv` (grep-clean repo-wide).
- Its only live linkage is `reanalysis` importing the `AD_LIKE_CTRL = {CTRL-07,08,10}` constant.
- `reanalysis` already reproduces its output as `mea_AD_vs_cleanCTRL.csv` "by the same code
  path" (its own docstring).

So either the "production" claim is stale (it's superseded by the reanalysis twin) or it
lost its task/consumer in an earlier consolidation. **Decision required** (see §4, Q2).

### 1d. Function-level dead code

None. Every top-level `def`/`class` in all 9 modules is reachable from a `__main__`
guard or an external import. The only "dead" unit is the orphan *module* in §1c.

---

## 2. Destination

New package `alz/ctrl_outlier_audit/` (name mirrors `docs/plans/human_ctrl_outlier_audit_*`
and `outputs/.../ctrl_audit/`). **Decision required** on exact name (see §4, Q1).

Moves (via `git mv`, preserving history):

```
alz/cross_reference/ctrl_outlier_audit.py            -> alz/ctrl_outlier_audit/audit.py
alz/cross_reference/ctrl_outlier_audit_kinases.py    -> alz/ctrl_outlier_audit/audit_kinases.py
alz/cross_reference/ctrl_outlier_audit_report_figs.py-> alz/ctrl_outlier_audit/report_figs.py
alz/cross_reference/ctrl_outlier_suspect_lfc_table.py-> alz/ctrl_outlier_audit/suspect_lfc_table.py
alz/cross_reference/human_group_mea_reanalysis.py    -> alz/ctrl_outlier_audit/group_mea_reanalysis.py
alz/cross_reference/human_group_mea.py               -> alz/ctrl_outlier_audit/group_mea.py   (pending Q2)
```

(Dropping the redundant `ctrl_outlier_`/`human_` filename prefixes now that the package
name carries that context. If you prefer verbatim filenames, say so and I'll keep them.)

New `alz/ctrl_outlier_audit/__init__.py`.

---

## 3. Reference fixups (the whole blast radius)

1. **pixi.toml** `mea-suspect-reanalysis` →
   `python -m alz.ctrl_outlier_audit.group_mea_reanalysis`.
2. **Intra-package import** in `group_mea_reanalysis.py`:
   `from alz.cross_reference.human_group_mea import AD_LIKE_CTRL`
   → `from alz.ctrl_outlier_audit.group_mea import AD_LIKE_CTRL` (or relocated per Q2).
3. **Docstrings/headers** referencing old paths inside the moved files.
4. **`alz/cross_reference/__init__.py`** + `alz/cross_reference/README.md` — drop the
   CTRL-outlier inventory; reduce to the 3 attribution modules.
5. **Repo-level docs** mentioning `cross_reference`: `README.md:181`,
   `docs/INDEX.md`, `alz/shared/__init__.py:4`, `alz/shared/README.md:10`,
   `docs/plans/kedro_argo_reintroduction_2026-05-26.md` (file inventory). Update the
   CTRL-outlier file list to the new package; leave historical plan prose intact.
6. No change to `evidence.py` import in `attribute.py` or the viewer's `sys.path` insert
   (those target files stay).

Note: `evidence.py`, `human_celltype_attribution.py`, `seaad_human_agreement.py` do **not**
move, so no runner/pixi changes for the live cross-reference path.

---

## 4. Decisions for the user

- **Q1 — package name:** `alz/ctrl_outlier_audit/` (recommended) vs `alz/human_ctrl_audit/`
  vs keep all under `cross_reference/` with no move.
- **Q2 — `human_group_mea.py`:** (a) move into the audit package as the corrected
  production metric and wire it a pixi task + consumer; (b) move it but treat the
  `AD_LIKE_CTRL` constant + clean-CTRL logic as belonging to `reanalysis`, archiving the
  standalone module; (c) leave it in `cross_reference/` as a live human-cohort deliverable.
- **Q3 — filename prefixes:** strip `ctrl_outlier_`/`human_` (recommended) vs keep verbatim.

---

## 5. Sequence (after approval)

1. `git mv` the 5–6 files into the new package; add `__init__.py`.
2. Fix imports (`group_mea_reanalysis`), pixi task, in-file docstrings.
3. Rewrite `cross_reference/README.md` (3 modules) + `cross_reference/__init__.py`.
4. Write new `alz/ctrl_outlier_audit/README.md`.
5. Update repo-level doc references (§3.5).
6. Verify: `python -c "import alz.ctrl_outlier_audit..."` import smoke +
   `pixi run mea-suspect-reanalysis --help`/dry path resolves + `git grep -n cross_reference`
   shows only intended residue.
