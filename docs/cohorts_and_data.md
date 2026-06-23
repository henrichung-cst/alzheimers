# Cohorts and Data — Status and Conventions

Sources: `project_5xfad_cohort.md`, `project_tcell_exhaustion_cohort.md`, `project_human_ctrl_ad_like_contamination.md`, `project_atlas_raws_removed_2026-06-08.md`, `project_solo_dev_repo.md`, `project_rclone_ingest_ssot.md`

## Repo context

Solo-dev repository — exactly two contributors: the user (hchung) and Claude (stated 2026-05-20). No other humans pushing, no downstream consumers of `public/main`, no open PRs from third parties, no CI deploys. When weighing git operations, skip multi-party-coordination caveats. The only real risks are local data loss (covered by tags + reflog) and the user's own recovery.

## Drive ingest: rclone-ingest SSOT

Drive/rclone ingest is consolidated (2026-06-08) into a shared engine at `~/Projects/work/rclone-ingest/`, vendored as a git submodule at `vendor/rclone-ingest` + installed as a pixi editable path-dep. Project-specific configuration lives only in `conf/data_sources.yaml`.

- CLI: `rclone-ingest sync <name> [--group G] [--dry-run]`, `… --all`, `list`, `check`. `check` verifies rclone + every remote is in `rclone listremotes`.
- Explore commands (v0.2.0): `ls [<source>] [PATH]` (metadata only, `-R`/`--max-depth`/`--json`), `peek <file> [--head N|--tail N]` (ranged GET, text only), `fetch <file>... [--dest DIR]` (ad-hoc copy; recurring pulls belong in the manifest).
- Manifest grammar: sources → transfers (mode copy|copyto, src, dest, group, max_depth, include, flags).
- pixi `ingest-*` tasks are now thin wrappers: `rclone-ingest sync <name>`.
- The submodule was added via local file transport (`-c protocol.file.allow=always`); not yet pushed to GitHub. `.gitignore` is whitelist-style — must `!`-allow `/vendor/**` and `/.gitmodules`.

Replaced (anti-shim, deleted): `alz/runners/supporting/ingest_tcells.sh`, `bench/perf/download_sce4_source.sh`. Plan/SSOT: `docs/plans/rclone_ingest_ssot_2026-06-08.md`.

## Atlas raw files deleted 2026-06-08

Three large external atlas raws were INTENTIONALLY deleted to reclaim ~121 GB (box was at 92%, now 74%). They are derivation-only and re-downloadable. Do NOT reflexively re-download them.

| Deleted file | Size | Derived artifacts (present on box) | Restore |
|---|---|---|---|
| `data/external/sea_ad/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad` | 34 GB | `effect_sizes{,_early,_late}.h5ad`, `aggregates/seaad/expression_by_supertype.csv`, seaad specificity CSVs | `atlas.py --sea-ad-expression` |
| `data/external/allen_abc/expression_matrices/WMB-10Xv3-subset/` | 51 GB | `wmb_kinase_expression.csv` (+scope.json), `wmb_proteome_expression.csv`, bridges | re-run `extract_wmb_gene_subset.py` from full WMB download |
| `data/external/allen_abc/expression_matrices/WHB-10Xv3/` | 36 GB | `aggregates/hbca/expression_by_class.csv`, hbca specificity CSVs, bridge | `atlas.py --hbca-download` |

**TRAP:** `run_all.sh` WITHOUT `--skip-atlas` checks for the full `WMB-10Xv3/` dir (absent) and triggers a needless download. Use `--skip-atlas` on normal runs — the derived CSVs cache-hit.

Also removed 2026-06-08: ~12 GB superseded/duplicate run outputs. NOT removed: `DEG_PRG_ma_2mo_AppP_WTyp_…-001/` (holds ~200 unique per-pair CSVs + a kinase rds absent from `outputs/`). Full record + restore commands: `docs/plans/data_footprint_audit_2026-06-08.md`.

## Human (Mukesh/NBB) cohort: AD-like control contamination

3 of 7 human controls — **CTRL-07, CTRL-08, CTRL-10** — carry a genuinely AD-like phospho-omic signature (audited 2026-05-25). CTRL-08 and CTRL-10 embed inside the AD cluster (PCA, AD-vs-clean axis alignment +0.86 / +0.78, within AD range); CTRL-07 is AD-leaning (+0.47) but also a low-coverage outlier. Confirmed genuine (not technical): survives complete-case coverage control, survives removal of the protein denominator, and is site-specific to AD-discriminating substrate motifs (per-kinase leading-edge: clean-baseline deviation corr AD-vs-suspicious 0.78–0.90).

**Impact:** the per-donor human contrast (`alz/ingest/mukesh_perdonor.py:_build_donor_deltas`) builds every donor's LFC vs `nanmean(all 7 CTRL)`. With 3/7 controls AD-like, the reference mean is pulled toward AD — clean controls (CTRL-01..04) show significant NES in the anti-AD direction, and AD-vs-CTRL effects are attenuated.

**How to apply:** any single human AD-vs-CTRL kinase metric (Concern 1) must use the clean control set **CTRL-01/02/03/04**, not all 7. Whether to drop CTRL-07/08/10 or model them as an "AD-like control" group is an open study-design decision. Audit scripts: `alz/cross_reference/ctrl_outlier_audit{,_kinases}.py`. Findings: `docs/plans/human_ctrl_outlier_audit_findings_2026-05-25.md`.

## 5xFAD cohort (Lucie, DIA label-free)

**Design:** 2 regions (Cortex, Hippocampus) × 4 ages (3/6/9/12 mo) × genotype TG (5xFAD) vs WT, ~30 mice/region.

**Status (verified 2026-06-18): UNBLOCKED.** The two formerly `.sne`-only cells have been re-exported as Spectronaut TSV reports and ingested:
- `…102325_LD_5xfad_IMAC_cortex_PTMSiteReport.tsv` (was `…_IMAC_cortex.sne`)
- `…Lucie_Hippocampus_male_Mo6-12_5xFAD_Report.tsv` (hippo total; `.sne` is now unused historical record)

Cohort module: `alz/cohorts/fivexfad/`.

On-disk outputs:
- Kinase MEA: `outputs/reports/kinase_attribution_5xfad/{cortex,hippocampus}_{st,py}_mea_stoichiometry.csv` (both regions, both tracks).
- Incytr inputs: `data/derived/5xfad_incytr_inputs/{cortex,hippocampus}/` — `incytr_obj.rds` (351M / 398M) + `pr/ps/py_deconvoluted.csv` (mass-identity passing).

**Still pending:**
- Incytr pair-mode run: `allmarkers.csv` needed first (`pixi run 5xfad-build-incytr-gene-list`), then the run. Uses DEG∪prG derivation (t-cell path; no sce4 frozen gene.use, no parity gate).
- AcK/KGG PTM data exists for 5xFAD only — see `docs/plans/todo7_incytr_acetyl_ubiquitin.md`.

Plan: `docs/plans/todo6_incytr_on_5xfad.md`.

5xFAD resolves to the same 31-cluster spine as Song via the per-cohort intrinsic filter (drop unnamed `cluster-N` from the 46-label set; verified set-equal 2026-06-16).

## T Cell Exhaustion cohort

Ingested 2026-05-27 from Drive folder `1YE_h1jIyBajtm6ArxJqevJ0rt0xLKQgX` → `data/datasets/tcells/{donor1,donor2}/`. Full design: `docs/plans/meeting_notes_triage_2026-05-27.md`.

### Hard constraints (do not re-litigate)

- **Donor 2 has NO IMAC** → skip kinase MEA for donor 2. Kinase MEA on bulk only, NEVER on deconvoluted.
- Contrast = (later timepoint) − Day 2 baseline, per donor independently; timepoints not comparable across donors.
  - Donor1: days 2/11/13/15/17/19/20 (2 reps); Donor2: days 2/5/7/9/11 (1 rep); IMAC days 2/13/15/17/19/20.
- **Representation = `ForPerseus`** (site-level, `PTM.FlankingRegion` motif, linear intensities). Donor1 IMAC = 18May Normalized site report (137 MB, has `PG.Genes`, no R parser).
- **Self-normalize** (`alz/ingest/tcells.py`): log2 + per-run median-center. Do NOT use the big May "Normalized_NotParsed" reports.
- Filter pY/IMAC to `PTM.ModificationTitle == "Phospho (STY)"` (tables include Carbamidomethyl/Oxidation contaminants).
- Replicates are technical re-injections (r1↔r2 corr 0.96–0.99) → averaged to one column per (donor, day).
- Out of scope: KGG/AcK/MME enrichments, flow cytometry.

### scRNA cell-state spine: ProjecTILs (not Seurat clusters)

Aggregate directly by per-cell `functional.cluster` from ProjecTILs (carmonalab figshare doi 10.6084/m9.figshare.23608308, cached at `data/external/projectils/`). Azimuth PBMC-ref was stripped (lost 30.2% / 64.8% donor1/2). Cluster-keyed path was also tried and dropped (lost 44.5% donor1 after recluster).

Pipeline order: `ingest-tcells-scrna → tcells-projectils-map → tcells-scrna-extract → tcells-decompose`. ProjecTILs-map MUST run before extract (extract reads per-cell predictions).

State names are sanitized via inline 14-entry `LABEL_MAP` in `tcells_scrna_extract.R` (alphanumeric only — Incytr constraint). Anti-shim (deleted): `tcells_annotate_clusters.py`, `tcells_recluster.R`, recluster runners, on-disk RDS + cluster_annotations.csv + audit JSONs, pixi tasks `tcells-recluster` / `tcells-annotate`.

**OOM lesson:** `GetAssayData(layer="scale.data")` on a 27486×25678 object materializes a ~5.6 GB dense matrix and OOMs the 30 GB box. Fix: immediately `DietSeurat(assays="RNA")` + drop scale.data/reductions + `gc()` after load, never access scale.data, load the object exactly once and extract all small artifacts in that pass.

### Key output counts

| Donor | pr genes | pY sites | IMAC sites | Samples |
|---|---|---|---|---|
| Donor1 | 8,125 | 1,180 | 62,807 | 7/6/6 (pr/py/IMAC) |
| Donor2 | 7,767 | 514 | — | 5 |

Donor1 scRNA: 27,486g × 25,678c, 14 ProjecTILs states, days 0/2/9/13/17/20.
Donor2 scRNA: 29,191g × 20,654c, 13 ProjecTILs states, days 2/5/7/9/11.

Donor2 pY has no `PTM.FlankingRegion` → empty motif (MEA skipped; Incytr uses phospho substrate, not motif).

Donor1 supports full 3-channel pair-mode (pr+ps+py+sclog2FC/DEG). Donor2 = pr+py only (no IMAC → no ps; verify `incytr_commandline.R` tolerates missing ps).

Mass identity verified to float roundoff (max |rel err| 1e-15).

## ApTt genotype — sub-additivity as a pipeline sanity check

Source: `project_aptt_subadditivity.md`

Biologically, co-expressing both AD genotypes (App + Tau → **ApTt**) should produce a **synergistic** response — stronger than either App or Tau alone. A prior analysis instead showed ApTt responding *less* than either single component (sub-additivity / antagonism), which pointed to a problem in that earlier pipeline rather than a real biological effect.

Use this as a validation criterion, not just a visualization: if the current stoichiometry-corrected pipeline still shows ApTt sub-additivity, treat it as a **red flag** on pipeline validity. The **Additivity Scatter (viewer Tab 5)** is the sanity check — points above the diagonal = synergy (expected); points below = antagonism (concerning). Frame ApTt results in this validation light when interpreting or fielding questions about the condition.
