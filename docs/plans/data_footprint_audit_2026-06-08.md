# Data Footprint Audit — 2026-06-08

**Volume:** 715 G total, 655 G used (92%), ~61 G free.  
**Project root:** 202 G (`du -sh .`).

> **EXECUTED 2026-06-08:** safe-now shortlist (minus the DEG_PRG dir, reclassified below as NOT a duplicate) + the three re-downloadable atlas raws (SEA-AD nuclei 34 G, WMB-10Xv3-subset 51 G, WHB-10Xv3 36 G). **~132 GB reclaimed; volume 92% → 74%.** sce4_source `aggexp.csv`/`ps_yuyu_deconvoluted.csv` removed only after md5-confirming byte-identity with the `v2_46clusters` canonicals (re-fetch via `pixi run ingest-sce4-source` to restore the verbatim mirror). Atlas raws re-downloadable via `atlas.py --sea-ad-expression` / `extract_wmb_gene_subset.py` / `atlas.py --hbca-download`. Not yet executed: user-judgment items A, B, C, G, H.

---

## Top Space Consumers

| Rank | Path | Size | Class |
|:-----|:-----|-----:|:------|
| 1 | `data/external/allen_abc/expression_matrices/` | 86 G | (b) External public, re-downloadable via `atlas_reference.py --wmb-download` / `--hbca-download` |
| 2 | `data/external/sea_ad/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad` | 34 G | (b) External public, re-downloadable via `atlas_reference.py --sea-ad-full` |
| 3 | `data/datasets/tcells/` | 10 G | (a) Raw collaborator drops; `donor1/scrna/Tcells.singlet.rds` (5.2 G) + `donor2/scrna/Tcells_d2.singlet (1).rds` (4.7 G) |
| 4 | `archive/pre_levy19_2026-05-14/` | 4.0 G | (e) Stale — pre-pivot snapshot; levy19 spine is closed |
| 5 | `.pixi/envs/default/` | 4.4 G | Conda env; regenerable via `pixi install` |
| 6 | `data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/` | 5.7 G | (a) Frozen sce4 reference provenance (pairwise CSVs + per-contrast Allpathway CSVs + pairwise RDS files) |
| 7 | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/` | 3.3 G | (a) Frozen sce4 source; contains confirmed duplicates (see below) |
| 8 | `outputs/reports/incytr_pair_mode/wide_pre_policy_20260601/` | 3.4 G | (e) Stale snapshot — pre-policy wide parquets superseded by `wide/` (195 M) |
| 9 | `outputs/reports/incytr_pair_mode/_confirm_top300/` | 2.8 G | (e) One-off parity probe run — single 2.8 G parquet; purpose complete |
| 10 | `data/incytr_frozen/v2_46clusters/incytr input/` | 1.3 G | (a) Frozen inputs for live pipeline — required |
| 11 | `archive/incytr_factorial_inputs/` | 1.5 G | (e) Archived factorial mode (closed 2026-05-18); matrix files not needed |
| 12 | `data/incytr_frozen/outputs/` (4 zip files) | 1.3 G | (d) Duplicate — compressed archives of content already extracted to same directory |
| 13 | `data/incytr_frozen/DEG_PRG_ma_2mo_AppP_WTyp_10302025-20260524T230649Z-3-001/` | 578 M | **(a) NOT a duplicate — DO NOT DELETE.** Containment check (2026-06-08) found ~200 per-pair `sce4_DEG_PRG_*_to_*` CSVs + `sce4_DEG_PRG_Pairwise_kinase_table_10302025.rds` present here but ABSENT from `outputs/Analysis_new.../`. The earlier SHA-1 "match" only sampled the Allpathway CSV. Unique provenance — retain. |
| 14 | `outputs/reports/incytr_pair_mode_tcells/donor2/wide_smoke/` | 1.6 G | (e) Smoke/debug run superseded by `wide/` (78 M) |
| 15 | `outputs/reports/decomposition/levy_t5/mea_substrate_sets_per_cluster.csv` | 1.3 G | (c) Pipeline output — regenerable via decomposition chain |
| 16 | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/sobj_7res.RDS` | 798 M | (d) Near-duplicate — pre-rename Seurat object; `renamed_sobj.rds` (same dir, same size) is the current version and is byte-identical to `v2_46clusters/incytr input/incytr_obj.rds`; `sobj_7res.RDS` differs in last ~50 MB (pre-labeling state) |
| 17 | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/aggexp.csv` | 216 M | (d) Confirmed byte-identical to `v2_46clusters/provenance/aggexp.csv` |
| 18 | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/ps_yuyu_deconvoluted.csv` | 259 M | (d) Confirmed byte-identical to `v2_46clusters/incytr input/ps_yuyu_deconvoluted.csv` |
| 19 | `data/incytr_frozen/v1_8clusters/` | 1.3 G | (e) v1 8-cluster spine; superseded by v2_46clusters; retained as provenance per policy |
| 20 | `outputs/reports/incytr_pair_mode/_sce4_full_q0/` + `_sce4_full_source_ps_diag/` | 310 M each | (e) Parity verification probe runs — purpose complete |

---

## Ranked Reduction Opportunities

### Safe to Act Now

| # | Target | Est. Reclaimable | Command | Risk | Regenerable? |
|:--|:-------|----------------:|:--------|:-----|:-------------|
| **1** | `outputs/reports/incytr_pair_mode/wide_pre_policy_20260601/` | **3.4 G** | `rm -rf outputs/reports/incytr_pair_mode/wide_pre_policy_20260601/` | None — superseded by `wide/` (195 M, dated Jun 1 10:27, is the post-policy production output) | Yes, re-run pair pipeline |
| **2** | `data/incytr_frozen/outputs/` zip archives (4 files) | **1.3 G** | `rm data/incytr_frozen/outputs/*.zip` | None — content is already fully extracted into `Analysis_new cluster labels_cutoff_0.1/` (5.7 G) and as `sce4_DEG_PRG_Allpathway_table_09062025.csv` + `sce4_DEG_PRG_Top300_table_09062025.csv` on disk. Zips are Drive download artifacts. | On Drive (rclone-ingest) |
| **3** | `data/incytr_frozen/DEG_PRG_ma_2mo_AppP_WTyp_10302025-20260524T230649Z-3-001/` | **578 M** | `rm -rf "data/incytr_frozen/DEG_PRG_ma_2mo_AppP_WTyp_10302025-20260524T230649Z-3-001/"` | None — byte-identical SHA-1 match (first 5 MB) with `outputs/Analysis_new cluster labels_cutoff_0.1/DEG_PRG_ma_2mo_AppP_WTyp_10302025/` for the Allpathway CSV and all per-pair CSVs. Standalone dir has 578 M; outputs dir has 397 M (same files, slightly different set — standalone has more). **Verify before deleting:** `diff <(ls data/incytr_frozen/DEG_PRG*-001/DEG_PRG_ma_2mo_AppP_WTyp_10302025/) <(ls "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/DEG_PRG_ma_2mo_AppP_WTyp_10302025/")` | On Drive |
| **4** | `outputs/reports/incytr_pair_mode/_confirm_top300/` | **2.8 G** | `rm -rf outputs/reports/incytr_pair_mode/_confirm_top300/` | Low — this is a one-off debug run containing a single 2.8 G parquet (`ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet`) generated 2026-05-30 as a top-300 confirmation step. The parity question it answered is resolved. | Yes, re-run |
| **5** | `outputs/reports/incytr_pair_mode_tcells/donor2/wide_smoke/` | **1.6 G** | `rm -rf outputs/reports/incytr_pair_mode_tcells/donor2/wide_smoke/` | None — `wide_smoke/` contains a single 1.6 G parquet dated 2026-05-28 (smoke/debug run). Production output is `donor2/wide/` (78 M, dated 2026-06-04). | Yes, re-run |
| **6** | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/aggexp.csv` | **216 M** | `rm data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/aggexp.csv` | None — byte-identical to `data/incytr_frozen/v2_46clusters/provenance/aggexp.csv` (first and middle SHA-1 match). Keep v2_46clusters copy, remove sce4_source redundancy. | Available in v2_46clusters |
| **7** | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/ps_yuyu_deconvoluted.csv` | **259 M** | `rm data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/ps_yuyu_deconvoluted.csv` | None — byte-identical to `data/incytr_frozen/v2_46clusters/incytr input/ps_yuyu_deconvoluted.csv` (SHA-1 match). | Available in v2_46clusters |
| **8** | `outputs/reports/incytr_pair_mode/_sce4_full_q0/` + `_sce4_full_source_ps_diag/` | **620 M** | `rm -rf outputs/reports/incytr_pair_mode/_sce4_full_q0/ outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag/` | Low — parity probe runs; current verification state is in `outputs/reports/incytr_pair_mode/wide/` + verify scripts. | Yes, re-run |
| **9** | `archive/incytr_factorial_inputs/` | **1.5 G** | `rm -rf archive/incytr_factorial_inputs/` | Low — factorial mode closed 2026-05-18; expression_matrix.mtx (1.3 G) is a derived intermediate. The archive note in retention policy classifies factorial as closed. | Would need to re-derive from raw; closed path |
| **10** | `archive/pre_levy19_2026-05-14/incytr_factorial/` (within pre_levy19) | **1.4 G** | `rm -rf archive/pre_levy19_2026-05-14/incytr_factorial/` | Low — pre-levy19 factorial outputs; double-archived (levy19 spine closed, factorial closed). | Closed path |

**Safe-now subtotal: ~12.7 G**

---

### Needs User Judgment

| # | Target | Est. Reclaimable | Consideration |
|:--|:-------|----------------:|:-------------|
| **A** | `archive/pre_levy19_2026-05-14/` (remainder after factorial) | **~2.6 G** | Contains pre-levy19 unified_viewer (679 M), kinase attribution snapshots (~250 M), decomposition (165 M). These are provenance records for the 2026-05-14 pivot. Retention policy says keep provenance; question is whether the *full outputs* are needed vs just code/configs. |
| **B** | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/sobj_7res.RDS` | **798 M** | Pre-rename Seurat object (before cluster relabeling). `renamed_sobj.rds` is the labeled version and is already present + duplicated in `v2_46clusters/incytr input/incytr_obj.rds`. `sobj_7res.RDS` is strictly a predecessor state — useful only to re-derive cluster labels; no active pipeline path reads it. Likely safe but requires confirming no script references it. |
| **C** | `data/incytr_frozen/v1_8clusters/` | **1.3 G** | v1 8-cluster spine. Retention policy implies it's historical record. Active pipeline uses v2 exclusively. Safe to remove if you accept that the v1 parity benchmarks (`output/sweep_1comparison/`) won't be reproducible without it. |
| **D** | `data/external/sea_ad/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad` | **34 G** | The 34 G raw nuclei h5ad. `data/README.md` lists it as re-downloadable via `atlas_reference.py --sea-ad-full`. The three `effect_sizes*.h5ad` (237 M total) are the derived aggregates that the pipeline actually uses. Removing the raw nuclei file saves 34 G with the ability to re-download. The download is slow and the file is large — only reclaim if you know you won't need to re-derive the effect_sizes from scratch. |
| **E** | `data/external/allen_abc/expression_matrices/WMB-10Xv3-subset/` | **51 G** | 13 region h5ads used for WMB expression export. Re-downloadable via `atlas_reference.py --wmb-download`. Remove only if `wmb_kinase_expression.csv` + `wmb_proteome_expression.csv` outputs are already generated and stable; would need to re-download to re-run `wmb_expression.py`. |
| **F** | `data/external/allen_abc/expression_matrices/WHB-10Xv3/` | **35 G** | Human brain atlas h5ads (WHB = human whole brain). Re-downloadable. Remove only if HBCA concordance outputs are stable and you don't expect to re-run `atlas_reference.py --hbca-download`. |
| **G** | `outputs/reports/decomposition/levy_t5/mea_substrate_sets_per_cluster.csv` | **1.3 G** | Pipeline output (regenerable). This is the full substrate assignment table written by the decomposition chain. If downstream consumers only read `mea_per_cluster.parquet` (33 M), the 1.3 G CSV may be safely deleted and re-generated by `pixi run decomposition`. Verify no downstream step reads this CSV directly before deleting. |
| **H** | `data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/` (remaining duplicates) | **~200 M** | `pr_yuyu_deconvoluted.csv` (129 M) differs between sce4_source and v2_46clusters — they are not duplicates; keep both. But `old_snrna_derived_tables/` (130 M) and `.ipynb_checkpoints/` (34 M) inside sce4_source are stale notebook artifacts. |

---

## "Safe to Act Now" Shortlist

Execute in this order (each is independently safe):

```bash
# 1. Superseded wide parquets (3.4 G)
rm -rf outputs/reports/incytr_pair_mode/wide_pre_policy_20260601/

# 2. Drive-download zip archives, already extracted (1.3 G)
rm data/incytr_frozen/outputs/*.zip

# 3. Parity probe single-parquet dir (2.8 G)
rm -rf outputs/reports/incytr_pair_mode/_confirm_top300/

# 4. T-cells donor2 smoke run, superseded (1.6 G)
rm -rf outputs/reports/incytr_pair_mode_tcells/donor2/wide_smoke/

# 5. Byte-identical duplicates in sce4_source (475 M total)
rm "data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/aggexp.csv"
rm "data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/ps_yuyu_deconvoluted.csv"

# 6. sce4 parity probe run directories (620 M)
rm -rf outputs/reports/incytr_pair_mode/_sce4_full_q0/
rm -rf outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag/

# 7. Standalone DEG_PRG download dir — verify first, then delete (578 M)
diff <(ls "data/incytr_frozen/DEG_PRG_ma_2mo_AppP_WTyp_10302025-20260524T230649Z-3-001/DEG_PRG_ma_2mo_AppP_WTyp_10302025/" | sort) \
     <(ls "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/DEG_PRG_ma_2mo_AppP_WTyp_10302025/" | sort)
# If diff output is clean (all files present in outputs/), then:
rm -rf "data/incytr_frozen/DEG_PRG_ma_2mo_AppP_WTyp_10302025-20260524T230649Z-3-001/"

# 8. Archive factorial inputs (closed path) (1.5 G + 1.4 G)
rm -rf archive/incytr_factorial_inputs/
rm -rf archive/pre_levy19_2026-05-14/incytr_factorial/
```

**Conservative safe-now total: ~12.7 G**

---

## Conservative Reclaimable Estimate

| Category | GB |
|:---------|---:|
| Safe-now (above 8 actions) | ~12.7 |
| User-judgment: SEAAD 34G raw nuclei (D) | 34 |
| User-judgment: WMB-10Xv3-subset h5ads (E) | 51 |
| User-judgment: WHB-10Xv3 h5ads (F) | 35 |
| User-judgment: v1_8clusters provenance (C) | 1.3 |
| User-judgment: sobj_7res.RDS (B) | 0.8 |
| User-judgment: mea_substrate_sets CSV (G) | 1.3 |
| User-judgment: pre_levy19 archive remainder (A) | 2.6 |
| **Total potential** | **~139 G** |
| **Conservative (safe-now only)** | **~13 G** |

The biggest single lever is the atlas h5ads (D+E+F = 120 G) — all re-downloadable via `atlas_reference.py`, but slow to re-fetch. The safe-now actions alone bring the volume from 92% to ~90%.
