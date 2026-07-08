# Incytr sce4 Reproduction — Status and Closed Dead Ends

Sources: `project_incytr_sce4_cutoffs_and_scoring_saturation.md`, `project_incytr_pair_pvalue_untrustworthy.md`, `project_incytr_proteome_boltdown_disqualified.md`

## Current production recipe (wired into `alz/incytr_pair/incytr_commandline.R`)

- **Sender gene.use:** DEG (this contrast's two conditions' markers, `avg_log2FC > 1 & p_val < 1e-4`) ∪ prG. No HEG. No `top_n(500)` cap.
- **Receiver gene.use for AD:** sce4's own pre-cap per-pair node sets extracted from `sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds` → `data/incytr_frozen/sce4_geneuse/<c1>_<c2>.csv`, consumed via `SCE4_GENEUSE_DIR`. Per-pair (not per-cluster) is required — a per-cluster union lets the engine recombine cross-pair nodes into chains sce4 never enumerated (Micro→Cholin ballooned to 2,952 under per-cluster).
- **Receiver gene.use for T-cell:** DEG ∪ prG derivation (no sce4 frozen gene.use; no parity gate).
- **prG definition:** `Incytr::proteomics_gene(style="aFC", cutoff=1, strict=TRUE)` — `|aFC| > 1` where `aFC = pr_log2FC · min(2·Vmax²/(Vmax²+a²), 1)`, and `pr_log2FC` is computed via `limma::normalizeBetweenArrays()` (quantile normalization across all genes) on floored `[cond1, cond2]` pr columns then `log2`. NOT the raw column ratio.
- **Transgene force-include:** `TRANSGENES <- intersect(c("App","Psen1","Mapt"), rownames(Data.input))` unioned into every cluster's `prg_by_cluster`.
- **Top300:** per-pair top-300 PDS↑ ∪ top-300 PDS↓.
- **SigProb/PDS driver cutoffs:** 0/0 (all paths emitted; downstream filter in `alz/incytr_pair/filter_significant_paths.py`).
- **Production filter:** `SigProb > 0.1 (either) AND |PDS| >= 0.2`, uncapped. No p_adj/FDR arm — sce4 never ran permutation test.

Full parity constants: see `CLAUDE.md` § "Pair-mode Incytr — sce4 parity constants".

## Parity status (2026-05-31, updated 2026-06-18)

| Benchmark | Enumeration recall | Cap (Top300) fidelity |
|---|---|---|
| Micro→Cholin (ma_2mo) | 599/600 (lone miss: Depdc5 knife-edge) | 53/600 — residual sender-ligand breadth (see below) |
| Ndnf×Ndnf (ma_2mo) | 599/600 | 276/600 |

**Gated path-set identity** (primary gate in `verify_sce4_parity.py`): symmetric diff vs sce4's Allpathway must be transgene-only. The top-300 cap fidelity is informational.

## Documented residuals (do not re-investigate)

### Depdc5 knife-edge (lone non-enumerated path)
aFC = 0.9974 vs sce4's recorded 1.000444. Proven 2026-05-30: the whole gap is in the `normalizeBetweenArrays` quantile-norm gene universe; scoring off sce4's OWN frozen pr gives aFC=0.99757. No input swap closes it. The `verify-incytr-sce4` recall floor is 599. Full mechanism: `docs/plans/sce4_reproduction.md` §4, §7.

### App transgene value residual
The 27 Micro→Cholin App-ligand paths enumerate (candidacy closed via force-include) but carry our flat App `Ligand_sclog2FC` ≈ 0.19 vs sce4's 7.65. Root cause: `hsAPP`=0 in Microglia, endogenous `App` is two-sided — this is a transcript-provenance gap not derivable from on-disk artifacts. Do NOT fabricate App's 7.65 to force value parity.

### Sender-ligand breadth (cap fidelity gap)
Our Microglia emits 8 ligands vs sce4's 4. Extra ligands: Ctsd, Adam17, Entpd1 — strong Microglia DEGs in sce4's own markers but absent from its 182-ligand universe. Ctsd→App alone = 5,226 spurious high-|PDS| paths that win the per-pair top-300 cap and evict real sce4 paths. The narrowing rule separating C1qa-kept from Ctsd-dropped is NOT determinable from on-disk artifacts (CellChat ligand-category hypothesis was REFUTED, §6.6). The `verify-incytr-sce4` now gates on `--min-cap-frac` so this no longer reports false-green.

### Acvr1/Grm7 sub-threshold ratio (irreducible, 2026-05-24)
Acvr1 in Cholinergic-Neurons is effectively n=1 vs n=1 (detected in one male animal per genotype of three). Every deconvolution variant gives ≈1.8×, never sce4's 4×. Do NOT lower the `|aFC|>1` cutoff to fit this threshold to one gene, and do NOT switch to both-sexes to chase it. Accepted as documented sparse-detection residual. Full record: `bench/bench.md` A34.

## Permanently disqualified: all-proteome receiver (global_prG bolt-on)

Passing the entire deconvoluted proteome (~6,067–6,407 genes) as `gene.use_Receiver` ("rule A") recovers ~595–600/600 of sce4's paths. This is permanently disqualified on two grounds:

1. **Correctness:** it is the absence of a filter, not a filter. Parity is a superset hit. A no-bolt-on gate is required.
2. **Cost:** ≈110K paths/pair ≈ 180× the T7 oracle minimum; across the 8,649-pair grid ≈ 10⁹ path-rows / ~17 single-core-days.

**Note (A24):** path count is NOT cubic in `|gene.use_Receiver|` — it is driven by the count of DB-connected receptors × EMs in the receiver list (its composition). A 5,743-gene presence-union produced 311K paths vs the 6,407-gene proteome's 110K.

This dead end has been re-derived as a "fresh finding" 3-4 times. Confirming "the proteome works" is not progress — it is the known dead end. Log any re-derivation as re-confirmation of the dead end. Reference: `bench/bench.md`, especially the recurring-relapse warning.

## Pair pvalue is untrustworthy — never use for filtering or ranking

The `pvalue` column in pair-mode Incytr output is untrustworthy (unlike factorial-mode's permutation-based pvalue). Two independent grounds:

1. **No calibrated null (production context):** the pair-mode driver does not produce a calibrated pvalue; treating it as significance leads to wrong conclusions.
2. **Degenerate in sce4 context (A28):** the 1-cell receiver (Cholinergic, 58/63,706 cells) gets ~58 cells on average under the global label shuffle → permuted SigProb ≫ observed → p≈0 for 92.8% of paths. All 573 reachable sce4 survivors tie at pvalue rank 1 with 102,245 others. Pvalue is RULED OUT as sce4's selection rule.

**Filter and rank on `|PDS|`** (production canonical default: `|PDS| >= 0.2` in `filter_significant_paths.py`). The `nboot=100 p_value_*` columns stay in `wide/` as informational only. Do NOT add a p_adj / FDR gate — it drops cell-sparse pairs (e.g. Micro→Cholin vanishes at 2mo).

## Pre-cap rds availability

5 of 9 contrasts have pre-cap rds on box; 4 (ma_2mo_ApTt, ma_2mo_Ttau, ma_4mo_ApTt, ma_4mo_Ttau) are pending from the Drive. Full record: `docs/plans/sce4_reproduction.md` §6.7.
