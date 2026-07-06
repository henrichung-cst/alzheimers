#!/usr/bin/env Rscript
# Pair-mode driver for the levy_t5 spine.
#
# Calls Incytr::Cal_pairwise_grid (upstream `Incytr` R package,
# ~/Projects/work/incytr/) ONCE PER PAIR with the pair's own tight gene
# filters (DG_sender ≈ 6K, DG_receiver ≈ 6K). The whole-grid-shared
# pathway_inference path was tried and abandoned: it requires
# gene.use_* = union over all 31 clusters ≈ 28K (essentially the whole
# transcriptome), which blew past the 30 GiB box on enumeration alone.
# Per-pair calls keep enumeration small (matching the historical
# per-pair runIncytr memory profile) while still routing through the
# upstream code paths that carry the sce4-parity defaults.
#
# The per-(gene, cluster, condition) trimean is pair-invariant, so it is
# computed ONCE over the union of all per-cluster gene.use sets and injected
# into every per-pair Cal_pairwise_grid call (expr_bygroup =), instead of
# recomputing Expr_bygroup 961x per contrast (Incytr perf Improvement 3).
#
# sce4 parity: five of the six historical sce4-calibration overrides are
# now upstream defaults (Cal_SigProb correction=0.01, cutoff_SigProb=0;
# Cal_scFC correction=0.01; Cal_PDS cutoff_PDS=0; Expr_bygroup mean_method
# = NULL trimean). Locked by tests/testthat/test-sce4_defaults.R in the
# Incytr package. The remaining two stay driver-side because they depend
# on AD-project inputs:
#   - DG no-cap + per-contrast gene.use (assembled here): per cluster, THIS
#     contrast's two conditions' DEG (one-vs-rest markers from allmarkers.csv,
#     avg_log2FC > 1) ∪ prG (genes with |aFC| > 1 in the cluster's deconvoluted
#     proteome). Per-contrast (NOT the union over all 12 genotype x age splits).
#     NO HEG: sce4 used DEG ∪ prG only (zero HEG labels in its reference).
#   - pmax(pr_*, 1) floor on deconvoluted proteomics (just below)
# Receiver gene.use = DEG ∪ prG per cluster (t-cells) / sce4's frozen per-pair
# node sets (AD, §6.7). Investigation history:
# bench/bench.md A31–A33; archive/sce4_reproduction_2026-06-08/README.md §6.5/§6.7.
# Verification gate: `pixi run verify-incytr-sce4`.
#
# Usage (from any working directory):
#   Rscript alz/incytr_pair/incytr_commandline.R <condition1> <condition2>
# The DEG gene.use is derived per-contrast from allmarkers.csv in INPUTS_DIR.

suppressPackageStartupMessages({
  library(Incytr)
  library(Seurat)
  library(readr)
  library(dplyr)
  library(arrow)
  library(data.table)
  library(parallel)
  library(DBI)
  library(duckdb)
})

REPO_ROOT  <- system("git rev-parse --show-toplevel", intern = TRUE)
# INPUTS_DIR can be overridden via env. The sce4 parity gate points this at
# the frozen 46-cluster provenance inputs (data/incytr_frozen/v2_46clusters);
# production runs use the derived levy_t5 31-cluster inputs (the default).
INPUTS_DIR <- Sys.getenv("INPUTS_DIR_OVERRIDE", unset = "")
if (!nzchar(INPUTS_DIR)) {
  INPUTS_DIR <- file.path(REPO_ROOT, "data", "derived", "incytr_inputs")
}
# OUTPUT_DIR can be overridden via env (used by smoke runs to write into a
# scratch directory without disturbing the canonical wide/ outputs).
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR_OVERRIDE", unset = "")
if (!nzchar(OUTPUT_DIR)) {
  OUTPUT_DIR <- file.path(REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide")
}
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# BACKBONE_OUT_DIR: root dir for backbone grain shards, parallel to wide/.
# Set to empty string to disable backbone emission (used by verify-incytr-sce4
# which only checks path parity).  When not set at all, defaults to the
# canonical backbone path.  Distinction between "not set" (NA) and "set to
# empty string" (deliberate disable) is made via unset=NA_character_.
.backbone_env <- Sys.getenv("BACKBONE_OUT_DIR", unset = NA_character_)
BACKBONE_OUT_DIR <- if (is.na(.backbone_env)) {
  # Not set in environment → canonical default
  file.path(REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "backbone")
} else {
  # Explicitly set (including to "" for deliberate disable)
  .backbone_env
}
rm(.backbone_env)

# =====================================================================
# Parallelism / memory config.
#
# The pair loop runs NPAIR_WORKERS pairs concurrently via mclapply over the
# 961-pair grid (each fork runs one single-pair Cal_pairwise_grid, writes its
# own shard, returns the path). Forks COW-inherit the post-rm(Data.input) base
# (~3 GB). Per audit 2026-05-27 Experiment A, in-fork gc() was REMOVED — it
# cost ~12% wall and ~400 MB cgroup memory.peak with no compensating reclaim;
# the parent still gc()s between scheduling waves. Measured on this 30 GB box:
# W=3 peaks ~13–15 GB (zero oom under a 24 G cgroup cap) at a near-linear
# 2.77× speedup. Whether the post-gc-removal footprint admits W=4 is pending
# Experiment B (parallel sweep re-run). ALWAYS run inside the cgroup scope in
# alz/incytr_pair/README.md. Permutation inside each pair stays single-core
# (perm.n.cores=NPERM_WORKERS, default 1) to avoid nested forking.
# See bench/bench.md.
# =====================================================================
N_PAIR_WORKERS <- as.integer(Sys.getenv("NPAIR_WORKERS", unset = "3"))
N_PERM_WORKERS <- as.integer(Sys.getenv("NPERM_WORKERS", unset = "1"))
PAIR_LIMIT     <- as.integer(Sys.getenv("PAIR_LIMIT",    unset = "0"))
NBOOT          <- as.integer(Sys.getenv("NBOOT",         unset = "100"))
stopifnot(!is.na(N_PAIR_WORKERS), N_PAIR_WORKERS >= 1L)
stopifnot(!is.na(N_PERM_WORKERS), N_PERM_WORKERS >= 1L)
stopifnot(!is.na(PAIR_LIMIT),     PAIR_LIMIT     >= 0L)
stopifnot(!is.na(NBOOT),          NBOOT          >= 0L)

Incytr::assert_core_budget(
  c(pairs = N_PAIR_WORKERS, perms = N_PERM_WORKERS),
  reserve = 2L
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: incytr_commandline.R <condition1> <condition2>", call. = FALSE)
}
condition1 <- args[1]
condition2 <- args[2]

cat(sprintf("[pair-driver] contrast=%s vs %s  nboot=%d  pair_workers=%d  perm_workers=%d  pair_limit=%d\n",
            condition1, condition2, NBOOT, N_PAIR_WORKERS, N_PERM_WORKERS, PAIR_LIMIT))

# =====================================================================
# Inputs
# =====================================================================
# Species-specific Ligand/Receptor/EM/Target DBs. Mouse for the AD cohort,
# human for the T-cell cohort. Gene symbols in the substrate must match the
# DB's casing (mouse: title-case `App`; human: upper `APP`); a mismatch
# silently yields 0 enumerated paths.
SPECIES <- tolower(Sys.getenv("SPECIES", unset = "mouse"))
stopifnot(SPECIES %in% c("mouse", "human"))
DB.M <- if (SPECIES == "human") {
  list(Incytr::DB_Layer1_human_filtered,
       Incytr::DB_Layer2_human_filtered,
       Incytr::DB_Layer3_human_filtered)
} else {
  list(Incytr::DB_Layer1_mouse_filtered,
       Incytr::DB_Layer2_mouse_filtered,
       Incytr::DB_Layer3_mouse_filtered)
}
cat(sprintf("[pair-driver] species=%s\n", SPECIES))

Data.input <- readRDS(file.path(INPUTS_DIR, "incytr_obj.rds"))
Data.input@meta.data$Type <- Data.input@active.ident
# Mouse Seurat carries `Group` (= "ma_<age>_<geno>"); T-cell Seurat sets
# `condition` directly (= "d<day>") in build_tcells_seurat.R. Use Group when
# present, otherwise trust the builder's `condition`.
if (!is.null(Data.input@meta.data$Group)) {
  Data.input@meta.data$condition <- as.factor(Data.input@meta.data$Group)
} else if (!is.null(Data.input@meta.data$condition)) {
  Data.input@meta.data$condition <- as.factor(Data.input@meta.data$condition)
} else {
  stop("Seurat metadata has neither `Group` nor `condition`", call. = FALSE)
}

# Channel set + filenames + gene-key column are env-parameterized so the same
# driver runs both the mouse cohort (3-channel, "Gene Symbol" key) and the
# T-cell cohort (donor1 3-channel, donor2 2-channel — no IMAC — with
# `gene_symbol` key). Defaults reproduce the mouse path byte-exactly.
CHANNELS    <- strsplit(Sys.getenv("CHANNELS", unset = "pr,py,ps"), ",", fixed = TRUE)[[1]]
CHANNELS    <- trimws(CHANNELS[nzchar(CHANNELS)])
stopifnot(all(CHANNELS %in% c("pr", "py", "ps", "Ack", "KGG")))
stopifnot("pr" %in% CHANNELS)  # pr drives prG receiver gene-set — required
PR_FILE     <- Sys.getenv("PR_FILE", unset = "pr_yuyu_deconvoluted.csv")
PY_FILE     <- Sys.getenv("PY_FILE", unset = "py_yuyu_deconvoluted.csv")
PS_FILE     <- Sys.getenv("PS_FILE", unset = "ps_yuyu_deconvoluted.csv")
PR_GENE_COL <- Sys.getenv("PR_GENE_COL", unset = "Gene Symbol")
PY_GENE_COL <- Sys.getenv("PY_GENE_COL", unset = "gene_symbol")
PS_GENE_COL <- Sys.getenv("PS_GENE_COL", unset = "gene_symbol")
# Ack (acetylation) and KGG (ubiquitination) are optional PTM channels. They are
# 5xFAD-specific: Song and T-cells have no AcK/KGG data. When ACK_FILE/KGG_FILE
# are unset (empty string, the default), the channels are inactive and the driver
# is byte-identical to a phospho-only run. The Incytr package already has Ack_FC
# and KGG_FC slots and scores them end-to-end — no package changes required.
# Do NOT apply floor_pr to Ack/KGG: the floor is a pr-specific sce4 parity
# constant for deconvolution residuals in the total proteome.
ACK_FILE    <- Sys.getenv("ACK_FILE",    unset = "")
KGG_FILE    <- Sys.getenv("KGG_FILE",    unset = "")
ACK_GENE_COL <- Sys.getenv("ACK_GENE_COL", unset = "gene_symbol")
KGG_GENE_COL <- Sys.getenv("KGG_GENE_COL", unset = "gene_symbol")
cat(sprintf("[pair-driver] channels=[%s]  pr_file=%s  pr_gene_col=%s\n",
            paste(CHANNELS, collapse = ","), PR_FILE, PR_GENE_COL))

pr  <- read_csv(file.path(INPUTS_DIR, PR_FILE))
py  <- if ("py"  %in% CHANNELS) read_csv(file.path(INPUTS_DIR, PY_FILE))  else NULL
ps  <- if ("ps"  %in% CHANNELS) read_csv(file.path(INPUTS_DIR, PS_FILE))  else NULL
ack <- if ("Ack" %in% CHANNELS && nzchar(ACK_FILE))
         read_csv(file.path(INPUTS_DIR, ACK_FILE)) else NULL
kgg <- if ("KGG" %in% CHANNELS && nzchar(KGG_FILE))
         read_csv(file.path(INPUTS_DIR, KGG_FILE)) else NULL

# Slice pr/ps/py into condition1 / condition2 wide-by-cluster tables.
# Each table is one row per gene_symbol, one column per cluster, suffixed
# `_pr` / `_ps` / `_py`. Mean-collapse duplicate gene rows.
#
# Column-name match is anchored: `^<condition>_` (NOT substring), so condition
# "d2" does not accidentally also pick up "d20_..." / "d13_..." columns on the
# T-cell cohort. Mouse columns "ma_2mo_AppP_<cluster>" still match cleanly.
slice_omics <- function(df, gene_col, condition, suffix) {
  pat <- paste0("^", condition, "_")
  out <- dplyr::select(df, matches(pat))
  colnames(out) <- paste0(sub(pat, "", colnames(out)), "_", suffix)
  out$gene_symbol <- df[[gene_col]]
  out %>% group_by(gene_symbol) %>% summarise_all(mean, na.rm = TRUE)
}
# When slice_omics finds no matching condition columns the result is a 1-col
# (gene_symbol-only) frame — a non-NULL empty table that Incytr cannot handle.
# Coerce to NULL so the layer is silently skipped, identical to a file that was
# never loaded. This fires when an optional PTM assay was not collected for a
# given timepoint (e.g. hippocampus ACK is 6mo/12mo only).
null_if_empty <- function(x, label) {
  if (!is.null(x) && ncol(x) <= 1L) {
    cat(sprintf("[pair-driver] %s: no condition columns found — treating as absent\n", label))
    return(NULL)
  }
  x
}
pr_1  <- slice_omics(pr, PR_GENE_COL, condition1, "pr")
pr_2  <- slice_omics(pr, PR_GENE_COL, condition2, "pr")
ps_1  <- if ("ps"  %in% CHANNELS) slice_omics(ps,  PS_GENE_COL,  condition1, "ps")  else NULL
ps_2  <- if ("ps"  %in% CHANNELS) slice_omics(ps,  PS_GENE_COL,  condition2, "ps")  else NULL
py_1  <- if ("py"  %in% CHANNELS) slice_omics(py,  PY_GENE_COL,  condition1, "py")  else NULL
py_2  <- if ("py"  %in% CHANNELS) slice_omics(py,  PY_GENE_COL,  condition2, "py")  else NULL
ack_1 <- null_if_empty(if (!is.null(ack)) slice_omics(ack, ACK_GENE_COL, condition1, "Ack") else NULL, paste0("Ack/", condition1))
ack_2 <- null_if_empty(if (!is.null(ack)) slice_omics(ack, ACK_GENE_COL, condition2, "Ack") else NULL, paste0("Ack/", condition2))
kgg_1 <- null_if_empty(if (!is.null(kgg)) slice_omics(kgg, KGG_GENE_COL, condition1, "KGG") else NULL, paste0("KGG/", condition1))
kgg_2 <- null_if_empty(if (!is.null(kgg)) slice_omics(kgg, KGG_GENE_COL, condition2, "KGG") else NULL, paste0("KGG/", condition2))

# sce4 reproduction (driver-side override #2): floor pr values < 1 to 1.
# pr_yuyu carries ~1e-5 deconvolution residuals where sce4-era values were
# hard zeros; without this Cal_foldchange's zero-check (which adds 1e-4
# only on exact zeros) never fires, producing 13-log2 outliers on
# one-sided genes (Olfm2, Fkbp15) vs sce4. Floor-to-1 matches sce4 within
# 0.015 log2 on every bug driver. Validated in bench/regen/.
floor_pr <- function(df) {
  num_cols <- setdiff(colnames(df), "gene_symbol")
  for (cc in num_cols) df[[cc]] <- pmax(df[[cc]], 1)
  df
}
pr_1 <- floor_pr(pr_1)
pr_2 <- floor_pr(pr_2)

# kldata (kinase-substrate library) is mouse-only for the AD cohort. For the
# T-cell cohort the kinase track lives on bulk (per meeting notes Stream D);
# pair-mode skips it via kldata=NULL. USE_KLDATA env controls.
USE_KLDATA <- as.logical(Sys.getenv("USE_KLDATA", unset = "TRUE"))
stopifnot(!is.na(USE_KLDATA))
if (USE_KLDATA) {
  kldata <- read_csv(file.path(INPUTS_DIR, "kldata.csv"))
  kldata <- kldata[, c("gene", "site_pos", "motif.geneName")]
} else {
  kldata <- NULL
  cat("[pair-driver] USE_KLDATA=FALSE  -> kinase scoring skipped (kldata=NULL)\n")
}
# Per-contrast DEG substrate, condition-keyed so the gene.use is assembled from
# only THIS contrast's two conditions (not all 12 genotype x age splits — that
# union was the dominant over-emission source vs sce4; see
# archive/sce4_reproduction_2026-06-08/README.md §6.2). Built per cluster below
# once `clusters` is known.
#   allmarkers.csv  one-vs-rest FindAllMarkers, cluster = "<cluster>_<condition>",
#                   run BROAD (logfc.threshold = 0.1) to match sce4's frozen
#                   table. DEG = the two contrast conditions' markers
#                   (avg_log2FC > 1 & p_val < 1e-4).
# sce4 used DEG ∪ prG, NO HEG (zero HEG labels across its ma_2mo + ma_4mo
# reference).
# See archive/sce4_reproduction_2026-06-08/README.md §6.5.
# Gene.use SOURCE (per-cohort, not a fallback toggle). When SCE4_GENEUSE_DIR is
# set and holds this contrast's reconstructed sce4 gene.use, the AD run CONSUMES
# sce4's own per-PAIR node sets (alz/incytr_pair/extract_sce4_geneuse.R, read off
# sce4's pre-cap per-pair pathway rds) INSTEAD of deriving DEG∪prG — this is what
# reproduces sce4's Allpathway exactly (the engine is monotonic in the gene set;
# feeding sce4's per-pair nodes yields 0 extra / 0 missing). The T-cell cohort has
# no sce4 reference, leaves SCE4_GENEUSE_DIR unset, and derives. Two legitimate
# sources for two datasets, selected by data availability — NOT a "switch back" flag.
# Record: archive/sce4_reproduction_2026-06-08/README.md §6.7
SCE4_GENEUSE_DIR <- Sys.getenv("SCE4_GENEUSE_DIR", unset = "")
geneuse_csv <- if (nzchar(SCE4_GENEUSE_DIR))
  file.path(SCE4_GENEUSE_DIR, paste0(condition1, "_", condition2, ".csv")) else ""
use_frozen_geneuse <- nzchar(geneuse_csv) && file.exists(geneuse_csv)
if (nzchar(SCE4_GENEUSE_DIR) && !use_frozen_geneuse) {
  stop(sprintf("SCE4_GENEUSE_DIR is set but no frozen gene.use for contrast %s_%s at %s",
               condition1, condition2, geneuse_csv), call. = FALSE)
}

# DEG markers are only needed when deriving gene.use (T-cell cohort).
if (!use_frozen_geneuse) {
  allmarkers <- read_csv(file.path(INPUTS_DIR, "allmarkers.csv"))
  stopifnot(all(c("gene", "cluster", "avg_log2FC", "p_val") %in% colnames(allmarkers)))
}

# =====================================================================
# Gene-set construction (per-cluster, tight)
# =====================================================================
# sce4 reproduction (driver-side override #1, "DG no-cap"): per-cluster DG
# is the union of this contrast's per-cluster DEGs (the two contrast conditions'
# markers, built below) with the proteomically-regulated gene set (prG): genes
# with |pr_log2FC| > 1 for that cluster.
#
# CRITICAL: prG must use the SAME pr_log2FC that the Incytr scorer computes,
# not a raw column ratio. Incytr::integrate_omics_layer (analysis.R) runs
# limma::normalizeBetweenArrays() on the cluster's [cond1, cond2] pr columns
# (quantile normalization across all genes) BEFORE Cal_foldchange. That
# re-maps each value by its within-column rank, so the raw deconvoluted
# ratio and the scored pr_log2FC differ substantially (e.g. Acvr1 in
# Cholinergic-Neurons: raw log2(50.74/28.22)=0.85 < 1, but quantile-
# normalized = ~2.0 > 1). Computing prG on the raw ratio dropped exactly the
# genes the scorer flags as pr-significant, which is why the Microglia->
# Cholinergic benchmark lost all 600 paths (Receptor overlap 0). prG now
# mirrors integrate_omics_layer term-for-term: normalize the floored pr
# columns, then |log2(c1_norm / c2_norm)| > 1. No correction term — floor_pr
# guarantees values >= 1 so Cal_foldchange's zero-correction never fires.
#
# This replaces the previous "full proteome" receiver (all pr genes with
# finite fold-magnitude after floor-to-1 ≈ all ~6k proteome genes) with
# a per-cluster tight set where proteomics is differentially regulated.
# Investigation history in bench/bench.md A31–A35.

# prG = proteomically-regulated genes per cluster via the package/paper-default
# aFC: |aFC| > 1, where aFC = quantile-normalized pr_log2FC * a magnitude
# adjustment (min(2*Vmax^2/(Vmax^2+a^2), 1), a = pr.q quantile) — see
# Incytr::proteomics_gene / R/math.R. pr_log2FC itself is computed the way the
# Incytr scorer computes it: quantile normalization (limma::normalizeBetweenArrays)
# on the cluster's [cond1, cond2] pr columns then log2, NOT the raw column ratio.
# This gene-selection is method logic and lives in the package. The AD app supplies
# only the floored pr tables, the strict |aFC| > 1 cutoff (strict=TRUE), and the
# rownames intersect. floor_pr guarantees values >= 1, so Cal_foldchange's
# zero-correction never fires. pr_1 / pr_2 share columns named <cluster>_pr;
# clusters absent from pr get character(0).
#
# WHY aFC, not log2FC (2026-05-30): aFC is the paper/package default. Its
# magnitude term down-weights only modest-abundance genes near the pr.q quantile
# (for most prG genes Vmax >> pr.q so adj saturates to 1 and aFC == log2FC), so
# the enumeration narrowing is modest, NOT dramatic: Cholinergic prG 2109->1965
# (~7% fewer genes), Microglia->Cholinergic enumerated paths 30067->19758 (~1.5x;
# over-emission vs sce4's 1283 goes 23x->15x). Its one cost on that benchmark is
# Depdc5 — the Target of C1qa|Cr1l|Cbfb|Depdc5 — which lands at aFC=0.9974 vs sce4's
# recorded 1.000, a 0.003 input-provenance gap of the SAME class as Acvr1/App (our
# deconvolution differs from sce4's off-box pr at the third decimal on a boundary
# gene). This drops Micro->Cholin recall 573->572; the verify-incytr-sce4 recall
# floor was lowered 595->572 to accept this documented residual. We adopt aFC as
# the canonical (paper-default) style despite the modest narrowing. See §6i.
clusters <- as.character(unique(Data.input@meta.data$Type))

if (use_frozen_geneuse) {
  # AD reproduction: gene.use IS sce4's own per-cluster node set (REPLACES the
  # DEG∪prG derivation below for the AD cohort). label = sce4's own DEG/prG label
  # (DEG-priority), so label_node() stays faithful to the reference. App/Psen1
  # are already present here as prG nodes (sce4's transgene force-include is baked
  # into the reconstructed set) — no separate TRANSGENES add.
  cat(sprintf("[pair-driver] gene.use SOURCE = frozen sce4 per-pair (%s)\n", geneuse_csv))
  gu <- read_csv(geneuse_csv)
  stopifnot(all(c("sender", "receiver", "gene", "role", "label") %in% colnames(gu)))
  bad <- setdiff(unique(c(gu$sender, gu$receiver)), clusters)
  if (length(bad) > 0L) {
    stop("frozen gene.use references clusters absent from the data spine: ",
         paste(bad, collapse = ", "), call. = FALSE)
  }
  gu <- gu[gu$gene %in% rownames(Data.input), , drop = FALSE]
  # PER-PAIR gene.use. The engine gates all receiver positions by ONE flat
  # gene.use_Receiver; sce4's per-cluster INPUT set is not recoverable from its
  # gated artifacts, and a per-cluster appearing-node union recombines cross-pair
  # nodes into chains sce4 never enumerated (the gate can't remove them on
  # cell-sparse receivers). Feeding each pair its OWN node set reproduces sce4's
  # Allpathway pair-for-pair (1283/699; bench/perf/sce4_identity_test.R, §6). The
  # engine call below keys gus/gur by the (sender, receiver) pair.
  pkey <- paste(gu$sender, gu$receiver, sep = "\t")
  is_s <- gu$role == "S"
  gus_by_pair <- split(gu$gene[is_s],  pkey[is_s])
  gur_by_pair <- split(gu$gene[!is_s], pkey[!is_s])
  # per-cluster DEG/prG membership for label_node()'s .label output: a gene's
  # cluster is its sender in role S, its receiver in role R; DEG-priority.
  cl_of <- ifelse(is_s, gu$sender, gu$receiver)
  deg_by_cluster <- lapply(setNames(clusters, clusters), function(cl)
    unique(gu$gene[cl_of == cl & gu$label == "DEG"]))
  prg_by_cluster <- lapply(setNames(clusters, clusters), function(cl)
    unique(gu$gene[cl_of == cl & gu$label != "DEG"]))
  gus_by_cluster <- NULL; gur_by_cluster <- NULL
} else {
pr_clusters <- clusters[vapply(clusters, function(cl) {
  col <- paste0(cl, "_pr")
  col %in% colnames(pr_1) && col %in% colnames(pr_2)
}, logical(1))]
pg <- if (length(pr_clusters)) {
  Incytr::proteomics_gene(
    as.data.frame(pr_1), as.data.frame(pr_2),
    cell_group = pr_clusters, style = "aFC", cutoff = 1,
    pr.correction = 0.001, strict = TRUE
  )
} else NULL
# Transgene force-inclusion (sce4 parity). The AD model's human transgenes
# (App / Psen1 / Mapt) are the disease drivers, and sce4 force-includes them
# into prG for EVERY cluster regardless of their |aFC|, bypassing the |aFC|>1
# rule (App aFC ~= -0.01 in the deconvoluted proteome). Without this, App is a
# candidate ligand only in the 28/31 clusters where it happens to be a
# transcriptomic marker — NOT Microglia/Astrocytes/OPC — so the 27 App-ligand
# Microglia->Cholinergic paths sce4 reports never enumerate from the Microglia
# sender. Forcing them in recovers those paths (Micro->Cholin recall 572->599;
# verified bench/perf/test_parity_levers.sh). The paths carry our (flat) App
# transcript value, NOT sce4's saturated Ligand sclog2FC=7.65 — that value gap
# is a separate, documented transcript-provenance residual (the verify gate
# tolerates Ligand/EM |Δ| outliers when the position gene is App). The
# intersect with rownames self-gates: cohorts lacking these symbols (e.g. the
# human T-cell cohort) get an empty add and are unaffected. label_node leaves
# these as prG (matching sce4's Ligand.label), since they are not in the DEG arm.
TRANSGENES <- intersect(c("App", "Psen1", "Mapt"), rownames(Data.input))
prg_by_cluster <- lapply(setNames(clusters, clusters), function(cl) {
  if (is.null(pg)) return(TRANSGENES)
  union(intersect(pg$gene_symbol[pg$cluster == cl], rownames(Data.input)),
        TRANSGENES)
})

# Per-contrast DEG: union of the two contrast conditions' one-vs-rest markers
# (avg_log2FC > 1 & p_val < 1e-4) for this cluster. allmarkers' `cluster` is
# "<cluster>_<condition>", so the contrast's idents are "<cl>_<condition1/2>".
# DEG cutoff = avg_log2FC > 1 (symmetric with prG's |aFC| > 1). sce4's DEG-label
# membership is reproduced at > 1 (80/82 of its Ndnf receiver DEG nodes) and lost
# at > 1.5 (55/82); the former 1.5 dropped real DEGs HEG then re-admitted.
deg_keep <- allmarkers$avg_log2FC > 1 & allmarkers$p_val < 1e-4
deg_by_cluster <- lapply(setNames(clusters, clusters), function(cl) {
  idents <- paste0(cl, "_", c(condition1, condition2))
  genes  <- allmarkers$gene[deg_keep & allmarkers$cluster %in% idents]
  intersect(unique(genes), rownames(Data.input))
})
# Per-cluster gene.use_* for the per-pair Cal_pairwise_grid calls.
# Each cluster gets DEG ∪ prG — sce4's candidate set (no HEG). The T-cell cohort
# has no per-role reference, so it uses the same set for both roles.
dg_by_cluster <- lapply(setNames(clusters, clusters), function(cl) {
  union(deg_by_cluster[[cl]], prg_by_cluster[[cl]])
})
gus_by_cluster <- dg_by_cluster
gur_by_cluster <- dg_by_cluster
}
if (use_frozen_geneuse) {
  cat(sprintf("[pair-driver] per-pair gene.use: %d pairs; sender median=%d/max=%d  receiver median=%d/max=%d\n",
              length(gus_by_pair),
              as.integer(median(lengths(gus_by_pair))), max(lengths(gus_by_pair)),
              as.integer(median(lengths(gur_by_pair))), max(lengths(gur_by_pair))))
} else {
  cat(sprintf("[pair-driver] per-cluster gene.use sizes: sender median=%d/max=%d  receiver median=%d/max=%d\n",
              as.integer(median(lengths(gus_by_cluster))), max(lengths(gus_by_cluster)),
              as.integer(median(lengths(gur_by_cluster))), max(lengths(gur_by_cluster))))
}

# =====================================================================
# KsG — kinase-substrate gene layer (TOGGLE = kinase-data presence).
# When KSG_MEA_FILE is unset the whole block is skipped and the driver is
# byte-identical to a no-kinase run (this is the path verify-incytr-sce4
# exercises — it supplies no kinase inputs, so sce4 parity is untouched).
# When supplied, kinase-supported substrate genes (Incytr::kinase_substrate_gene,
# the method in the package) are admitted into gene.use per cluster (unioned in
# process_pair, so it applies to both frozen and derived bases). The selector is
# data presence, never a mode flag (mirrors SCE4_GENEUSE_DIR / ACK_FILE).
# =====================================================================
KSG_MEA_FILE   <- Sys.getenv("KSG_MEA_FILE", unset = "")
use_ksg        <- nzchar(KSG_MEA_FILE) && file.exists(KSG_MEA_FILE)
ksg_by_cluster <- NULL
if (use_ksg) {
  KSG_MEA_PY_FILE <- Sys.getenv("KSG_MEA_PY_FILE",      unset = "")
  KSG_MOTIF_FILE  <- Sys.getenv("KSG_MOTIF_FILE",       unset = "")
  KSG_ATTR_FILE   <- Sys.getenv("KSG_ATTRIBUTION_FILE", unset = "")
  KSG_CONTRAST    <- Sys.getenv("KSG_CONTRAST",         unset = "")
  if (!nzchar(KSG_MOTIF_FILE) || !nzchar(KSG_ATTR_FILE) || !nzchar(KSG_CONTRAST)) {
    stop("KsG active (KSG_MEA_FILE set) but KSG_MOTIF_FILE / KSG_ATTRIBUTION_FILE / KSG_CONTRAST missing",
         call. = FALSE)
  }
  mea <- as.data.frame(read_csv(KSG_MEA_FILE))
  if (nzchar(KSG_MEA_PY_FILE) && file.exists(KSG_MEA_PY_FILE)) {
    mea <- rbind(mea, as.data.frame(read_csv(KSG_MEA_PY_FILE)))
  }
  mea <- mea[!is.na(mea$contrast) & mea$contrast == KSG_CONTRAST, , drop = FALSE]
  motif_map   <- as.data.frame(read_csv(KSG_MOTIF_FILE))[, c("motif", "gene_symbol")]
  attribution <- as.data.frame(read_csv(KSG_ATTR_FILE))[, c("kinase", "cell_type")]

  ksg <- Incytr::kinase_substrate_gene(
    mea = mea, motif_map = motif_map, attribution = attribution,
    cell_group = clusters, fdr_cutoff = 0.25)
  # self-gate to the data matrix (same as prG/DEG); split per cluster.
  ksg <- ksg[ksg$gene_symbol %in% rownames(Data.input), , drop = FALSE]
  .by <- split(ksg$gene_symbol, ksg$cluster)
  ksg_by_cluster <- lapply(setNames(clusters, clusters), function(cl) unique(.by[[cl]]))
  cat(sprintf("[pair-driver] KsG active: contrast=%s  %d (gene,cluster) admitted across %d clusters\n",
              KSG_CONTRAST, nrow(ksg), sum(lengths(ksg_by_cluster) > 0)))
}

# Every pair's pathway genes are bounded by the union of all gene.use sets, so the
# per-(gene, cluster, condition) trimean — which is pair-invariant — is computed
# ONCE over that union below and injected into each per-pair grid call
# (Improvement 3), replacing 961 redundant Expr_bygroup passes per contrast.
# KsG genes are added here because per-pair KsG widening (lines below) expands
# gene_use_S/R beyond DEG∪prG; inject_precomputed_expr requires the substrate
# to cover every pathway node that enumerate_paths_dt can produce.
ksg_extra <- if (use_ksg) unlist(ksg_by_cluster, use.names = FALSE) else character(0)
gene_union <- if (use_frozen_geneuse) {
  intersect(unique(c(unique(gu$gene), ksg_extra)), rownames(Data.input))
} else {
  intersect(unique(c(unlist(gus_by_cluster, use.names = FALSE),
                     unlist(gur_by_cluster, use.names = FALSE),
                     ksg_extra)),
            rownames(Data.input))
}
cat(sprintf("[pair-driver] precompute-trimean gene union: %d genes\n",
            length(gene_union)))

# =====================================================================
# Template object + pairs grid
# =====================================================================
# sender/receiver here are placeholders — Cal_pairwise_grid overwrites
# `obj@sender` / `obj@receiver` on each per-pair clone.
template <- create_Incytr(
  object     = GetAssayData(Data.input, assay = DefaultAssay(Data.input), layer = "data"),
  meta       = Data.input@meta.data,
  sender     = clusters[1],
  receiver   = clusters[1],
  group.by   = "Type",
  conditions = c(condition1, condition2),
  assay      = NULL,
  do.sparse  = TRUE
)

pairs <- expand.grid(sender = clusters, receiver = clusters,
                     stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)

# PAIR_SUBSET: comma-separated `sender:receiver` list, taking precedence
# over PAIR_LIMIT. Used by smoke runs to hit exactly the parity-gate
# pairs without depending on cluster-ordering in Data.input.
pair_subset_env <- Sys.getenv("PAIR_SUBSET", unset = "")
if (nzchar(pair_subset_env)) {
  spec <- strsplit(pair_subset_env, ",", fixed = TRUE)[[1]]
  spec <- trimws(spec[nzchar(spec)])
  subset_df <- do.call(rbind, lapply(spec, function(p) {
    sr <- strsplit(p, ":", fixed = TRUE)[[1]]
    if (length(sr) != 2L) {
      stop(sprintf("PAIR_SUBSET entry '%s' is not 'sender:receiver'", p),
           call. = FALSE)
    }
    data.frame(sender = sr[1], receiver = sr[2], stringsAsFactors = FALSE)
  }))
  bad <- setdiff(unique(c(subset_df$sender, subset_df$receiver)), clusters)
  if (length(bad) > 0L) {
    stop("PAIR_SUBSET references unknown clusters: ",
         paste(bad, collapse = ", "), call. = FALSE)
  }
  pairs <- subset_df
  cat(sprintf("[pair-driver] PAIR_SUBSET active; running %d pairs\n",
              nrow(pairs)))
} else if (PAIR_LIMIT > 0L && PAIR_LIMIT < nrow(pairs)) {
  pairs <- pairs[seq_len(PAIR_LIMIT), , drop = FALSE]
  cat(sprintf("[pair-driver] PAIR_LIMIT active; running first %d pairs\n",
              nrow(pairs)))
}

multiomics_args <- list(
  pr.data_condition1  = pr_1,  pr.data_condition2  = pr_2,  pr.correction  = 0.001, pr.q  = NULL,
  ps.data_condition1  = ps_1,  ps.data_condition2  = ps_2,  ps.correction  = 0.001, ps.q  = NULL,
  py.data_condition1  = py_1,  py.data_condition2  = py_2,  py.correction  = 0.001, py.q  = NULL,
  Ack.data_condition1 = ack_1, Ack.data_condition2 = ack_2, Ack.correction = 0.001, Ack.q = NULL,
  KGG.data_condition1 = kgg_1, KGG.data_condition2 = kgg_2, KGG.correction = 0.001, KGG.q = NULL
)
# When ack_1/ack_2 or kgg_1/kgg_2 are NULL (the default when ACK_FILE/KGG_FILE
# are unset), Integr_multiomics skips those layers silently — the same pattern
# used for ps/py on donor2 T-cells. The phospho-only path is byte-identical.

# Condition->barcodes split must be captured before rm(Data.input) below.
cond_cells <- lapply(c(condition1, condition2), function(cc) {
  rownames(Data.input@meta.data)[
    as.character(Data.input@meta.data$condition) == cc]
})
names(cond_cells) <- c(condition1, condition2)

# =====================================================================
# Per-pair Cal_pairwise_grid loop
# =====================================================================
# Drop the redundant per-(node, omics) `_aFC` raw fold mirrors and the SiK
# sub-score breakdowns; the viewer keeps `<Node>_<omics>_log2FC` instead.
drop_pat <- "^(Ligand|Receptor|EM|Target)_(pr|ps|py|Ack|KGG)_aFC$|^SiK_(R|EM|T)_of_(EM|T|R)(_EI_.*)?$"
num_pat  <- "^(SigProb|p_value|SiK|log2FC|aFC|PDS|TPDS|PPDS|PhPDS|Ack_score|KGG_score|Rme1_score|multimodel_score|pr_|ps_|py_)"

# Shards persist across runs so an interrupted contrast resumes per-pair: a
# completed shard is reused, never recomputed. The dir is removed only after the
# final concat succeeds. A contrast that already finished is skipped upstream by
# the runner (its final parquet exists), so we only reach here for unfinished
# work. Shards are written atomically (tmp + rename) so a crash mid-write leaves
# either a complete shard or none — never a truncated one that resume would trust.
shard_dir <- file.path(OUTPUT_DIR, ".shards",
                       paste0(condition1, "_", condition2))
dir.create(shard_dir, recursive = TRUE, showWarnings = FALSE)

# Per-pair DEG/prG node labels. Pair-mode Export_results does not track
# per-node provenance; the upstream factorial path emitted these via
# assign_path_labels().
# Precedence DEG > prG > KsG: a "KsG" label marks a node admitted ONLY by
# kinase evidence (not a DEG/prG candidate). When use_ksg is FALSE, ksg is NULL,
# `set` reduces to union(deg, prg) and every member is overwritten to DEG/prG —
# so the output is byte-identical to the prior two-tier labelling.
label_node <- function(node_genes, cluster) {
  deg <- deg_by_cluster[[cluster]]
  prg <- prg_by_cluster[[cluster]]
  ksg <- if (use_ksg) ksg_by_cluster[[cluster]] else NULL
  set <- Reduce(union, list(deg, prg, ksg))
  lbl <- setNames(rep("KsG", length(set)), set)
  lbl[prg] <- "prG"
  lbl[deg] <- "DEG"
  unname(lbl[node_genes])
}

rss_mb <- function() {
  s <- "/proc/self/status"; if (!file.exists(s)) return(NA_real_)
  v <- grep("^VmRSS:", readLines(s, warn = FALSE), value = TRUE)
  if (!length(v)) NA_real_ else
    as.numeric(sub("^VmRSS:\\s+([0-9]+)\\s+kB$", "\\1", v)) / 1024
}
hwm_mb <- function() {
  s <- "/proc/self/status"; if (!file.exists(s)) return(NA_real_)
  v <- grep("^VmHWM:", readLines(s, warn = FALSE), value = TRUE)
  if (!length(v)) NA_real_ else
    as.numeric(sub("^VmHWM:\\s+([0-9]+)\\s+kB$", "\\1", v)) / 1024
}

# Free the redundant full Seurat object before the pair loop. Everything the
# loop reads — `template`, `gus/gur_by_cluster`, `deg/prg_by_cluster`, `pr/ps/py_*`,
# `kldata` — is already built; Data.input's counts + scale.data + meta are dead
# weight (~1.2 GB resident) that every Permutation_test fork would COW-inherit.
# The expression matrix stays alive via `template` (shared refcount), so this
# cannot change any computed value (verified byte-identical, max |Δ|=0 across
# all numeric columns on both sce4 benchmark pairs).
cat(sprintf("[pair-driver] rss before rm(Data.input): %.0f MB\n", rss_mb()))
rm(Data.input); gc(verbose = FALSE)
cat(sprintf("[pair-driver] rss after  rm(Data.input): %.0f MB\n", rss_mb()))

# Build the pair-invariant trimean substrate once (Improvement 3): the per-(gene,
# cluster, condition) trimean is pair-invariant, so it is computed ONCE over
# gene_union and injected into every per-pair Cal_pairwise_grid call
# (expr_bygroup =) instead of recomputing Expr_bygroup 961x. This is the method's
# Expr_bygroup, batched over an explicit gene set — it lives in the package
# (Incytr::precompute_expr_bygroup), gene-chunked to bound the dense submatrix.
# template@data shares the expression matrix (refcount) with the dropped
# Data.input, so this reads the same values. Forks below COW-inherit it read-only.
t_sub <- proc.time()[["elapsed"]]
expr_substrate <- Incytr::precompute_expr_bygroup(template@data, template@idents,
                                                  cond_cells, gene_union)
cat(sprintf("[pair-driver] built expr substrate in %.1fs  rss=%.0fMB\n",
            proc.time()[["elapsed"]] - t_sub, rss_mb()))

# One pair: run Cal_pairwise_grid, post-process, write its shard, return the
# shard path (or NA if the pair produced no rows / errored). Self-contained so
# it runs identically in-process (NPAIR_WORKERS=1) or in an mclapply fork.
process_pair <- function(i) {
  s <- pairs$sender[i]; r <- pairs$receiver[i]
  pair_df <- data.frame(sender = s, receiver = r, stringsAsFactors = FALSE)
  # Per-pair gene.use in frozen (AD reproduction) mode; per-cluster otherwise
  # (T-cell). A pair absent from sce4's Allpathway produced no gated chains -> its
  # gene.use is empty -> the engine emits 0 paths (caught below), matching sce4.
  # Guard NULL -> character(0): NULL would make pathway_inference default to ALL
  # genes (analysis.R), silently enumerating the whole genome for an empty pair.
  if (use_frozen_geneuse) {
    pk <- paste(s, r, sep = "\t")
    gene_use_S <- gus_by_pair[[pk]]; gene_use_R <- gur_by_pair[[pk]]
  } else {
    gene_use_S <- gus_by_cluster[[s]]; gene_use_R <- gur_by_cluster[[r]]
  }
  if (is.null(gene_use_S)) gene_use_S <- character(0)
  if (is.null(gene_use_R)) gene_use_R <- character(0)
  # KsG admission (toggle): union the kinase-supported genes for this pair's
  # sender (-> Ligand candidates) and receiver (-> Receptor/EM/Target). Placed
  # before the empty-gene.use check so KsG can also mint a pair sce4 never had.
  if (use_ksg) {
    if (!is.null(ksg_by_cluster[[s]])) gene_use_S <- union(gene_use_S, ksg_by_cluster[[s]])
    if (!is.null(ksg_by_cluster[[r]])) gene_use_R <- union(gene_use_R, ksg_by_cluster[[r]])
  }
  if (length(gene_use_S) == 0L || length(gene_use_R) == 0L) {
    cat(sprintf("[pair-driver] pair %d/%d  %s -> %s  empty gene.use (0 paths)\n",
                i, nrow(pairs), s, r))
    return(NA_character_)
  }
  sub_path <- file.path(shard_dir, sprintf("pair_%04d.parquet", i))
  if (file.exists(sub_path) && file.info(sub_path)$size > 0) {
    cat(sprintf("[pair-driver] pair %d/%d  %s -> %s  RESUME (shard exists)\n",
                i, nrow(pairs), s, r))
    return(sub_path)
  }
  t_one <- proc.time()[["elapsed"]]
  res <- tryCatch(
    Incytr::Cal_pairwise_grid(
      template          = template,
      pairs             = pair_df,
      DB                = DB.M,
      gene.use_Sender   = gene_use_S,
      gene.use_Receiver = gene_use_R,
      multiomics        = multiomics_args,
      kldata            = kldata,
      mean_method       = NULL,
      fold_threshold    = 10,
      n.cores           = 1L,
      nboot             = NBOOT,
      seed.use          = 1L,
      perm.n.cores      = N_PERM_WORKERS,
      expr_bygroup      = expr_substrate,
      backbone_out_dir  = if (nzchar(BACKBONE_OUT_DIR)) BACKBONE_OUT_DIR else NULL
    ),
    error = function(e) {
      warning(sprintf("[pair-driver] Cal_pairwise_grid failed for %s -> %s: %s",
                      s, r, conditionMessage(e)), call. = FALSE)
      NULL
    }
  )
  dt_sec <- proc.time()[["elapsed"]] - t_one
  obj <- if (!is.null(res)) res[[paste(s, r, sep = "__")]] else NULL
  out <- if (!is.null(obj)) tryCatch(Export_results(obj, indicator = TRUE),
                                     error = function(e) NULL) else NULL
  # No in-fork gc() — see audit 2026-05-27 Experiment A. `rm()` releases the
  # bindings; fork teardown returns the pages; an explicit gc() here dirties
  # COW shared pages with no compensating reclaim (measured -3.6% cgroup
  # memory.peak and -12% wall on W=2 after removal).
  rm(res, obj)

  if (is.null(out) || nrow(out) == 0L) {
    cat(sprintf("[pair-driver] pair %d/%d  %s -> %s  rows=0  %.1fs  rss=%.0fMB\n",
                i, nrow(pairs), s, r, dt_sec, rss_mb()))
    return(NA_character_)
  }
  out$Sender   <- s
  out$Receiver <- r
  out$Ligand.label   <- label_node(out$Ligand,   s)
  out$Receptor.label <- label_node(out$Receptor, r)
  out$EM.label       <- label_node(out$EM,       r)
  out$Target.label   <- label_node(out$Target,   r)
  if (use_frozen_geneuse && all(c("log2FC", "aFC", "TPDS", "multimodel_score", "PDS") %in% colnames(out))) {
    # sce4's frozen Pairwise RDS stores path-level SigProb aFC equal to log2FC
    # (Cal_foldchange q=0 behavior). The package default q=0.75 compresses
    # TPDS and moves threshold-edge rows across |PDS|=0.2.
    logi <- function(x, k = 2) 2 / (1 + exp(-k * x)) - 1
    nz <- function(x) {
      x[is.na(x)] <- 0
      x
    }
    out$aFC <- out$log2FC
    out$TPDS <- logi(out$aFC, 2)
    score_cols <- c("PPDS", "PhPDS_ps", "PhPDS_py",
                    "Ack_score", "KGG_score", "Rme1_score")
    for (cc in score_cols) {
      if (!cc %in% colnames(out)) out[[cc]] <- 0
    }
    out$multimodel_score <- out$TPDS +
      0.5 * (nz(out$PPDS) + nz(out$PhPDS_ps) + nz(out$PhPDS_py) +
               nz(out$Ack_score) + nz(out$KGG_score) + nz(out$Rme1_score))
    sik1 <- paste0("SiK_score_", condition1)
    sik2 <- paste0("SiK_score_", condition2)
    if (all(c(sik1, sik2) %in% colnames(out))) {
      s1v <- nz(out[[sik1]])
      s2v <- nz(out[[sik2]])
      out$PDS <- ifelse(
        out$multimodel_score > 0,
        out$multimodel_score + 0.5 * s1v,
        ifelse(
          out$multimodel_score < 0,
          out$multimodel_score - 0.5 * s2v,
          out$multimodel_score + 0.5 * (s1v - s2v)
        )
      )
    } else {
      out$PDS <- out$multimodel_score
    }
  }
  out <- out[, !grepl(drop_pat, colnames(out)), drop = FALSE]
  dt <- as.data.table(out)
  for (col in grep(num_pat, names(dt), value = TRUE)) {
    if (is.character(dt[[col]])) {
      set(dt, j = col, value = suppressWarnings(as.numeric(dt[[col]])))
    }
  }
  tmp_path <- paste0(sub_path, ".tmp")
  arrow::write_parquet(as.data.frame(dt), tmp_path, compression = "zstd")
  file.rename(tmp_path, sub_path)
  cat(sprintf("[pair-driver] pair %d/%d  %s -> %s  rows=%d  %.1fs  rss=%.0fMB  hwm=%.0fMB\n",
              i, nrow(pairs), s, r, nrow(out), dt_sec, rss_mb(), hwm_mb()))
  # No in-fork gc() here either — see Experiment A note above.
  rm(out, dt)
  sub_path
}

t_loop <- proc.time()[["elapsed"]]
results <- if (N_PAIR_WORKERS > 1L) {
  mclapply(seq_len(nrow(pairs)), process_pair,
           mc.cores = N_PAIR_WORKERS, mc.preschedule = FALSE)
} else {
  lapply(seq_len(nrow(pairs)), process_pair)
}
shard_paths <- unlist(results, use.names = FALSE)
shard_paths <- shard_paths[!is.na(shard_paths)]
cat(sprintf("[pair-driver] pair loop done in %.1f min (%d shards)\n",
            (proc.time()[["elapsed"]] - t_loop) / 60, length(shard_paths)))

if (length(shard_paths) == 0L) {
  stop("[pair-driver] no shards produced", call. = FALSE)
}

# =====================================================================
# Concat
# =====================================================================
out_path <- file.path(OUTPUT_DIR,
                      paste0(condition1, "_", condition2, "_incytr_output.parquet"))
# Stream the per-pair shards into one parquet via DuckDB. dplyr::collect() here
# materialized the full combined table (~all 961 pairs) in RAM before writing,
# which blew the 24G cgroup cap on the final concat (loop itself peaks ~5GB).
# DuckDB streams the read_parquet glob to disk and spills to DUCKDB_TEMP_DIR,
# staying well under the cap. union_by_name handles the per-shard schema
# differences that unify_schemas used to absorb.
tmp_out <- paste0(out_path, ".tmp")
con <- DBI::dbConnect(duckdb::duckdb())
DBI::dbExecute(con, "PRAGMA memory_limit='10GB'")
DBI::dbExecute(con, sprintf("PRAGMA temp_directory='%s'",
                            Sys.getenv("DUCKDB_TEMP_DIR",
                                       file.path(Sys.getenv("HOME"), ".cache/duckdb"))))
DBI::dbExecute(con, sprintf(
  "COPY (SELECT * FROM read_parquet([%s], union_by_name=true)) TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)",
  paste(sprintf("'%s'", shard_paths), collapse = ", "), tmp_out))
DBI::dbDisconnect(con, shutdown = TRUE)
file.rename(tmp_out, out_path)
sz <- file.info(out_path)$size
cat(sprintf("[pair-driver] wrote %s (%.1f MB)\n", out_path, sz / 1e6))
unlink(shard_dir, recursive = TRUE)

# =====================================================================
# Backbone shard concat — one parquet per grain, parallel to wide/ output.
# Shards written by .emit_backbone_shards inside Cal_pairwise_grid are
# concatenated here in the same DuckDB-stream pattern as path shards.
# Skipped when BACKBONE_OUT_DIR is empty (e.g. during verify-incytr-sce4).
# =====================================================================
if (nzchar(BACKBONE_OUT_DIR)) {
  contrast_key    <- paste(condition1, condition2, sep = "_")
  backbone_grains <- c("R-EM", "L-R-EM", "R-EM-T")
  for (grain in backbone_grains) {
    bshard_dir <- file.path(BACKBONE_OUT_DIR, grain, ".shards", contrast_key)
    bshards    <- if (dir.exists(bshard_dir))
                    sort(list.files(bshard_dir, pattern = "\\.parquet$", full.names = TRUE))
                  else character(0)
    if (length(bshards) == 0L) {
      cat(sprintf("[pair-driver] backbone %s/%s: no shards, skipping concat\n",
                  grain, contrast_key))
      next
    }
    bout_path <- file.path(BACKBONE_OUT_DIR, grain,
                           paste0(condition1, "_", condition2, "_backbone_output.parquet"))
    btmp_out  <- paste0(bout_path, ".tmp")
    dir.create(dirname(bout_path), recursive = TRUE, showWarnings = FALSE)
    bcon <- DBI::dbConnect(duckdb::duckdb())
    DBI::dbExecute(bcon, "PRAGMA memory_limit='10GB'")
    DBI::dbExecute(bcon, sprintf("PRAGMA temp_directory='%s'",
                                 Sys.getenv("DUCKDB_TEMP_DIR",
                                            file.path(Sys.getenv("HOME"), ".cache/duckdb"))))
    DBI::dbExecute(bcon, sprintf(
      "COPY (SELECT * FROM read_parquet([%s], union_by_name=true)) TO '%s' (FORMAT PARQUET, COMPRESSION ZSTD)",
      paste(sprintf("'%s'", bshards), collapse = ", "), btmp_out))
    DBI::dbDisconnect(bcon, shutdown = TRUE)
    file.rename(btmp_out, bout_path)
    bsz <- file.info(bout_path)$size
    cat(sprintf("[pair-driver] backbone %s/%s: wrote %s (%.1f MB, %d shards)\n",
                grain, contrast_key, bout_path, bsz / 1e6, length(bshards)))
    unlink(bshard_dir, recursive = TRUE)
  }
}
