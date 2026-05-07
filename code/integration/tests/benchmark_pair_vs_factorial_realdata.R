# Benchmark: pair vs factorial Incytr on real Song snRNA-seq data.
#
# Single sender-receiver pair (L2/3 IT -> Astrocyte). Compares pair mode
# (2-condition collapse: WT_2mo vs App_2mo, 4 animals total) against
# factorial mode (full 15-animal design with App_2mo - WT_2mo contrast).
#
# Both modes use the same pathway DB (edge_list_l1/l2/l3) and the same
# detection-threshold gene filter. Reports wallclock per stage and
# numerical comparison of TPDS/multimodel_score.
#
# Usage: Rscript code/integration/tests/benchmark_pair_vs_factorial_realdata.R [INCYTR_DIR]

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
incytr_dir <- if (length(args) >= 1) args[1] else "../incytr"

suppressMessages(pkgload::load_all(incytr_dir, quiet = TRUE))

repo_root <- normalizePath(getwd())
fac_dir   <- file.path(repo_root, "code", "integration", "intermediates", "factorial")
int_dir   <- file.path(repo_root, "code", "integration", "intermediates")

# --------------------------------------------------------------------------
# 1. Load shared inputs
# --------------------------------------------------------------------------
cat("=== 1. Load shared inputs ===\n")
t_load <- system.time({
  mat <- as(readMM(file.path(fac_dir, "expression_matrix.mtx")), "dgCMatrix")
  genes    <- read.csv(file.path(fac_dir, "expression_genes.csv"))$gene
  barcodes <- read.csv(file.path(fac_dir, "expression_barcodes.csv"))$barcode
  rownames(mat) <- genes; colnames(mat) <- barcodes

  meta <- read.csv(file.path(fac_dir, "expression_metadata.csv"),
                   row.names = 1, check.names = FALSE)
  animal_md <- read.csv(file.path(fac_dir, "animal_metadata.csv"),
                        row.names = 1, check.names = FALSE)

  L1 <- read.csv(file.path(int_dir, "edge_list_l1.csv"))[, c("from", "to")]
  L2 <- read.csv(file.path(int_dir, "edge_list_l2.csv"))[, c("from", "to")]
  L3 <- read.csv(file.path(int_dir, "edge_list_l3.csv"))[, c("from", "to")]
  DB <- list(L1, L2, L3)
})
cat(sprintf("  loaded mat %d x %d, meta %d rows, animals %d, DB layers %d/%d/%d in %.2fs\n",
            nrow(mat), ncol(mat), nrow(meta), nrow(animal_md),
            nrow(L1), nrow(L2), nrow(L3), t_load["elapsed"]))

# --------------------------------------------------------------------------
# 2. Subset to one sender-receiver pair
# --------------------------------------------------------------------------
SENDER   <- "L2/3 IT"
RECEIVER <- "Astrocyte"
cat(sprintf("\n=== 2. Subset to pair: %s -> %s ===\n", SENDER, RECEIVER))

pair_cells <- rownames(meta)[meta$labels %in% c(SENDER, RECEIVER)]
mat_pair   <- mat[, pair_cells, drop = FALSE]
meta_pair  <- meta[pair_cells, , drop = FALSE]

# Detection-threshold gene filter (>=10% in either cell type)
detect_rate <- function(M) Matrix::rowMeans(M > 0)
gene_keep <- detect_rate(mat_pair[, meta_pair$labels == SENDER]) >= 0.10 |
             detect_rate(mat_pair[, meta_pair$labels == RECEIVER]) >= 0.10
mat_pair  <- mat_pair[gene_keep, , drop = FALSE]
cat(sprintf("  cells: %d (%s=%d, %s=%d)\n",
            ncol(mat_pair), SENDER, sum(meta_pair$labels == SENDER),
            RECEIVER, sum(meta_pair$labels == RECEIVER)))
cat(sprintf("  genes after >=10%% detection filter: %d\n", nrow(mat_pair)))

# --------------------------------------------------------------------------
# 3. Pair mode setup: collapse to 2 conditions (WT_2mo vs App_2mo)
# --------------------------------------------------------------------------
PAIR_CONDS <- c("WT_2mo", "App_2mo")
animals_2mo <- animal_md[animal_md$timepoint == "2mo" &
                         animal_md$genotype %in% c("WTyp", "AppP"), ]
animals_2mo$condition <- ifelse(animals_2mo$genotype == "WTyp", "WT_2mo", "App_2mo")

meta_pair2 <- meta_pair
meta_pair2$condition <- animals_2mo$condition[match(meta_pair2$animal_id,
                                                     rownames(animals_2mo))]
keep_pair <- !is.na(meta_pair2$condition)
mat_pair2  <- mat_pair[, keep_pair, drop = FALSE]
meta_pair2 <- meta_pair2[keep_pair, , drop = FALSE]
cat(sprintf("\n=== 3. Pair-mode subset: WT_2mo vs App_2mo, %d cells, %d animals ===\n",
            ncol(mat_pair2), length(unique(meta_pair2$animal_id))))

# --------------------------------------------------------------------------
# 4. Factorial mode setup: all 15 animals + App_2mo - WT_2mo contrast
# --------------------------------------------------------------------------
design_cols <- c("const", "App", "Tau", "Int", "time_4mo", "time_6mo",
                 "App_x_time4", "App_x_time6", "Tau_x_time4", "Tau_x_time6")
design <- as.matrix(animal_md[, design_cols])
rownames(design) <- rownames(animal_md)

# Contrast: App_2mo - WT_2mo
# At 2mo, time_4mo=time_6mo=0, so App_2mo = const + App; WT_2mo = const.
# Difference = App term only.
contrasts_fac <- list(
  App_2mo_vs_WT_2mo = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
)
cat(sprintf("\n=== 4. Factorial-mode setup: %d animals, contrast App_2mo - WT_2mo ===\n",
            nrow(design)))

# --------------------------------------------------------------------------
# 5. Run pair mode
# --------------------------------------------------------------------------
cat("\n=== 5. Run pair mode ===\n")
t_pair <- system.time({
  obj_p <- create_Incytr(mat_pair2, meta = meta_pair2,
                         sender = SENDER, receiver = RECEIVER,
                         conditions = PAIR_CONDS)
  obj_p <- pathway_inference(obj_p, DB)
  obj_p <- Expr_bygroup(obj_p)
  obj_p <- Cal_SigProb(obj_p, cutoff_SigProb = 0.05)
  obj_p <- Cal_scFC(obj_p)
  obj_p <- Pathway_evaluation(obj_p)
  obj_p <- Cal_PDS(obj_p)
})
cat(sprintf("  enumerated %d pathways, scored %d\n",
            nrow(obj_p@pathways), nrow(obj_p@Evaluation)))
cat(sprintf("  wallclock: user=%.2fs sys=%.2fs elapsed=%.2fs\n",
            t_pair["user.self"], t_pair["sys.self"], t_pair["elapsed"]))

# --------------------------------------------------------------------------
# 5b. Pair mode on ALL WT vs ALL App animals (pooled across timepoints).
# Compares against factorial full (15 animals), so we can see what pair gets
# when given roughly the same animal pool but no design structure.
# --------------------------------------------------------------------------
cat("\n=== 5b. Pair mode (WT vs App pooled across timepoints) ===\n")
animals_wt_app <- animal_md[animal_md$genotype %in% c("WTyp", "AppP"), ]
animals_wt_app$condition <- ifelse(animals_wt_app$genotype == "WTyp", "WT_pooled", "App_pooled")

meta_pairW <- meta_pair
meta_pairW$condition <- animals_wt_app$condition[match(meta_pairW$animal_id,
                                                        rownames(animals_wt_app))]
keep_pairW <- !is.na(meta_pairW$condition)
mat_pairW  <- mat_pair[, keep_pairW, drop = FALSE]
meta_pairW <- meta_pairW[keep_pairW, , drop = FALSE]
cat(sprintf("  %d cells, %d animals (WT=%d, App=%d)\n",
            ncol(mat_pairW), length(unique(meta_pairW$animal_id)),
            sum(unique(meta_pairW$animal_id) %in%
                rownames(animals_wt_app)[animals_wt_app$genotype == "WTyp"]),
            sum(unique(meta_pairW$animal_id) %in%
                rownames(animals_wt_app)[animals_wt_app$genotype == "AppP"])))

t_pairW <- system.time({
  obj_pW <- create_Incytr(mat_pairW, meta = meta_pairW,
                          sender = SENDER, receiver = RECEIVER,
                          conditions = c("WT_pooled", "App_pooled"))
  obj_pW <- pathway_inference(obj_pW, DB)
  obj_pW <- Expr_bygroup(obj_pW)
  obj_pW <- Cal_SigProb(obj_pW, cutoff_SigProb = 0.05)
  obj_pW <- Cal_scFC(obj_pW)
  obj_pW <- Pathway_evaluation(obj_pW)
  obj_pW <- Cal_PDS(obj_pW)
})
cat(sprintf("  enumerated %d pathways, scored %d, elapsed=%.2fs\n",
            nrow(obj_pW@pathways), nrow(obj_pW@Evaluation), t_pairW["elapsed"]))

# --------------------------------------------------------------------------
# 6a. Run factorial mode RESTRICTED to the same 4 animals as pair mode.
# Isolates the algorithmic difference (Hill order + OLS-vs-log-ratio) from
# the "more data" effect that the full-15-animal factorial run mixes in.
# --------------------------------------------------------------------------
cat("\n=== 6a. Run factorial mode (restricted to same 4 animals as pair) ===\n")
animals_pair4 <- unique(meta_pair2$animal_id)
design_p4 <- cbind(const = 1,
                   App = ifelse(animal_md[animals_pair4, "genotype"] == "AppP", 1, 0))
rownames(design_p4) <- animals_pair4
contrasts_p4 <- list(App_vs_WT = c(0, 1))

t_fac_r <- system.time({
  obj_fr <- create_Incytr(mat_pair2, meta = meta_pair2,
                          sender = SENDER, receiver = RECEIVER,
                          conditions = PAIR_CONDS,
                          animal_id = "animal_id",
                          design = design_p4,
                          contrasts = contrasts_p4)
  obj_fr <- pathway_inference(obj_fr, DB)
  obj_fr <- Expr_bygroup(obj_fr)
  obj_fr <- Expr_bygroup_animal(obj_fr, min_cells = 5)
  obj_fr <- Cal_SigProb(obj_fr, compute_fc = FALSE, cutoff_SigProb = 0.05)
  obj_fr <- Cal_SigProb_animal(obj_fr)
  obj_fr <- Contrast_SigProb(obj_fr)
  obj_fr <- Cal_scFC(obj_fr)
  obj_fr <- Pathway_evaluation(obj_fr)
  obj_fr <- Cal_PDS(obj_fr)
})
cat(sprintf("  enumerated %d pathways, scored %d\n",
            nrow(obj_fr@pathways), nrow(obj_fr@Evaluation)))
cat(sprintf("  wallclock: elapsed=%.2fs\n", t_fac_r["elapsed"]))

# --------------------------------------------------------------------------
# 6b. Run factorial mode with full 15-animal design
# --------------------------------------------------------------------------
cat("\n=== 6b. Run factorial mode (full 15-animal design) ===\n")
t_fac <- system.time({
  obj_f <- create_Incytr(mat_pair, meta = meta_pair,
                         sender = SENDER, receiver = RECEIVER,
                         conditions = unique(meta_pair$animal_id)[1:1],  # placeholder
                         animal_id = "animal_id",
                         design = design,
                         contrasts = contrasts_fac)
  # Set conditions from genotype to match pair convention
  obj_f@meta$condition <- factor(meta_pair$genotype)
  obj_f@conditions <- as.character(levels(obj_f@meta$condition))
  obj_f <- pathway_inference(obj_f, DB)
  obj_f <- Expr_bygroup(obj_f)
  obj_f <- Expr_bygroup_animal(obj_f, min_cells = 5)
  obj_f <- Cal_SigProb(obj_f, compute_fc = FALSE, cutoff_SigProb = 0.05)
  obj_f <- Cal_SigProb_animal(obj_f)
  obj_f <- Contrast_SigProb(obj_f)
  obj_f <- Cal_scFC(obj_f)
  obj_f <- Pathway_evaluation(obj_f)
  obj_f <- Cal_PDS(obj_f)
})
cat(sprintf("  enumerated %d pathways, scored %d\n",
            nrow(obj_f@pathways), nrow(obj_f@Evaluation)))
cat(sprintf("  wallclock: user=%.2fs sys=%.2fs elapsed=%.2fs\n",
            t_fac["user.self"], t_fac["sys.self"], t_fac["elapsed"]))

# --------------------------------------------------------------------------
# 7. Compare numerical outputs
# --------------------------------------------------------------------------
cat("\n=== 7. Numerical comparison ===\n")

pe <- obj_p@Evaluation
fe <- obj_f@Evaluation

tpds_col <- grep("^TPDS_", colnames(fe), value = TRUE)[1]
mm_col   <- grep("^multimodel_score_", colnames(fe), value = TRUE)[1]

common <- intersect(pe$Path, fe$Path)
cat(sprintf("  pair pathways: %d, factorial pathways: %d, common: %d\n",
            nrow(pe), nrow(fe), length(common)))

# Pair convention: aFC = log2(cond1/cond2) = log2(WT_2mo/App_2mo) -> WT>App is positive.
# Factorial contrast c(0,1,...): estimates App term -> App>WT is positive.
# Negate pair_TPDS to put both on App-vs-WT direction for fair comparison.
cmp <- data.frame(
  Path = common,
  pair_TPDS       = -pe$TPDS[match(common, pe$Path)],   # sign-flipped to App-vs-WT
  factorial_TPDS  =  fe[[tpds_col]][match(common, fe$Path)],
  pair_score      = -pe$multimodel_score[match(common, pe$Path)],
  factorial_score =  fe[[mm_col]][match(common, fe$Path)]
)
cmp$delta_TPDS  <- cmp$factorial_TPDS  - cmp$pair_TPDS
cmp$delta_score <- cmp$factorial_score - cmp$pair_score
cat("(pair_TPDS shown sign-flipped to align with factorial App>WT direction)\n")

cat("\nFirst 12 pathways:\n")
print(head(cmp[, c("Path", "pair_TPDS", "factorial_TPDS", "delta_TPDS",
                   "pair_score", "factorial_score", "delta_score")], 12),
      row.names = FALSE, digits = 4)

cat("\nDelta summary across all common pathways:\n")
cat(sprintf("  TPDS:  max|d|=%.4g  mean|d|=%.4g  median|d|=%.4g  cor=%.4f\n",
            max(abs(cmp$delta_TPDS), na.rm = TRUE),
            mean(abs(cmp$delta_TPDS), na.rm = TRUE),
            median(abs(cmp$delta_TPDS), na.rm = TRUE),
            cor(cmp$pair_TPDS, cmp$factorial_TPDS, use = "complete.obs")))

# Restricted-factorial vs pair (apples-to-apples: same 4 animals, only Hill+OLS differs)
fer <- obj_fr@Evaluation
tpds_col_r <- grep("^TPDS_", colnames(fer), value = TRUE)[1]
common_r <- intersect(pe$Path, fer$Path)
cmp_r <- data.frame(
  Path = common_r,
  pair_TPDS         = -pe$TPDS[match(common_r, pe$Path)],
  factorial_r_TPDS  =  fer[[tpds_col_r]][match(common_r, fer$Path)]
)

# Pair-pooled (WT vs App across timepoints) vs factorial-full (15 animals)
peW <- obj_pW@Evaluation
common_W_full <- intersect(peW$Path, fe$Path)
cmp_W_full <- data.frame(
  Path = common_W_full,
  pair_pooled_TPDS  = -peW$TPDS[match(common_W_full, peW$Path)],
  factorial_TPDS    =  fe[[tpds_col]][match(common_W_full, fe$Path)]
)
# Pair-pooled vs pair-2mo-only (different pair-mode contrasts on same data)
common_pp <- intersect(pe$Path, peW$Path)
cmp_pp <- data.frame(
  Path = common_pp,
  pair_2mo_TPDS    = -pe$TPDS[match(common_pp, pe$Path)],
  pair_pooled_TPDS = -peW$TPDS[match(common_pp, peW$Path)]
)

cat("\n--- Comparisons matrix ---\n")
cat(sprintf("  factorial restricted (4 animals)    vs pair (4 animals, 2mo):           cor=%.3f sign-agree=%.0f%% mean|d|=%.3g\n",
            cor(cmp_r$pair_TPDS, cmp_r$factorial_r_TPDS, use = "complete.obs"),
            100 * mean(sign(cmp_r$pair_TPDS) == sign(cmp_r$factorial_r_TPDS), na.rm = TRUE),
            mean(abs(cmp_r$factorial_r_TPDS - cmp_r$pair_TPDS), na.rm = TRUE)))
cat(sprintf("  factorial full (15 animals)         vs pair (4 animals, 2mo):           cor=%.3f sign-agree=%.0f%% mean|d|=%.3g\n",
            cor(cmp$pair_TPDS, cmp$factorial_TPDS, use = "complete.obs"),
            100 * mean(sign(cmp$pair_TPDS) == sign(cmp$factorial_TPDS), na.rm = TRUE),
            mean(abs(cmp$delta_TPDS), na.rm = TRUE)))
cat(sprintf("  factorial full (15 animals)         vs pair pooled (8 animals, WT-App): cor=%.3f sign-agree=%.0f%% mean|d|=%.3g\n",
            cor(cmp_W_full$pair_pooled_TPDS, cmp_W_full$factorial_TPDS, use = "complete.obs"),
            100 * mean(sign(cmp_W_full$pair_pooled_TPDS) == sign(cmp_W_full$factorial_TPDS), na.rm = TRUE),
            mean(abs(cmp_W_full$factorial_TPDS - cmp_W_full$pair_pooled_TPDS), na.rm = TRUE)))
cat(sprintf("  pair pooled (8 animals)             vs pair (4 animals, 2mo):           cor=%.3f sign-agree=%.0f%% mean|d|=%.3g\n",
            cor(cmp_pp$pair_2mo_TPDS, cmp_pp$pair_pooled_TPDS, use = "complete.obs"),
            100 * mean(sign(cmp_pp$pair_2mo_TPDS) == sign(cmp_pp$pair_pooled_TPDS), na.rm = TRUE),
            mean(abs(cmp_pp$pair_pooled_TPDS - cmp_pp$pair_2mo_TPDS), na.rm = TRUE)))
cat(sprintf("  score: max|d|=%.4g  mean|d|=%.4g  median|d|=%.4g  cor=%.4f\n",
            max(abs(cmp$delta_score), na.rm = TRUE),
            mean(abs(cmp$delta_score), na.rm = TRUE),
            median(abs(cmp$delta_score), na.rm = TRUE),
            cor(cmp$pair_score, cmp$factorial_score, use = "complete.obs")))

cat("\nSign-agreement (TPDS): pair and factorial agree on direction in",
    sprintf("%.1f%% of pathways\n",
            100 * mean(sign(cmp$pair_TPDS) == sign(cmp$factorial_TPDS),
                       na.rm = TRUE)))

# --------------------------------------------------------------------------
# 8. Wallclock summary
# --------------------------------------------------------------------------
cat("\n=== 8. Wallclock summary ===\n")
cat(sprintf("  pair (4 animals, 2mo only):               %.2fs\n", t_pair["elapsed"]))
cat(sprintf("  pair pooled (8 animals, WT vs App):       %.2fs\n", t_pairW["elapsed"]))
cat(sprintf("  factorial restricted (4 animals, C=1):    %.2fs   (%.2fx pair-2mo)\n",
            t_fac_r["elapsed"], t_fac_r["elapsed"] / t_pair["elapsed"]))
cat(sprintf("  factorial full (15 animals, C=1):         %.2fs   (%.2fx pair-2mo)\n",
            t_fac["elapsed"], t_fac["elapsed"] / t_pair["elapsed"]))

cat("\nDone.\n")
