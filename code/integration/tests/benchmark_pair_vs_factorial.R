# Benchmark: pair Incytr vs factorial Incytr on a single sender-receiver pair.
#
# Goal: side-by-side numerical and wallclock comparison on identical synthetic
# data, with a single contrast (condB - condA) so factorial output is directly
# comparable to pair mode's one comparison.
#
# Three runs, all on the same input matrix:
#   1. Pair mode (legacy else branch on current HEAD).
#   2. Factorial mode with C=1 contrast (condB - condA).
#   3. Factorial mode with C=4 contrasts (illustrates per-contrast scaling).
#
# Audit context: native pair (commit 93b9881) is bitwise-identical to current
# pair on a 2-condition input — verified by run_degenerate_2cond.sh Check 1 —
# so "current pair" stands in for "native pair without performance" as well.
# Factorial has no pre-perf analog (perf was co-developed); the C=1 vs C=4
# columns characterize per-contrast scaling within the factorial branch.
#
# Usage: Rscript code/integration/tests/benchmark_pair_vs_factorial.R [INCYTR_DIR]

suppressMessages({
  args <- commandArgs(trailingOnly = TRUE)
  incytr_dir <- if (length(args) >= 1) args[1] else "../incytr"
  if (!dir.exists(incytr_dir)) {
    incytr_dir <- file.path(dirname(dirname(dirname(dirname(normalizePath(sys.frame(1)$ofile))))), "incytr")
  }
  pkgload::load_all(incytr_dir, quiet = TRUE)
})

set.seed(1234)

# ------------------------------------------------------------------------------
# Synthetic data: one sender-receiver pair, 10 animals, 2 conditions, 2 cell types
# ------------------------------------------------------------------------------
n_animals <- 10
animals_A <- sprintf("A%02d", 1:5)            # condA
animals_B <- sprintf("A%02d", 6:10)           # condB
animals   <- c(animals_A, animals_B)

cells_per_animal_per_type <- 6
cell_types <- c("TypeA", "TypeB")

cell_meta <- do.call(rbind, lapply(animals, function(a) {
  cond <- if (a %in% animals_A) "condA" else "condB"
  do.call(rbind, lapply(cell_types, function(ct) {
    data.frame(
      cell    = paste0(a, "_", ct, "_", seq_len(cells_per_animal_per_type)),
      animal  = a,
      labels  = ct,
      condition = cond,
      stringsAsFactors = FALSE
    )
  }))
}))
rownames(cell_meta) <- cell_meta$cell

n_genes <- 12
gene_names <- paste0("G", seq_len(n_genes))

# Per-animal × per-cell-type baselines, with a small condition shift
animal_baseline <- matrix(abs(rnorm(n_animals * n_genes, mean = 1.5, sd = 0.4)),
                          nrow = n_genes, ncol = n_animals,
                          dimnames = list(gene_names, animals))
# Inject a condition-driven shift on a subset of genes
shift_genes <- c("G3", "G5", "G7", "G10")
animal_baseline[shift_genes, animals_B] <- animal_baseline[shift_genes, animals_B] * 1.6

# Generate cells from animal baselines + small noise
mat <- matrix(0, nrow = n_genes, ncol = nrow(cell_meta),
              dimnames = list(gene_names, cell_meta$cell))
for (i in seq_len(nrow(cell_meta))) {
  a <- cell_meta$animal[i]
  mat[, i] <- pmax(0, animal_baseline[, a] + rnorm(n_genes, sd = 0.1))
}

DB <- list(
  data.frame(from = c("G1", "G2", "G9"),
             to   = c("G3", "G4", "G10"), stringsAsFactors = FALSE),
  data.frame(from = c("G3", "G4", "G10"),
             to   = c("G5", "G6", "G11"), stringsAsFactors = FALSE),
  data.frame(from   = c("G5", "G6", "G11"),
             to     = c("G7", "G8", "G12"),
             source = c("src1", "src2", "src1"), stringsAsFactors = FALSE)
)

# ------------------------------------------------------------------------------
# Run 1: pair mode
# ------------------------------------------------------------------------------
run_pair <- function() {
  obj <- create_Incytr(mat, meta = cell_meta,
                       sender = "TypeA", receiver = "TypeB",
                       conditions = c("condA", "condB"))
  obj <- pathway_inference(obj, DB)
  obj <- Expr_bygroup(obj)
  obj <- Cal_SigProb(obj)
  obj <- Cal_scFC(obj)
  obj <- Pathway_evaluation(obj)
  obj <- Cal_PDS(obj)
  obj
}

# ------------------------------------------------------------------------------
# Run 2 / 3: factorial mode
# ------------------------------------------------------------------------------
animal_cond <- setNames(c(rep("condA", 5), rep("condB", 5)), animals)
design <- model.matrix(~ animal_cond)        # 10 x 2: (Intercept, animal_condcondB)
rownames(design) <- animals
colnames(design) <- c("Intercept", "condB_vs_condA")

contrasts_C1 <- list(
  condB_vs_condA = c(0, 1)
)
contrasts_C4 <- list(
  condB_vs_condA = c(0, 1),
  intercept_only = c(1, 0),     # not biologically meaningful; tests scaling
  half_shift     = c(0, 0.5),
  combined       = c(0.5, 0.5)
)

run_factorial <- function(contr) {
  obj <- create_Incytr(mat, meta = cell_meta,
                       sender = "TypeA", receiver = "TypeB",
                       conditions = c("condA", "condB"),
                       animal_id = "animal",
                       design = design,
                       contrasts = contr)
  obj <- pathway_inference(obj, DB)
  obj <- Expr_bygroup(obj)
  obj <- Expr_bygroup_animal(obj, min_cells = 3)
  obj <- Cal_SigProb(obj, compute_fc = FALSE)
  obj <- Cal_SigProb_animal(obj)
  obj <- Contrast_SigProb(obj)
  obj <- Cal_scFC(obj)
  obj <- Pathway_evaluation(obj)
  obj <- Cal_PDS(obj)
  obj
}

cat("\n=== Benchmark: pair vs factorial Incytr ===\n")
cat(sprintf("Synthetic: %d animals (%d condA / %d condB), %d cells/animal/type, %d genes, %d pathways\n",
            n_animals, length(animals_A), length(animals_B),
            cells_per_animal_per_type, n_genes, nrow(DB[[1]]) * nrow(DB[[2]]) / 1))

cat("\n--- Run 1: pair mode (current HEAD legacy else branch) ---\n")
t1 <- system.time(pair_obj <- run_pair())
cat(sprintf("Wallclock: user=%.3fs system=%.3fs elapsed=%.3fs\n",
            t1["user.self"], t1["sys.self"], t1["elapsed"]))

cat("\n--- Run 2: factorial mode, C=1 (condB - condA) ---\n")
t2 <- system.time(fact1_obj <- run_factorial(contrasts_C1))
cat(sprintf("Wallclock: user=%.3fs system=%.3fs elapsed=%.3fs\n",
            t2["user.self"], t2["sys.self"], t2["elapsed"]))

cat("\n--- Run 3: factorial mode, C=4 contrasts ---\n")
t3 <- system.time(fact4_obj <- run_factorial(contrasts_C4))
cat(sprintf("Wallclock: user=%.3fs system=%.3fs elapsed=%.3fs\n",
            t3["user.self"], t3["sys.self"], t3["elapsed"]))

# ------------------------------------------------------------------------------
# Numerical comparison: pathway-level TPDS and multimodel_score
# ------------------------------------------------------------------------------
cat("\n=== Numerical comparison (per pathway, condB - condA contrast) ===\n")

pair_eval <- pair_obj@Evaluation
fact1_eval <- fact1_obj@Evaluation

# Identify per-contrast TPDS / score column names in factorial output
tpds_col <- grep("^TPDS_condB_vs_condA$", colnames(fact1_eval), value = TRUE)
mm_col   <- grep("^multimodel_score_condB_vs_condA$", colnames(fact1_eval), value = TRUE)
if (length(tpds_col) == 0) tpds_col <- grep("^TPDS_", colnames(fact1_eval), value = TRUE)[1]
if (length(mm_col) == 0)   mm_col   <- grep("^multimodel_score_", colnames(fact1_eval), value = TRUE)[1]

cmp <- data.frame(
  Path = pair_eval$Path,
  pair_TPDS         = pair_eval$TPDS,
  factorial_TPDS    = fact1_eval[[tpds_col]],
  delta_TPDS        = fact1_eval[[tpds_col]] - pair_eval$TPDS,
  pair_score        = pair_eval$multimodel_score,
  factorial_score   = fact1_eval[[mm_col]],
  delta_score       = fact1_eval[[mm_col]] - pair_eval$multimodel_score,
  stringsAsFactors = FALSE
)

print(cmp, row.names = FALSE, digits = 4)

cat("\nSummary statistics on |delta|:\n")
cat(sprintf("  TPDS:            max=%.4g  mean=%.4g  median=%.4g\n",
            max(abs(cmp$delta_TPDS)), mean(abs(cmp$delta_TPDS)), median(abs(cmp$delta_TPDS))))
cat(sprintf("  multimodel_score: max=%.4g  mean=%.4g  median=%.4g\n",
            max(abs(cmp$delta_score)), mean(abs(cmp$delta_score)), median(abs(cmp$delta_score))))

# ------------------------------------------------------------------------------
# sc_FC comparison (sender side)
# ------------------------------------------------------------------------------
cat("\n=== sc_FC comparison (sender, condB - condA) ===\n")
cat(sprintf("pair sc_FC names:       %s\n", paste(names(pair_obj@sc_FC), collapse = ", ")))
cat(sprintf("factorial sc_FC names:  %s\n", paste(names(fact1_obj@sc_FC), collapse = ", ")))

pair_sc_sender <- pair_obj@sc_FC[[1]]
fact1_sc <- fact1_obj@sc_FC
contrast_nm <- names(fact1_sc)[1]
fact1_entry <- fact1_sc[[contrast_nm]]
cat(sprintf("factorial sc_FC[[%s]] structure: %s\n",
            contrast_nm, paste(names(fact1_entry), collapse = ", ")))
fact1_sc_sender <- fact1_entry$sender
cat(sprintf("pair sender rows=%d cols=[%s]\n",
            nrow(pair_sc_sender), paste(colnames(pair_sc_sender), collapse = ",")))
cat(sprintf("factorial sender rows=%d cols=[%s]\n",
            nrow(fact1_sc_sender), paste(colnames(fact1_sc_sender), collapse = ",")))

if (nrow(fact1_sc_sender) > 0 && nrow(pair_sc_sender) > 0) {
  pair_fc_col <- if ("log2FC" %in% colnames(pair_sc_sender)) "log2FC" else "log2FoldChange"
  fact_fc_col <- if ("log2FC" %in% colnames(fact1_sc_sender)) "log2FC" else "log2FoldChange"
  sc_cmp <- data.frame(
    Gene             = pair_sc_sender$gene_symbol,
    pair_log2FC      = pair_sc_sender[[pair_fc_col]],
    factorial_beta   = fact1_sc_sender[[fact_fc_col]][match(pair_sc_sender$gene_symbol,
                                                            fact1_sc_sender$gene_symbol)],
    stringsAsFactors = FALSE
  )
  sc_cmp$delta <- sc_cmp$factorial_beta - sc_cmp$pair_log2FC
  print(sc_cmp, row.names = FALSE, digits = 4)
  cat(sprintf("\nsc_FC sender: max|delta|=%.4g  mean|delta|=%.4g\n",
              max(abs(sc_cmp$delta), na.rm = TRUE),
              mean(abs(sc_cmp$delta), na.rm = TRUE)))
} else {
  cat("(skipping sc_FC numeric compare; one side empty)\n")
}

# ------------------------------------------------------------------------------
# Wallclock summary
# ------------------------------------------------------------------------------
cat("\n=== Wallclock summary ===\n")
cat(sprintf("  pair mode (1 comparison):   %.3fs elapsed\n", t1["elapsed"]))
cat(sprintf("  factorial mode (C=1):       %.3fs elapsed   ratio_vs_pair=%.2fx\n",
            t2["elapsed"], t2["elapsed"] / t1["elapsed"]))
cat(sprintf("  factorial mode (C=4):       %.3fs elapsed   ratio_vs_pair=%.2fx   ratio_vs_C1=%.2fx\n",
            t3["elapsed"], t3["elapsed"] / t1["elapsed"], t3["elapsed"] / t2["elapsed"]))

cat("\n=== Output column counts ===\n")
cat(sprintf("  pair Evaluation cols:                 %d\n", ncol(pair_obj@Evaluation)))
cat(sprintf("  factorial Evaluation cols (C=1):      %d\n", ncol(fact1_obj@Evaluation)))
cat(sprintf("  factorial Evaluation cols (C=4):      %d\n", ncol(fact4_obj@Evaluation)))

cat("\nDone.\n")
