#!/usr/bin/env Rscript
# Per-donor T-cell input_gene_list.csv = DEG ∪ prG per state.
#
# Method mirrors alz/incytr_pair/build_input_gene_list.R (mouse) but:
#   - condition vocabulary is `d{day}` (e.g. "d13"), not "ma_<age>_<geno>"
#   - prG is computed from the donor's pr_deconvoluted.csv (state-keyed
#     `d{day}_{state}` columns), using the same quantile-normalization rule the
#     scorer applies (limma::normalizeBetweenArrays on the floored [c1,c2] pr
#     columns -> log2 -> |log2FC|>1). Matches incytr_commandline.R:161-172.
#   - HEG is dropped — sce4-parity override #1 says no top_n(500) cap, but the
#     receiver rule for pair-mode is DEG ∪ prG, and prG already covers the
#     proteomically-regulated half. Adding HEG here would inflate the receiver
#     set with high-expression genes that aren't differentially regulated.
#
# Outputs (under data/derived/tcells_incytr_inputs/<donor>/):
#   allmarkers.csv       raw FindAllMarkers output (Type_condition idents)
#   input_gene_list.csv  (gene, cluster) union of DEG + prG, dedup
#
# Usage:  pixi run Rscript alz/incytr_pair/build_tcells_input_gene_list.R <donor1|donor2>
suppressPackageStartupMessages({
  library(Seurat)
  library(stringr)
  library(Matrix)
  library(presto)
  library(future)
  library(limma)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
donor <- args[[1]]

REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
DONOR_DIR <- file.path(REPO_ROOT, "data", "derived", "tcells_incytr_inputs", donor)
OBJ_PATH  <- file.path(DONOR_DIR, "incytr_obj.rds")
PR_PATH   <- file.path(DONOR_DIR, "pr_deconvoluted.csv")
stopifnot(file.exists(OBJ_PATH), file.exists(PR_PATH))

N_WORKERS <- min(4L, max(1L, parallel::detectCores() - 1L))
plan(multisession, workers = N_WORKERS)
options(future.globals.maxSize = 8 * 1024^3)
cat("[input_gene_list] future plan: multisession, workers=", N_WORKERS, "\n",
    sep = "")

cat("[input_gene_list] reading", OBJ_PATH, "\n")
obj <- readRDS(OBJ_PATH)
cat("[input_gene_list] dim:", paste(dim(obj), collapse = " x "), "\n")
stopifnot("Type" %in% colnames(obj@meta.data),
          "condition" %in% colnames(obj@meta.data),
          "Type_condition" %in% colnames(obj@meta.data))

Idents(obj) <- "Type_condition"
cat("[input_gene_list] Type_condition idents (n=",
    length(levels(Idents(obj))), ")\n", sep = "")

t0 <- Sys.time()
cat("[input_gene_list] FindAllMarkers (only.pos=TRUE, presto + future)\n")
markers <- FindAllMarkers(
  obj,
  only.pos        = TRUE,
  logfc.threshold = 1.2,
  verbose         = FALSE
)
cat("[input_gene_list] markers rows:", nrow(markers), "\n")
write.csv(markers, file.path(DONOR_DIR, "allmarkers.csv"), row.names = FALSE)

# DEG: avg_log2FC > 1.5 & p_val < 1e-4. Recover Type from Type_condition by
# stripping the `_d<digits>` suffix (Type is alphanumeric; condition is `d<int>`).
deg <- markers[markers$avg_log2FC > 1.5 & markers$p_val < 1e-4, ]
deg$cluster <- sub("_d\\d+$", "", as.character(deg$cluster))
deg <- unique(deg[, c("gene", "cluster")])
cat("[input_gene_list] DEG (gene, cluster) rows:", nrow(deg), "\n")

# prG: from pr_deconvoluted.csv columns `d{day}_{state}`. For each state, take
# the set of conditions (days) present; pairwise quantile-normalize each
# (later, baseline=d2) pair and union the |log2FC|>1 genes. This is the same
# rule the driver's prg_by_cluster applies per-pair contrast, lifted to the
# state-level gene set for input_gene_list (so a gene flagged on any later
# day vs d2 is admitted to that state's receiver set).
pr <- read.csv(PR_PATH, check.names = FALSE)
stopifnot("gene_symbol" %in% colnames(pr))
genes_pr <- as.character(pr$gene_symbol)
val_cols <- setdiff(colnames(pr), "gene_symbol")
m <- do.call(rbind, strsplit(val_cols, "_", fixed = TRUE))
stopifnot(ncol(m) == 2L)  # `d<int>_<state>`
col_day   <- m[, 1]
col_state <- m[, 2]

states <- sort(unique(col_state))
baseline_day <- "d2"
stopifnot(baseline_day %in% col_day)

prg_rows <- vector("list", length(states))
for (i in seq_along(states)) {
  st <- states[i]
  base_col_idx <- which(col_day == baseline_day & col_state == st)
  later_days <- setdiff(unique(col_day[col_state == st]), baseline_day)
  if (length(base_col_idx) != 1L || length(later_days) == 0L) next
  base_vec <- as.numeric(pr[[val_cols[base_col_idx]]])
  base_vec <- pmax(base_vec, 1)  # floor — matches incytr_commandline.R override #2
  hits <- character(0)
  for (ld in later_days) {
    li <- which(col_day == ld & col_state == st)
    if (length(li) != 1L) next
    later_vec <- pmax(as.numeric(pr[[val_cols[li]]]), 1)
    yn <- limma::normalizeBetweenArrays(cbind(later_vec, base_vec))
    abs_lfc <- abs(log2(yn[, 1] / yn[, 2]))
    keep <- is.finite(abs_lfc) & abs_lfc > 1
    hits <- union(hits, genes_pr[keep])
  }
  if (length(hits) > 0L) {
    prg_rows[[i]] <- data.frame(gene = hits, cluster = st, stringsAsFactors = FALSE)
  }
}
prg <- do.call(rbind, prg_rows)
if (is.null(prg)) prg <- data.frame(gene = character(), cluster = character())
cat("[input_gene_list] prG (gene, cluster) rows:", nrow(prg), "\n")

out <- unique(rbind(deg, prg))
out_path <- file.path(DONOR_DIR, "input_gene_list.csv")
write.csv(out, out_path, row.names = FALSE)
cat("[input_gene_list] combined unique (gene, cluster):", nrow(out), "\n")
cat("[input_gene_list] DONE in",
    round(difftime(Sys.time(), t0, units = "mins"), 2), "min\n")
