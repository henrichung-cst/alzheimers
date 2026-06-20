#!/usr/bin/env Rscript
# TODO #2: ProjecTILs projection of the FULL 10x NSCLC matrix (all 897,733 cells).
#
# scGate (inside Run.ProjecTILs filter.cells=TRUE) is the authoritative T-cell
# gate: we project EVERY cell and let scGate accept/reject it, rather than pre-
# filtering with coarse markers. The cohort's own attribution uses ProjecTILs the
# same way, so the public NSCLC reference lands in the SAME 14-state vocabulary
# and is directly comparable. Markers (nsclc_expression.py --label-clusters) now
# only label the non-T compartment (which ProjecTILs structurally cannot) and
# serve as a sanity check on these scGate calls.
#
#   Emits data/external/nsclc_10x/projectils_predictions.csv
#     barcode, lineage_gate, functional.cluster, functional.cluster.conf
#
# Memory-safe: the full matrix (1.3 B nnz, ~15 GB as a sparse dgCMatrix) is
# NEVER loaded whole. We hyperslab-read BATCH_CELLS-column ranges directly from
# the 10x CSC h5 via hdf5r, build a per-batch dgCMatrix (genes x batch), and
# project. plan(sequential) — a future worker would duplicate the ~1.5 GiB ref.
# Peak RAM = both refs (~3 GiB) + one batch's scGate working set. Run under a
# systemd MemoryMax scope.
#
# Usage:  pixi run Rscript alz/ingest/nsclc_projectils_map.R
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
  library(future)
  library(hdf5r)
  library(ProjecTILs)
  library(scGate)
  library(STACAS)
  library(UCell)
})

H5      <- "data/external/nsclc_10x/sample_feature_bc_matrix.h5"
OUT_CSV <- "data/external/nsclc_10x/projectils_predictions.csv"
REF_DIR <- "data/external/projectils"
CD8_REF <- file.path(REF_DIR, "CD8T_human_ref_v1.rds")
CD4_REF <- file.path(REF_DIR, "CD4T_human_ref_v1.rds")
stopifnot(file.exists(H5), file.exists(CD8_REF), file.exists(CD4_REF))

plan(sequential)
options(future.globals.maxSize = +Inf)

BATCH_CELLS <- 25000

memline <- function(tag) {
  gc(full = TRUE)
  mi <- tryCatch(readLines("/proc/meminfo", n = 3), error = function(e) character())
  avail <- sub(".*?:\\s*", "", grep("MemAvailable", mi, value = TRUE))
  cat(sprintf("[mem %-26s] MemAvailable: %s\n", tag, ifelse(length(avail), avail, "?")))
}

# Contiguous 1-D read [start1, start1+n-1] (1-based inclusive). hdf5r detects the
# regular range and reads it as a hyperslab; the transient index vector (~n
# doubles, ~290 MB for a 36 M-nnz batch) is freed immediately after.
read_slab <- function(ds, start1, n) {
  if (n <= 0) return(numeric(0))
  ds[start1:(start1 + n - 1)]
}

cat("==== NSCLC PROJECTILS MAP (full cohort) ====\n")
memline("start")

h5 <- H5File$new(H5, mode = "r")
m <- h5[["matrix"]]
shp <- as.integer(m[["shape"]]$read())
n_genes <- shp[1]; n_cells <- shp[2]
# 10x gene symbols contain duplicates; make.unique mirrors Read10X_h5 (the first
# occurrence keeps the canonical symbol, so kinase matching is unaffected).
gene_names <- make.unique(m[["features"]][["name"]]$read())
all_barcodes <- m[["barcodes"]]$read()
indptr <- as.numeric(m[["indptr"]]$read())   # length n_cells+1; pointer offsets
data_ds <- m[["data"]]
idx_ds  <- m[["indices"]]
cat(sprintf("  matrix: %d genes x %d cells; %s nnz\n",
            n_genes, n_cells, format(indptr[length(indptr)], big.mark = ",")))
memline("after h5 open")

cd8_ref <- readRDS(CD8_REF)
cd4_ref <- readRDS(CD4_REF)
memline("after refs loaded")

# Per-cell accumulators (indexed by global cell position).
gate <- rep("none", n_cells)        # CD8 | CD4 | doublet | none
fc   <- rep(NA_character_, n_cells)  # functional.cluster
conf <- rep(NA_real_, n_cells)

project_batch <- function(counts, ref) {
  obj <- CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0)
  q <- tryCatch(
    Run.ProjecTILs(query = obj, ref = ref,
                   filter.cells = TRUE, split.by = NULL, ncores = 1),
    error = function(e) { cat("    Run.ProjecTILs error:", conditionMessage(e), "\n"); NULL })
  if (is.null(q) || ncol(q) == 0) return(NULL)
  data.frame(barcode = colnames(q),
             state = as.character(q$functional.cluster),
             conf = as.numeric(q$functional.cluster.conf),
             stringsAsFactors = FALSE)
}

t0 <- Sys.time()
n_batches <- ceiling(n_cells / BATCH_CELLS)
for (b in seq_len(n_batches)) {
  c0 <- (b - 1) * BATCH_CELLS          # 0-based first col of batch
  c1 <- min(c0 + BATCH_CELLS, n_cells) # 0-based one-past-last
  ncol_b <- c1 - c0
  p0 <- indptr[c0 + 1]                  # nnz offset before col c0 (0-based val)
  p1 <- indptr[c1 + 1]                  # nnz offset before col c1
  nnz <- p1 - p0
  d <- as.numeric(read_slab(data_ds, p0 + 1, nnz))
  ii <- as.numeric(read_slab(idx_ds, p0 + 1, nnz))     # 0-based gene rows
  local_p <- indptr[(c0 + 1):(c1 + 1)] - p0            # length ncol_b+1
  cols <- (c0 + 1):c1
  mat <- sparseMatrix(i = ii + 1, p = local_p, x = d,
                      dims = c(n_genes, ncol_b),
                      dimnames = list(gene_names, all_barcodes[cols]))
  bc_pos <- setNames(cols, all_barcodes[cols])

  r8 <- project_batch(mat, cd8_ref)
  r4 <- project_batch(mat, cd4_ref)
  if (!is.null(r8)) {
    pos <- bc_pos[r8$barcode]
    gate[pos] <- ifelse(gate[pos] == "CD4", "doublet", "CD8")
    fc[pos] <- r8$state; conf[pos] <- r8$conf
  }
  if (!is.null(r4)) {
    pos <- bc_pos[r4$barcode]
    isdbl <- gate[pos] == "CD8"
    gate[pos] <- ifelse(isdbl, "doublet", "CD4")
    # CD4 wins the label only where the cell was not already a clean CD8 call
    fc[pos[!isdbl]] <- r4$state[!isdbl]
    conf[pos[!isdbl]] <- r4$conf[!isdbl]
  }
  n8 <- if (is.null(r8)) 0L else nrow(r8)
  n4 <- if (is.null(r4)) 0L else nrow(r4)
  cat(sprintf("  batch %d/%d cells %d-%d: CD8 kept %d, CD4 kept %d  [%.1f min]\n",
              b, n_batches, c0 + 1, c1, n8, n4,
              as.numeric(Sys.time() - t0, units = "mins")))
  rm(mat, d, ii, r8, r4); gc(full = TRUE)
}
h5$close_all()
memline("after projection")

# doublet cells (gated by both lineages) carry no single-lineage state
fc[gate == "doublet"] <- NA_character_
conf[gate == "doublet"] <- NA_real_

pred <- data.frame(
  barcode = all_barcodes,
  lineage_gate = gate,
  functional.cluster = fc,
  functional.cluster.conf = conf,
  stringsAsFactors = FALSE, check.names = FALSE)
write.csv(pred, OUT_CSV, row.names = FALSE)
cat("\nwrote", nrow(pred), "x", ncol(pred), "->", OUT_CSV, "\n")

cat("\nlineage_gate distribution:\n"); print(table(pred$lineage_gate, useNA = "ifany"))
cat("\nfunctional.cluster distribution (gated cells only):\n")
print(sort(table(pred$functional.cluster, useNA = "no"), decreasing = TRUE))
cat("==== DONE ====\n")
