#!/usr/bin/env Rscript
# D4 ProjecTILs reference mapping for the T-cell exhaustion cohort donors.
# Replaces the prior Azimuth path (deleted 2026-05-28).
#
# Method (per donor, in one DietSeurat load):
#   1. Memory-safe one-load (DietSeurat → RNA only, drop scale.data + reductions)
#   2. Run.ProjecTILs against the human CD8 atlas with filter.cells=TRUE
#      (ProjecTILs runs scGate internally to keep only CD8-like cells)
#   3. Run.ProjecTILs against the human CD4 atlas, same gate
#   4. Merge per-cell: a cell passing only one gate gets that lineage's
#      functional.cluster; a cell passing both gates gets `doublet_suspect`;
#      a cell passing neither gets `ungated` (downstream → unresolved)
#
# Emits data/derived/tcells_incytr_inputs/<donor>/scrna/projectils_predictions.csv
#   barcode, seurat_clusters, day, lineage_gate, functional.cluster,
#   functional.cluster.conf
#
# Usage:  pixi run tcells-projectils-map <donor1|donor2>
#         alz/runners/supporting/tcells_projectils_map.sh [donor]
suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratObject)
  library(Matrix)
  library(future)
  library(ProjecTILs)
  library(scGate)
  library(STACAS)
  library(UCell)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
donor <- args[[1]]

cfg <- list(
  donor1 = list(
    rds     = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
    day_col = "sample_ID"),
  donor2 = list(
    rds     = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds",
    day_col = "Sample_Label")
)[[donor]]
stopifnot(!is.null(cfg))

outdir <- file.path("data/derived/tcells_incytr_inputs", donor, "scrna")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

REF_DIR  <- "data/external/projectils"
CD8_REF  <- file.path(REF_DIR, "CD8T_human_ref_v1.rds")
CD4_REF  <- file.path(REF_DIR, "CD4T_human_ref_v1.rds")
stopifnot(file.exists(CD8_REF), file.exists(CD4_REF))

# Force single-threaded — ProjecTILs internally calls SCT and STACAS anchors;
# futures' parallel workers each duplicate the reference object (~1.5 GiB) and
# OOM the 30 GB box.
plan(sequential)
options(future.globals.maxSize = +Inf)

memline <- function(tag) {
  gc(full = TRUE)
  mi <- tryCatch(readLines("/proc/meminfo", n = 3), error = function(e) character())
  avail <- sub(".*?:\\s*", "", grep("MemAvailable", mi, value = TRUE))
  cat(sprintf("[mem %-26s] MemAvailable: %s\n", tag, ifelse(length(avail), avail, "?")))
}

cat("==== PROJECTILS MAP", donor, "====\n")
memline("start")

# --- query: memory-safe one-load ----------------------------------------
t0 <- Sys.time()
obj <- readRDS(cfg$rds)
cat("read query in", round(as.numeric(Sys.time() - t0, units = "secs"), 1),
    "s ;", "dim", paste(dim(obj), collapse = " x "), "\n")
memline("after query readRDS")

DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = "RNA", dimreducs = NULL, graphs = NULL)
obj[["RNA"]]$scale.data <- NULL

day_raw <- as.character(obj@meta.data[[cfg$day_col]])
day <- as.integer(sub(".*[Dd]ay[_ ]?(\\d+).*", "\\1", day_raw))
if (any(is.na(day))) stop("unparsed day labels: ",
                          paste(sort(unique(day_raw[is.na(day)])), collapse = " | "))
obj$ts_day <- day
all_barcodes <- colnames(obj)
all_clusters <- as.character(obj$seurat_clusters)
memline("after DietSeurat")

# --- CD8 projection ------------------------------------------------------
cat("\n--- CD8 projection ---\n")
cd8_ref <- readRDS(CD8_REF)
cat("CD8 ref:", paste(dim(cd8_ref), collapse = " x "), "; states:",
    paste(sort(unique(as.character(cd8_ref$functional.cluster))), collapse = ", "),
    "\n")
memline("after CD8 ref load")

t0 <- Sys.time()
# filter.cells=TRUE runs scGate against the reference's built-in lineage
# gating model and keeps only cells matching the reference's lineage (CD8 T).
# split.by=NULL → one global projection.
q_cd8 <- Run.ProjecTILs(query = obj, ref = cd8_ref,
                        filter.cells = TRUE,
                        split.by = NULL,
                        ncores = 1)
cat("CD8 projection done in",
    round(as.numeric(Sys.time() - t0, units = "mins"), 2), "min\n")
cd8_bc <- if (!is.null(q_cd8)) colnames(q_cd8) else character(0)
cd8_states <- if (length(cd8_bc)) as.character(q_cd8$functional.cluster) else character(0)
cd8_conf   <- if (length(cd8_bc)) as.numeric(q_cd8$functional.cluster.conf) else numeric(0)
cat("CD8: kept", length(cd8_bc), "/", length(all_barcodes), "cells (",
    round(100 * length(cd8_bc) / length(all_barcodes), 1), "%)\n")
rm(q_cd8, cd8_ref); gc(full = TRUE)
memline("after CD8 projection")

# --- CD4 projection ------------------------------------------------------
cat("\n--- CD4 projection ---\n")
cd4_ref <- readRDS(CD4_REF)
cat("CD4 ref:", paste(dim(cd4_ref), collapse = " x "), "; states:",
    paste(sort(unique(as.character(cd4_ref$functional.cluster))), collapse = ", "),
    "\n")
memline("after CD4 ref load")

t0 <- Sys.time()
q_cd4 <- Run.ProjecTILs(query = obj, ref = cd4_ref,
                        filter.cells = TRUE,
                        split.by = NULL,
                        ncores = 1)
cat("CD4 projection done in",
    round(as.numeric(Sys.time() - t0, units = "mins"), 2), "min\n")
cd4_bc <- if (!is.null(q_cd4)) colnames(q_cd4) else character(0)
cd4_states <- if (length(cd4_bc)) as.character(q_cd4$functional.cluster) else character(0)
cd4_conf   <- if (length(cd4_bc)) as.numeric(q_cd4$functional.cluster.conf) else numeric(0)
cat("CD4: kept", length(cd4_bc), "/", length(all_barcodes), "cells (",
    round(100 * length(cd4_bc) / length(all_barcodes), 1), "%)\n")
rm(q_cd4, cd4_ref, obj); gc(full = TRUE)
memline("after CD4 projection")

# --- merge into per-cell prediction table -------------------------------
cd8_map <- setNames(cd8_states, cd8_bc)
cd4_map <- setNames(cd4_states, cd4_bc)
cd8_conf_map <- setNames(cd8_conf, cd8_bc)
cd4_conf_map <- setNames(cd4_conf, cd4_bc)

in_cd8 <- all_barcodes %in% cd8_bc
in_cd4 <- all_barcodes %in% cd4_bc

lineage_gate <- rep(NA_character_, length(all_barcodes))
lineage_gate[in_cd8 & !in_cd4] <- "CD8"
lineage_gate[in_cd4 & !in_cd8] <- "CD4"
lineage_gate[in_cd4 &  in_cd8] <- "doublet"
lineage_gate[!in_cd4 & !in_cd8] <- "none"

functional_cluster <- rep(NA_character_, length(all_barcodes))
functional_conf    <- rep(NA_real_,      length(all_barcodes))
functional_cluster[in_cd8 & !in_cd4] <- cd8_map[all_barcodes[in_cd8 & !in_cd4]]
functional_cluster[in_cd4 & !in_cd8] <- cd4_map[all_barcodes[in_cd4 & !in_cd8]]
functional_conf[in_cd8 & !in_cd4]    <- cd8_conf_map[all_barcodes[in_cd8 & !in_cd4]]
functional_conf[in_cd4 & !in_cd8]    <- cd4_conf_map[all_barcodes[in_cd4 & !in_cd8]]

pred <- data.frame(
  barcode               = all_barcodes,
  seurat_clusters       = all_clusters,
  day                   = day,
  lineage_gate          = lineage_gate,
  functional.cluster    = functional_cluster,
  functional.cluster.conf = functional_conf,
  stringsAsFactors = FALSE, check.names = FALSE
)
out_path <- file.path(outdir, "projectils_predictions.csv")
write.csv(pred, out_path, row.names = FALSE)
cat("\nwrote", nrow(pred), "x", ncol(pred), "→", out_path, "\n")

cat("\nlineage_gate distribution:\n"); print(table(pred$lineage_gate, useNA = "ifany"))
cat("\nfunctional.cluster distribution (resolved cells only):\n")
print(sort(table(pred$functional.cluster, useNA = "no"), decreasing = TRUE))
res <- pred[!is.na(pred$functional.cluster), ]
if (nrow(res)) {
  cat("\nmedian functional.cluster.conf:",
      round(median(res$functional.cluster.conf, na.rm = TRUE), 3), "\n")
}
cat("==== DONE", donor, "====\n")
