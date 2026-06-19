#!/usr/bin/env Rscript
# TODO #2: ProjecTILs projection of the 10x NSCLC T/NK compartment.
#
# Mirrors alz/ingest/tcells_projectils_map.R (the cohort's own projection) so
# the public NSCLC reference lands in the SAME 14-state ProjecTILs vocabulary
# as the within-cohort attribution. Reads the T/NK subset 10x h5 exported by
# alz/ingest/nsclc_subset_tnk.py (~182 K cells), projects against the human
# CD8 then CD4 references with filter.cells=TRUE (scGate keeps only the
# matching lineage), and writes per-barcode functional.cluster.
#
#   Emits data/external/nsclc_10x/projectils_predictions.csv
#     barcode, lineage_gate, functional.cluster, functional.cluster.conf
#
# Memory: plan(sequential) — each future worker would duplicate the ~1.5 GiB
# reference and OOM the box. Run under a systemd MemoryMax scope.
#
# Usage:  pixi run Rscript alz/ingest/nsclc_projectils_map.R
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

SUBSET_H5 <- "data/external/nsclc_10x/tnk_subset_feature_bc_matrix.h5"
OUT_CSV   <- "data/external/nsclc_10x/projectils_predictions.csv"
REF_DIR   <- "data/external/projectils"
CD8_REF   <- file.path(REF_DIR, "CD8T_human_ref_v1.rds")
CD4_REF   <- file.path(REF_DIR, "CD4T_human_ref_v1.rds")
stopifnot(file.exists(SUBSET_H5), file.exists(CD8_REF), file.exists(CD4_REF))

plan(sequential)
options(future.globals.maxSize = +Inf)

memline <- function(tag) {
  gc(full = TRUE)
  mi <- tryCatch(readLines("/proc/meminfo", n = 3), error = function(e) character())
  avail <- sub(".*?:\\s*", "", grep("MemAvailable", mi, value = TRUE))
  cat(sprintf("[mem %-26s] MemAvailable: %s\n", tag, ifelse(length(avail), avail, "?")))
}

cat("==== NSCLC PROJECTILS MAP ====\n")
memline("start")

t0 <- Sys.time()
counts <- Read10X_h5(SUBSET_H5)
obj <- CreateSeuratObject(counts = counts, min.cells = 0, min.features = 0)
rm(counts)
cat("loaded query", paste(dim(obj), collapse = " x "), "in",
    round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s\n")
all_barcodes <- colnames(obj)
memline("after query load")

# --- CD8 projection ------------------------------------------------------
cat("\n--- CD8 projection ---\n")
cd8_ref <- readRDS(CD8_REF)
t0 <- Sys.time()
q_cd8 <- Run.ProjecTILs(query = obj, ref = cd8_ref,
                        filter.cells = TRUE, split.by = NULL, ncores = 1)
cat("CD8 done in", round(as.numeric(Sys.time() - t0, units = "mins"), 2), "min\n")
cd8_bc    <- if (!is.null(q_cd8)) colnames(q_cd8) else character(0)
cd8_state <- if (length(cd8_bc)) as.character(q_cd8$functional.cluster) else character(0)
cd8_conf  <- if (length(cd8_bc)) as.numeric(q_cd8$functional.cluster.conf) else numeric(0)
cat("CD8: kept", length(cd8_bc), "/", length(all_barcodes), "cells (",
    round(100 * length(cd8_bc) / length(all_barcodes), 1), "%)\n")
rm(q_cd8, cd8_ref); gc(full = TRUE)
memline("after CD8 projection")

# --- CD4 projection ------------------------------------------------------
cat("\n--- CD4 projection ---\n")
cd4_ref <- readRDS(CD4_REF)
t0 <- Sys.time()
q_cd4 <- Run.ProjecTILs(query = obj, ref = cd4_ref,
                        filter.cells = TRUE, split.by = NULL, ncores = 1)
cat("CD4 done in", round(as.numeric(Sys.time() - t0, units = "mins"), 2), "min\n")
cd4_bc    <- if (!is.null(q_cd4)) colnames(q_cd4) else character(0)
cd4_state <- if (length(cd4_bc)) as.character(q_cd4$functional.cluster) else character(0)
cd4_conf  <- if (length(cd4_bc)) as.numeric(q_cd4$functional.cluster.conf) else numeric(0)
cat("CD4: kept", length(cd4_bc), "/", length(all_barcodes), "cells (",
    round(100 * length(cd4_bc) / length(all_barcodes), 1), "%)\n")
rm(q_cd4, cd4_ref, obj); gc(full = TRUE)
memline("after CD4 projection")

# --- merge per-cell ------------------------------------------------------
cd8_map      <- setNames(cd8_state, cd8_bc)
cd4_map      <- setNames(cd4_state, cd4_bc)
cd8_conf_map <- setNames(cd8_conf, cd8_bc)
cd4_conf_map <- setNames(cd4_conf, cd4_bc)

in_cd8 <- all_barcodes %in% cd8_bc
in_cd4 <- all_barcodes %in% cd4_bc

lineage_gate <- rep(NA_character_, length(all_barcodes))
lineage_gate[in_cd8 & !in_cd4] <- "CD8"
lineage_gate[in_cd4 & !in_cd8] <- "CD4"
lineage_gate[in_cd4 &  in_cd8] <- "doublet"
lineage_gate[!in_cd4 & !in_cd8] <- "none"

fc   <- rep(NA_character_, length(all_barcodes))
conf <- rep(NA_real_,      length(all_barcodes))
fc[in_cd8 & !in_cd4]   <- cd8_map[all_barcodes[in_cd8 & !in_cd4]]
fc[in_cd4 & !in_cd8]   <- cd4_map[all_barcodes[in_cd4 & !in_cd8]]
conf[in_cd8 & !in_cd4] <- cd8_conf_map[all_barcodes[in_cd8 & !in_cd4]]
conf[in_cd4 & !in_cd8] <- cd4_conf_map[all_barcodes[in_cd4 & !in_cd8]]

pred <- data.frame(
  barcode = all_barcodes,
  lineage_gate = lineage_gate,
  functional.cluster = fc,
  functional.cluster.conf = conf,
  stringsAsFactors = FALSE, check.names = FALSE)
write.csv(pred, OUT_CSV, row.names = FALSE)
cat("\nwrote", nrow(pred), "x", ncol(pred), "->", OUT_CSV, "\n")

cat("\nlineage_gate distribution:\n"); print(table(pred$lineage_gate, useNA = "ifany"))
cat("\nfunctional.cluster distribution (resolved cells only):\n")
print(sort(table(pred$functional.cluster, useNA = "no"), decreasing = TRUE))
cat("==== DONE ====\n")
