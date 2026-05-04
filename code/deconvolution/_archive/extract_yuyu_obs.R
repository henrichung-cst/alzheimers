#!/usr/bin/env Rscript
# Extract per-nucleus metadata from Yuyu's renamed Seurat object.
# Writes a thin CSV: barcode, raw cluster id (resolution 7), 46-name label.
# Source: yuyu01/documentation/incytr/deconvolution/deconvolution_with_new_clusters_20250721/renamed_sobj.rds
#
# Usage: Rscript code/deconvolution/extract_yuyu_obs.R

suppressPackageStartupMessages({
  library(Seurat)
})

RDS <- "data/raw/external/gdrive_shared/integrations/yuyu01/documentation/incytr/deconvolution/deconvolution_with_new_clusters_20250721/renamed_sobj.rds"
OUT <- "data/raw/external/gdrive_shared/integrations/yuyu01/documentation/incytr/deconvolution/deconvolution_with_new_clusters_20250721/renamed_sobj_obs.csv"

cat("Reading", RDS, "...\n")
sobj <- readRDS(RDS)
cat("Class:", class(sobj)[1], "  n_cells:", ncol(sobj), "\n")
md <- sobj@meta.data
md$barcode <- rownames(md)
md$Idents <- as.character(Idents(sobj))
cat("Meta columns:\n"); print(colnames(md))
cat("Idents head:\n"); print(head(table(md$Idents)))

# Take everything; downstream Python picks the columns it needs.
write.csv(md, OUT, row.names = FALSE)
cat("Wrote", OUT, "  rows:", nrow(md), "\n")
