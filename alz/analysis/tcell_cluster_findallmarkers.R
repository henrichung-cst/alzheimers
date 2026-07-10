#!/usr/bin/env Rscript
# Data-driven annotation of the T-cell Seurat clusters: FindAllMarkers grouped by
# the object-native `seurat_clusters` (NOT ProjecTILs state). This answers "what
# genes define each unsupervised cluster?" over the whole transcriptome — the
# open-vocabulary counterpart to scoring clusters against the fixed 39-gene panel.
# Params match alz/ingest/tcells_scrna_extract.R's state-grouped allmarkers so the
# two marker tables are directly comparable; only the grouping (Idents) differs.
#
# Loads the multi-GB cell-level .rds, so run under a memory cap:
#   systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G \
#     Rscript alz/analysis/tcell_cluster_findallmarkers.R <donor>
#
# Args: <donor> (donor1|donor2).

suppressPackageStartupMessages({library(Seurat); library(SeuratObject)})

args <- commandArgs(trailingOnly = TRUE)
donor <- if (length(args) >= 1) args[[1]] else stop("usage: Rscript ... <donor>")

rds <- list(
  donor1 = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
  donor2 = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds"
)[[donor]]
outdir <- "outputs/reports/tcell_labeling/clusters"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

message("[", donor, "] loading ", rds)
obj <- readRDS(rds)
if (!inherits(obj, "Seurat")) stop("not a Seurat object: ", class(obj))

DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = "RNA", dimreducs = NULL, graphs = NULL)
if (!"seurat_clusters" %in% colnames(obj@meta.data))
  stop("no seurat_clusters column in meta.data")

dat <- SeuratObject::GetAssayData(obj, assay = "RNA", layer = "data")
if (is.null(dat) || nrow(dat) == 0L) stop("RNA 'data' layer empty — not log-normalized")

Idents(obj) <- "seurat_clusters"
message("[", donor, "] FindAllMarkers over ", length(levels(Idents(obj))),
        " clusters, ", ncol(obj), " cells, ", nrow(obj), " genes")
mk <- FindAllMarkers(obj, only.pos = TRUE, min.pct = 0.1,
                     logfc.threshold = 0.25, verbose = FALSE)

out <- file.path(outdir, paste0(donor, "_cluster_allmarkers.csv"))
write.csv(mk, out, row.names = FALSE)
message("[", donor, "] wrote ", out, " (", nrow(mk), " rows across ",
        length(unique(mk$cluster)), " clusters)")
