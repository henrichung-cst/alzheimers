#!/usr/bin/env Rscript
# Cell-cycle-corrected re-clustering of the T-cell objects. The native
# seurat_clusters are confounded by proliferation (~20% of cells cluster by
# division phase, not identity). This scores cell cycle, regresses S/G2M out
# during scaling, re-clusters on the corrected space, and runs FindAllMarkers on
# the new clusters — to test whether the non-cytotoxic CD4 clusters carry any
# identity once proliferation is removed.
#
# Loads the multi-GB .rds, so run under a memory cap:
#   systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G \
#     Rscript alz/analysis/tcell_cellcycle_recluster.R <donor>
#
# Args: <donor> (donor1|donor2).

suppressPackageStartupMessages({library(Seurat); library(SeuratObject)})

args <- commandArgs(trailingOnly = TRUE)
donor <- if (length(args) >= 1) args[[1]] else stop("usage: Rscript ... <donor>")
umap_seed <- 42L

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
old_clusters <- obj@meta.data$seurat_clusters   # preserve native partition

s.genes   <- cc.genes.updated.2019$s.genes
g2m.genes <- cc.genes.updated.2019$g2m.genes

obj <- NormalizeData(obj, verbose = FALSE)
obj <- CellCycleScoring(obj, s.features = s.genes, g2m.features = g2m.genes,
                        set.ident = FALSE)
message("[", donor, "] phase table: ",
        paste(names(table(obj$Phase)), table(obj$Phase), sep = "=", collapse = " "))

obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE)
obj <- ScaleData(obj, vars.to.regress = c("S.Score", "G2M.Score"), verbose = FALSE)
obj <- RunPCA(obj, npcs = 30, verbose = FALSE)
obj <- FindNeighbors(obj, dims = 1:30, verbose = FALSE)
obj <- FindClusters(obj, resolution = 0.8, verbose = FALSE)
obj <- RunUMAP(obj, dims = 1:30, seed.use = umap_seed, verbose = FALSE)
message("[", donor, "] cc-regressed clusters: ", length(levels(Idents(obj))))

# per-cell: old vs new cluster + phase/scores for cross-tab downstream
cells <- data.frame(
  barcode = colnames(obj),
  old_cluster = as.character(old_clusters),
  cc_cluster  = as.character(Idents(obj)),
  UMAP_1 = Embeddings(obj, "umap")[colnames(obj), 1],
  UMAP_2 = Embeddings(obj, "umap")[colnames(obj), 2],
  Phase = obj$Phase, S.Score = round(obj$S.Score, 3), G2M.Score = round(obj$G2M.Score, 3),
  check.names = FALSE)
write.csv(cells, file.path(outdir, paste0(donor, "_cc_recluster_cells.csv")), row.names = FALSE)

mk <- FindAllMarkers(obj, only.pos = TRUE, min.pct = 0.1,
                     logfc.threshold = 0.25, verbose = FALSE)
out <- file.path(outdir, paste0(donor, "_cc_recluster_allmarkers.csv"))
write.csv(mk, out, row.names = FALSE)
message("[", donor, "] wrote ", out, " (", nrow(mk), " rows across ",
        length(unique(mk$cluster)), " cc-regressed clusters)")
