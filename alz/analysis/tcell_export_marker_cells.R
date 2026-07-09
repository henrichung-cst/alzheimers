#!/usr/bin/env Rscript
# Extract per-cell log-normalized expression for the marker genes ONLY, from the
# cell-level Seurat object, to a compact CSV (cells x markers). This is the one
# step that must touch the multi-GB .rds; it drops everything else immediately so
# nothing large is written out. Per-cell state labels are assigned downstream by
# tcell_state_labels.py, which joins this marker evidence to the cell-cycle and
# native-cluster exports.
#
# Run under a memory cap (the object is ~5 GB compressed, ~15-30 GB in RAM):
#   systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G \
#     pixi run Rscript alz/analysis/tcell_export_marker_cells.R <donor>
#
# Args: <donor> (donor1|donor2). Reads the marker list written by
#   `pixi run python alz/analysis/tcell_percell_auroc.py --write-markers <path>`.

suppressPackageStartupMessages({library(Seurat); library(Matrix)})

args <- commandArgs(trailingOnly = TRUE)
donor <- if (length(args) >= 1) args[[1]] else stop("usage: Rscript ... <donor>")

rds <- list(
  donor1 = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
  donor2 = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds"
)[[donor]]
marker_file <- "outputs/reports/tcell_labeling/auroc/marker_genes.txt"
outdir <- "outputs/reports/tcell_labeling/auroc"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

markers <- readLines(marker_file)
markers <- markers[nzchar(markers)]

message("[", donor, "] loading ", rds)
obj <- readRDS(rds)
if (!inherits(obj, "Seurat")) stop("not a Seurat object: ", class(obj))

assay <- "RNA"
if (!assay %in% Assays(obj)) stop("RNA assay is required for exhaustion-marker evidence")
DefaultAssay(obj) <- assay
message("[", donor, "] assay=", assay, " cells=", ncol(obj), " genes=", nrow(obj))

# log-normalized 'data' slot — matches the log-space choice in the pseudobulk path.
mat <- GetAssayData(obj, assay = assay, layer = "data")
present <- intersect(markers, rownames(mat))
missing <- setdiff(markers, rownames(mat))
if (length(missing)) message("[", donor, "] markers absent: ", paste(missing, collapse = ", "))

sub <- as.matrix(t(mat[present, , drop = FALSE]))  # cells x markers, dense & tiny
df <- data.frame(barcode = colnames(obj), sub, check.names = FALSE)

out <- file.path(outdir, paste0(donor, "_marker_cell_expr.csv"))
write.csv(df, out, row.names = FALSE)
message("[", donor, "] wrote ", out, " (", nrow(df), " cells x ", length(present), " markers)")
