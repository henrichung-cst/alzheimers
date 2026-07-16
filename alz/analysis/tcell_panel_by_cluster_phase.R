#!/usr/bin/env Rscript
# Absolute panel expression per native seurat_cluster x proliferation state.
# Answers the label review: is TOX/TIM3 actually expressed, and does the
# cytotoxic/exhausted/TPEX/TEX assignment hold once cells are split by dividing
# (S/G2M) vs non-dividing (G1)? Proliferation is treated as a labeling axis, not
# regressed out.
#
# Loads the multi-GB .rds — run under a memory cap:
#   systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G \
#     Rscript alz/analysis/tcell_panel_by_cluster_phase.R <donor>

suppressPackageStartupMessages({library(Seurat); library(SeuratObject)})

args <- commandArgs(trailingOnly = TRUE)
donor <- if (length(args) >= 1) args[[1]] else stop("usage: Rscript ... <donor>")

rds <- list(
  donor1 = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
  donor2 = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds"
)[[donor]]
outdir <- "outputs/reports/tcell_labeling/clusters"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

panel <- c("CD4","CD8A","CD8B","GZMB","GZMK","PRF1","NKG7","IFNG",
           "TOX","HAVCR2","PDCD1","LAG3","TIGIT","ENTPD1","CTLA4",
           "TCF7","SLAMF6","IL7R","CCR7","SELL","LEF1")

message("[", donor, "] loading ", rds)
obj <- readRDS(rds)
DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = "RNA", dimreducs = NULL, graphs = NULL)
obj <- NormalizeData(obj, verbose = FALSE)
obj <- CellCycleScoring(obj, s.features = cc.genes.updated.2019$s.genes,
                        g2m.features = cc.genes.updated.2019$g2m.genes, set.ident = FALSE)

present <- panel[panel %in% rownames(obj)]
missing <- setdiff(panel, present)
if (length(missing)) message("[", donor, "] panel genes absent from object: ",
                             paste(missing, collapse = ", "))

fd <- FetchData(obj, vars = c(present, "seurat_clusters"), layer = "data")
fd$cluster <- as.character(obj$seurat_clusters)
fd$div <- ifelse(obj$Phase %in% c("S", "G2M"), "dividing", "resting")

# per-cell day + cluster lineage (CD4/CD8), joined from the report CSVs by barcode
coords <- read.csv(file.path(outdir, "..", "umap",
                             paste0(donor, "_native_umap_coords.csv")), check.names = FALSE)
day_by_bc <- setNames(as.integer(sub(".*Day_(\\d+).*", "\\1", coords$day_label)), coords$barcode)
ann <- read.csv(file.path(outdir, paste0(donor, "_cluster_annotation.csv")))
lineage_by_cl <- setNames(ann$marker_lineage, as.character(ann$cluster))
fd$day <- day_by_bc[colnames(obj)]
fd$lineage <- lineage_by_cl[fd$cluster]

summarise <- function(df, keys) {
  rows <- list()
  grp <- interaction(df[keys], drop = TRUE)
  for (lv in levels(grp)) {
    sub <- df[grp == lv, , drop = FALSE]
    kv <- sub[1, keys, drop = FALSE]
    for (g in present) {
      x <- sub[[g]]
      rows[[length(rows) + 1]] <- cbind(kv, data.frame(n = nrow(sub), gene = g,
        mean_logexpr = round(mean(x), 3), pct_expr = round(mean(x > 0) * 100, 1),
        check.names = FALSE), row.names = NULL)
    }
  }
  do.call(rbind, rows)
}

out1 <- file.path(outdir, paste0(donor, "_panel_by_cluster_phase.csv"))
write.csv(summarise(fd, c("cluster", "div")), out1, row.names = FALSE)

# raw exhaustion program over the day-course, split by lineage (the exhaustion question)
out2 <- file.path(outdir, paste0(donor, "_panel_by_day_lineage.csv"))
write.csv(summarise(fd[!is.na(fd$day), ], c("lineage", "day")), out2, row.names = FALSE)
message("[", donor, "] wrote ", out1, " and ", out2)
