#!/usr/bin/env Rscript
# Design inspection: dump the full CITE-seq Protein (ADT) antibody panel so we
# know which exhaustion/state markers were measured at protein level.
#   systemd-run --user --scope -p MemoryMax=26G -p MemorySwapMax=2G \
#     Rscript alz/analysis/tcell_object_inspect.R <donor>
suppressPackageStartupMessages({library(Seurat); library(SeuratObject)})
donor <- commandArgs(trailingOnly = TRUE)[[1]]
rds <- list(donor1 = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
            donor2 = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds")[[donor]]
obj <- readRDS(rds)
cat("\n===== [", donor, "] full Protein (ADT) panel =====\n")
print(sort(rownames(obj[["Protein"]])))
