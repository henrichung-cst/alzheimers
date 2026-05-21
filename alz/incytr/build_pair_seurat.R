#!/usr/bin/env Rscript
# Build the males-only Seurat object for pair-mode on the levy_t5 spine.
#
# Source: data/incytr_frozen/v2_46clusters/incytr input/incytr_obj.rds (46 named
# clusters, both sexes, 63,706 cells). Filters to (Sex == "ma") × (Idents()
# in the levy_t5 spine), copies Idents() → Type, Group → condition, and writes
# the trimmed object to data/derived/incytr_inputs/incytr_obj.rds.
#
# Run from any working directory (paths resolved via git rev-parse).

suppressPackageStartupMessages({
  library(Seurat)
  library(readr)
})

# Resolve repo root so the script runs from any cwd.
REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)

SRC   <- file.path(REPO_ROOT, "data", "incytr_frozen", "v2_46clusters", "incytr input", "incytr_obj.rds")
SPINE <- file.path(REPO_ROOT, "data", "incytr_frozen", "v2_46clusters", "spines", "levy_t5", "cluster_spine.csv")
DST   <- file.path(REPO_ROOT, "data", "derived", "incytr_inputs", "incytr_obj.rds")
cat("[seurat] spine=levy_t5  src=", SRC, "  spine_csv=", SPINE,
    "  dst=", DST, "\n", sep = "")

t0 <- Sys.time()
cat("[seurat] reading", SRC, "\n")
sobj <- readRDS(SRC)
cat("[seurat] source dim:", paste(dim(sobj), collapse = " x "),
    "(genes x cells)\n")

spine <- read_csv(SPINE, show_col_types = FALSE)
in_spine <- spine$cluster_name[spine$in_spine]
cat("[seurat] in_spine count: ", length(in_spine), "\n", sep = "")
stopifnot(length(in_spine) >= 1)
# Required by build_input_gene_list.R: cluster names must contain no underscores
# (it splits Type_condition on the first "_" to recover Type).
stopifnot(!any(grepl("_", in_spine)))

idents_chr <- as.character(Idents(sobj))
sex_chr    <- as.character(sobj$Sex)
keep <- sex_chr == "ma" & idents_chr %in% in_spine
cat("[seurat] keeping", sum(keep), "/", length(keep),
    "cells (males x in-spine)\n")

sobj <- subset(sobj, cells = colnames(sobj)[keep])

# Normalize metadata vocabulary to what build_input_gene_list.R + the driver expect.
sobj$Type      <- as.character(Idents(sobj))
sobj$condition <- as.character(sobj$Group)   # already "ma_<age>_<geno>"
sobj$Genotype  <- as.character(sobj$Genotype)
sobj$age       <- as.character(sobj$Time)
sobj$sex       <- as.character(sobj$Sex)

stopifnot(all(grepl("^ma_(2|4|6)mo_(WTyp|AppP|Ttau|ApTt)$", sobj$condition)))
cat("[seurat] condition levels (n=", length(unique(sobj$condition)), "):\n  ",
    paste(sort(unique(sobj$condition)), collapse = "  "), "\n", sep = "")

# Drop cluster levels emptied by the spine filter so FindAllMarkers does not
# spend time on zero-cell idents.
Idents(sobj) <- factor(sobj$Type, levels = sort(unique(sobj$Type)))

dir.create(dirname(DST), showWarnings = FALSE, recursive = TRUE)
cat("[seurat] writing", DST, "\n")
saveRDS(sobj, DST)
cat("[seurat] final dim:", paste(dim(sobj), collapse = " x "),
    "  (", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1),
    "s)\n", sep = "")
