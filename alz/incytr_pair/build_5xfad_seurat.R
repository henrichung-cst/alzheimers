#!/usr/bin/env Rscript
# Per-tissue 5xFAD Seurat for Incytr pair-mode.
#
# Cohort analog of build_tcells_seurat.R. One RDS holds both tissues, so we load
# once and write a slim per-tissue object with:
#   - Idents()           = new_clusters (Type)
#   - obj$Type           = new_clusters
#   - obj$condition      = `<geno>_<age>`            (e.g. "TG_3mo")
#   - obj$Type_condition = paste0(Type, "_", condition)
#
# cell_type = new_clusters with unnamed `cluster-N` dropped (-> 31; the levy
# 46-name spine is a NAME cross-check only, never an in_spine whitelist). condition
# comes from the omics join manifest `use` rows (the pooled-only WildT_06mo_C_11 is
# excluded there). Pair-mode driver consumes each as
# data/derived/5xfad_incytr_inputs/<tissue>/incytr_obj.rds.
#
# Unlike the AD/t-cell cohorts, 5xFAD cell-type labels carry hyphens/spaces and
# condition carries an internal `_` — this is identical to the AD
# `ma_<age>_<geno>` + spaced-cluster scheme the driver already handles (it reads
# Type/condition as separate metadata columns and matches the deconvoluted columns
# by the anchored `^<condition>_` prefix), so no alphanumeric sanitization here.
#
# Usage:  pixi run Rscript alz/incytr_pair/build_5xfad_seurat.R
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

RDS_PATH  <- "data/datasets/5xFAD/primary/scrna/reclustering/fivex_renamed_from_merged.RDS"
JOIN_PATH <- "data/datasets/5xFAD/metadata/omics_join_manifest.csv"
SPINE_PATH <- "data/incytr_frozen/v2_46clusters/spines/levy_t5/cluster_spine.csv"
OUTROOT   <- "data/derived/5xfad_incytr_inputs"
stopifnot(file.exists(RDS_PATH), file.exists(JOIN_PATH), file.exists(SPINE_PATH))

memline <- function(tag) {
  gc(full = TRUE)
  mi <- tryCatch(readLines("/proc/meminfo", n = 3), error = function(e) character())
  avail <- sub(".*?:\\s*", "", grep("MemAvailable", mi, value = TRUE))
  cat(sprintf("[mem %-18s] MemAvailable: %s\n", tag, ifelse(length(avail), avail, "?")))
}

cat("==== BUILD 5xFAD INCYTR SEURAT ====\n")
memline("start")

t0 <- Sys.time()
obj <- readRDS(RDS_PATH)
cat("read in", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s ;",
    "dim", paste(dim(obj), collapse = " x "), "\n")
memline("after readRDS")

assay <- DefaultAssay(obj)
obj <- DietSeurat(obj, assays = assay, dimreducs = NULL, graphs = NULL)
suppressWarnings(try(obj[[assay]]$scale.data <- NULL, silent = TRUE))
memline("after DietSeurat")

# condition / tissue from the join manifest (use rows) ----------------------
stopifnot(all(c("sample", "new_clusters") %in% colnames(obj@meta.data)))
join <- read.csv(JOIN_PATH, stringsAsFactors = FALSE, check.names = FALSE)
use  <- join[join$per_animal_integration_action == "use", , drop = FALSE]
samp2cond   <- setNames(paste0(use$transcriptomics_genotype, "_", use$age),
                        use$transcriptomics_sample_id)
samp2tissue <- setNames(use$tissue, use$transcriptomics_sample_id)
obj$condition  <- unname(samp2cond[obj$sample])
obj$tissue_grp <- unname(samp2tissue[obj$sample])
obj$Type       <- as.character(obj$new_clusters)

# Drop non-use samples + unnamed clusters (cohort-intrinsic 46 -> 31).
unnamed <- grepl("^cluster-[0-9]+$", obj$Type)
keep <- !is.na(obj$condition) & !is.na(obj$Type) & nzchar(obj$Type) & !unnamed
cat(sprintf("cells: total=%d  kept=%d (%.1f%%)\n",
            ncol(obj), sum(keep), 100 * sum(keep) / ncol(obj)))
obj <- subset(obj, cells = colnames(obj)[keep])
memline("after subset")

# Name cross-check vs the levy 46-name spine (NOT an in_spine gate).
spine_names <- as.character(read.csv(SPINE_PATH, stringsAsFactors = FALSE,
                                     check.names = FALSE)$cluster_name)
unknown <- setdiff(unique(obj$Type), spine_names)
if (length(unknown)) {
  stop("new_clusters labels absent from the 46-name spine: ",
       paste(unknown, collapse = "; "))
}

obj$Type_condition <- paste0(obj$Type, "_", obj$condition)
stopifnot(all(grepl("^(TG|WT)_(3|6|9|12)mo$", obj$condition)))

write_tissue <- function(tissue) {
  cells <- colnames(obj)[obj$tissue_grp == tissue]
  if (!length(cells)) { cat("[", tissue, "] no cells -- skip\n"); return(invisible()) }
  sub <- subset(obj, cells = cells)
  Idents(sub) <- factor(sub$Type, levels = sort(unique(sub$Type)))
  outdir <- file.path(OUTROOT, tissue)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  dst <- file.path(outdir, "incytr_obj.rds")
  saveRDS(sub, dst)
  cat(sprintf("[%s] dim %s  types=%d  conditions=%d -> %s\n",
              tissue, paste(dim(sub), collapse = " x "),
              length(unique(sub$Type)), length(unique(sub$condition)), dst))
}

for (tissue in c("cortex", "hippocampus")) write_tissue(tissue)
cat("==== DONE ====\n")
