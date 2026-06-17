#!/usr/bin/env Rscript
# 5xFAD snRNA extraction for the Incytr pair-mode deconvolution.
#
# The cohort analog of alz/ingest/tcells_scrna_extract.R. Aggregates the
# log-normalized `data` layer by `(cell_type, condition)` where
#   cell_type = new_clusters  (unnamed `cluster-N` dropped -> 31; the levy
#               46-name spine is a NAME cross-check only, never an in_spine
#               whitelist -- both cohorts reach the same 31 by this intrinsic
#               rule, see docs/plans and build_5xfad_snrna_decomposition_pseudobulk.R)
#   condition = `<geno>_<age>`  (e.g. "TG_3mo"), from the omics join manifest
#               `use` rows (the pooled-only WildT_06mo_C_11 is excluded there).
#
# One RDS holds BOTH tissues, so we load once (memory-safe: DietSeurat to the
# default assay, drop scale.data) and emit per tissue under
# data/derived/5xfad_incytr_inputs/<tissue>/scrna/:
#   aggexp_data.csv  (gene x `<Type>__<condition>`)  AggregateExpression(slot=data) sum
#   cell_counts.csv  (cell_type, condition, n_cells) -> P_c size factors
#   extract_manifest.json
#
# Usage:  pixi run Rscript alz/ingest/fivexfad_scrna_extract.R
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(jsonlite)
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

cat("==== 5xFAD snRNA extract (Incytr deconvolution) ====\n")
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

# --- condition / tissue from the omics join manifest (use rows) ------------
md <- obj@meta.data
stopifnot(all(c("sample", "new_clusters") %in% colnames(md)))
md$cell_barcode <- rownames(md)

join <- read.csv(JOIN_PATH, stringsAsFactors = FALSE, check.names = FALSE)
use  <- join[join$per_animal_integration_action == "use", , drop = FALSE]
samp2cond   <- setNames(paste0(use$transcriptomics_genotype, "_", use$age),
                        use$transcriptomics_sample_id)
samp2tissue <- setNames(use$tissue, use$transcriptomics_sample_id)
md$condition <- unname(samp2cond[md$sample])
md$tissue    <- unname(samp2tissue[md$sample])

# Drop cells from samples not in the `use` set (incl. the pooled-only exclusion).
keep <- !is.na(md$condition)
# Drop unnamed clusters (cohort-intrinsic 46 -> 31 filter).
unnamed <- grepl("^cluster-[0-9]+$", md$new_clusters)
keep <- keep & !is.na(md$new_clusters) & nzchar(md$new_clusters) & !unnamed
cat(sprintf("cells: total=%d  kept=%d (%.1f%%)  [dropped %d unnamed-cluster cells]\n",
            nrow(md), sum(keep), 100 * sum(keep) / nrow(md), sum(unnamed & !is.na(md$condition))))

md <- md[keep, , drop = FALSE]

# Name cross-check vs the levy 46-name spine (NOT an in_spine gate).
spine_names <- as.character(read.csv(SPINE_PATH, stringsAsFactors = FALSE,
                                     check.names = FALSE)$cluster_name)
types <- sort(unique(md$new_clusters))
unknown <- setdiff(types, spine_names)
if (length(unknown)) {
  stop("new_clusters labels absent from the 46-name spine: ",
       paste(unknown, collapse = "; "))
}
cat("cell types (named):", length(types), "\n")

# --- data layer, kept cells only -------------------------------------------
dat <- SeuratObject::GetAssayData(obj, assay = assay, layer = "data")
if (is.null(dat) || nrow(dat) == 0L) stop("'data' layer empty -- not log-normalized")
dat <- dat[, md$cell_barcode, drop = FALSE]
memline("after data slice")

write_tissue <- function(tissue) {
  sel <- md$tissue == tissue
  if (!any(sel)) { cat("[", tissue, "] no cells -- skip\n"); return(invisible()) }
  sub_md  <- md[sel, , drop = FALSE]
  sub_dat <- dat[, sub_md$cell_barcode, drop = FALSE]
  outdir  <- file.path(OUTROOT, tissue, "scrna")
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  grp <- paste0(sub_md$new_clusters, "__", sub_md$condition)  # <Type>__<condition>
  gf  <- factor(grp)
  ind <- Matrix::sparseMatrix(i = seq_along(gf), j = as.integer(gf), x = 1,
                              dims = c(length(gf), nlevels(gf)),
                              dimnames = list(NULL, levels(gf)))

  # AggregateExpression(slot="data"): sum of log-normalized data per group
  # (matches tcells_scrna_extract.R + the AD provenance aggexp).
  agg <- as.matrix(sub_dat %*% ind)
  agg_df <- data.frame(gene = rownames(agg), agg, check.names = FALSE, row.names = NULL)
  write.csv(agg_df, file.path(outdir, "aggexp_data.csv"), row.names = FALSE)

  cc <- as.data.frame(table(cell_type = sub_md$new_clusters,
                            condition = sub_md$condition), stringsAsFactors = FALSE)
  cc <- cc[cc$Freq > 0, ]
  colnames(cc)[colnames(cc) == "Freq"] <- "n_cells"
  cc <- cc[order(cc$condition, cc$cell_type), ]
  write.csv(cc, file.path(outdir, "cell_counts.csv"), row.names = FALSE)

  manifest <- list(
    tissue = tissue, rds = RDS_PATH, assay = assay,
    n_cells = sum(sel), n_genes = nrow(agg),
    cell_types = sort(unique(sub_md$new_clusters)),
    conditions = sort(unique(sub_md$condition)),
    n_group_cols = ncol(agg),
    generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
  )
  write_json(manifest, file.path(outdir, "extract_manifest.json"),
             auto_unbox = TRUE, pretty = TRUE)

  cat(sprintf("[%s] %d cells; aggexp %d genes x %d (type__cond) cols; %d cell-count groups; conditions=%s -> %s\n",
              tissue, sum(sel), nrow(agg), ncol(agg), nrow(cc),
              paste(sort(unique(sub_md$condition)), collapse = ","), outdir))
}

for (tissue in c("cortex", "hippocampus")) write_tissue(tissue)
cat("==== DONE ====\n")
