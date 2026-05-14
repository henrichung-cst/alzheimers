#!/usr/bin/env Rscript
# Emit data/incytr/v2_46clusters/barcode_to_cluster.csv
# (barcode, seurat_cluster_id) from incytr_obj.rds@meta.data.
#
# Source taxonomy: kr_cluster_id_key.csv (Cluster.ID 0-109 -> New_ID).
# Downstream consumers join cluster_id -> cluster_name via that key.

suppressPackageStartupMessages({
  library(SeuratObject)
})

obj_path <- "data/incytr/v2_46clusters/incytr input/incytr_obj.rds"
key_path <- "data/incytr/v2_46clusters/provenance/kr_cluster_id_key.csv"
out_path <- "data/incytr/v2_46clusters/barcode_to_cluster.csv"
meta_path <- "data/incytr/v2_46clusters/cell_metadata.csv"

obj <- readRDS(obj_path)
md <- obj@meta.data
key <- read.csv(key_path, stringsAsFactors = FALSE)

sc <- as.integer(as.character(md$seurat_clusters))
stopifnot(!anyNA(sc))
missing_ids <- setdiff(unique(sc), key$Cluster.ID)
if (length(missing_ids) > 0) {
  stop("seurat_clusters values absent from kr_cluster_id_key.csv: ",
       paste(missing_ids, collapse = ", "))
}

name_for_id <- setNames(key$New_ID, key$Cluster.ID)
extra_cols <- c("Cluster_coarse", "Cluster_fine")
missing_extra <- setdiff(extra_cols, colnames(md))
if (length(missing_extra) > 0) {
  stop("missing taxonomy columns in incytr_obj.rds: ",
       paste(missing_extra, collapse = ", "))
}
out <- data.frame(
  barcode = rownames(md),
  seurat_cluster_id = sc,
  cluster_subclass = name_for_id[as.character(sc)],
  cluster_coarse = as.character(md$Cluster_coarse),
  cluster_fine = as.character(md$Cluster_fine),
  stringsAsFactors = FALSE
)
stopifnot(nrow(out) == nrow(md))
stopifnot(!anyDuplicated(out$barcode))
stopifnot(!anyNA(out$cluster_subclass))

write.csv(out, out_path, row.names = FALSE, quote = FALSE)
cat(sprintf("wrote %s (%d rows)\n", out_path, nrow(out)))

meta_cols <- c("sample", "Genotype", "Time", "Sex")
missing_cols <- setdiff(meta_cols, colnames(md))
if (length(missing_cols) > 0) {
  stop("missing metadata columns in incytr_obj.rds: ",
       paste(missing_cols, collapse = ", "))
}
meta <- data.frame(
  barcode = rownames(md),
  sample = as.character(md$sample),
  Genotype = as.character(md$Genotype),
  Time = as.character(md$Time),
  Sex = as.character(md$Sex),
  stringsAsFactors = FALSE
)
stopifnot(nrow(meta) == nrow(md))
write.csv(meta, meta_path, row.names = FALSE, quote = FALSE)
cat(sprintf("wrote %s (%d rows)\n", meta_path, nrow(meta)))
