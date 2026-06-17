#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
})

repo_root <- tryCatch(
  system("git rev-parse --show-toplevel", intern = TRUE),
  error = function(e) getwd()
)
if (length(repo_root) == 0 || !nzchar(repo_root[[1]])) {
  repo_root <- getwd()
}

kinase_dir <- file.path(repo_root, "outputs", "reports", "kinase_attribution_5xfad")
rds_path <- file.path(
  repo_root, "data", "datasets", "5xFAD", "primary", "scrna",
  "reclustering", "fivex_renamed_from_merged.RDS"
)
join_path <- file.path(repo_root, "data", "datasets", "5xFAD", "metadata", "omics_join_manifest.csv")
spine_path <- file.path(
  repo_root, "data", "incytr_frozen", "v2_46clusters", "spines",
  "levy_t5", "cluster_spine.csv"
)
out_dir <- file.path(kinase_dir, "celltype_mea")
out_path <- file.path(out_dir, "fivexfad_snrna_pseudobulk_linear.csv.gz")
counts_path <- file.path(out_dir, "fivexfad_snrna_pseudobulk_counts.csv")
gene_map_path <- file.path(out_dir, "fivexfad_snrna_gene_map.csv")

stopifnot(file.exists(rds_path), file.exists(join_path), file.exists(spine_path))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_required_genes <- function(kinase_dir) {
  files <- list.files(
    kinase_dir,
    pattern = "^(cortex|hippocampus)_(st|py)_raw_phospho_normalized\\.csv$",
    full.names = TRUE
  )
  genes <- character()
  for (path in files) {
    x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
    if ("gene_symbol" %in% names(x)) {
      genes <- c(genes, x$gene_symbol)
    }
  }
  sort(unique(genes[!is.na(genes) & nzchar(genes)]))
}

safe_get_data <- function(obj, assay) {
  tryCatch(
    GetAssayData(obj, assay = assay, layer = "data"),
    error = function(e) GetAssayData(obj, assay = assay, slot = "data")
  )
}

required_genes <- read_required_genes(kinase_dir)
if (!length(required_genes)) {
  stop("No 5xFAD phosphosite gene symbols found under ", kinase_dir)
}

obj <- readRDS(rds_path)
md <- obj@meta.data
required_cols <- c("sample", "new_clusters")
missing <- setdiff(required_cols, names(md))
if (length(missing)) {
  stop("Missing Seurat metadata columns: ", paste(missing, collapse = ", "))
}

join <- read.csv(join_path, stringsAsFactors = FALSE, check.names = FALSE)
join <- join[join$per_animal_integration_action == "use", , drop = FALSE]
join$age_months <- as.integer(sub("mo$", "", join$age))

md$cell_barcode <- rownames(md)
md <- merge(
  md,
  join[, c(
    "transcriptomics_sample_id", "proposed_proteomics_biological_sample_id",
    "tissue", "age", "age_months", "transcriptomics_genotype"
  )],
  by.x = "sample",
  by.y = "transcriptomics_sample_id",
  all.x = FALSE,
  all.y = FALSE
)
md <- md[!is.na(md$new_clusters) & nzchar(md$new_clusters), , drop = FALSE]
unnamed_clusters <- grepl("^cluster-[0-9]+$", md$new_clusters)
if (any(unnamed_clusters)) {
  cat("[5xfad-snrna-decomp-pb] excluding unnamed new_clusters labels: ",
      paste(sort(unique(md$new_clusters[unnamed_clusters])), collapse = ", "),
      "\n", sep = "")
  md <- md[!unnamed_clusters, , drop = FALSE]
}

spine <- read.csv(spine_path, stringsAsFactors = FALSE, check.names = FALSE)
clusters <- as.character(spine$cluster_name)
unknown <- setdiff(unique(md$new_clusters), clusters)
if (length(unknown)) {
  stop("new_clusters labels absent from 46-cluster spine: ", paste(sort(unknown), collapse = "; "))
}

expr <- safe_get_data(obj, DefaultAssay(obj))
genes <- rownames(expr)
gene_key <- setNames(genes, toupper(genes))
matched <- unname(gene_key[toupper(required_genes)])
gene_map <- data.frame(
  gene_symbol = required_genes,
  matched_gene = matched,
  stringsAsFactors = FALSE
)
gene_map <- gene_map[!is.na(gene_map$matched_gene) & nzchar(gene_map$matched_gene), , drop = FALSE]
gene_map <- gene_map[!duplicated(gene_map$gene_symbol), , drop = FALSE]
if (!nrow(gene_map)) {
  stop("No phosphosite parent genes matched Seurat expression rownames.")
}

expr <- expr[unique(gene_map$matched_gene), md$cell_barcode, drop = FALSE]
expr@x <- expm1(expr@x)

group <- paste(
  md$tissue,
  md$age_months,
  md$transcriptomics_genotype,
  md$proposed_proteomics_biological_sample_id,
  md$new_clusters,
  sep = "|"
)
group_fac <- factor(group, levels = unique(group))
design <- sparseMatrix(
  i = seq_along(group_fac),
  j = as.integer(group_fac),
  x = 1,
  dims = c(length(group_fac), nlevels(group_fac)),
  dimnames = list(md$cell_barcode, levels(group_fac))
)
cell_n <- Matrix::colSums(design)
pb <- expr %*% design
pb <- t(t(pb) / as.numeric(cell_n))

parts <- do.call(rbind, strsplit(colnames(pb), "\\|", fixed = FALSE))
colnames(parts) <- c("tissue", "age_months", "genotype", "sample_id", "cell_type")
group_meta <- as.data.frame(parts, stringsAsFactors = FALSE)
group_meta$age_months <- as.integer(group_meta$age_months)
group_meta$n_cells <- as.integer(cell_n)

pb_df <- as.data.frame(t(as.matrix(pb)), check.names = FALSE)
colnames(pb_df) <- rownames(pb)
out <- cbind(group_meta, pb_df)

write.csv(
  group_meta[, c("tissue", "age_months", "genotype", "sample_id", "cell_type", "n_cells")],
  counts_path,
  row.names = FALSE
)
write.csv(gene_map, gene_map_path, row.names = FALSE)
gz <- gzfile(out_path, "wt")
write.csv(out, gz, row.names = FALSE)
close(gz)

cat("[5xfad-snrna-decomp-pb] wrote ", out_path, " rows=", nrow(out),
    " genes=", ncol(pb_df), "\n", sep = "")
cat("[5xfad-snrna-decomp-pb] wrote ", counts_path, "\n", sep = "")
cat("[5xfad-snrna-decomp-pb] wrote ", gene_map_path, "\n", sep = "")
