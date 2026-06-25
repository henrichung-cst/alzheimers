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

rds_path <- file.path(
  repo_root, "data", "datasets", "5xFAD", "primary", "scrna",
  "reclustering", "fivex_renamed_from_merged.RDS"
)
join_path <- file.path(
  repo_root, "data", "datasets", "5xFAD", "metadata", "omics_join_manifest.csv"
)
spine_path <- file.path(
  repo_root, "data", "incytr_frozen", "v2_46clusters", "spines",
  "levy_t5", "cluster_spine.csv"
)
out_dir <- file.path(repo_root, "outputs", "reports", "kinase_attribution_5xfad")
out_path <- file.path(out_dir, "fivexfad_snrna_attribution.csv")
counts_path <- file.path(out_dir, "fivexfad_snrna_cell_counts.csv")
expr_path <- file.path(out_dir, "fivexfad_snrna_expression.csv")
mapping_path <- file.path(repo_root, "data", "derived", "caches", "kinase_to_gene_mapping.csv")

stopifnot(file.exists(rds_path), file.exists(join_path), file.exists(spine_path))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_kinase_gene_map <- function(path) {
  if (!file.exists(path)) return(character())
  x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("kinase_abbreviation", "gene_symbol")
  if (!all(required %in% names(x))) return(character())
  x <- x[!is.na(x$kinase_abbreviation) & nzchar(x$kinase_abbreviation)
         & !is.na(x$gene_symbol) & nzchar(x$gene_symbol), , drop = FALSE]
  stats::setNames(x$gene_symbol, toupper(x$kinase_abbreviation))
}

read_kinases <- function(out_dir) {
  files <- list.files(out_dir, pattern = "_mea_(stoichiometry|raw_phospho)\\.csv$", full.names = TRUE)
  if (!length(files)) return(data.frame(kinase = character(), gene_symbol = character()))
  rows <- do.call(rbind, lapply(files, function(path) {
    x <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
    if (!all(c("kinase", "gene_symbol") %in% names(x))) {
      if (!"kinase" %in% names(x)) return(NULL)
      x$gene_symbol <- x$kinase
    }
    unique(x[, c("kinase", "gene_symbol")])
  }))
  rows <- rows[!is.na(rows$kinase) & nzchar(rows$kinase), , drop = FALSE]
  rows <- unique(rows)
  gene_map <- read_kinase_gene_map(mapping_path)
  if (length(gene_map)) {
    mapped <- unname(gene_map[toupper(rows$kinase)])
    use_mapped <- !is.na(mapped) & nzchar(mapped) & (
      is.na(rows$gene_symbol) | !nzchar(rows$gene_symbol) | rows$gene_symbol == rows$kinase
    )
    rows$gene_symbol[use_mapped] <- mapped[use_mapped]
  }
  unique(rows)
}

safe_get_data <- function(obj, assay) {
  tryCatch(
    GetAssayData(obj, assay = assay, layer = "data"),
    error = function(e) GetAssayData(obj, assay = assay, slot = "data")
  )
}

obj <- readRDS(rds_path)
md <- obj@meta.data
required <- c("sample", "new_clusters")
missing <- setdiff(required, names(md))
if (length(missing)) stop("Missing required Seurat metadata columns: ", paste(missing, collapse = ", "))

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

spine <- read.csv(spine_path, stringsAsFactors = FALSE, check.names = FALSE)
clusters <- as.character(spine$cluster_name)
unknown <- setdiff(unique(md$new_clusters), clusters)
if (length(unknown)) {
  stop("new_clusters labels absent from 46-cluster spine: ", paste(sort(unknown), collapse = "; "))
}

kin <- read_kinases(out_dir)
if (!nrow(kin)) stop("No 5xFAD MEA kinase rows found in ", out_dir)

expr <- safe_get_data(obj, DefaultAssay(obj))
genes <- rownames(expr)
gene_key <- setNames(genes, toupper(genes))
kin$matched_gene <- unname(gene_key[toupper(kin$gene_symbol)])
kin$matched_gene[is.na(kin$matched_gene)] <- unname(gene_key[toupper(kin$kinase[is.na(kin$matched_gene)])])
kin <- kin[!is.na(kin$matched_gene) & nzchar(kin$matched_gene), , drop = FALSE]
if (!nrow(kin)) stop("No kinase symbols matched Seurat expression rownames.")

expr <- expr[unique(kin$matched_gene), md$cell_barcode, drop = FALSE]
colnames(expr) <- md$cell_barcode

# Raw counts for the detection gate (fraction_cells_expressing is count-based and
# normalization-free, so it means the same thing in every cohort's pipeline).
counts <- tryCatch(
  GetAssayData(obj, assay = DefaultAssay(obj), layer = "counts"),
  error = function(e) GetAssayData(obj, assay = DefaultAssay(obj), slot = "counts")
)
counts <- counts[unique(kin$matched_gene), md$cell_barcode, drop = FALSE]

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

write.csv(
  group_meta[, c("tissue", "age_months", "genotype", "sample_id", "cell_type", "n_cells")],
  counts_path,
  row.names = FALSE
)

# --- Within-cohort detection + expression for the standard attribution metric ---
# Pool cells by (tissue, new_clusters) and emit, per (gene, tissue, cell_type):
#   fraction_cells_expressing  count-based detection gate (# cells with count > 0)
#   mean_log2_expression       mean over cells of log2(x+1) = mean_ln / log(2)
# specificity.compute (Python, alz/cross_reference/specificity.py) turns these into
# the standard detection/concentration columns — one definition, every cohort.
det_group <- paste(md$tissue, md$new_clusters, sep = "|")
det_fac <- factor(det_group, levels = unique(det_group))
det_design <- sparseMatrix(
  i = seq_along(det_fac),
  j = as.integer(det_fac),
  x = 1,
  dims = c(length(det_fac), nlevels(det_fac)),
  dimnames = list(md$cell_barcode, levels(det_fac))
)
det_ncell <- Matrix::colSums(det_design)
sum_ln_tc <- as.matrix(expr %*% det_design)        # genes x groups, Σ ln(1 + x)
nnz_tc <- as.matrix((counts > 0) %*% det_design)    # genes x groups, # expressing cells
mean_ln_tc <- t(t(sum_ln_tc) / pmax(det_ncell, 1))
frac_tc <- t(t(nnz_tc) / pmax(det_ncell, 1))
det_tissue <- sub("\\|.*$", "", colnames(det_design))
det_ct <- sub("^[^|]*\\|", "", colnames(det_design))
genes_u <- rownames(mean_ln_tc)
expr_long <- data.frame(
  matched_gene = rep(genes_u, times = ncol(mean_ln_tc)),
  tissue = rep(det_tissue, each = nrow(mean_ln_tc)),
  cell_type = rep(det_ct, each = nrow(mean_ln_tc)),
  mean_log2_expression = as.numeric(mean_ln_tc) / log(2),
  fraction_cells_expressing = as.numeric(frac_tc),
  n_cells = rep(det_ncell, each = nrow(mean_ln_tc)),
  stringsAsFactors = FALSE
)
# Complete to the full cluster x tissue grid so N_total (the tier baseline) is the
# whole cohort, not just observed combos; absent combos are zero (undetected).
full_grid <- expand.grid(
  matched_gene = genes_u,
  tissue = c("cortex", "hippocampus"),
  cell_type = clusters,
  stringsAsFactors = FALSE
)
expr_long <- merge(full_grid, expr_long, by = c("matched_gene", "tissue", "cell_type"), all.x = TRUE)
expr_long$mean_log2_expression[is.na(expr_long$mean_log2_expression)] <- 0
expr_long$fraction_cells_expressing[is.na(expr_long$fraction_cells_expressing)] <- 0
expr_long$n_cells[is.na(expr_long$n_cells)] <- 0
# Denormalize per kinase (a gene can back several kinases); specificity.compute
# dedups on matched_gene before computing the per-gene metric.
expr_long <- merge(
  unique(kin[, c("kinase", "gene_symbol", "matched_gene")]),
  expr_long, by = "matched_gene", all.x = FALSE
)
write.csv(
  expr_long[, c(
    "kinase", "gene_symbol", "matched_gene", "tissue", "cell_type",
    "mean_log2_expression", "fraction_cells_expressing", "n_cells"
  )],
  expr_path,
  row.names = FALSE
)

# Disease-direction LFC per (kinase, tissue, age, cell_type): TG - WT mean pseudobulk
# log-expression. Specificity (above) is contrast-invariant and lives in expr_path;
# this CSV carries only the direction signal + cell support.
out <- list()
idx <- 0L
for (i in seq_len(nrow(kin))) {
  gene <- kin$matched_gene[[i]]
  vals <- as.numeric(pb[gene, ])
  for (tissue in c("cortex", "hippocampus")) {
    for (age in c(3L, 6L, 9L, 12L)) {
      for (cell_type in clusters) {
        mask <- group_meta$tissue == tissue & group_meta$age_months == age & group_meta$cell_type == cell_type
        wt <- mask & group_meta$genotype == "WT"
        tg <- mask & group_meta$genotype == "TG"
        wt_vals <- vals[wt]
        tg_vals <- vals[tg]
        n_wt <- length(unique(group_meta$sample_id[wt]))
        n_tg <- length(unique(group_meta$sample_id[tg]))
        n_cells_wt <- sum(group_meta$n_cells[wt])
        n_cells_tg <- sum(group_meta$n_cells[tg])
        lfc <- if (n_wt > 0 && n_tg > 0) mean(tg_vals, na.rm = TRUE) - mean(wt_vals, na.rm = TRUE) else NA_real_
        pval <- NA_real_
        if (n_wt >= 2 && n_tg >= 2 && stats::sd(c(wt_vals, tg_vals), na.rm = TRUE) > 0) {
          pval <- tryCatch(stats::t.test(tg_vals, wt_vals)$p.value, error = function(e) NA_real_)
        }
        idx <- idx + 1L
        out[[idx]] <- data.frame(
          kinase = kin$kinase[[i]],
          gene_symbol = kin$gene_symbol[[i]],
          matched_gene = gene,
          tissue = tissue,
          age_months = age,
          cell_type = cell_type,
          fivexfad_lfc = lfc,
          fivexfad_pval = pval,
          n_snrna_samples_wt = n_wt,
          n_snrna_samples_tg = n_tg,
          n_cells_wt = n_cells_wt,
          n_cells_tg = n_cells_tg,
          cluster_source = "new_clusters",
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

df <- do.call(rbind, out)
df$fivexfad_fdr <- ave(df$fivexfad_pval, df$tissue, df$age_months, FUN = function(x) {
  ok <- is.finite(x)
  y <- rep(NA_real_, length(x))
  if (any(ok)) y[ok] <- p.adjust(x[ok], method = "BH")
  y
})
df <- df[, c(
  "kinase", "gene_symbol", "matched_gene", "tissue", "age_months", "cell_type",
  "fivexfad_lfc", "fivexfad_pval", "fivexfad_fdr",
  "n_snrna_samples_wt", "n_snrna_samples_tg", "n_cells_wt", "n_cells_tg",
  "cluster_source"
)]
write.csv(df, out_path, row.names = FALSE)

cat("[5xfad-snrna-attribution] wrote ", out_path, " rows=", nrow(df), "\n", sep = "")
cat("[5xfad-snrna-attribution] wrote ", counts_path, "\n", sep = "")
cat("[5xfad-snrna-attribution] wrote ", expr_path, " rows=", nrow(expr_long), "\n", sep = "")
