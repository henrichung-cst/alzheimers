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

specificity_tau <- function(x) {
  x <- as.numeric(x)
  if (length(x) <= 1 || all(!is.finite(x))) return(NA_real_)
  mx <- max(x, na.rm = TRUE)
  if (!is.finite(mx) || mx <= 0) return(NA_real_)
  sum(1 - (x / mx), na.rm = TRUE) / (length(x) - 1)
}

location_tier <- function(fold) {
  if (!is.finite(fold)) return("none")
  if (fold >= 2) return("high")
  if (fold >= 1) return("moderate")
  "low"
}

min_cells_per_contrast <- 3L

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

uniform <- 1 / length(clusters)
out <- list()
idx <- 0L
for (i in seq_len(nrow(kin))) {
  gene <- kin$matched_gene[[i]]
  vals <- as.numeric(pb[gene, ])
  for (tissue in c("cortex", "hippocampus")) {
    tissue_mask <- group_meta$tissue == tissue
    tissue_linear <- expm1(vals[tissue_mask])
    tissue_meta <- group_meta[tissue_mask, , drop = FALSE]
    spec_by_cluster <- tapply(tissue_linear, tissue_meta$cell_type, mean, na.rm = TRUE)
    spec_by_cluster <- spec_by_cluster[clusters]
    spec_by_cluster[!is.finite(spec_by_cluster)] <- 0
    total <- sum(spec_by_cluster)
    share <- if (is.finite(total) && total > 0) spec_by_cluster / total else rep(NA_real_, length(clusters))
    tau <- specificity_tau(spec_by_cluster)
    top_cluster <- if (any(is.finite(share))) names(which.max(share)) else ""

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
        spec <- unname(share[cell_type])
        fold <- spec / uniform
        tier <- location_tier(fold)
        basis <- paste0("5xFAD snRNA tissue-specific new_clusters location tier: ", tier)
        if ((n_cells_wt + n_cells_tg) < min_cells_per_contrast) {
          spec <- NA_real_
          fold <- NA_real_
          tier <- "none"
          basis <- paste0(
            "Fewer than ", min_cells_per_contrast,
            " 5xFAD snRNA cells for this tissue, age, and new_clusters label; ",
            "tissue-specific location tier not applied"
          )
        }
        idx <- idx + 1L
        out[[idx]] <- data.frame(
          kinase = kin$kinase[[i]],
          gene_symbol = kin$gene_symbol[[i]],
          matched_gene = gene,
          tissue = tissue,
          age_months = age,
          cell_type = cell_type,
          confidence_tier = tier,
          confidence_basis = basis,
          fivexfad_specificity = spec,
          fivexfad_fold_over_uniform = fold,
          fivexfad_tau = tau,
          fivexfad_top_cluster = top_cluster,
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
  "confidence_tier", "confidence_basis", "fivexfad_specificity",
  "fivexfad_fold_over_uniform", "fivexfad_tau", "fivexfad_top_cluster",
  "fivexfad_lfc", "fivexfad_pval", "fivexfad_fdr",
  "n_snrna_samples_wt", "n_snrna_samples_tg", "n_cells_wt", "n_cells_tg",
  "cluster_source"
)]
write.csv(df, out_path, row.names = FALSE)

cat("[5xfad-snrna-attribution] wrote ", out_path, " rows=", nrow(df), "\n", sep = "")
cat("[5xfad-snrna-attribution] wrote ", counts_path, "\n", sep = "")
