#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(dplyr)
  library(limma)
  library(readr)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

condition1 <- "ma_2mo_AppP"
condition2 <- "ma_2mo_WTyp"

paths <- list(
  ours_parquet = file.path(repo, "outputs/reports/incytr_pair_mode/_sce4_one_contrast_q0/ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet"),
  ref_rds = file.path(repo, "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/DEG_PRG_ma_2mo_AppP_WTyp_10302025/sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds"),
  allmarkers = file.path(repo, "data/derived/incytr_inputs/allmarkers.csv"),
  derived = file.path(repo, "data/derived/incytr_inputs"),
  source = file.path(repo, "data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721"),
  out_dir = file.path(repo, "outputs/reports/incytr_pair_mode/forensics")
)

dir.create(paths$out_dir, recursive = TRUE, showWarnings = FALSE)

norm_key <- function(s) tolower(trimws(gsub("[._ -]+", " ", s)))

build_crosswalk <- function(allmarkers) {
  cl <- unique(fread(allmarkers, select = "cluster")$cluster)
  types <- sort(unique(sub("_ma_[0-9]+mo_(AppP|WTyp|Ttau|ApTt)$", "", cl)))
  keys <- norm_key(types)
  if (anyDuplicated(keys)) {
    dup <- types[duplicated(keys) | duplicated(keys, fromLast = TRUE)]
    stop("spine label collision under norm(): ", paste(dup, collapse = " | "), call. = FALSE)
  }
  setNames(types, keys)
}

map_clusters <- function(values, xwalk, where) {
  k <- norm_key(values)
  miss <- unique(values[!(k %in% names(xwalk))])
  if (length(miss)) {
    stop(sprintf("%s: sce4 cluster(s) with no spine match: %s",
                 where, paste(miss, collapse = ", ")), call. = FALSE)
  }
  unname(xwalk[k])
}

path_key <- function(dt) {
  paste(dt$Sender, dt$Receiver, dt$Ligand, dt$Receptor, dt$EM, dt$Target, sep = "\001")
}

simple_key <- function(dt) {
  paste(dt$Sender, dt$Receiver, dt$Ligand, dt$Receptor, dt$EM, dt$Target, sep = "|")
}

sig_col <- function(dt, nm) {
  if (nm %in% names(dt)) return(dt[[nm]])
  short <- sub("^ma_[0-9]+mo_", "", nm)
  if (short %in% names(dt)) return(dt[[short]])
  stop("missing column: ", nm, call. = FALSE)
}

read_ref <- function() {
  xwalk <- build_crosswalk(paths$allmarkers)
  cols <- c(
    "Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target",
    "SigProb_AppP", "SigProb_WTyp", "PDS",
    "Ligand_pr_aFC", "Receptor_pr_aFC", "EM_pr_aFC", "Target_pr_aFC",
    "Ligand_ps_aFC", "Receptor_ps_aFC", "EM_ps_aFC", "Target_ps_aFC",
    "Ligand_py_aFC", "Receptor_py_aFC", "EM_py_aFC", "Target_py_aFC",
    "PhPDS_ps", "PhPDS_py"
  )
  rbindlist(lapply(readRDS(paths$ref_rds), function(e) {
    d <- as.data.table(e)
    keep <- intersect(cols, names(d))
    d <- d[, ..keep]
    d[, Sender := map_clusters(Sender.group, xwalk, "sce4 ref")]
    d[, Receiver := map_clusters(Receiver.group, xwalk, "sce4 ref")]
    d
  }), use.names = TRUE, fill = TRUE)
}

read_ours <- function() {
  d <- as.data.table(read_parquet(paths$ours_parquet))
  d[, `:=`(
    SigProb_AppP = get(paste0("SigProb_", condition1)),
    SigProb_WTyp = get(paste0("SigProb_", condition2))
  )]
  d
}

slice_omics <- function(df, gene_col, condition, suffix) {
  pat <- paste0("^", condition, "_")
  out <- dplyr::select(df, matches(pat))
  colnames(out) <- paste0(sub(pat, "", colnames(out)), "_", suffix)
  out$gene_symbol <- df[[gene_col]]
  out %>% group_by(gene_symbol) %>% summarise_all(mean, na.rm = TRUE)
}

floor_pr <- function(df) {
  num_cols <- setdiff(colnames(df), "gene_symbol")
  for (cc in num_cols) df[[cc]] <- pmax(df[[cc]], 1)
  df
}

cal_foldchange <- function(df, correction = 0.0001, q = 0.75) {
  c1 <- df$condition1
  c2 <- df$condition2
  if (any(c1 == 0 | c2 == 0, na.rm = TRUE)) {
    df$condition1 <- df$condition1 + correction
    df$condition2 <- df$condition2 + correction
  }
  df$log2FC <- log2(df$condition1 / df$condition2)
  th <- quantile(c(df$condition1, df$condition2), q, na.rm = TRUE)
  vmax <- pmax(df$condition1, df$condition2)
  adj <- pmin(2 * (vmax^2) / (vmax^2 + th^2), 1)
  df$aFC <- df$log2FC * adj
  df
}

compute_afc_table <- function(input_dir, suffix, gene_col, clusters, floor_values = FALSE) {
  df <- read_csv(file.path(input_dir, paste0(suffix, "_yuyu_deconvoluted.csv")), show_col_types = FALSE)
  c1 <- slice_omics(df, gene_col, condition1, suffix)
  c2 <- slice_omics(df, gene_col, condition2, suffix)
  if (floor_values) {
    c1 <- floor_pr(c1)
    c2 <- floor_pr(c2)
  }
  rbindlist(lapply(clusters, function(cl) {
    col <- paste0(cl, "_", suffix)
    if (!(col %in% names(c1)) || !(col %in% names(c2))) {
      return(data.table(cluster = cl, gene_symbol = character(), log2FC = numeric(), aFC = numeric()))
    }
    common <- intersect(c1$gene_symbol, c2$gene_symbol)
    fc <- data.frame(
      gene_symbol = common,
      condition1 = c1[[col]][match(common, c1$gene_symbol)],
      condition2 = c2[[col]][match(common, c2$gene_symbol)]
    )
    fc[, c("condition1", "condition2")] <- normalizeBetweenArrays(as.matrix(fc[, c("condition1", "condition2")]))
    fc <- cal_foldchange(fc)
    data.table(cluster = cl, gene_symbol = fc$gene_symbol, log2FC = fc$log2FC, aFC = fc$aFC)
  }), use.names = TRUE, fill = TRUE)
}

make_node_long <- function(dt, value_prefix) {
  roles <- c("Ligand", "Receptor", "EM", "Target")
  rbindlist(lapply(roles, function(role) {
    out <- dt[, .(Sender, Receiver, Ligand, Receptor, EM, Target)]
    out[, `:=`(
      role = role,
      cluster = if (role == "Ligand") Sender else Receiver,
      gene_symbol = dt[[role]]
    )]
    for (suffix in c("pr", "ps", "py")) {
      col <- paste0(role, "_", suffix, "_aFC")
      if (col %in% names(dt)) {
        out[[paste0(value_prefix, "_", suffix, "_aFC")]] <- dt[[col]]
      }
    }
    out
  }), use.names = TRUE, fill = TRUE)
}

join_source_values <- function(nodes, source_name, input_dir, clusters) {
  specs <- list(
    pr = list(gene_col = "Gene Symbol", floor_values = TRUE),
    ps = list(gene_col = "gene_symbol", floor_values = FALSE),
    py = list(gene_col = "gene_symbol", floor_values = FALSE)
  )
  out <- copy(nodes)
  for (suffix in names(specs)) {
    fc <- compute_afc_table(input_dir, suffix, specs[[suffix]]$gene_col, clusters, specs[[suffix]]$floor_values)
    setnames(fc, c("log2FC", "aFC"), c(paste0(source_name, "_", suffix, "_log2FC"), paste0(source_name, "_", suffix, "_aFC")))
    out <- merge(out, fc, by = c("cluster", "gene_symbol"), all.x = TRUE, sort = FALSE)
  }
  out
}

summarise_delta <- function(dt, subset_name) {
  rows <- list()
  for (suffix in c("pr", "ps", "py")) {
    for (src in c("derived", "source")) {
      a <- paste0(src, "_", suffix, "_aFC")
      b <- paste0("ref_", suffix, "_aFC")
      if (!(a %in% names(dt)) || !(b %in% names(dt))) next
      delta <- abs(dt[[a]] - dt[[b]])
      rows[[length(rows) + 1L]] <- data.table(
        subset = subset_name,
        source = src,
        channel = suffix,
        n = sum(!is.na(delta)),
        median_abs = median(delta, na.rm = TRUE),
        p95_abs = as.numeric(quantile(delta, 0.95, na.rm = TRUE)),
        p99_abs = as.numeric(quantile(delta, 0.99, na.rm = TRUE)),
        max_abs = max(delta, na.rm = TRUE),
        n_gt_0.01 = sum(delta > 0.01, na.rm = TRUE),
        n_gt_0.05 = sum(delta > 0.05, na.rm = TRUE)
      )
    }
  }
  rbindlist(rows)
}

summarise_source_vs_derived <- function(dt, subset_name) {
  rows <- list()
  for (suffix in c("pr", "ps", "py")) {
    dcol <- paste0("derived_", suffix, "_aFC")
    scol <- paste0("source_", suffix, "_aFC")
    rcol <- paste0("ref_", suffix, "_aFC")
    if (!(dcol %in% names(dt)) || !(scol %in% names(dt)) || !(rcol %in% names(dt))) next
    keep <- !is.na(dt[[dcol]]) & !is.na(dt[[scol]]) & !is.na(dt[[rcol]])
    if (!any(keep)) next
    ds <- abs(dt[[dcol]][keep] - dt[[scol]][keep])
    dr <- abs(dt[[dcol]][keep] - dt[[rcol]][keep])
    sr <- abs(dt[[scol]][keep] - dt[[rcol]][keep])
    rows[[length(rows) + 1L]] <- data.table(
      subset = subset_name,
      channel = suffix,
      n = length(ds),
      median_abs_derived_source = median(ds, na.rm = TRUE),
      p95_abs_derived_source = as.numeric(quantile(ds, 0.95, na.rm = TRUE)),
      max_abs_derived_source = max(ds, na.rm = TRUE),
      source_closer_to_ref = sum(sr < dr, na.rm = TRUE),
      derived_closer_to_ref = sum(dr < sr, na.rm = TRUE),
      ties = sum(dr == sr, na.rm = TRUE)
    )
  }
  rbindlist(rows)
}

ref <- read_ref()
ours <- read_ours()
ref[, key := path_key(ref)]
ours[, key := path_key(ours)]
ref <- unique(ref, by = "key")
ours <- unique(ours, by = "key")

ref[, gated_ref := (SigProb_AppP > 0.1 | SigProb_WTyp > 0.1) & abs(PDS) >= 0.2]
ours[, gated_ours := (SigProb_AppP > 0.1 | SigProb_WTyp > 0.1) & abs(PDS) >= 0.2]

shared_keys <- intersect(ref[gated_ref == TRUE, key], ours[gated_ours == TRUE, key])
missing_keys <- setdiff(ref[gated_ref == TRUE, key], ours[gated_ours == TRUE, key])

ref_shared <- ref[key %in% shared_keys]
ref_missing <- ref[key %in% missing_keys]

clusters <- sort(unique(c(ours$Sender, ours$Receiver, ref$Sender, ref$Receiver)))
nodes_shared <- make_node_long(ref_shared, "ref")
nodes_missing <- make_node_long(ref_missing, "ref")

nodes_shared <- join_source_values(nodes_shared, "derived", paths$derived, clusters)
nodes_shared <- join_source_values(nodes_shared, "source", paths$source, clusters)
nodes_missing <- join_source_values(nodes_missing, "derived", paths$derived, clusters)
nodes_missing <- join_source_values(nodes_missing, "source", paths$source, clusters)

nodes_shared[, row_key := simple_key(.SD), .SDcols = c("Sender", "Receiver", "Ligand", "Receptor", "EM", "Target")]
nodes_missing[, row_key := simple_key(.SD), .SDcols = c("Sender", "Receiver", "Ligand", "Receptor", "EM", "Target")]

summary <- rbindlist(list(
  summarise_delta(nodes_shared, "shared_gated_nodes"),
  summarise_delta(nodes_missing, "missing_ref_gated_nodes")
), use.names = TRUE)

source_compare <- rbindlist(list(
  summarise_source_vs_derived(nodes_shared, "shared_gated_nodes"),
  summarise_source_vs_derived(nodes_missing, "missing_ref_gated_nodes")
), use.names = TRUE)

fwrite(summary, file.path(paths$out_dir, "ma_2mo_AppP_afc_delta_summary.csv"))
fwrite(source_compare, file.path(paths$out_dir, "ma_2mo_AppP_source_vs_derived_afc_summary.csv"))
fwrite(nodes_missing, file.path(paths$out_dir, "ma_2mo_AppP_missing_ref_nodes_afc.csv"))

top_ps <- copy(nodes_shared)
top_ps[, derived_ps_abs_delta := abs(derived_ps_aFC - ref_ps_aFC)]
top_ps[, source_ps_abs_delta := abs(source_ps_aFC - ref_ps_aFC)]
top_ps <- top_ps[!is.na(derived_ps_abs_delta)]
setorder(top_ps, -derived_ps_abs_delta)
fwrite(top_ps[1:min(.N, 200)], file.path(paths$out_dir, "ma_2mo_AppP_top_shared_ps_afc_deltas.csv"))

top_py <- copy(nodes_shared)
top_py[, derived_py_abs_delta := abs(derived_py_aFC - ref_py_aFC)]
top_py[, source_py_abs_delta := abs(source_py_aFC - ref_py_aFC)]
top_py <- top_py[!is.na(derived_py_abs_delta)]
setorder(top_py, -derived_py_abs_delta)
fwrite(top_py[1:min(.N, 200)], file.path(paths$out_dir, "ma_2mo_AppP_top_shared_py_afc_deltas.csv"))

cat("wrote forensic summaries to ", paths$out_dir, "\n", sep = "")
cat("shared gated rows: ", length(shared_keys), "\n", sep = "")
cat("missing ref gated rows: ", length(missing_keys), "\n", sep = "")
