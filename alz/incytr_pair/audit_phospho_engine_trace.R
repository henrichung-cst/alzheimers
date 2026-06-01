#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(limma)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  missing_glob = file.path(repo, "outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/ma_*_missing_nontransgene_audit.csv"),
  input_dir = file.path(repo, "data/derived/incytr_inputs_source_ps_diag"),
  out_dir = file.path(repo, "outputs/reports/incytr_pair_mode/forensics_source_ps_full_diag/phospho_engine_trace")
)

parse_args <- function(args) {
  x <- defaults
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    need <- function() {
      if (i == length(args)) stop("missing value after ", a, call. = FALSE)
      args[[i + 1L]]
    }
    if (a == "--missing-glob") {
      x$missing_glob <- need(); i <- i + 2L
    } else if (a == "--input-dir") {
      x$input_dir <- need(); i <- i + 2L
    } else if (a == "--out-dir") {
      x$out_dir <- need(); i <- i + 2L
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
  }
  x
}

logi <- function(x, k = 2) 2 / (1 + exp(-k * x)) - 1

cal_foldchange <- function(dt, correction = 0.001, q = 0.75) {
  x <- copy(dt)
  has_zero <- any(x$condition1 == 0 | x$condition2 == 0, na.rm = TRUE)
  if (has_zero && correction == 0) stop("zero values with correction=0", call. = FALSE)
  if (has_zero) {
    x[, condition1 := condition1 + correction]
    x[, condition2 := condition2 + correction]
  }
  x[, log2FC := log2(condition1 / condition2)]
  th <- stats::quantile(c(x$condition1, x$condition2), q, na.rm = TRUE)
  vmax <- pmax(x$condition1, x$condition2)
  adj <- pmin(2 * (vmax^2) / (vmax^2 + th^2), 1)
  x[, aFC := log2FC * adj]
  x
}

read_omics <- function(input_dir, suffix) {
  path <- file.path(input_dir, paste0(suffix, "_yuyu_deconvoluted.csv"))
  d <- fread(path)
  if (!"gene_symbol" %in% names(d) && "Gene Symbol" %in% names(d)) {
    setnames(d, "Gene Symbol", "gene_symbol")
  }
  d
}

prepare_fc <- function(raw, contrast, cluster, suffix) {
  age <- sub("^ma_([0-9]+mo)_.*$", "\\1", contrast)
  c2 <- paste0("ma_", age, "_WTyp")
  col1 <- paste0(contrast, "_", cluster)
  col2 <- paste0(c2, "_", cluster)
  if (!all(c(col1, col2, "gene_symbol") %in% names(raw))) {
    return(data.table(gene_symbol = character(), log2FC = numeric(), aFC = numeric()))
  }
  x <- raw[, .(gene_symbol, condition1 = get(col1), condition2 = get(col2))]
  x <- x[!is.na(gene_symbol)]
  x <- x[, .(
    condition1 = mean(condition1, na.rm = TRUE),
    condition2 = mean(condition2, na.rm = TRUE)
  ), by = gene_symbol]
  if (nrow(x)) {
    x[, c("condition1", "condition2") := as.data.table(
      normalizeBetweenArrays(as.matrix(.SD))
    ), .SDcols = c("condition1", "condition2")]
  }
  cal_foldchange(x, correction = 0.001, q = 0.75)[, .(gene_symbol, log2FC, aFC)]
}

node_requests <- function(dt) {
  roles <- c("Ligand", "Receptor", "EM", "Target")
  rbindlist(lapply(roles, function(role) {
    data.table(
      contrast = dt$contrast,
      row_id = dt$row_id,
      role = role,
      cluster = if (role == "Ligand") dt$Sender else dt$Receiver,
      gene_symbol = dt[[role]]
    )
  }), use.names = TRUE)
}

score_rows <- function(rows, fc, suffix) {
  roles <- c("Ligand", "Receptor", "EM", "Target")
  vals <- lapply(roles, function(role) {
    nodes <- data.table(
      row_id = rows$row_id,
      role = role,
      contrast = rows$contrast,
      cluster = if (role == "Ligand") rows$Sender else rows$Receiver,
      gene_symbol = rows[[role]]
    )
    y <- merge(nodes, fc, by = c("contrast", "cluster", "gene_symbol"), all.x = TRUE, sort = FALSE)
    y[order(row_id), aFC]
  })
  mat <- do.call(cbind, vals)
  mat[is.na(mat)] <- 0
  rowMeans(logi(mat, 2))
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(opts$out_dir, recursive = TRUE, showWarnings = FALSE)
files <- Sys.glob(opts$missing_glob)
if (!length(files)) stop("no files matched: ", opts$missing_glob, call. = FALSE)

lst <- lapply(files, fread)
names(lst) <- basename(files)
rows <- rbindlist(lst, use.names = TRUE, fill = TRUE, idcol = "file")
rows[, contrast := sub("_missing_nontransgene_audit.csv$", "", file)]
rows[, row_id := .I]

raw <- list(ps = read_omics(opts$input_dir, "ps"), py = read_omics(opts$input_dir, "py"))
requests <- unique(node_requests(rows)[, .(contrast, cluster, gene_symbol)])
fc <- rbindlist(lapply(c("ps", "py"), function(suffix) {
  rbindlist(lapply(seq_len(nrow(requests)), function(i) {
    x <- prepare_fc(raw[[suffix]], requests$contrast[i], requests$cluster[i], suffix)
    if (!nrow(x)) return(NULL)
    x[gene_symbol == requests$gene_symbol[i]][
      , `:=`(contrast = requests$contrast[i], cluster = requests$cluster[i], suffix = suffix)]
  }), use.names = TRUE, fill = TRUE)
}), use.names = TRUE, fill = TRUE)

rows[, candidate_PhPDS_ps := score_rows(rows, fc[suffix == "ps"], "ps")]
rows[, candidate_PhPDS_py := score_rows(rows, fc[suffix == "py"], "py")]
rows[, `:=`(
  candidate_vs_ours_PhPDS_ps = candidate_PhPDS_ps - ours_PhPDS_ps,
  candidate_vs_ref_PhPDS_ps = candidate_PhPDS_ps - ref_PhPDS_ps,
  candidate_vs_ours_PhPDS_py = candidate_PhPDS_py - ours_PhPDS_py,
  candidate_vs_ref_PhPDS_py = candidate_PhPDS_py - ref_PhPDS_py
)]

summary <- rbindlist(lapply(c("PhPDS_ps", "PhPDS_py"), function(metric) {
  data.table(
    metric = metric,
    rows = nrow(rows),
    median_abs_candidate_vs_ours = median(abs(rows[[paste0("candidate_vs_ours_", metric)]]), na.rm = TRUE),
    p95_abs_candidate_vs_ours = as.numeric(quantile(abs(rows[[paste0("candidate_vs_ours_", metric)]]), 0.95, na.rm = TRUE)),
    max_abs_candidate_vs_ours = max(abs(rows[[paste0("candidate_vs_ours_", metric)]]), na.rm = TRUE),
    median_abs_candidate_vs_ref = median(abs(rows[[paste0("candidate_vs_ref_", metric)]]), na.rm = TRUE),
    p95_abs_candidate_vs_ref = as.numeric(quantile(abs(rows[[paste0("candidate_vs_ref_", metric)]]), 0.95, na.rm = TRUE)),
    max_abs_candidate_vs_ref = max(abs(rows[[paste0("candidate_vs_ref_", metric)]]), na.rm = TRUE)
  )
}), use.names = TRUE)

detail_path <- file.path(opts$out_dir, "phospho_engine_trace_detail.csv")
summary_path <- file.path(opts$out_dir, "phospho_engine_trace_summary.csv")
fwrite(rows, detail_path)
fwrite(summary, summary_path)
print(summary)
cat("Wrote:\n  ", detail_path, "\n  ", summary_path, "\n", sep = "")
