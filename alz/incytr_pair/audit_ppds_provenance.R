#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(limma)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  missing_glob = file.path(repo, "outputs/reports/incytr_pair_mode/forensics_v2_46bundle/ma_4mo_Ttau_missing_nontransgene_audit.csv"),
  derived_dir = file.path(repo, "data/derived/incytr_inputs"),
  v2_dir = file.path(repo, "data/incytr_frozen/v2_46clusters/incytr input"),
  out_dir = file.path(repo, "outputs/reports/incytr_pair_mode/forensics")
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
    } else if (a == "--derived-dir") {
      x$derived_dir <- need(); i <- i + 2L
    } else if (a == "--v2-dir") {
      x$v2_dir <- need(); i <- i + 2L
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
  if (is.null(q)) q <- 0.75
  th <- stats::quantile(c(x$condition1, x$condition2), q, na.rm = TRUE)
  vmax <- pmax(x$condition1, x$condition2)
  adj <- pmin(2 * (vmax^2) / (vmax^2 + th^2), 1)
  x[, aFC := log2FC * adj]
  x
}

read_pr <- function(input_dir) {
  path <- file.path(input_dir, "pr_yuyu_deconvoluted.csv")
  if (!file.exists(path)) stop("missing protein input: ", path, call. = FALSE)
  d <- fread(path)
  if (!"gene_symbol" %in% names(d) && "Gene Symbol" %in% names(d)) {
    setnames(d, "Gene Symbol", "gene_symbol")
  }
  if (!"gene_symbol" %in% names(d)) stop("no gene_symbol/Gene Symbol column in ", path, call. = FALSE)
  d
}

node_table <- function(files) {
  roles <- c("Ligand", "Receptor", "EM", "Target")
  rbindlist(lapply(files, function(f) {
    d <- fread(f)
    contrast <- sub("_missing_nontransgene_audit.csv$", "", basename(f))
    d[, row_id := .I]
    rbindlist(lapply(roles, function(role) {
      cluster <- if (role == "Ligand") d$Sender else d$Receiver
      data.table(
        contrast = contrast,
        row_id = d$row_id,
        path_key = d$key,
        role = role,
        cluster = cluster,
        gene_symbol = d[[role]],
        ref_pr_aFC = d[[paste0("ref_", role, "_pr_aFC")]],
        ref_PPDS = d$ref_PPDS,
        ours_PPDS = d$ours_PPDS,
        delta_PPDS = d$delta_PPDS,
        ref_PDS = d$ref_PDS,
        ours_PDS = d$ours_PDS,
        largest_abs_delta_component = d$largest_abs_delta_component
      )
    }), use.names = TRUE, fill = TRUE)
  }), use.names = TRUE, fill = TRUE)
}

collapse_gene_values <- function(raw, c1, c2, cluster, duplicate_mode, floor_mode, normalize) {
  col1 <- paste0(c1, "_", cluster)
  col2 <- paste0(c2, "_", cluster)
  if (!all(c(col1, col2, "gene_symbol") %in% names(raw))) {
    return(data.table(gene_symbol = character(), condition1 = numeric(), condition2 = numeric()))
  }

  x <- raw[, .(gene_symbol, condition1 = get(col1), condition2 = get(col2))]
  x <- x[!is.na(gene_symbol)]

  if (duplicate_mode == "first") {
    x <- x[!duplicated(gene_symbol)]
  } else if (duplicate_mode %in% c("mean_raw", "median_raw")) {
    fun <- if (duplicate_mode == "mean_raw") mean else median
    x <- x[, .(
      condition1 = fun(condition1, na.rm = TRUE),
      condition2 = fun(condition2, na.rm = TRUE)
    ), by = gene_symbol]
  } else {
    stop("unknown duplicate_mode: ", duplicate_mode, call. = FALSE)
  }

  if (floor_mode == "floor_lt1") {
    x[, `:=`(condition1 = pmax(condition1, 1), condition2 = pmax(condition2, 1))]
  } else if (floor_mode != "none") {
    stop("unknown floor_mode: ", floor_mode, call. = FALSE)
  }

  if (normalize && nrow(x)) {
    x[, c("condition1", "condition2") := as.data.table(normalizeBetweenArrays(as.matrix(.SD))),
      .SDcols = c("condition1", "condition2")]
  }
  x
}

score_one_spec <- function(nodes, raw, input_name, duplicate_mode, floor_mode, normalize, q, correction) {
  targets <- unique(nodes[, .(contrast, cluster)])
  vals <- rbindlist(lapply(seq_len(nrow(targets)), function(i) {
    contrast <- targets$contrast[i]
    age <- sub("^ma_([0-9]+mo)_.*$", "\\1", contrast)
    c2 <- paste0("ma_", age, "_WTyp")
    base <- collapse_gene_values(
      raw, contrast, c2, targets$cluster[i],
      duplicate_mode = duplicate_mode,
      floor_mode = floor_mode,
      normalize = normalize
    )
    if (!nrow(base)) return(NULL)
    fc <- cal_foldchange(base, correction = correction, q = q)
    fc <- fc[, .(gene_symbol, candidate_pr_aFC = aFC)]
    fc[, `:=`(contrast = contrast, cluster = targets$cluster[i])]
    fc
  }), use.names = TRUE, fill = TRUE)

  detail <- merge(
    nodes,
    vals,
    by = c("contrast", "cluster", "gene_symbol"),
    all.x = TRUE,
    sort = FALSE,
    allow.cartesian = TRUE
  )
  detail[, `:=`(
    input = input_name,
    duplicate_mode = duplicate_mode,
    floor_mode = floor_mode,
    normalize = normalize,
    q = if (is.null(q)) NA_real_ else q,
    correction = correction,
    node_abs_delta = abs(candidate_pr_aFC - ref_pr_aFC)
  )]

  ppds <- dcast(
    detail,
    input + duplicate_mode + floor_mode + normalize + q + correction +
      contrast + row_id + path_key + ref_PPDS + ours_PPDS + delta_PPDS + ref_PDS + ours_PDS +
      largest_abs_delta_component ~ role,
    value.var = c("candidate_pr_aFC", "ref_pr_aFC")
  )
  role_cols <- paste0("candidate_pr_aFC_", c("Ligand", "Receptor", "EM", "Target"))
  ref_cols <- paste0("ref_pr_aFC_", c("Ligand", "Receptor", "EM", "Target"))
  for (cc in c(role_cols, ref_cols)) {
    if (!cc %in% names(ppds)) ppds[[cc]] <- NA_real_
  }
  cand_mat <- as.matrix(ppds[, ..role_cols])
  ref_mat <- as.matrix(ppds[, ..ref_cols])
  cand_mat[is.na(cand_mat)] <- 0
  ref_mat[is.na(ref_mat)] <- 0
  ppds[, `:=`(
    candidate_PPDS = rowMeans(logi(cand_mat, 2)),
    ref_recalc_PPDS = rowMeans(logi(ref_mat, 2))
  )]
  ppds[, `:=`(
    candidate_vs_ref_PPDS_delta = candidate_PPDS - ref_PPDS,
    candidate_vs_ref_PPDS_abs_delta = abs(candidate_PPDS - ref_PPDS),
    candidate_vs_ours_PPDS_delta = candidate_PPDS - ours_PPDS,
    candidate_vs_ours_PPDS_abs_delta = abs(candidate_PPDS - ours_PPDS),
    ref_recalc_vs_ref_PPDS_abs_delta = abs(ref_recalc_PPDS - ref_PPDS)
  )]

  list(detail = detail, ppds = ppds)
}

summarise_nodes <- function(dt) {
  dt[, .(
    n_nodes = .N,
    n_matched = sum(!is.na(candidate_pr_aFC)),
    median_node_abs = median(node_abs_delta, na.rm = TRUE),
    p95_node_abs = as.numeric(quantile(node_abs_delta, 0.95, na.rm = TRUE)),
    max_node_abs = max(node_abs_delta, na.rm = TRUE),
    n_node_gt_0.01 = sum(node_abs_delta > 0.01, na.rm = TRUE),
    n_node_gt_0.05 = sum(node_abs_delta > 0.05, na.rm = TRUE),
    n_node_gt_0.10 = sum(node_abs_delta > 0.10, na.rm = TRUE)
  ), by = .(input, duplicate_mode, floor_mode, normalize, q, correction)]
}

summarise_ppds <- function(dt) {
  dt[, .(
    n_rows = .N,
    median_abs_vs_ref_PPDS = median(candidate_vs_ref_PPDS_abs_delta, na.rm = TRUE),
    p95_abs_vs_ref_PPDS = as.numeric(quantile(candidate_vs_ref_PPDS_abs_delta, 0.95, na.rm = TRUE)),
    max_abs_vs_ref_PPDS = max(candidate_vs_ref_PPDS_abs_delta, na.rm = TRUE),
    median_abs_vs_ours_PPDS = median(candidate_vs_ours_PPDS_abs_delta, na.rm = TRUE),
    p95_abs_vs_ours_PPDS = as.numeric(quantile(candidate_vs_ours_PPDS_abs_delta, 0.95, na.rm = TRUE)),
    max_abs_vs_ours_PPDS = max(candidate_vs_ours_PPDS_abs_delta, na.rm = TRUE),
    max_ref_recalc_abs = max(ref_recalc_vs_ref_PPDS_abs_delta, na.rm = TRUE),
    n_abs_vs_ref_gt_0.01 = sum(candidate_vs_ref_PPDS_abs_delta > 0.01, na.rm = TRUE),
    n_abs_vs_ref_gt_0.05 = sum(candidate_vs_ref_PPDS_abs_delta > 0.05, na.rm = TRUE),
    n_abs_vs_ref_gt_0.10 = sum(candidate_vs_ref_PPDS_abs_delta > 0.10, na.rm = TRUE)
  ), by = .(input, duplicate_mode, floor_mode, normalize, q, correction)]
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(opts$out_dir, recursive = TRUE, showWarnings = FALSE)
files <- Sys.glob(opts$missing_glob)
if (!length(files)) stop("no missing audit files matched: ", opts$missing_glob, call. = FALSE)

nodes <- unique(node_table(files))
inputs <- list(derived = opts$derived_dir, v2 = opts$v2_dir)
duplicate_modes <- c("first", "mean_raw", "median_raw")
floor_modes <- c("floor_lt1", "none")
normalizes <- c(TRUE, FALSE)
qs <- list(0.75, 0)
corrections <- c(0.001, 0.0001)

detail_parts <- list()
ppds_parts <- list()
for (input_name in names(inputs)) {
  raw <- read_pr(inputs[[input_name]])
  for (dm in duplicate_modes) {
    for (fm in floor_modes) {
      for (nm in normalizes) {
        for (qv in qs) {
          for (corr in corrections) {
            scored <- score_one_spec(nodes, raw, input_name, dm, fm, nm, qv[[1]], corr)
            detail_parts[[length(detail_parts) + 1L]] <- scored$detail
            ppds_parts[[length(ppds_parts) + 1L]] <- scored$ppds
          }
        }
      }
    }
  }
}

detail <- rbindlist(detail_parts, use.names = TRUE, fill = TRUE)
ppds <- rbindlist(ppds_parts, use.names = TRUE, fill = TRUE)
node_summary <- summarise_nodes(detail)
ppds_summary <- summarise_ppds(ppds)
setorder(node_summary, median_node_abs, p95_node_abs, max_node_abs)
setorder(ppds_summary, median_abs_vs_ref_PPDS, p95_abs_vs_ref_PPDS, max_abs_vs_ref_PPDS)

tag <- if (length(files) == 1L) sub("_missing_nontransgene_audit.csv$", "", basename(files[[1]])) else "multi_contrast"
node_detail_path <- file.path(opts$out_dir, paste0(tag, "_ppds_provenance_node_detail.csv"))
row_detail_path <- file.path(opts$out_dir, paste0(tag, "_ppds_provenance_row_detail.csv"))
node_summary_path <- file.path(opts$out_dir, paste0(tag, "_ppds_provenance_node_summary.csv"))
row_summary_path <- file.path(opts$out_dir, paste0(tag, "_ppds_provenance_row_summary.csv"))

fwrite(detail, node_detail_path)
fwrite(ppds, row_detail_path)
fwrite(node_summary, node_summary_path)
fwrite(ppds_summary, row_summary_path)

cat("Top candidate protein preprocessing specs by PPDS agreement:\n")
print(head(ppds_summary, 16))
cat("Top candidate protein preprocessing specs by node pr_aFC agreement:\n")
print(head(node_summary, 16))
cat("Wrote:\n")
cat("  ", node_detail_path, "\n", sep = "")
cat("  ", row_detail_path, "\n", sep = "")
cat("  ", node_summary_path, "\n", sep = "")
cat("  ", row_summary_path, "\n", sep = "")
