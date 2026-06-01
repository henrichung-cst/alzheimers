#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(limma)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  missing_glob = file.path(repo, "outputs/reports/incytr_pair_mode/forensics/ma_*_missing_nontransgene_audit.csv"),
  derived_dir = file.path(repo, "data/derived/incytr_inputs"),
  source_dir = file.path(repo, "data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721"),
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
    } else if (a == "--source-dir") {
      x$source_dir <- need(); i <- i + 2L
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

read_omics <- function(input_dir, suffix) {
  path <- file.path(input_dir, paste0(suffix, "_yuyu_deconvoluted.csv"))
  if (!file.exists(path)) return(NULL)
  d <- fread(path)
  if (!"gene_symbol" %in% names(d) && "Gene Symbol" %in% names(d)) {
    setnames(d, "Gene Symbol", "gene_symbol")
  }
  d
}

node_table <- function(files) {
  roles <- c("Ligand", "Receptor", "EM", "Target")
  rbindlist(lapply(files, function(f) {
    d <- fread(f)
    contrast <- sub("_missing_nontransgene_audit.csv$", "", basename(f))
    rbindlist(lapply(roles, function(role) {
      cluster <- if (role == "Ligand") d$Sender else d$Receiver
      data.table(
        contrast = contrast,
        role = role,
        cluster = cluster,
        gene_symbol = d[[role]],
        ref_ps_aFC = d[[paste0("ref_", role, "_ps_aFC")]],
        ref_py_aFC = d[[paste0("ref_", role, "_py_aFC")]]
      )
    }), use.names = TRUE, fill = TRUE)
  }), use.names = TRUE, fill = TRUE)
}

prepare_gene_values <- function(raw, c1, c2, cluster, suffix, duplicate_mode, normalize) {
  col1 <- paste0(c1, "_", cluster)
  col2 <- paste0(c2, "_", cluster)
  if (!all(c(col1, col2, "gene_symbol") %in% names(raw))) {
    return(data.table(gene_symbol = character(), condition1 = numeric(), condition2 = numeric()))
  }
  x <- raw[, .(gene_symbol, condition1 = get(col1), condition2 = get(col2))]
  x <- x[!is.na(gene_symbol)]

  if (duplicate_mode == "first") {
    x <- x[!duplicated(gene_symbol)]
    if (normalize && nrow(x)) {
      x[, c("condition1", "condition2") := as.data.table(normalizeBetweenArrays(as.matrix(.SD))),
        .SDcols = c("condition1", "condition2")]
    }
    return(x)
  }

  if (duplicate_mode %in% c("mean_raw", "median_raw")) {
    fun <- if (duplicate_mode == "mean_raw") mean else median
    x <- x[, .(
      condition1 = fun(condition1, na.rm = TRUE),
      condition2 = fun(condition2, na.rm = TRUE)
    ), by = gene_symbol]
    if (normalize && nrow(x)) {
      x[, c("condition1", "condition2") := as.data.table(normalizeBetweenArrays(as.matrix(.SD))),
        .SDcols = c("condition1", "condition2")]
    }
    return(x)
  }

  if (duplicate_mode == "site_mean_afc" || duplicate_mode == "site_max_abs_afc") {
    if (normalize && nrow(x)) {
      x[, c("condition1", "condition2") := as.data.table(normalizeBetweenArrays(as.matrix(.SD))),
        .SDcols = c("condition1", "condition2")]
    }
    return(x)
  }

  stop("unknown duplicate_mode: ", duplicate_mode, call. = FALSE)
}

score_one_spec <- function(nodes, raw_by_suffix, input_name, duplicate_mode, normalize, q, correction) {
  out <- list()
  for (suffix in c("ps", "py")) {
    raw <- raw_by_suffix[[suffix]]
    if (is.null(raw)) next
    targets <- unique(nodes[, .(contrast, cluster)])
    vals <- rbindlist(lapply(seq_len(nrow(targets)), function(i) {
      contrast <- targets$contrast[i]
      age <- sub("^ma_([0-9]+mo)_.*$", "\\1", contrast)
      c2 <- paste0("ma_", age, "_WTyp")
      base <- prepare_gene_values(raw, contrast, c2, targets$cluster[i], suffix,
                                  duplicate_mode, normalize)
      if (!nrow(base)) return(NULL)
      fc <- cal_foldchange(base, correction = correction, q = q)
      if (duplicate_mode == "site_mean_afc") {
        fc <- fc[, .(aFC = mean(aFC, na.rm = TRUE)), by = gene_symbol]
      } else if (duplicate_mode == "site_max_abs_afc") {
        fc <- fc[, .(aFC = aFC[which.max(abs(aFC))]), by = gene_symbol]
      } else {
        fc <- fc[, .(gene_symbol, aFC)]
      }
      fc[, `:=`(contrast = contrast, cluster = targets$cluster[i])]
      fc
    }), use.names = TRUE, fill = TRUE)
    if (!nrow(vals)) next
    setnames(vals, "aFC", "candidate_aFC")
    n <- merge(
      nodes[, .(contrast, role, cluster, gene_symbol,
                ref_aFC = get(paste0("ref_", suffix, "_aFC")))],
      vals,
      by = c("contrast", "cluster", "gene_symbol"),
      all.x = TRUE,
      sort = FALSE,
      allow.cartesian = TRUE
    )
    n <- n[!is.na(ref_aFC)]
    n[, `:=`(
      input = input_name,
      suffix = suffix,
      duplicate_mode = duplicate_mode,
      normalize = normalize,
      q = if (is.null(q)) NA_real_ else q,
      correction = correction,
      abs_delta = abs(candidate_aFC - ref_aFC)
    )]
    out[[suffix]] <- n
  }
  rbindlist(out, use.names = TRUE, fill = TRUE)
}

summarise_spec <- function(dt) {
  dt[, .(
    n = .N,
    n_matched = sum(!is.na(candidate_aFC)),
    median_abs = median(abs_delta, na.rm = TRUE),
    p95_abs = as.numeric(quantile(abs_delta, 0.95, na.rm = TRUE)),
    max_abs = max(abs_delta, na.rm = TRUE),
    n_gt_0.01 = sum(abs_delta > 0.01, na.rm = TRUE),
    n_gt_0.05 = sum(abs_delta > 0.05, na.rm = TRUE),
    n_gt_0.10 = sum(abs_delta > 0.10, na.rm = TRUE)
  ), by = .(input, suffix, duplicate_mode, normalize, q, correction)]
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(opts$out_dir, recursive = TRUE, showWarnings = FALSE)
files <- Sys.glob(opts$missing_glob)
if (!length(files)) stop("no missing audit files matched: ", opts$missing_glob, call. = FALSE)

nodes <- unique(node_table(files))

inputs <- list(
  derived = opts$derived_dir,
  source = opts$source_dir
)
duplicate_modes <- c("first", "mean_raw", "median_raw", "site_mean_afc", "site_max_abs_afc")
normalizes <- c(TRUE, FALSE)
qs <- list(0.75, 0)
corrections <- c(0.001, 0.0001)

details <- list()
for (input_name in names(inputs)) {
  raw_by_suffix <- list(ps = read_omics(inputs[[input_name]], "ps"),
                        py = read_omics(inputs[[input_name]], "py"))
  for (dm in duplicate_modes) {
    for (nm in normalizes) {
      for (qv in qs) {
        for (corr in corrections) {
          details[[length(details) + 1L]] <- score_one_spec(
            nodes, raw_by_suffix, input_name, dm, nm, qv[[1]], corr
          )
        }
      }
    }
  }
}

detail <- rbindlist(details, use.names = TRUE, fill = TRUE)
summary <- summarise_spec(detail)
setorder(summary, suffix, median_abs, p95_abs)

detail_path <- file.path(opts$out_dir, "sce4_pds_provenance_node_detail.csv")
summary_path <- file.path(opts$out_dir, "sce4_pds_provenance_summary.csv")
fwrite(detail, detail_path)
fwrite(summary, summary_path)

cat("Top candidate preprocessing specs by suffix:\n")
print(summary[, head(.SD, 12), by = suffix])
cat("Wrote:\n")
cat("  ", detail_path, "\n", sep = "")
cat("  ", summary_path, "\n", sep = "")
