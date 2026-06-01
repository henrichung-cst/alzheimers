#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  contrast = "ma_4mo_Ttau",
  wide_dir = file.path(repo, "outputs/reports/incytr_pair_mode/_sce4_full_q0"),
  ref_dir = file.path(repo, "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1"),
  allmarkers = file.path(repo, "data/derived/incytr_inputs/allmarkers.csv"),
  out_dir = file.path(repo, "outputs/reports/incytr_pair_mode/forensics"),
  sigprob_cutoff = 0.1,
  pds_gate = 0.2
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
    if (a == "--contrast") {
      x$contrast <- sub("_WTyp$", "", need()); i <- i + 2L
    } else if (a == "--wide-dir") {
      x$wide_dir <- need(); i <- i + 2L
    } else if (a == "--ref-dir") {
      x$ref_dir <- need(); i <- i + 2L
    } else if (a == "--allmarkers") {
      x$allmarkers <- need(); i <- i + 2L
    } else if (a == "--out-dir") {
      x$out_dir <- need(); i <- i + 2L
    } else if (a == "--sigprob-cutoff") {
      x$sigprob_cutoff <- as.numeric(need()); i <- i + 2L
    } else if (a == "--pds-gate") {
      x$pds_gate <- as.numeric(need()); i <- i + 2L
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
  }
  x
}

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

contrast_paths <- function(opts) {
  age <- sub("^ma_([0-9]+mo)_.*$", "\\1", opts$contrast)
  c2 <- sprintf("ma_%s_WTyp", age)
  token <- sprintf("%s_WTyp", opts$contrast)
  list(
    c1 = opts$contrast,
    c2 = c2,
    c1_short = sub("^ma_[0-9]+mo_", "", opts$contrast),
    c2_short = "WTyp",
    parquet = file.path(opts$wide_dir, sprintf("%s_%s_incytr_output.parquet", opts$contrast, c2)),
    rds = file.path(opts$ref_dir, sprintf("DEG_PRG_%s_10302025", token),
                    "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds")
  )
}

make_key <- function(dt) {
  paste(dt$Sender, dt$Receiver, dt$Ligand, dt$Receptor, dt$EM, dt$Target, sep = "\001")
}

is_transgene_path <- function(dt) {
  tg <- c("App", "Psen1", "Mapt")
  dt$Ligand %in% tg | dt$Receptor %in% tg | dt$EM %in% tg | dt$Target %in% tg
}

read_ref <- function(rds, xwalk, paths) {
  score_cols <- c(
    "Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target",
    paste0("SigProb_", paths$c1_short), paste0("SigProb_", paths$c2_short),
    "TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score", "PDS",
    "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC",
    "Ligand_pr_aFC", "Receptor_pr_aFC", "EM_pr_aFC", "Target_pr_aFC",
    "Ligand_ps_aFC", "Receptor_ps_aFC", "EM_ps_aFC", "Target_ps_aFC",
    "Ligand_py_aFC", "Receptor_py_aFC", "EM_py_aFC", "Target_py_aFC"
  )
  ref <- rbindlist(lapply(readRDS(rds), function(e) {
    d <- as.data.table(e)
    keep <- intersect(score_cols, names(d))
    d[, ..keep]
  }), use.names = TRUE, fill = TRUE)
  ref[, Sender := map_clusters(Sender.group, xwalk, "sce4 Pairwise RDS")]
  ref[, Receiver := map_clusters(Receiver.group, xwalk, "sce4 Pairwise RDS")]
  setnames(ref, paste0("SigProb_", paths$c1_short), "ref_SigProb_c1")
  setnames(ref, paste0("SigProb_", paths$c2_short), "ref_SigProb_c2")
  for (cc in setdiff(names(ref), c("Sender", "Receiver", "Sender.group", "Receiver.group",
                                   "Ligand", "Receptor", "EM", "Target",
                                   "ref_SigProb_c1", "ref_SigProb_c2"))) {
    setnames(ref, cc, paste0("ref_", cc))
  }
  ref[, key := make_key(ref)]
  unique(ref, by = "key")
}

read_ours <- function(parquet, paths, opts) {
  ours <- as.data.table(read_parquet(parquet))
  setnames(ours, paste0("SigProb_", paths$c1), "ours_SigProb_c1")
  setnames(ours, paste0("SigProb_", paths$c2), "ours_SigProb_c2")
  keep <- c(
    "Sender", "Receiver", "Ligand", "Receptor", "EM", "Target",
    "ours_SigProb_c1", "ours_SigProb_c2",
    "TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score", "PDS",
    "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC",
    "Ligand_pr_log2FC", "Receptor_pr_log2FC", "EM_pr_log2FC", "Target_pr_log2FC",
    "Ligand_ps_log2FC", "Receptor_ps_log2FC", "EM_ps_log2FC", "Target_ps_log2FC",
    "Ligand_py_log2FC", "Receptor_py_log2FC", "EM_py_log2FC", "Target_py_log2FC"
  )
  ours <- ours[, ..keep]
  for (cc in setdiff(names(ours), c("Sender", "Receiver", "Ligand", "Receptor", "EM", "Target",
                                    "ours_SigProb_c1", "ours_SigProb_c2"))) {
    setnames(ours, cc, paste0("ours_", cc))
  }
  ours[, key := make_key(ours)]
  ours[, ours_sigprob_pass := ours_SigProb_c1 > opts$sigprob_cutoff | ours_SigProb_c2 > opts$sigprob_cutoff]
  ours[, ours_pds_pass := abs(ours_PDS) >= opts$pds_gate]
  ours[, ours_gated := ours_sigprob_pass & ours_pds_pass]
  unique(ours, by = "key")
}

add_missing_reason <- function(dt) {
  dt <- copy(dt)
  score_cols <- c("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score", "PDS")
  for (cc in score_cols) {
    r <- paste0("ref_", cc)
    o <- paste0("ours_", cc)
    if (all(c(r, o) %in% names(dt))) dt[[paste0("delta_", cc)]] <- dt[[o]] - dt[[r]]
  }
  delta_cols <- paste0("delta_", score_cols)
  present_delta_cols <- intersect(delta_cols, names(dt))
  dt[, largest_abs_delta_component := NA_character_]
  if (length(present_delta_cols)) {
    mat <- as.matrix(abs(dt[, ..present_delta_cols]))
    max_i <- max.col(replace(mat, is.na(mat), -Inf), ties.method = "first")
    dt[is.finite(mat[cbind(seq_len(.N), max_i)]),
       largest_abs_delta_component := sub("^delta_", "", present_delta_cols[max_i])]
  }
  dt[, reason := fifelse(!raw_present, "absent_from_ours_raw",
                  fifelse(!ours_sigprob_pass, "ours_sigprob_below_gate",
                  fifelse(!ours_pds_pass & largest_abs_delta_component %in% c("PhPDS_ps", "PhPDS_py"),
                          "phospho_threshold_residual",
                  fifelse(!ours_pds_pass, "nonphospho_pds_threshold_residual",
                          "unresolved_present_in_raw"))))]
  dt
}

summarise_audit <- function(ref, ours, missing_nt, extra_nt, opts, paths) {
  extra_path <- paste(extra_nt$Ligand, extra_nt$Receptor, extra_nt$EM, extra_nt$Target, sep = "\001")
  ref_path <- unique(paste(ref$Ligand, ref$Receptor, ref$EM, ref$Target, sep = "\001"))
  data.table(
    contrast = opts$contrast,
    sce4_pairwise_rows = nrow(ref),
    ours_raw_rows = nrow(ours),
    ours_gated_rows = sum(ours$ours_gated),
    missing_non_transgene = nrow(missing_nt),
    missing_raw_present = sum(missing_nt$raw_present),
    missing_raw_absent = sum(!missing_nt$raw_present),
    missing_ours_sigprob_below_gate = sum(missing_nt$reason == "ours_sigprob_below_gate"),
    missing_phospho_threshold_residual = sum(missing_nt$reason == "phospho_threshold_residual"),
    missing_nonphospho_pds_threshold_residual = sum(missing_nt$reason == "nonphospho_pds_threshold_residual"),
    missing_unresolved_present_in_raw = sum(missing_nt$reason == "unresolved_present_in_raw"),
    extra_non_transgene = nrow(extra_nt),
    extra_absent_from_sce4_pairwise_reference = nrow(extra_nt),
    extra_path_seen_elsewhere_in_sce4_pairwise = sum(extra_path %in% ref_path),
    ref_rds = paths$rds,
    ours_parquet = paths$parquet
  )
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
paths <- contrast_paths(opts)
if (!file.exists(paths$rds)) stop("missing sce4 RDS: ", paths$rds, call. = FALSE)
if (!file.exists(paths$parquet)) stop("missing rerun parquet: ", paths$parquet, call. = FALSE)
dir.create(opts$out_dir, recursive = TRUE, showWarnings = FALSE)

xwalk <- build_crosswalk(opts$allmarkers)
ref <- read_ref(paths$rds, xwalk, paths)
ours <- read_ours(paths$parquet, paths, opts)

ref[, transgene_path := is_transgene_path(ref)]
ours[, transgene_path := is_transgene_path(ours)]

missing_keys <- setdiff(ref$key, ours[ours_gated == TRUE, key])
extra_keys <- setdiff(ours[ours_gated == TRUE, key], ref$key)

missing <- ref[key %in% missing_keys & transgene_path == FALSE]
missing <- merge(
  missing,
  ours[, .(key, raw_present = TRUE, ours_SigProb_c1, ours_SigProb_c2,
           ours_sigprob_pass, ours_pds_pass, ours_gated,
           ours_TPDS, ours_PPDS, ours_PhPDS_ps, ours_PhPDS_py,
           ours_multimodel_score, ours_PDS)],
  by = "key", all.x = TRUE, sort = FALSE
)
missing[is.na(raw_present), raw_present := FALSE]
missing <- add_missing_reason(missing)

extra <- ours[key %in% extra_keys & transgene_path == FALSE]
extra[, reason := "absent_from_sce4_pairwise_reference"]

summary <- summarise_audit(ref, ours, missing, extra, opts, paths)

prefix <- file.path(opts$out_dir, opts$contrast)
fwrite(missing, paste0(prefix, "_missing_nontransgene_audit.csv"))
fwrite(extra, paste0(prefix, "_extra_nontransgene_audit.csv"))
fwrite(summary, paste0(prefix, "_mismatch_audit_summary.csv"))

print(summary)
cat("Wrote:\n")
cat("  ", paste0(prefix, "_missing_nontransgene_audit.csv"), "\n", sep = "")
cat("  ", paste0(prefix, "_extra_nontransgene_audit.csv"), "\n", sep = "")
cat("  ", paste0(prefix, "_mismatch_audit_summary.csv"), "\n", sep = "")
