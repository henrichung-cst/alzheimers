#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(duckdb)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  wide_dir = file.path(repo, "outputs/reports/incytr_pair_mode/_sce4_full_q0"),
  ref_dir = file.path(repo, "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1"),
  allmarkers = file.path(repo, "data/derived/incytr_inputs/allmarkers.csv"),
  out_csv = file.path(repo, "outputs/reports/incytr_pair_mode/forensics/sce4_shared_score_residual_summary.csv")
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
    if (a == "--wide-dir") {
      x$wide_dir <- need(); i <- i + 2L
    } else if (a == "--ref-dir") {
      x$ref_dir <- need(); i <- i + 2L
    } else if (a == "--allmarkers") {
      x$allmarkers <- need(); i <- i + 2L
    } else if (a == "--out-csv") {
      x$out_csv <- need(); i <- i + 2L
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

make_key <- function(dt) {
  paste(dt$Sender, dt$Receiver, dt$Ligand, dt$Receptor, dt$EM, dt$Target, sep = "\001")
}

sql_str <- function(x) {
  paste0("'", gsub("'", "''", normalizePath(x, mustWork = FALSE), fixed = TRUE), "'")
}

qident <- function(x) {
  paste0('"', gsub('"', '""', x, fixed = TRUE), '"')
}

contrast_paths <- function(contrast, opts) {
  age <- sub("^ma_([0-9]+mo)_.*$", "\\1", contrast)
  c2 <- sprintf("ma_%s_WTyp", age)
  token <- sprintf("%s_WTyp", contrast)
  list(
    c1 = contrast,
    c2 = c2,
    c1_short = sub("^ma_[0-9]+mo_", "", contrast),
    c2_short = "WTyp",
    rds = file.path(opts$ref_dir, sprintf("DEG_PRG_%s_10302025", token),
                    "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds"),
    parquet = file.path(opts$wide_dir, sprintf("%s_%s_incytr_output.parquet", contrast, c2))
  )
}

read_ref <- function(paths, xwalk) {
  score_cols <- c("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score", "PDS")
  ref_cols <- c("Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target",
                paste0("SigProb_", paths$c1_short), paste0("SigProb_", paths$c2_short),
                score_cols)
  ref <- rbindlist(lapply(readRDS(paths$rds), function(e) {
    d <- as.data.table(e)
    d[, intersect(ref_cols, names(d)), with = FALSE]
  }), use.names = TRUE, fill = TRUE)
  ref[, Sender := map_clusters(Sender.group, xwalk, "sce4 Pairwise RDS")]
  ref[, Receiver := map_clusters(Receiver.group, xwalk, "sce4 Pairwise RDS")]
  setnames(ref, paste0("SigProb_", paths$c1_short), "ref_SigProb_c1")
  setnames(ref, paste0("SigProb_", paths$c2_short), "ref_SigProb_c2")
  for (cc in score_cols) setnames(ref, cc, paste0("ref_", cc))
  ref[, key := make_key(ref)]
  unique(ref[, c("key", paste0("ref_", c("SigProb_c1", "SigProb_c2", score_cols))), with = FALSE],
         by = "key")
}

audit_one <- function(contrast, opts, xwalk) {
  paths <- contrast_paths(contrast, opts)
  if (!file.exists(paths$rds)) stop("missing RDS: ", paths$rds, call. = FALSE)
  if (!file.exists(paths$parquet)) stop("missing parquet: ", paths$parquet, call. = FALSE)
  ref <- read_ref(paths, xwalk)
  score_cols <- c("SigProb_c1", "SigProb_c2", "TPDS", "PPDS", "PhPDS_ps",
                  "PhPDS_py", "multimodel_score", "PDS")

  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
  DBI::dbExecute(con, "PRAGMA memory_limit='8GB'")
  DBI::dbWriteTable(con, "ref", as.data.frame(ref), temporary = TRUE, overwrite = TRUE)

  sp1 <- qident(paste0("SigProb_", paths$c1))
  sp2 <- qident(paste0("SigProb_", paths$c2))
  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE ours AS
    SELECT
      concat(Sender, chr(1), Receiver, chr(1), Ligand, chr(1), Receptor, chr(1), EM, chr(1), Target) AS key,
      %s AS ours_SigProb_c1,
      %s AS ours_SigProb_c2,
      TPDS AS ours_TPDS,
      PPDS AS ours_PPDS,
      PhPDS_ps AS ours_PhPDS_ps,
      PhPDS_py AS ours_PhPDS_py,
      multimodel_score AS ours_multimodel_score,
      PDS AS ours_PDS
    FROM read_parquet(%s)
    WHERE ((%s > 0.1) OR (%s > 0.1))
      AND abs(PDS) >= 0.2
  ", sp1, sp2, sql_str(paths$parquet), sp1, sp2))

  rbindlist(lapply(score_cols, function(cc) {
    q <- sprintf("
      SELECT
        count(*) AS shared,
        sum(CASE WHEN abs(ours_%s - ref_%s) > 1e-8 THEN 1 ELSE 0 END) AS n_gt_1e8,
        sum(CASE WHEN abs(ours_%s - ref_%s) > 1e-4 THEN 1 ELSE 0 END) AS n_gt_1e4,
        median(abs(ours_%s - ref_%s)) AS median_abs,
        quantile_cont(abs(ours_%s - ref_%s), 0.95) AS p95_abs,
        quantile_cont(abs(ours_%s - ref_%s), 0.99) AS p99_abs,
        max(abs(ours_%s - ref_%s)) AS max_abs
      FROM ref INNER JOIN ours USING(key)
    ", cc, cc, cc, cc, cc, cc, cc, cc, cc, cc, cc, cc)
    res <- as.data.table(DBI::dbGetQuery(con, q))
    res[, `:=`(contrast = contrast, component = cc)]
    res
  }), use.names = TRUE, fill = TRUE)
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(dirname(opts$out_csv), recursive = TRUE, showWarnings = FALSE)
xwalk <- build_crosswalk(opts$allmarkers)
contrasts <- c(
  "ma_2mo_AppP", "ma_2mo_ApTt", "ma_2mo_Ttau",
  "ma_4mo_AppP", "ma_4mo_ApTt", "ma_4mo_Ttau",
  "ma_6mo_AppP", "ma_6mo_ApTt", "ma_6mo_Ttau"
)

summary <- rbindlist(lapply(contrasts, audit_one, opts = opts, xwalk = xwalk),
                     use.names = TRUE, fill = TRUE)
setcolorder(summary, c("contrast", "component", "shared", "n_gt_1e8", "n_gt_1e4",
                       "median_abs", "p95_abs", "p99_abs", "max_abs"))
fwrite(summary, opts$out_csv)

print(summary[component %in% c("PDS", "TPDS", "PPDS", "PhPDS_ps", "PhPDS_py"),
              .(shared_total = sum(shared),
                n_gt_1e4 = sum(n_gt_1e4),
                median_abs_max = max(median_abs, na.rm = TRUE),
                p95_abs_max = max(p95_abs, na.rm = TRUE),
                max_abs = max(max_abs, na.rm = TRUE)),
              by = component][order(component)])
cat("Wrote:\n  ", opts$out_csv, "\n", sep = "")
