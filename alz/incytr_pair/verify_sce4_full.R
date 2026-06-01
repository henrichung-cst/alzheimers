#!/usr/bin/env Rscript
# Full sce4 reproduction gate.
#
# This verifies the reproducible invariant for the AD pair-mode run:
# our gated path tuples must match sce4's pre-cap Pairwise_pathway_table RDS
# pair-for-pair. Top300 membership is reported only as an informational PDS-rank
# check because documented phospho/App value residuals can move the cap boundary.
#
# The input parquets must be the driver's unfiltered outputs. This script applies
# sce4's gate itself:
#   (SigProb_condition1 > 0.1 OR SigProb_condition2 > 0.1) AND abs(PDS) >= 0.2
#
# Usage:
#   pixi run Rscript alz/incytr_pair/verify_sce4_full.R
#   pixi run Rscript alz/incytr_pair/verify_sce4_full.R --available-only
#   pixi run Rscript alz/incytr_pair/verify_sce4_full.R --contrast ma_2mo_AppP

suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(duckdb)
})

repo_root <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  wide_dir = file.path(repo_root, "outputs/reports/incytr_pair_mode/wide"),
  ref_dir = file.path(repo_root, "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1"),
  allmarkers = file.path(repo_root, "data/derived/incytr_inputs/allmarkers.csv"),
  sigprob_cutoff = 0.1,
  pds_gate = 0.2,
  tol = 1e-4,
  available_only = FALSE,
  contrasts = character(0),
  report_csv = ""
)

usage <- function() {
  cat("Usage: verify_sce4_full.R [--wide-dir DIR] [--ref-dir DIR] [--allmarkers CSV]\n")
  cat("                         [--available-only] [--contrast ma_2mo_AppP[,ma_4mo_AppP]]\n")
  cat("                         [--report-csv CSV]\n")
}

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
    } else if (a == "--sigprob-cutoff") {
      x$sigprob_cutoff <- as.numeric(need()); i <- i + 2L
    } else if (a == "--pds-gate") {
      x$pds_gate <- as.numeric(need()); i <- i + 2L
    } else if (a == "--tol") {
      x$tol <- as.numeric(need()); i <- i + 2L
    } else if (a == "--available-only") {
      x$available_only <- TRUE; i <- i + 1L
    } else if (a == "--contrast") {
      vals <- strsplit(need(), ",", fixed = TRUE)[[1]]
      x$contrasts <- unique(c(x$contrasts, trimws(vals[nzchar(vals)])))
      i <- i + 2L
    } else if (a == "--report-csv") {
      x$report_csv <- need(); i <- i + 2L
    } else if (a %in% c("-h", "--help")) {
      usage(); quit(status = 0L)
    } else {
      stop("unknown argument: ", a, call. = FALSE)
    }
  }
  stopifnot(!is.na(x$sigprob_cutoff), !is.na(x$pds_gate), !is.na(x$tol))
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

contrast_info <- function(dir_path) {
  token <- sub("^DEG_PRG_(ma_.*)_10302025$", "\\1", basename(dir_path))
  c1 <- sub("_WTyp$", "", token)
  age <- sub("^ma_([0-9]+mo)_.*$", "\\1", c1)
  c2 <- sprintf("ma_%s_WTyp", age)
  list(token = token, c1 = c1, c2 = c2)
}

path_key <- function(dt) {
  paste(dt$Ligand, dt$Receptor, dt$EM, dt$Target, sep = "\001")
}

pair_key <- function(dt) {
  paste(dt$Sender, dt$Receiver, sep = "\001")
}

is_transgene_path <- function(dt) {
  transgenes <- c("App", "Psen1", "Mapt")
  dt$Ligand %in% transgenes | dt$Receptor %in% transgenes |
    dt$EM %in% transgenes | dt$Target %in% transgenes
}

read_ref_rds <- function(rds_path, xwalk) {
  cols <- c("Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target",
            "Ligand_sclog2FC", "Receptor_sclog2FC", "EM_sclog2FC", "Target_sclog2FC")
  x <- readRDS(rds_path)
  ref <- rbindlist(lapply(x, function(e) {
    d <- as.data.table(e)
    d[, ..cols]
  }), use.names = TRUE, fill = TRUE)
  where <- basename(dirname(rds_path))
  ref[, Sender := map_clusters(Sender.group, xwalk, where)]
  ref[, Receiver := map_clusters(Receiver.group, xwalk, where)]
  ref[, k := path_key(ref)]
  unique(ref, by = c("Sender", "Receiver", "k"))
}

sql_str <- function(x) {
  paste0("'", gsub("'", "''", normalizePath(x, mustWork = FALSE), fixed = TRUE), "'")
}

sql_lit <- function(x) {
  paste0("'", gsub("'", "''", x, fixed = TRUE), "'")
}

qident <- function(x) {
  paste0('"', gsub('"', '""', x, fixed = TRUE), '"')
}

run_duckdb <- function(ref, parquet, c1, c2, top300_path, xwalk) {
  if (!file.exists(parquet)) {
    stop("missing parquet: ", parquet, call. = FALSE)
  }
  con <- DBI::dbConnect(duckdb::duckdb())
  on.exit(DBI::dbDisconnect(con, shutdown = TRUE), add = TRUE)
  DBI::dbExecute(con, "PRAGMA memory_limit='8GB'")
  temp_dir <- Sys.getenv("DUCKDB_TEMP_DIR", unset = file.path(Sys.getenv("HOME"), ".cache/duckdb"))
  dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
  DBI::dbExecute(con, sprintf("PRAGMA temp_directory=%s", sql_lit(temp_dir)))

  DBI::dbWriteTable(con, "ref", as.data.frame(ref), temporary = TRUE, overwrite = TRUE)
  sp1 <- qident(paste0("SigProb_", c1))
  sp2 <- qident(paste0("SigProb_", c2))
  path_expr <- "concat(Ligand, chr(1), Receptor, chr(1), EM, chr(1), Target)"
  parquet_sql <- sql_str(parquet)

  DBI::dbExecute(con, sprintf("
    CREATE TEMP TABLE ours AS
    SELECT DISTINCT
      Sender, Receiver, Ligand, Receptor, EM, Target,
      Ligand_sclog2FC, Receptor_sclog2FC, EM_sclog2FC, Target_sclog2FC,
      PDS, %s AS k
    FROM read_parquet(%s)
    WHERE ((%s > %f) OR (%s > %f))
      AND abs(PDS) >= %f
  ", path_expr, parquet_sql, sp1, opts$sigprob_cutoff, sp2, opts$sigprob_cutoff, opts$pds_gate))

  join_on <- paste(
    "r.Sender = o.Sender", "r.Receiver = o.Receiver",
    "r.Ligand = o.Ligand", "r.Receptor = o.Receptor",
    "r.EM = o.EM", "r.Target = o.Target",
    sep = " AND "
  )
  tg_r <- "r.Ligand IN ('App','Psen1','Mapt') OR r.Receptor IN ('App','Psen1','Mapt') OR r.EM IN ('App','Psen1','Mapt') OR r.Target IN ('App','Psen1','Mapt')"
  tg_o <- "o.Ligand IN ('App','Psen1','Mapt') OR o.Receptor IN ('App','Psen1','Mapt') OR o.EM IN ('App','Psen1','Mapt') OR o.Target IN ('App','Psen1','Mapt')"

  counts <- DBI::dbGetQuery(con, sprintf("
    WITH
      ref_n AS (SELECT count(*) AS n FROM ref),
      ours_n AS (SELECT count(*) AS n FROM ours),
      shared_n AS (
        SELECT count(*) AS n FROM ref r INNER JOIN ours o ON %s
      ),
      missing AS (
        SELECT r.* FROM ref r LEFT JOIN ours o ON %s WHERE o.Sender IS NULL
      ),
      extra AS (
        SELECT o.* FROM ours o LEFT JOIN ref r ON %s WHERE r.Sender IS NULL
      )
    SELECT
      (SELECT n FROM ours_n) AS ours_gated,
      (SELECT n FROM ref_n) AS sce4_gated,
      (SELECT n FROM shared_n) AS shared,
      (SELECT count(*) FROM missing) AS missing,
      (SELECT count(*) FROM missing r WHERE NOT (%s)) AS missing_non_transgene,
      (SELECT count(*) FROM extra) AS extra,
      (SELECT count(*) FROM extra o WHERE NOT (%s)) AS extra_non_transgene
  ", join_on, join_on, join_on, tg_r, tg_o))

  val <- DBI::dbGetQuery(con, sprintf("
    SELECT
      max(abs(o.Ligand_sclog2FC - r.Ligand_sclog2FC)) AS Ligand_max_delta,
      sum(CASE WHEN abs(o.Ligand_sclog2FC - r.Ligand_sclog2FC) > %f THEN 1 ELSE 0 END) AS Ligand_n_over,
      sum(CASE WHEN abs(o.Ligand_sclog2FC - r.Ligand_sclog2FC) > %f
                AND o.Ligand NOT IN ('App','Psen1','Mapt') THEN 1 ELSE 0 END) AS Ligand_non_tg_over,
      max(abs(o.Receptor_sclog2FC - r.Receptor_sclog2FC)) AS Receptor_max_delta,
      sum(CASE WHEN abs(o.Receptor_sclog2FC - r.Receptor_sclog2FC) > %f THEN 1 ELSE 0 END) AS Receptor_n_over,
      max(abs(o.EM_sclog2FC - r.EM_sclog2FC)) AS EM_max_delta,
      sum(CASE WHEN abs(o.EM_sclog2FC - r.EM_sclog2FC) > %f THEN 1 ELSE 0 END) AS EM_n_over,
      sum(CASE WHEN abs(o.EM_sclog2FC - r.EM_sclog2FC) > %f
                AND o.EM NOT IN ('App','Psen1','Mapt') THEN 1 ELSE 0 END) AS EM_non_tg_over,
      max(abs(o.Target_sclog2FC - r.Target_sclog2FC)) AS Target_max_delta,
      sum(CASE WHEN abs(o.Target_sclog2FC - r.Target_sclog2FC) > %f THEN 1 ELSE 0 END) AS Target_n_over
    FROM ref r INNER JOIN ours o ON %s
  ", opts$tol, opts$tol, opts$tol, opts$tol, opts$tol, opts$tol, join_on))

  cap <- duckdb_top300_info(con, top300_path, xwalk)
  list(counts = as.data.table(counts), values = as.data.table(val), top300 = cap)
}

duckdb_top300_info <- function(con, top300_path, xwalk) {
  if (!file.exists(top300_path)) return(list(shared = NA_integer_, total = NA_integer_))
  ref <- fread(top300_path)
  needed <- c("Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target")
  if (!all(needed %in% names(ref))) return(list(shared = NA_integer_, total = NA_integer_))
  ref[, Sender := map_clusters(Sender.group, xwalk, basename(dirname(top300_path)))]
  ref[, Receiver := map_clusters(Receiver.group, xwalk, basename(dirname(top300_path)))]
  ref[, k := path_key(ref)]
  ref_keys <- unique(ref[, .(Sender, Receiver, k)])
  DBI::dbWriteTable(con, "top300_ref", as.data.frame(ref_keys), temporary = TRUE, overwrite = TRUE)
  shared <- DBI::dbGetQuery(con, "
    WITH cap AS (
      SELECT Sender, Receiver, k FROM (
        SELECT Sender, Receiver, k,
               row_number() OVER (PARTITION BY Sender, Receiver ORDER BY PDS DESC) AS rn
        FROM ours WHERE PDS > 0
      ) WHERE rn <= 300
      UNION
      SELECT Sender, Receiver, k FROM (
        SELECT Sender, Receiver, k,
               row_number() OVER (PARTITION BY Sender, Receiver ORDER BY PDS ASC) AS rn
        FROM ours WHERE PDS < 0
      ) WHERE rn <= 300
    )
    SELECT count(*) AS n
    FROM top300_ref t
    INNER JOIN cap c
      ON t.Sender = c.Sender AND t.Receiver = c.Receiver AND t.k = c.k
  ")$n
  list(shared = shared, total = nrow(ref_keys))
}

check_one <- function(dir_path, xwalk) {
  ci <- contrast_info(dir_path)
  rds <- file.path(dir_path, "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds")
  parquet <- file.path(opts$wide_dir, sprintf("%s_%s_incytr_output.parquet", ci$c1, ci$c2))
  top300 <- file.path(dir_path, "sce4_DEG_PRG_Top300_table_10302025.csv")

  ref <- read_ref_rds(rds, xwalk)
  res <- run_duckdb(ref, parquet, ci$c1, ci$c2, top300, xwalk)
  counts <- res$counts[1]
  val <- res$values[1]
  cap <- res$top300

  value_ok <- val$Receptor_n_over == 0L && val$Target_n_over == 0L &&
    val$Ligand_non_tg_over == 0L && val$EM_non_tg_over == 0L
  ok <- counts$missing_non_transgene == 0L &&
    counts$extra_non_transgene == 0L && isTRUE(value_ok)
  cat(sprintf("[%s] ours_gated=%d sce4=%d shared=%d missing=%d(non-tg %d) extra=%d(non-tg %d)\n",
              ci$c1, counts$ours_gated, counts$sce4_gated, counts$shared,
              counts$missing, counts$missing_non_transgene,
              counts$extra, counts$extra_non_transgene))
  cat(sprintf("    %-8s max|delta|=%.4f n>%g=%d\n",
              "Ligand", val$Ligand_max_delta, opts$tol, val$Ligand_n_over))
  cat(sprintf("    %-8s max|delta|=%.4f n>%g=%d\n",
              "Receptor", val$Receptor_max_delta, opts$tol, val$Receptor_n_over))
  cat(sprintf("    %-8s max|delta|=%.4f n>%g=%d\n",
              "EM", val$EM_max_delta, opts$tol, val$EM_n_over))
  cat(sprintf("    %-8s max|delta|=%.4f n>%g=%d\n",
              "Target", val$Target_max_delta, opts$tol, val$Target_n_over))
  if (!is.na(cap$total) && cap$total > 0L) {
    cat(sprintf("    Top300 info: %d/%d (%.1f%%)\n",
                cap$shared, cap$total, 100 * cap$shared / cap$total))
  }
  cat(sprintf("    STATUS: %s\n", if (ok) "PASS" else "FAIL"))

  data.table(
    contrast = ci$c1,
    ours_gated = counts$ours_gated,
    sce4_gated = counts$sce4_gated,
    shared = counts$shared,
    missing = counts$missing,
    missing_non_transgene = counts$missing_non_transgene,
    extra = counts$extra,
    extra_non_transgene = counts$extra_non_transgene,
    top300_shared = cap$shared,
    top300_total = cap$total,
    pass = ok
  )
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
expected <- c(
  "ma_2mo_AppP_WTyp", "ma_2mo_ApTt_WTyp", "ma_2mo_Ttau_WTyp",
  "ma_4mo_AppP_WTyp", "ma_4mo_ApTt_WTyp", "ma_4mo_Ttau_WTyp",
  "ma_6mo_AppP_WTyp", "ma_6mo_ApTt_WTyp", "ma_6mo_Ttau_WTyp"
)
dirs <- file.path(opts$ref_dir, sprintf("DEG_PRG_%s_10302025", expected))
names(dirs) <- sub("_WTyp$", "", expected)

if (length(opts$contrasts)) {
  opts$contrasts <- sub("_WTyp$", "", opts$contrasts)
  unknown <- setdiff(opts$contrasts, names(dirs))
  if (length(unknown)) stop("unknown contrast(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  dirs <- dirs[opts$contrasts]
}

missing_rds <- names(dirs)[!file.exists(file.path(dirs, "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds"))]
if (length(missing_rds)) {
  cat("Missing sce4 pre-cap RDS files:\n")
  for (c in missing_rds) {
    cat(sprintf("  %s -> %s\n", c,
                file.path(dirs[[c]], "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds")))
  }
  if (!opts$available_only) {
    cat("verify_sce4_full: FAIL (full 9-contrast reproduction is blocked by missing RDS files)\n")
    quit(status = 1L)
  }
  dirs <- dirs[setdiff(names(dirs), missing_rds)]
}

if (!length(dirs)) {
  stop("no contrasts left to verify", call. = FALSE)
}

xwalk <- build_crosswalk(opts$allmarkers)
reports <- rbindlist(lapply(dirs, function(d) {
  tryCatch(
    check_one(d, xwalk),
    error = function(e) {
      ci <- contrast_info(d)
      cat(sprintf("[%s] FAIL: %s\n", ci$c1, conditionMessage(e)))
      data.table(
        contrast = ci$c1, ours_gated = NA_integer_, sce4_gated = NA_integer_,
        shared = NA_integer_, missing = NA_integer_, missing_non_transgene = NA_integer_,
        extra = NA_integer_, extra_non_transgene = NA_integer_,
        top300_shared = NA_integer_, top300_total = NA_integer_, pass = FALSE
      )
    }
  )
}), fill = TRUE)

if (nzchar(opts$report_csv)) {
  dir.create(dirname(opts$report_csv), recursive = TRUE, showWarnings = FALSE)
  fwrite(reports, opts$report_csv)
}

cat("\nSummary:\n")
print(reports)

if (all(reports$pass)) {
  cat("verify_sce4_full: PASS\n")
  quit(status = 0L)
}
cat("verify_sce4_full: FAIL\n")
quit(status = 1L)
