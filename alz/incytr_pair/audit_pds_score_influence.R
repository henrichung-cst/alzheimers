#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(DBI)
  library(duckdb)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  ref_dir = file.path(repo, "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1"),
  allmarkers = file.path(repo, "data/derived/incytr_inputs/allmarkers.csv"),
  canonical_dir = file.path(repo, "outputs/reports/incytr_pair_mode/_sce4_full_q0"),
  source_ps_dir = file.path(repo, "outputs/reports/incytr_pair_mode/_sce4_full_source_ps_diag"),
  out_csv = file.path(repo, "outputs/reports/incytr_pair_mode/forensics/sce4_pds_score_influence_summary.csv")
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
    if (a == "--ref-dir") {
      x$ref_dir <- need(); i <- i + 2L
    } else if (a == "--allmarkers") {
      x$allmarkers <- need(); i <- i + 2L
    } else if (a == "--canonical-dir") {
      x$canonical_dir <- need(); i <- i + 2L
    } else if (a == "--source-ps-dir") {
      x$source_ps_dir <- need(); i <- i + 2L
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

is_transgene_path <- function(dt) {
  transgenes <- c("App", "Psen1", "Mapt")
  dt$Ligand %in% transgenes | dt$Receptor %in% transgenes |
    dt$EM %in% transgenes | dt$Target %in% transgenes
}

sql_str <- function(x) {
  paste0("'", gsub("'", "''", normalizePath(x, mustWork = FALSE), fixed = TRUE), "'")
}

qident <- function(x) {
  paste0('"', gsub('"', '""', x, fixed = TRUE), '"')
}

contrast_paths <- function(contrast, wide_dir, ref_dir) {
  age <- sub("^ma_([0-9]+mo)_.*$", "\\1", contrast)
  c2 <- sprintf("ma_%s_WTyp", age)
  token <- sprintf("%s_WTyp", contrast)
  list(
    c1 = contrast,
    c2 = c2,
    rds = file.path(ref_dir, sprintf("DEG_PRG_%s_10302025", token),
                    "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds"),
    parquet = file.path(wide_dir, sprintf("%s_%s_incytr_output.parquet", contrast, c2))
  )
}

read_ref <- function(paths, xwalk) {
  cols <- c("Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target",
            "TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score", "PDS")
  ref <- rbindlist(lapply(readRDS(paths$rds), function(e) {
    d <- as.data.table(e)
    d[, intersect(cols, names(d)), with = FALSE]
  }), use.names = TRUE, fill = TRUE)
  ref[, Sender := map_clusters(Sender.group, xwalk, basename(dirname(paths$rds)))]
  ref[, Receiver := map_clusters(Receiver.group, xwalk, basename(dirname(paths$rds)))]
  ref[, key := make_key(ref)]
  ref[, transgene_path := is_transgene_path(ref)]
  setnames(ref, c("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score", "PDS"),
           paste0("ref_", c("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "multimodel_score", "PDS")))
  unique(ref[, .(key, transgene_path, ref_TPDS, ref_PPDS, ref_PhPDS_ps, ref_PhPDS_py,
                 ref_multimodel_score, ref_PDS)], by = "key")
}

audit_one <- function(run_name, wide_dir, contrast, opts, xwalk) {
  paths <- contrast_paths(contrast, wide_dir, opts$ref_dir)
  if (!file.exists(paths$rds)) stop("missing RDS: ", paths$rds, call. = FALSE)
  if (!file.exists(paths$parquet)) stop("missing parquet: ", paths$parquet, call. = FALSE)

  ref <- read_ref(paths, xwalk)
  con <- dbConnect(duckdb::duckdb())
  on.exit(dbDisconnect(con, shutdown = TRUE), add = TRUE)
  dbExecute(con, "PRAGMA memory_limit='10GB'")
  dbWriteTable(con, "ref", as.data.frame(ref), temporary = TRUE, overwrite = TRUE)

  sp1 <- qident(paste0("SigProb_", paths$c1))
  sp2 <- qident(paste0("SigProb_", paths$c2))
  dbExecute(con, sprintf("
    CREATE TEMP TABLE joined AS
    WITH ours AS (
      SELECT
        concat(Sender, chr(1), Receiver, chr(1), Ligand, chr(1), Receptor, chr(1), EM, chr(1), Target) AS key,
        TPDS AS ours_TPDS,
        PPDS AS ours_PPDS,
        PhPDS_ps AS ours_PhPDS_ps,
        PhPDS_py AS ours_PhPDS_py,
        multimodel_score AS ours_multimodel_score,
        PDS AS ours_PDS
      FROM read_parquet(%s)
      WHERE ((%s > 0.1) OR (%s > 0.1))
        AND abs(PDS) >= 0.2
    ),
    d AS (
      SELECT
        ref.transgene_path,
        coalesce(ours_TPDS, 0) - coalesce(ref_TPDS, 0) AS d_TPDS,
        coalesce(ours_PPDS, 0) - coalesce(ref_PPDS, 0) AS d_PPDS,
        coalesce(ours_PhPDS_ps, 0) - coalesce(ref_PhPDS_ps, 0) AS d_PhPDS_ps,
        coalesce(ours_PhPDS_py, 0) - coalesce(ref_PhPDS_py, 0) AS d_PhPDS_py,
        coalesce(ours_multimodel_score, 0) - coalesce(ref_multimodel_score, 0) AS d_multimodel,
        coalesce(ours_PDS, 0) - coalesce(ref_PDS, 0) AS d_PDS,
        (coalesce(ours_PDS, 0) - coalesce(ours_multimodel_score, 0)) -
          (coalesce(ref_PDS, 0) - coalesce(ref_multimodel_score, 0)) AS c_SiK_adjustment
      FROM ref INNER JOIN ours USING(key)
    )
    SELECT
      *,
      d_TPDS AS c_TPDS,
      0.5 * d_PPDS AS c_PPDS,
      0.5 * d_PhPDS_ps AS c_PhPDS_ps,
      0.5 * d_PhPDS_py AS c_PhPDS_py,
      d_multimodel - (d_TPDS + 0.5 * (d_PPDS + d_PhPDS_ps + d_PhPDS_py)) AS c_other_multimodel
    FROM d
  ", sql_str(paths$parquet), sp1, sp2))

  by_scope <- as.data.table(dbGetQuery(con, "
    WITH base AS (
      SELECT row_number() OVER () AS rowid, * FROM joined
    ),
    long AS (
      SELECT rowid, transgene_path, d_PDS, 'TPDS' AS contributor, abs(c_TPDS) AS abs_contrib FROM base
      UNION ALL SELECT rowid, transgene_path, d_PDS, 'PPDS', abs(c_PPDS) FROM base
      UNION ALL SELECT rowid, transgene_path, d_PDS, 'PhPDS_ps', abs(c_PhPDS_ps) FROM base
      UNION ALL SELECT rowid, transgene_path, d_PDS, 'PhPDS_py', abs(c_PhPDS_py) FROM base
      UNION ALL SELECT rowid, transgene_path, d_PDS, 'SiK_adjustment', abs(c_SiK_adjustment) FROM base
      UNION ALL SELECT rowid, transgene_path, d_PDS, 'other_multimodel', abs(c_other_multimodel) FROM base
    ),
    ranked AS (
      SELECT *,
        row_number() OVER (
          PARTITION BY transgene_path, rowid
          ORDER BY abs_contrib DESC, contributor ASC
        ) AS rn
      FROM long
    )
    SELECT
      CASE WHEN transgene_path THEN 'transgene' ELSE 'non_transgene' END AS scope,
      contributor AS dominant_contributor,
      count(*) AS rows
    FROM ranked
    WHERE rn = 1
    GROUP BY 1, 2
  "))

  stats <- as.data.table(dbGetQuery(con, "
    SELECT
      CASE WHEN transgene_path THEN 'transgene' ELSE 'non_transgene' END AS scope,
      count(*) AS shared_rows,
      sum(CASE WHEN abs(d_PDS) > 1e-4 THEN 1 ELSE 0 END) AS pds_gt_1e4,
      sum(CASE WHEN abs(d_PDS) > 0.01 THEN 1 ELSE 0 END) AS pds_gt_0_01,
      sum(CASE WHEN abs(d_PDS) > 0.05 THEN 1 ELSE 0 END) AS pds_gt_0_05,
      median(abs(d_PDS)) AS median_abs_pds,
      quantile_cont(abs(d_PDS), 0.95) AS p95_abs_pds,
      quantile_cont(abs(d_PDS), 0.99) AS p99_abs_pds,
      max(abs(d_PDS)) AS max_abs_pds,
      median(abs(c_TPDS)) AS median_abs_c_TPDS,
      median(abs(c_PPDS)) AS median_abs_c_PPDS,
      median(abs(c_PhPDS_ps)) AS median_abs_c_PhPDS_ps,
      median(abs(c_PhPDS_py)) AS median_abs_c_PhPDS_py,
      median(abs(c_SiK_adjustment)) AS median_abs_c_SiK_adjustment,
      median(abs(c_other_multimodel)) AS median_abs_c_other_multimodel,
      quantile_cont(abs(c_TPDS), 0.95) AS p95_abs_c_TPDS,
      quantile_cont(abs(c_PPDS), 0.95) AS p95_abs_c_PPDS,
      quantile_cont(abs(c_PhPDS_ps), 0.95) AS p95_abs_c_PhPDS_ps,
      quantile_cont(abs(c_PhPDS_py), 0.95) AS p95_abs_c_PhPDS_py,
      quantile_cont(abs(c_SiK_adjustment), 0.95) AS p95_abs_c_SiK_adjustment,
      quantile_cont(abs(c_other_multimodel), 0.95) AS p95_abs_c_other_multimodel
      ,
      quantile_cont(abs(c_TPDS), 0.99) AS p99_abs_c_TPDS,
      quantile_cont(abs(c_PPDS), 0.99) AS p99_abs_c_PPDS,
      quantile_cont(abs(c_PhPDS_ps), 0.99) AS p99_abs_c_PhPDS_ps,
      quantile_cont(abs(c_PhPDS_py), 0.99) AS p99_abs_c_PhPDS_py,
      quantile_cont(abs(c_SiK_adjustment), 0.99) AS p99_abs_c_SiK_adjustment,
      quantile_cont(abs(c_other_multimodel), 0.99) AS p99_abs_c_other_multimodel,
      max(abs(c_TPDS)) AS max_abs_c_TPDS,
      max(abs(c_PPDS)) AS max_abs_c_PPDS,
      max(abs(c_PhPDS_ps)) AS max_abs_c_PhPDS_ps,
      max(abs(c_PhPDS_py)) AS max_abs_c_PhPDS_py,
      max(abs(c_SiK_adjustment)) AS max_abs_c_SiK_adjustment,
      max(abs(c_other_multimodel)) AS max_abs_c_other_multimodel
    FROM joined
    GROUP BY 1
  "))

  dom <- dcast(by_scope, scope ~ dominant_contributor, value.var = "rows", fill = 0)
  out <- merge(stats, dom, by = "scope", all.x = TRUE)
  for (cc in c("TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_adjustment", "other_multimodel")) {
    if (!cc %in% names(out)) out[[cc]] <- 0
  }
  out[, `:=`(run = run_name, contrast = contrast)]
  out
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(dirname(opts$out_csv), recursive = TRUE, showWarnings = FALSE)
xwalk <- build_crosswalk(opts$allmarkers)
contrasts <- c(
  "ma_2mo_AppP", "ma_2mo_ApTt", "ma_2mo_Ttau",
  "ma_4mo_AppP", "ma_4mo_ApTt", "ma_4mo_Ttau",
  "ma_6mo_AppP", "ma_6mo_ApTt", "ma_6mo_Ttau"
)
runs <- list(canonical = opts$canonical_dir, source_ps = opts$source_ps_dir)

res <- rbindlist(lapply(names(runs), function(run_name) {
  rbindlist(lapply(contrasts, function(contrast) {
    audit_one(run_name, runs[[run_name]], contrast, opts, xwalk)
  }), use.names = TRUE, fill = TRUE)
}), use.names = TRUE, fill = TRUE)
setcolorder(res, c("run", "contrast", "scope", setdiff(names(res), c("run", "contrast", "scope"))))
fwrite(res, opts$out_csv)
print(res[, .(
  shared_rows = sum(shared_rows),
  pds_gt_1e4 = sum(pds_gt_1e4),
  pds_gt_0_05 = sum(pds_gt_0_05),
  median_abs_pds_max = max(median_abs_pds, na.rm = TRUE),
  p95_abs_pds_max = max(p95_abs_pds, na.rm = TRUE),
  PhPDS_ps = sum(PhPDS_ps, na.rm = TRUE),
  PhPDS_py = sum(PhPDS_py, na.rm = TRUE),
  PPDS = sum(PPDS, na.rm = TRUE),
  TPDS = sum(TPDS, na.rm = TRUE),
  SiK_adjustment = sum(SiK_adjustment, na.rm = TRUE),
  other_multimodel = sum(other_multimodel, na.rm = TRUE)
), by = .(run, scope)])
cat("Wrote: ", opts$out_csv, "\n", sep = "")
