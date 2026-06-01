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
  out_csv = file.path(repo, "outputs/reports/incytr_pair_mode/forensics/transgene_removed_top300_summary.csv")
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

empty_key <- function() {
  d <- data.table(x = character())
  setnames(d, "x", "key")
  d
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

full_key <- function(dt) {
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

read_pairwise_ref <- function(contrast, opts, xwalk) {
  token <- sprintf("%s_WTyp", contrast)
  rds <- file.path(opts$ref_dir, sprintf("DEG_PRG_%s_10302025", token),
                   "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds")
  x <- readRDS(rds)
  ref <- rbindlist(lapply(x, function(e) {
    d <- as.data.table(e)
    d[, .(Sender.group, Receiver.group, Ligand, Receptor, EM, Target, PDS)]
  }), use.names = TRUE, fill = TRUE)
  where <- basename(dirname(rds))
  ref[, Sender := map_clusters(Sender.group, xwalk, where)]
  ref[, Receiver := map_clusters(Receiver.group, xwalk, where)]
  ref[, key := full_key(ref)]
  unique(ref[!is_transgene_path(ref), .(key, Sender, Receiver, PDS)], by = "key")
}

top300_from_dt <- function(dt) {
  if (!nrow(dt)) return(empty_key())
  pos <- copy(dt[PDS > 0])
  neg <- copy(dt[PDS < 0])
  if (nrow(pos)) {
    setorder(pos, Sender, Receiver, -PDS)
    pos[, rn := seq_len(.N), by = .(Sender, Receiver)]
    pos <- pos[rn <= 300, .(key)]
  } else {
    pos <- empty_key()
  }
  if (nrow(neg)) {
    setorder(neg, Sender, Receiver, PDS)
    neg[, rn := seq_len(.N), by = .(Sender, Receiver)]
    neg <- neg[rn <= 300, .(key)]
  } else {
    neg <- empty_key()
  }
  unique(rbind(pos, neg), by = "key")
}

ours_tables <- function(wide_dir, contrast) {
  age <- sub("^ma_([0-9]+mo)_.*$", "\\1", contrast)
  c2 <- sprintf("ma_%s_WTyp", age)
  parquet <- file.path(wide_dir, sprintf("%s_%s_incytr_output.parquet", contrast, c2))
  if (!file.exists(parquet)) stop("missing parquet: ", parquet, call. = FALSE)

  con <- dbConnect(duckdb::duckdb())
  on.exit(dbDisconnect(con, shutdown = TRUE), add = TRUE)
  dbExecute(con, "PRAGMA memory_limit='8GB'")
  sp1 <- qident(paste0("SigProb_", contrast))
  sp2 <- qident(paste0("SigProb_", c2))
  transgene_expr <- paste(
    "Ligand IN ('App','Psen1','Mapt')",
    "Receptor IN ('App','Psen1','Mapt')",
    "EM IN ('App','Psen1','Mapt')",
    "Target IN ('App','Psen1','Mapt')",
    sep = " OR "
  )
  path_expr <- paste(
    "concat(Sender, chr(1), Receiver, chr(1), Ligand, chr(1),",
    "Receptor, chr(1), EM, chr(1), Target)"
  )

  gated_full <- as.data.table(dbGetQuery(con, sprintf("
    SELECT DISTINCT %s AS key, Sender, Receiver, PDS
    FROM read_parquet(%s)
    WHERE ((%s > 0.1) OR (%s > 0.1))
      AND abs(PDS) >= 0.2
      AND NOT (%s)
  ", path_expr, sql_str(parquet), sp1, sp2, transgene_expr)))

  list(
    gated = unique(gated_full[, .(key)], by = "key"),
    top300 = top300_from_dt(gated_full)
  )
}

cmp <- function(a, b) {
  if (!("key" %in% names(a))) a <- empty_key()
  if (!("key" %in% names(b))) b <- empty_key()
  setkey(a, key)
  setkey(b, key)
  list(
    a_n = nrow(a),
    b_n = nrow(b),
    shared = nrow(fintersect(a, b)),
    missing = nrow(fsetdiff(b, a)),
    extra = nrow(fsetdiff(a, b))
  )
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
xwalk <- build_crosswalk(opts$allmarkers)
runs <- list(
  canonical = opts$canonical_dir,
  source_ps = opts$source_ps_dir
)
contrasts <- c(
  "ma_2mo_AppP", "ma_2mo_ApTt", "ma_2mo_Ttau",
  "ma_4mo_AppP", "ma_4mo_ApTt", "ma_4mo_Ttau",
  "ma_6mo_AppP", "ma_6mo_ApTt", "ma_6mo_Ttau"
)

res <- rbindlist(lapply(names(runs), function(run_name) {
  rbindlist(lapply(contrasts, function(contrast) {
    ours <- ours_tables(runs[[run_name]], contrast)
    ref_full <- read_pairwise_ref(contrast, opts, xwalk)
    ref_g <- ref_full[, .(key)]
    ref_t <- top300_from_dt(ref_full)
    g <- cmp(ours$gated, ref_g)
    t <- cmp(ours$top300, ref_t)
    data.table(
      run = run_name,
      contrast = contrast,
      gated_ours = g$a_n,
      gated_sce4 = g$b_n,
      gated_shared = g$shared,
      gated_missing = g$missing,
      gated_extra = g$extra,
      top300_ours = t$a_n,
      top300_sce4 = t$b_n,
      top300_shared = t$shared,
      top300_missing = t$missing,
      top300_extra = t$extra
    )
  }), use.names = TRUE)
}), use.names = TRUE)

dir.create(dirname(opts$out_csv), recursive = TRUE, showWarnings = FALSE)
fwrite(res, opts$out_csv)
print(res)
print(res[, .(
  gated_ours = sum(gated_ours),
  gated_sce4 = sum(gated_sce4),
  gated_shared = sum(gated_shared),
  gated_missing = sum(gated_missing),
  gated_extra = sum(gated_extra),
  top300_ours = sum(top300_ours),
  top300_sce4 = sum(top300_sce4),
  top300_shared = sum(top300_shared),
  top300_missing = sum(top300_missing),
  top300_extra = sum(top300_extra)
), by = run])
cat("Wrote: ", opts$out_csv, "\n", sep = "")
