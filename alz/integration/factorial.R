#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Incytr)
})

parse_args <- function(argv) {
  args <- list(
    input_dir = "data/incytr_factorial_inputs",
    out_dir = "outputs/reports/incytr_factorial",
    pair_filter = Sys.getenv("PAIR_FILTER", "")
  )

  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    val <- if (i < length(argv)) argv[[i + 1L]] else NA_character_
    if (key %in% c("--input-dir", "--out-dir", "--pair-filter")) {
      if (is.na(val)) stop("Missing value for ", key)
      args[[sub("^--", "", gsub("-", "_", key))]] <- val
      i <- i + 2L
    } else {
      stop("Unknown argument: ", key)
    }
  }

  args
}

require_production_engine <- function() {
  ns <- asNamespace("Incytr")
  required <- c("construct_factorial_paths", "score_factorial_paths")
  missing <- required[!vapply(required, exists, logical(1), envir = ns,
                              mode = "function", inherits = FALSE)]
  if (length(missing) > 0) {
    stop(
      "The production Incytr factorial path functions are not implemented in ",
      "the installed package: ", paste(missing, collapse = ", "),
      ". Refusing to load AD inputs or fall back to run_factorial_all_pairs(), ",
      "run_factorial_receivers(), or the single-pair pathway_inference() path."
    )
  }
  list(
    construct = get("construct_factorial_paths", envir = ns, mode = "function",
                    inherits = FALSE),
    score = get("score_factorial_paths", envir = ns, mode = "function",
                inherits = FALSE)
  )
}

normalize_receiver_results <- function(results, receiver) {
  df <- as.data.frame(results)
  if (!"receiver" %in% names(df)) {
    df$receiver <- receiver
  }
  df
}

normalize_receiver_metadata <- function(metadata, receiver) {
  if (is.null(metadata)) {
    return(data.frame(
      receiver = receiver,
      status = "ok",
      stringsAsFactors = FALSE
    ))
  }
  df <- as.data.frame(metadata)
  if (!"receiver" %in% names(df)) {
    df$receiver <- receiver
  }
  df
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
factorial_engine <- require_production_engine()

suppressPackageStartupMessages({
  library(arrow)
})

source("alz/integration/load.R")
source("alz/integration/persist.R")

inputs <- load_ad_factorial_inputs(args$input_dir)
selected <- apply_pair_filter(inputs$senders, inputs$receivers, args$pair_filter)
contrasts <- build_factorial_contrasts(inputs$animal_meta)

message(sprintf(
  "Running native Incytr factorial pipeline for %d sender(s) x %d receiver(s)",
  length(selected$senders),
  length(selected$receivers)
))

message("DEBUG selected senders: ", paste(selected$senders, collapse = ", "))
message("DEBUG selected receivers: ", paste(selected$receivers, collapse = ", "))
message("DEBUG deg_lists names: ", if (is.null(inputs$deg_lists)) "NULL" else paste(names(inputs$deg_lists), collapse = ", "))
message("DEBUG prg_list length: ", if (is.null(inputs$prg_list)) "NULL" else length(inputs$prg_list))

paths <- factorial_engine$construct(
  expression = inputs$expr,
  metadata = inputs$meta,
  senders = selected$senders,
  receivers = selected$receivers,
  group.by = "labels",
  deg_lists = inputs$deg_lists,
  prg_list = inputs$prg_list
)
message(sprintf("DEBUG paths constructed: nrow=%d  ncol=%d", nrow(paths), ncol(paths)))
if (nrow(paths) > 0) message("DEBUG paths cols: ", paste(colnames(paths), collapse = ", "))

if (!is.null(inputs$cond_pairs)) {
  message(sprintf(
    "Loaded explicit cond_pairs for %d contrast(s) from MANIFEST.json",
    length(inputs$cond_pairs)
  ))
} else {
  message(
    "No cond_pairs in MANIFEST.json; SigProb will use single-coefficient ",
    "heuristic (multi-coef contrasts will get NA SigProb_ref/alt)."
  )
}

# Multi-omic mode (default since 2026-05): the Incytr engine returns one
# parquet per (sender, receiver) pair with populated PPDS / PhPDS_ps /
# PhPDS_py / multimodel_score / PDS columns. Required inputs (omics,
# output_dir) are passed through; the engine writes per-pair parquets to
# the staging directory and returns a summary frame with `path`, `status`,
# and `n_paths`. The wrapper then re-shards into the existing
# receiver_cache layout that downstream consumers expect.
omics <- list(
  pr = list(data_wide = inputs$pr_mat),
  ps = list(data_wide = inputs$ps_mat),
  py = list(data_wide = inputs$py_mat)
)

pair_staging <- file.path(args$out_dir, ".staging", "pair_parquets")
dir.create(pair_staging, recursive = TRUE, showWarnings = FALSE)

n_perm_env <- Sys.getenv("INCYTR_N_PERM", "0")
n_perm <- as.integer(n_perm_env)
if (is.na(n_perm) || n_perm < 0) {
  stop("INCYTR_N_PERM must be a non-negative integer; got '", n_perm_env, "'")
}

results_summary <- factorial_engine$score(
  expression = inputs$expr,
  metadata = inputs$meta,
  paths = paths,
  contrasts = contrasts,
  design = inputs$design,
  animal_id = "animal_id",
  condition_col = "condition",
  cond_pairs = inputs$cond_pairs,
  omics = omics,
  kldata = inputs$kldata,
  output_dir = pair_staging,
  n_perm = n_perm
)
message("DEBUG results_summary:")
print(results_summary)

ok_rows <- results_summary[
  results_summary$status %in% c("ok", "checkpointed"), ,
  drop = FALSE
]

if (nrow(ok_rows) > 0) {
  # Stream pair-at-a-time and write Hive-partitioned part-files. Building a
  # full per-receiver long frame in memory (rbind of 18+ pair dfs) hit the
  # 16 GB ulimit on the 100-pair sweep; per-pair peak is ~150 MB.
  condition_names <- as.character(unique(inputs$meta[["condition"]]))
  for (i in seq_len(nrow(ok_rows))) {
    fp <- ok_rows$path[[i]]
    receiver <- ok_rows$receiver[[i]]
    sender <- ok_rows$sender[[i]]
    recv_dir <- file.path(args$out_dir, "receiver_cache",
                          paste0("receiver=", sanitize_celltype(receiver)))
    ensure_dir(recv_dir)
    out_file <- file.path(recv_dir,
                          paste0("part-", sanitize_celltype(sender), ".parquet"))
    wide <- as.data.frame(arrow::read_parquet(fp))
    long <- Incytr::factorial_results_long(wide,
                                            contrast_names = names(contrasts),
                                            condition_names = condition_names)
    rm(wide); gc(verbose = FALSE)
    arrow::write_parquet(normalize_receiver_results(long, receiver), out_file)
    rm(long); gc(verbose = FALSE)
  }

  metadata_rows <- data.frame(
    sender = ok_rows$sender,
    receiver = ok_rows$receiver,
    n_post = ok_rows$n_paths,
    n_pre = ok_rows$n_paths,
    status = ok_rows$status,
    stringsAsFactors = FALSE
  )
  write_pair_metadata_parquet(metadata_rows, out_dir = args$out_dir)
}
write_views_sql(args$out_dir)

message("Wrote native Incytr factorial outputs to ", args$out_dir)
