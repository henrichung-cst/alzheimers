#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Incytr)
})

parse_args <- function(argv) {
  args <- list(
    input_dir = "alz/integration/intermediates/factorial",
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

paths <- factorial_engine$construct(
  expression = inputs$expr,
  metadata = inputs$meta,
  senders = selected$senders,
  receivers = selected$receivers,
  group.by = "labels"
)

results <- factorial_engine$score(
  expression = inputs$expr,
  metadata = inputs$meta,
  paths = paths,
  contrasts = contrasts,
  design = inputs$design,
  animal_id = "animal_id",
  condition_col = "condition"
)

if (nrow(results) > 0) {
  for (receiver in sort(unique(results$receiver))) {
    receiver_results <- results[results$receiver == receiver, , drop = FALSE]
    write_receiver_parquet(
      normalize_receiver_results(receiver_results, receiver),
      receiver,
      out_dir = args$out_dir
    )
  }

  metadata_rows <- aggregate(
    results$Path,
    by = list(sender = results$sender, receiver = results$receiver),
    FUN = function(x) length(unique(x))
  )
  names(metadata_rows)[names(metadata_rows) == "x"] <- "n_post"
  metadata_rows$n_pre <- metadata_rows$n_post
  metadata_rows$status <- "ok"
  write_pair_metadata_parquet(metadata_rows, out_dir = args$out_dir)
}
write_views_sql(args$out_dir)

message("Wrote native Incytr factorial outputs to ", args$out_dir)
