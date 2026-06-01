#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
})

repo <- normalizePath(system("git rev-parse --show-toplevel", intern = TRUE))

defaults <- list(
  extra_glob = file.path(repo, "outputs/reports/incytr_pair_mode/forensics/ma_*_extra_nontransgene_audit.csv"),
  ref_dir = file.path(repo, "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1"),
  allmarkers = file.path(repo, "data/derived/incytr_inputs/allmarkers.csv"),
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
    if (a == "--extra-glob") {
      x$extra_glob <- need(); i <- i + 2L
    } else if (a == "--ref-dir") {
      x$ref_dir <- need(); i <- i + 2L
    } else if (a == "--allmarkers") {
      x$allmarkers <- need(); i <- i + 2L
    } else if (a == "--out-dir") {
      x$out_dir <- need(); i <- i + 2L
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

path_key <- function(dt) {
  paste(dt$Ligand, dt$Receptor, dt$EM, dt$Target, sep = "\001")
}

pair_key <- function(sender, receiver) paste(sender, receiver, sep = "\001")

contrast_paths <- function(contrast, ref_dir) {
  age <- sub("^ma_([0-9]+mo)_.*$", "\\1", contrast)
  token <- paste0(contrast, "_WTyp")
  list(
    c2 = paste0("ma_", age, "_WTyp"),
    dir = file.path(ref_dir, paste0("DEG_PRG_", token, "_10302025")),
    rds = file.path(ref_dir, paste0("DEG_PRG_", token, "_10302025"),
                    "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds"),
    allpath = file.path(ref_dir, paste0("DEG_PRG_", token, "_10302025"),
                        "sce4_DEG_PRG_Allpathway_table_10302025.csv"),
    top300 = file.path(ref_dir, paste0("DEG_PRG_", token, "_10302025"),
                       "sce4_DEG_PRG_Top300_table_10302025.csv")
  )
}

read_ref_pairwise <- function(rds_path, xwalk) {
  x <- readRDS(rds_path)
  cols <- c("Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target")
  ref <- rbindlist(lapply(x, function(e) {
    d <- as.data.table(e)
    keep <- intersect(cols, names(d))
    d[, ..keep]
  }), use.names = TRUE, fill = TRUE)
  where <- basename(dirname(rds_path))
  ref[, Sender := map_clusters(Sender.group, xwalk, where)]
  ref[, Receiver := map_clusters(Receiver.group, xwalk, where)]
  ref[, path_key := path_key(ref)]
  ref[, pair_key := pair_key(Sender, Receiver)]
  unique(ref, by = c("pair_key", "path_key"))
}

read_csv_universe <- function(csv_path, xwalk) {
  if (!file.exists(csv_path)) return(NULL)
  cols <- c("Sender.group", "Receiver.group", "Ligand", "Receptor", "EM", "Target")
  d <- fread(csv_path, select = cols)
  where <- basename(dirname(csv_path))
  d[, Sender := map_clusters(Sender.group, xwalk, where)]
  d[, Receiver := map_clusters(Receiver.group, xwalk, where)]
  d[, path_key := path_key(d)]
  d[, pair_key := pair_key(Sender, Receiver)]
  unique(d[, .(pair_key, path_key)])
}

role_sets_for_pair <- function(ref) {
  ref[, .(
    ligands = list(unique(Ligand)),
    receptors = list(unique(Receptor)),
    ems = list(unique(EM)),
    targets = list(unique(Target)),
    receiver_flat = list(unique(c(Receptor, EM, Target))),
    paths = list(unique(path_key))
  ), by = .(pair_key, Sender, Receiver)]
}

audit_one <- function(extra_path, opts, xwalk) {
  contrast <- sub("_extra_nontransgene_audit.csv$", "", basename(extra_path))
  paths <- contrast_paths(contrast, opts$ref_dir)
  if (!file.exists(paths$rds)) stop("missing sce4 Pairwise RDS: ", paths$rds, call. = FALSE)
  extra <- fread(extra_path)
  extra[, contrast := contrast]
  if (!nrow(extra)) {
    return(list(detail = data.table(), summary = data.table(
      contrast = contrast,
      extra_non_transgene = 0L,
      in_pairwise_rds = 0L,
      in_top300 = 0L,
      in_allpathway = NA_integer_,
      all_nodes_role_present = 0L,
      all_nodes_flat_present = 0L,
      role_violation_rows = 0L,
      ligand_role_missing = 0L,
      receptor_role_missing = 0L,
      em_role_missing = 0L,
      target_role_missing = 0L,
      path_seen_elsewhere_pairwise = 0L,
      allpathway_available = file.exists(paths$allpath)
    )))
  }
  extra[, path_key := path_key(extra)]
  extra[, pair_key := pair_key(Sender, Receiver)]

  ref <- read_ref_pairwise(paths$rds, xwalk)
  rs <- role_sets_for_pair(ref)
  ref_paths_any_pair <- unique(ref$path_key)
  ref_key_set <- unique(paste(ref$pair_key, ref$path_key, sep = "\001"))

  top300 <- read_csv_universe(paths$top300, xwalk)
  allpath <- read_csv_universe(paths$allpath, xwalk)
  top_key_set <- if (is.null(top300)) character(0) else unique(paste(top300$pair_key, top300$path_key, sep = "\001"))
  all_key_set <- if (is.null(allpath)) character(0) else unique(paste(allpath$pair_key, allpath$path_key, sep = "\001"))

  detail <- merge(extra, rs, by = c("pair_key", "Sender", "Receiver"), all.x = TRUE, sort = FALSE)
  detail[, `:=`(
    in_pairwise_rds = paste(pair_key, path_key, sep = "\001") %in% ref_key_set,
    in_top300 = paste(pair_key, path_key, sep = "\001") %in% top_key_set,
    in_allpathway = if (is.null(allpath)) NA else paste(pair_key, path_key, sep = "\001") %in% all_key_set,
    path_seen_elsewhere_pairwise = path_key %in% ref_paths_any_pair
  )]

  detail[, `:=`(
    ligand_role_present = mapply(`%in%`, Ligand, ligands),
    receptor_role_present = mapply(`%in%`, Receptor, receptors),
    em_role_present = mapply(`%in%`, EM, ems),
    target_role_present = mapply(`%in%`, Target, targets),
    receptor_flat_present = mapply(`%in%`, Receptor, receiver_flat),
    em_flat_present = mapply(`%in%`, EM, receiver_flat),
    target_flat_present = mapply(`%in%`, Target, receiver_flat)
  )]
  detail[, `:=`(
    all_nodes_role_present = ligand_role_present & receptor_role_present & em_role_present & target_role_present,
    all_nodes_flat_present = ligand_role_present & receptor_flat_present & em_flat_present & target_flat_present
  )]
  detail[, role_violation := fifelse(!ligand_role_present, "ligand",
                              fifelse(!receptor_role_present, "receptor",
                              fifelse(!em_role_present, "em",
                              fifelse(!target_role_present, "target", "none"))))]
  detail[, receiver_role_violation_count :=
           (!receptor_role_present) + (!em_role_present) + (!target_role_present)]

  # List columns are useful internally but noisy in the row artifact.
  detail[, c("ligands", "receptors", "ems", "targets", "receiver_flat", "paths") := NULL]

  summary <- data.table(
    contrast = contrast,
    extra_non_transgene = nrow(detail),
    in_pairwise_rds = sum(detail$in_pairwise_rds),
    in_top300 = sum(detail$in_top300),
    in_allpathway = if (is.null(allpath)) NA_integer_ else sum(detail$in_allpathway),
    all_nodes_role_present = sum(detail$all_nodes_role_present),
    all_nodes_flat_present = sum(detail$all_nodes_flat_present),
    role_violation_rows = sum(!detail$all_nodes_role_present),
    ligand_role_missing = sum(!detail$ligand_role_present),
    receptor_role_missing = sum(!detail$receptor_role_present),
    em_role_missing = sum(!detail$em_role_present),
    target_role_missing = sum(!detail$target_role_present),
    path_seen_elsewhere_pairwise = sum(detail$path_seen_elsewhere_pairwise),
    allpathway_available = file.exists(paths$allpath),
    ref_rds = paths$rds,
    allpathway_csv = if (file.exists(paths$allpath)) paths$allpath else NA_character_,
    top300_csv = if (file.exists(paths$top300)) paths$top300 else NA_character_
  )
  list(detail = detail, summary = summary)
}

opts <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(opts$out_dir, recursive = TRUE, showWarnings = FALSE)
files <- Sys.glob(opts$extra_glob)
if (!length(files)) stop("no extra audit files matched: ", opts$extra_glob, call. = FALSE)
xwalk <- build_crosswalk(opts$allmarkers)

details <- list()
summaries <- list()
for (f in files) {
  res <- audit_one(f, opts, xwalk)
  if (nrow(res$detail)) details[[length(details) + 1L]] <- res$detail
  summaries[[length(summaries) + 1L]] <- res$summary
}
detail <- rbindlist(details, use.names = TRUE, fill = TRUE)
summary <- rbindlist(summaries, use.names = TRUE, fill = TRUE)
setorder(summary, contrast)
setorder(detail, contrast, Sender, Receiver, Ligand, Receptor, EM, Target)

detail_path <- file.path(opts$out_dir, "sce4_extra_universe_detail.csv")
summary_path <- file.path(opts$out_dir, "sce4_extra_universe_summary.csv")
fwrite(detail, detail_path)
fwrite(summary, summary_path)

print(summary[, .(
  contrast,
  extra_non_transgene,
  in_pairwise_rds,
  in_top300,
  in_allpathway,
  all_nodes_role_present,
  all_nodes_flat_present,
  role_violation_rows,
  ligand_role_missing,
  receptor_role_missing,
  em_role_missing,
  target_role_missing,
  path_seen_elsewhere_pairwise,
  allpathway_available
)])
cat("Wrote:\n")
cat("  ", detail_path, "\n", sep = "")
cat("  ", summary_path, "\n", sep = "")
