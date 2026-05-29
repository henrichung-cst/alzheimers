#!/usr/bin/env Rscript
# Per-donor T-cell Seurat for Incytr pair-mode.
#
# Reads the raw donor RDS, joins per-cell ProjecTILs state from the extract
# step's projectils_predictions.csv, sanitizes state names alphanumeric, drops
# NA-state cells, and writes a slim Seurat with:
#   - Idents()       = state (Type)
#   - obj$Type       = state
#   - obj$condition  = sprintf("d%d", day)         (e.g. "d13")
#   - obj$Type_condition = paste0(Type, "_", condition)
#
# Pair-mode driver consumes this as data/derived/tcells_incytr_inputs/<donor>/incytr_obj.rds.
#
# Memory pattern carried from tcells_scrna_extract.R: DietSeurat → RNA only,
# drop scale.data immediately, subset down to labeled cells before any compute.
#
# Usage:  pixi run Rscript alz/incytr_pair/build_tcells_seurat.R <donor1|donor2>
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
donor <- args[[1]]

cfg <- list(
  donor1 = list(
    rds     = "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
    day_col = "sample_ID"),
  donor2 = list(
    rds     = "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds",
    day_col = "Sample_Label")
)[[donor]]
stopifnot(!is.null(cfg))

outdir <- file.path("data/derived/tcells_incytr_inputs", donor)
scrnadir <- file.path(outdir, "scrna")
pred_path <- file.path(scrnadir, "projectils_predictions.csv")
if (!file.exists(pred_path)) {
  stop("missing projectils predictions — run `pixi run tcells-projectils-map ",
       donor, "` first: ", pred_path)
}

# 14-state sanitization map (identical to tcells_scrna_extract.R). Alphanumeric
# only because the pair-mode driver splits `<condition>_<cluster>` on `_`.
LABEL_MAP <- c(
  "CD8.CM"         = "CD8CM",
  "CD8.EM"         = "CD8EM",
  "CD8.MAIT"       = "CD8MAIT",
  "CD8.NaiveLike"  = "CD8Naive",
  "CD8.TEMRA"      = "CD8TEMRA",
  "CD8.TEX"        = "CD8Tex",
  "CD8.TPEX"       = "CD8Tpex",
  "CD4.CTL_EOMES"  = "CD4CTLeomes",
  "CD4.CTL_Exh"    = "CD4CTLexh",
  "CD4.CTL_GNLY"   = "CD4CTLgnly",
  "CD4.NaiveLike"  = "CD4Naive",
  "CD4.Tfh"        = "CD4Tfh",
  "CD4.Th17"       = "CD4Th17",
  "CD4.Treg"       = "Treg"
)

memline <- function(tag) {
  gc(full = TRUE)
  mi <- tryCatch(readLines("/proc/meminfo", n = 3), error = function(e) character())
  avail <- sub(".*?:\\s*", "", grep("MemAvailable", mi, value = TRUE))
  cat(sprintf("[mem %-18s] MemAvailable: %s\n", tag, ifelse(length(avail), avail, "?")))
}

cat("==== BUILD INCYTR SEURAT", donor, "====\n")
memline("start")

t0 <- Sys.time()
obj <- readRDS(cfg$rds)
cat("read in", round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s ;",
    "dim", paste(dim(obj), collapse = " x "), "\n")
memline("after readRDS")

DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = "RNA", dimreducs = NULL, graphs = NULL)
obj[["RNA"]]$scale.data <- NULL
memline("after DietSeurat")

# Per-cell ProjecTILs state -------------------------------------------------
pred <- read.csv(pred_path, stringsAsFactors = FALSE)
stopifnot(all(c("barcode", "functional.cluster") %in% colnames(pred)))
bc_to_state_raw <- setNames(pred$functional.cluster, pred$barcode)
state_raw <- bc_to_state_raw[colnames(obj)]

state <- rep(NA_character_, length(state_raw))
mask  <- !is.na(state_raw)
unknown <- setdiff(unique(state_raw[mask]), names(LABEL_MAP))
if (length(unknown)) {
  stop("unknown ProjecTILs functional.cluster value(s): ",
       paste(unknown, collapse = ", "), " — add to LABEL_MAP")
}
state[mask] <- LABEL_MAP[state_raw[mask]]
obj$state <- state

# Day parsing ---------------------------------------------------------------
day_raw <- as.character(obj@meta.data[[cfg$day_col]])
day <- as.integer(sub(".*[Dd]ay[_ ]?(\\d+).*", "\\1", day_raw))
if (any(is.na(day))) {
  bad_day <- sort(unique(day_raw[is.na(day)]))
  stop("unparsed day labels: ", paste(bad_day, collapse = " | "))
}
obj$ts_day <- day

# Drop NA-state cells -------------------------------------------------------
n_total <- ncol(obj)
keep <- !is.na(obj$state)
n_kept <- sum(keep)
cat("cells: total=", n_total, " kept=", n_kept, " (",
    round(100 * n_kept / n_total, 1), "%)\n", sep = "")

obj <- subset(obj, cells = colnames(obj)[keep])
memline("after subset")

# Set driver-expected metadata ---------------------------------------------
obj$Type      <- as.character(obj$state)
obj$condition <- sprintf("d%d", obj$ts_day)
obj$Type_condition <- paste0(obj$Type, "_", obj$condition)

# Alphanumeric guard (Incytr driver splits Type_condition on `_`).
stopifnot(!any(grepl("[^A-Za-z0-9]", obj$Type)))
stopifnot(all(grepl("^d\\d+$", obj$condition)))

Idents(obj) <- factor(obj$Type, levels = sort(unique(obj$Type)))

dst <- file.path(outdir, "incytr_obj.rds")
cat("[seurat] writing", dst, "\n")
saveRDS(obj, dst)
cat("[seurat] final dim:", paste(dim(obj), collapse = " x "),
    " conditions=", length(unique(obj$condition)),
    " types=", length(unique(obj$Type)), "\n", sep = "")
cat("==== DONE", donor, "->", dst, "====\n")
