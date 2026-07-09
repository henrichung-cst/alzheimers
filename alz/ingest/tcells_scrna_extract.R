#!/usr/bin/env Rscript
# D4 scRNA extraction keyed on evidence-backed per-cell state.
#
# The substrate aggregates by `(state, day)` where `state` is the sanitized
# per-cell `type` from the evidence report. Blank types are contaminants and are
# dropped; all retained cells have an explicit CD4/CD8 state call.
#
# Memory pattern unchanged: load ONCE, DietSeurat to RNA-only + drop scale.data
# immediately, extract every artifact in one pass.
#
# Emits under data/derived/tcells_incytr_inputs/donor<N>/scrna/:
#   cell_counts.csv     (state, day, n_cells)        -> P_s size factors
#   aggexp_data.csv     (gene x state__day)          -> specific_s share numerator
#   allmarkers.csv      (FindAllMarkers on state)    -> spine markers
#   extract_manifest.json
#   state_audit.json    (per-state totals, drop accounting)
#
# Prereq: Phase 1 per-cell labels under outputs/reports/tcell_labeling/cells/
# Usage:  pixi run Rscript alz/ingest/tcells_scrna_extract.R <donor1|donor2>
suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(jsonlite)
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

REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
source(file.path(REPO_ROOT, "alz", "ingest", "tcells_state_labels.R"))

outdir <- file.path("data/derived/tcells_incytr_inputs", donor, "scrna")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

memline <- function(tag) {
  gc(full = TRUE)
  mi <- tryCatch(readLines("/proc/meminfo", n = 3), error = function(e) character())
  avail <- sub(".*?:\\s*", "", grep("MemAvailable", mi, value = TRUE))
  cat(sprintf("[mem %-18s] MemAvailable: %s\n", tag, ifelse(length(avail), avail, "?")))
}

cat("==== EXTRACT", donor, "(state-keyed) ====\n")
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

# Day parsing
day_raw <- as.character(obj@meta.data[[cfg$day_col]])
day <- as.integer(sub(".*[Dd]ay[_ ]?(\\d+).*", "\\1", day_raw))
if (any(is.na(day))) {
  bad_day <- sort(unique(day_raw[is.na(day)]))
  stop("unparsed day labels: ", paste(bad_day, collapse = " | "))
}
obj$ts_day <- day

# --- join evidence-backed per-cell state ---------------------------------
joined <- load_tcell_state_labels(
  donor = donor,
  barcodes = colnames(obj),
  days = day,
  seurat_clusters = obj$seurat_clusters,
  repo_root = REPO_ROOT
)
state <- joined$type
obj$state <- state
obj$state_label <- joined$label

# --- drop contaminants ---------------------------------------------------
n_total <- ncol(obj)
keep <- joined$keep
n_kept <- sum(keep)
n_drop <- n_total - n_kept

cat("cells: total=", n_total, " kept=", n_kept, " (",
    round(100 * n_kept / n_total, 1), "%)  dropped=", n_drop, "\n", sep = "")
drop_label_tab <- table(label = joined$label[!keep])
cat("drop breakdown by evidence label:\n"); print(drop_label_tab)

obj <- subset(obj, cells = colnames(obj)[keep])
state <- obj$state
day   <- obj$ts_day
obj$ts_group <- sprintf("%s__d%d", state, day)
memline("after subset to labeled cells")

# --- 1. cell counts per (state, day) ------------------------------------
cc <- as.data.frame(table(state = state, day = day), stringsAsFactors = FALSE)
cc <- cc[cc$Freq > 0, ]
colnames(cc)[colnames(cc) == "Freq"] <- "n_cells"
cc$day <- as.integer(cc$day)
cc <- cc[order(cc$day, cc$state), ]
write.csv(cc, file.path(outdir, "cell_counts.csv"), row.names = FALSE)
cat("cell_counts:", nrow(cc), "(state,day) groups,", length(unique(cc$state)), "unique states\n")

# --- 2. AggregateExpression(slot="data") per (state, day) ---------------
dat <- SeuratObject::GetAssayData(obj, assay = "RNA", layer = "data")
if (is.null(dat) || nrow(dat) == 0L) stop("RNA 'data' layer empty — not log-normalized")
g <- factor(obj$ts_group)
ind <- Matrix::sparseMatrix(i = seq_along(g), j = as.integer(g), x = 1,
                            dims = c(length(g), nlevels(g)),
                            dimnames = list(NULL, levels(g)))
agg <- as.matrix(dat %*% ind)
agg_df <- data.frame(gene = rownames(agg), agg, check.names = FALSE,
                     row.names = NULL)
write.csv(agg_df, file.path(outdir, "aggexp_data.csv"), row.names = FALSE)
cat("aggexp_data:", nrow(agg), "genes x", ncol(agg), "state__day cols\n")
memline("after aggexp")

# --- 2b. pct_expressing per (state, day) --------------------------------
# Fraction of cells in each (state, day) group with a non-zero count — the
# DETECTION foundation of the standard attribution metric (frac >= 0.10). It is
# count-based / normalization-free (count > 0 == data > 0, since log-norm
# preserves zeros), so it means the same thing as the NSCLC-reference detection
# and lets the within-cohort share tier be retired. Same sparse indicator
# product as the sum above, on a binarized copy of the data layer.
bin <- dat
bin@x <- rep(1.0, length(bin@x))           # mark every stored (expressed) entry
nz <- as.matrix(bin %*% ind)               # genes x groups: # cells expressing
ncells_per_group <- Matrix::colSums(ind)   # cells per (state, day) group
pct <- sweep(nz, 2, ncells_per_group, "/") # fraction of cells expressing
pct_df <- data.frame(gene = rownames(pct), pct, check.names = FALSE,
                     row.names = NULL)
write.csv(pct_df, file.path(outdir, "pct_expressing.csv"), row.names = FALSE)
cat("pct_expressing:", nrow(pct), "genes x", ncol(pct), "state__day cols\n")
memline("after pct_expressing")

# --- 3. FindAllMarkers on state ----------------------------------------
Idents(obj) <- "state"
mk <- FindAllMarkers(obj, only.pos = TRUE, min.pct = 0.1,
                     logfc.threshold = 0.25, verbose = FALSE)
write.csv(mk, file.path(outdir, "allmarkers.csv"), row.names = FALSE)
cat("allmarkers:", nrow(mk), "rows across", length(unique(mk$cluster)), "states\n")
memline("after markers")

# --- manifests ----------------------------------------------------------
manifest <- list(
  donor = donor, rds = cfg$rds, day_col = cfg$day_col,
  state_labels = joined$path,
  n_cells_kept = n_kept, n_cells_dropped = n_drop, n_cells_total = n_total,
  n_genes = nrow(obj),
  states = sort(unique(state)),
  days = sort(unique(day)),
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)
write_json(manifest, file.path(outdir, "extract_manifest.json"),
           auto_unbox = TRUE, pretty = TRUE)

audit <- list(
  donor = donor,
  n_total = n_total, n_kept = n_kept, n_dropped = n_drop,
  drop_pct = round(100 * n_drop / n_total, 2),
  drop_by_label = as.list(setNames(as.integer(drop_label_tab), names(drop_label_tab))),
  state_totals = as.list(setNames(as.integer(table(state)), names(table(state)))),
  state_by_day = lapply(sort(unique(state)), function(s) {
    rows <- cc[cc$state == s, ]
    list(state = s,
         total = sum(rows$n_cells),
         by_day = as.list(setNames(as.integer(rows$n_cells), as.character(rows$day))))
  }),
  generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)
write_json(audit, file.path(outdir, "state_audit.json"),
           auto_unbox = TRUE, pretty = TRUE)

cat("==== DONE", donor, "->", outdir, "====\n")
