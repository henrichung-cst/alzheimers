#!/usr/bin/env Rscript
# Pair-mode driver for the Levy-t5 spine.
#
# Calls Incytr::Cal_pairwise_grid (upstream `Incytr` R package,
# ~/Projects/work/incytr/) which vectorizes the hot loops (Cal_SigProb,
# Permutation_test, Pathway_evaluation) so pair-mode runs an order of
# magnitude faster than the pre-package code.
#
# Mouse DB layers ship as lazy-loaded package data — no Database/*.rda load
# needed.
#
# Usage (from any working directory):
#   Rscript alz/incytr/incytr_commandline.R <condition1> <condition2> <input_gene_list.csv>

suppressPackageStartupMessages({
  library(Incytr)
  library(Seurat)
  library(readr)
  library(DESeq2)
  library(purrr)
  library(dplyr)
  library(future)
  library(future.apply)
  library(arrow)
})

# Resolve repo root so all paths are absolute and the script runs from any cwd.
REPO_ROOT <- system("git rev-parse --show-toplevel", intern = TRUE)
INPUTS_DIR <- file.path(REPO_ROOT, "data", "derived", "incytr_inputs")
OUTPUT_DIR <- file.path(REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide")

# =====================================================================
# Parallelism config — READ THIS BEFORE TUNING ANY KNOB.
#
# Three multiplicative layers of parallelism exist inside one contrast:
#
#   total_cores_per_contrast = CHUNK_PARALLEL × NPAIR_WORKERS × NPERM_WORKERS
#
#   CHUNK_PARALLEL  — orchestrator subprocess fan-out (one Rscript / chunk)
#   NPAIR_WORKERS   — in-chunk pair-level mclapply workers (fork)
#   NPERM_WORKERS   — per-pair Permutation_test workers (fork over nboot)
#
# A fourth layer (contrasts) is handled by the outer wrapper script
# (alz/incytr/run_pair_mode.sh). If that wrapper ever runs N contrasts at once,
# the box-wide ceiling becomes N × (CHUNK_PARALLEL × NPAIR_WORKERS × NPERM_WORKERS).
# Mirror this assertion there before adding contrast parallelism.
#
# MEMORY (not just cores) is the binding constraint on this 32 GB shared
# box. Observed per-contrast peak RSS spans 4–18 GB (ApTt is the heavy
# tail). Two heavy contrasts in parallel will OOM. The cores assertion
# below does not protect against OOM — it only prevents core
# oversubscription. The driver's subprocess-per-chunk design (see
# SUBPROCESS_CHUNKS below) is the existing memory-safety floor; do not
# replace it with in-process fork parallelism without re-measuring peak
# RSS on the worst contrast (ma_2mo_ApTt / ma_2mo_Ttau historically).
#
# Defaults: serial everywhere (1 × 1 × 1). Opt in deliberately.
# =====================================================================
N_PAIR_WORKERS <- as.integer(Sys.getenv("NPAIR_WORKERS", unset = "1"))
N_PERM_WORKERS <- as.integer(Sys.getenv("NPERM_WORKERS", unset = "1"))
CHUNK_PARALLEL <- as.integer(Sys.getenv("CHUNK_PARALLEL", unset = "1"))
PAIR_LIMIT <- as.integer(Sys.getenv("PAIR_LIMIT", unset = "0"))
# Subprocess-per-chunk mode (default ON). Each chunk runs in a fresh Rscript
# so OS reclaims the dense `cond_mats` allocations from upstream
# Permutation_test between chunks. Set SUBPROCESS_CHUNKS=0 to fall back to
# the in-process mclapply path (broken at scale: per-pair RSS climb leads
# to OOM by chunk 3-4 even with 1 worker).
# CHUNK_INDEX > 0 marks a worker invocation: process only that one chunk.
SUBPROCESS_CHUNKS <- as.integer(Sys.getenv("SUBPROCESS_CHUNKS", unset = "1"))
CHUNK_INDEX <- as.integer(Sys.getenv("CHUNK_INDEX", unset = "0"))
stopifnot(!is.na(N_PAIR_WORKERS), N_PAIR_WORKERS >= 1L)
stopifnot(!is.na(N_PERM_WORKERS), N_PERM_WORKERS >= 1L)
stopifnot(!is.na(CHUNK_PARALLEL), CHUNK_PARALLEL >= 1L)
stopifnot(!is.na(PAIR_LIMIT), PAIR_LIMIT >= 0L)
stopifnot(!is.na(SUBPROCESS_CHUNKS), SUBPROCESS_CHUNKS %in% c(0L, 1L))
stopifnot(!is.na(CHUNK_INDEX), CHUNK_INDEX >= 0L)

# Multiplicative core-budget guard. Delegates the arithmetic check to
# Incytr::assert_core_budget() (machine-agnostic; stops if product
# exceeds detectCores() - reserve). reserve=2 is the shared-infra
# politeness convention; bump locally if this box becomes dedicated.
#
# IMPORTANT: the helper validates CORES only. Per-contrast RSS on this
# 32 GB box has been observed at up to 18 GB (ApTt). Two heavy contrasts
# in parallel will OOM regardless of the core check. Any contrast-level
# wrapper above this script must enforce its own memory budget.
Incytr::assert_core_budget(
  c(chunks = CHUNK_PARALLEL,
    pairs  = N_PAIR_WORKERS,
    perms  = N_PERM_WORKERS),
  reserve = 2L
)
if (.Platform$OS.type == "unix" && parallelly::supportsMulticore()) {
  plan(multicore, workers = N_PAIR_WORKERS)
} else {
  plan(multisession, workers = N_PAIR_WORKERS)
}
options(future.globals.maxSize = 16 * 1024^3)

# Orchestrator short-circuit: if we are SUBPROCESS_CHUNKS=1 + CHUNK_INDEX=0, do
# NOT load Data.input / pr / ps / py / kldata. The orchestrator only spawns
# Rscript subprocesses and waits — every byte it loads is held resident for
# the entire run and stacks on top of the worker's peak RSS. Previously the
# orchestrator sat at 12+ GB while waiting, blowing the 30 GB ceiling once
# the worker hit its 13–15 GB peak on 2mo Ttau / ApTt.
args_early <- commandArgs(trailingOnly = TRUE)
if (length(args_early) >= 2L &&
    SUBPROCESS_CHUNKS == 1L && CHUNK_INDEX == 0L) {
  condition1 <- args_early[1]
  condition2 <- args_early[2]
  input_gene_path <- args_early[3]
  nboot <- as.integer(Sys.getenv("NBOOT", unset = "100"))
  stopifnot(!is.na(nboot), nboot >= 1L)
  N_CHUNK_MULT <- as.integer(Sys.getenv("N_CHUNK_MULT", unset = "8"))
  stopifnot(!is.na(N_CHUNK_MULT), N_CHUNK_MULT >= 1L)
  n_chunks <- N_PAIR_WORKERS * N_CHUNK_MULT
  shard_dir <- file.path(OUTPUT_DIR, ".shards",
                         paste0(condition1, "_", condition2))
  unlink(shard_dir, recursive = TRUE, force = TRUE)
  dir.create(shard_dir, recursive = TRUE, showWarnings = FALSE)
  rscript_bin <- file.path(R.home("bin"), "Rscript")
  driver_path <- normalizePath(sub("^--file=", "",
    grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
  # CHUNK_PARALLEL is set at the top of the file alongside NPAIR_WORKERS /
  # NPERM_WORKERS so the multiplicative core-budget assertion covers it.
  cat(sprintf("[pair-driver] orchestrator (lean): %d chunks, nboot=%d, parallel=%d\n",
              n_chunks, nboot, CHUNK_PARALLEL))
  t_loop <- proc.time()[["elapsed"]]
  spawn_chunk <- function(i) {
    cat(sprintf("[pair-driver] spawn chunk %d/%d\n", i, n_chunks))
    t_chunk <- proc.time()[["elapsed"]]
    rc <- system2(
      "/usr/bin/env",
      args = shQuote(c(
        sprintf("CHUNK_INDEX=%d", i),
        sprintf("NPAIR_WORKERS=%d", N_PAIR_WORKERS),
        sprintf("NPERM_WORKERS=%d", N_PERM_WORKERS),
        sprintf("NBOOT=%d", nboot),
        sprintf("PAIR_LIMIT=%d", PAIR_LIMIT),
        sprintf("N_CHUNK_MULT=%d", N_CHUNK_MULT),
        "SUBPROCESS_CHUNKS=0",
        rscript_bin,
        driver_path, condition1, condition2, input_gene_path
      )),
      wait = TRUE
    )
    cat(sprintf("[pair-driver] chunk %d/%d rc=%d in %.1f min\n",
                i, n_chunks, rc,
                (proc.time()[["elapsed"]] - t_chunk) / 60))
    expected <- file.path(shard_dir, sprintf("chunk_%03d.parquet", i))
    if (rc == 0L && file.exists(expected)) expected else NA_character_
  }
  if (CHUNK_PARALLEL == 1L) {
    shard_paths <- lapply(seq_len(n_chunks), spawn_chunk)
  } else {
    shard_paths <- parallel::mclapply(
      seq_len(n_chunks), spawn_chunk,
      mc.cores = CHUNK_PARALLEL, mc.preschedule = FALSE
    )
  }
  shard_paths <- vapply(shard_paths,
                        function(x) if (is.character(x) && length(x) == 1L) x else NA_character_,
                        character(1))
  shard_paths <- shard_paths[!is.na(shard_paths)]
  shard_paths <- shard_paths[file.exists(shard_paths)]
  cat(sprintf("[pair-driver] pair loop done in %.1f min (%d shards)\n",
              (proc.time()[["elapsed"]] - t_loop) / 60, length(shard_paths)))
  if (length(shard_paths) == 0L) stop("[pair-driver] no shards produced", call. = FALSE)
  out_path <- file.path(OUTPUT_DIR, paste0(condition1, "_", condition2, "_incytr_output.parquet"))
  ds <- arrow::open_dataset(shard_paths, format = "parquet")
  arrow::write_parquet(dplyr::collect(ds), out_path, compression = "zstd")
  sz <- file.info(out_path)$size
  cat(sprintf("[pair-driver] wrote %s (%.1f MB)\n", out_path, sz / 1e6))
  unlink(shard_dir, recursive = TRUE)
  quit(status = 0L)
}

rss_mb <- function() {
  status <- "/proc/self/status"
  if (!file.exists(status)) return(NA_real_)
  rss <- grep("^VmRSS:", readLines(status, warn = FALSE), value = TRUE)
  if (length(rss) == 0L) return(NA_real_)
  as.numeric(sub("^VmRSS:\\s+([0-9]+)\\s+kB$", "\\1", rss)) / 1024
}

log_stage <- function(label, t0 = NULL) {
  elapsed <- if (is.null(t0)) NA_real_ else proc.time()[["elapsed"]] - t0
  mem <- rss_mb()
  elapsed_txt <- if (is.na(elapsed)) "NA" else sprintf("%.2f", elapsed)
  rss_txt <- if (is.na(mem)) "NA" else sprintf("%.1f", mem)
  cat(sprintf("[pair-driver] %s elapsed_s=%s rss_mb=%s\n",
              label, elapsed_txt, rss_txt))
}

args = commandArgs(trailingOnly = TRUE)

if (length(args)<2) {
  stop("Please pass 2 condition names as they appear in seurat_obj.rds and the path to an input gene list with columns: gene and cluster", call.=FALSE)
}

condition1 = args[1]
condition2 = args[2]
input_gene_path = args[3]

#other parameters that could be changed
# Override nboot via NBOOT env var (e.g. NBOOT=2 for a smoke run).
nboot <- as.integer(Sys.getenv("NBOOT", unset = "100"))
stopifnot(!is.na(nboot), nboot >= 1L)
cat("[pair-driver] nboot =", nboot, "\n")
cat("[pair-driver] pair_workers =", N_PAIR_WORKERS,
    "perm_workers =", N_PERM_WORKERS,
    "pair_limit =", PAIR_LIMIT, "\n")
cutoff_SigProb = 0.25
cutoff_PDS = 0.1

# exFINDER mouse layers ship as package data in the upstream Incytr package
# (../incytr/data/DB_Layer{1,2,3}_mouse_filtered.rda). Accessing the namespaced
# symbols triggers lazy load.
DB.M <- list(
  Incytr::DB_Layer1_mouse_filtered,
  Incytr::DB_Layer2_mouse_filtered,
  Incytr::DB_Layer3_mouse_filtered
)

#Load scrnaseq object
Data.input <- readRDS(file.path(INPUTS_DIR, "incytr_obj.rds"))

Data.input@meta.data$Type <- Data.input@active.ident
Data.input@meta.data$condition <- as.factor(Data.input@meta.data$Group)

#LOAD INPUT PROTEOMICS AND KINOMICS DATA
pr <- read_csv(file.path(INPUTS_DIR, "pr_yuyu_deconvoluted.csv"))
ps <- read_csv(file.path(INPUTS_DIR, "ps_yuyu_deconvoluted.csv"))
py <- read_csv(file.path(INPUTS_DIR, "py_yuyu_deconvoluted.csv"))

#make df for each condition/modality
#for each table: select the columns for that condition, add back the gene symbol, summarize
#any redundant rows (must have only one row per gene symbol)

pr_1 <- select(pr,contains(condition1))
colnames(pr_1) <- paste0(sub(paste0(condition1,".*_"), "", colnames(pr_1)), "_pr")
pr_1$gene_symbol <- pr$`Gene Symbol`

pr_1 <- pr_1 %>%
  group_by(gene_symbol) %>%
  summarise_all(mean,na.rm=T)

pr_2 <- select(pr,contains(condition2))
colnames(pr_2) <- paste0(sub(paste0(condition2,".*_"), "", colnames(pr_2)), "_pr")
pr_2$gene_symbol <- pr$`Gene Symbol`

pr_2 <- pr_2 %>%
  group_by(gene_symbol) %>%
  summarise_all(mean,na.rm=T)

ps_1 <- select(ps,contains(condition1))
colnames(ps_1) <- paste0(sub(paste0(condition1,".*_"), "", colnames(ps_1)), "_ps")
ps_1$gene_symbol <- ps$gene_symbol

ps_1 <- ps_1 %>%
  group_by(gene_symbol) %>%
  summarise_all(mean,na.rm=T)

ps_2 <- select(ps,contains(condition2))
colnames(ps_2) <- paste0(sub(paste0(condition2,".*_"), "", colnames(ps_2)), "_ps")
ps_2$gene_symbol <- ps$gene_symbol

ps_2 <- ps_2 %>%
  group_by(gene_symbol) %>%
  summarise_all(mean,na.rm=T)

py_1 <- select(py,contains(condition1))
colnames(py_1) <- paste0(sub(paste0(condition1,".*_"), "", colnames(py_1)), "_py")
py_1$gene_symbol <- py$gene_symbol

py_1 <- py_1 %>%
  group_by(gene_symbol) %>%
  summarise_all(mean,na.rm=T)

py_2 <- select(py,contains(condition2))
colnames(py_2) <- paste0(sub(paste0(condition2,".*_"), "", colnames(py_2)), "_py")
py_2$gene_symbol <- py$gene_symbol

py_2 <- py_2 %>%
  group_by(gene_symbol) %>%
  summarise_all(mean,na.rm=T)

#read kldata
kldata <- read_csv(file.path(INPUTS_DIR, "kldata.csv"))
kldata <- kldata[, c("gene", "site_pos", "motif.geneName")]

#read input gene list
input_gene_list <- read_csv(input_gene_path)

#define entire Incytr workflow: see Incytr source code for descriptions of individual functions
runIncytr <- function(Sender.group, Receiver.group) {

  #select differentially expressed genes from input gene list
  DG.Sender <- unique(input_gene_list$gene[input_gene_list$cluster == Sender.group])
  DG.Receiver <- unique(input_gene_list$gene[input_gene_list$cluster ==Receiver.group])

  #add genes differentially expressed in deconvoluted proteomics data
  fc_total <- data.frame(gene_symbol = pr_1[ , c("gene_symbol")] ,
                         Sender.1 = pr_1[ , paste0(Sender.group,"_pr")] ,
                         Sender.2 = pr_2[ , paste0(Sender.group,"_pr")] ,
                         Receiver.1 = pr_1[ , paste0(Receiver.group,"_pr")] ,
                         Receiver.2 = pr_2[ , paste0(Receiver.group,"_pr")]
  )

  names(fc_total) = c("gene_symbol", "Sender.1", "Sender.2", "Receiver.1", "Receiver.2")

  fold_magnitude <- function(alt, ref, cap = 10) {
    score <- rep(NA_real_, length(alt))

    valid <- is.finite(alt) & is.finite(ref)
    both_positive <- valid & alt > 0 & ref > 0
    fc <- alt[both_positive] / ref[both_positive]
    score[both_positive] <- pmin(pmax(fc, 1 / fc), cap)

    one_sided <- valid & ((alt > 0 & ref == 0) | (alt == 0 & ref > 0))
    score[one_sided] <- cap

    score
  }

  top_omics_genes <- function(df, score_col, n = 500L) {
    score <- df[[score_col]]
    keep <- !is.na(score) & is.finite(score)
    if (!any(keep)) return(character())
    genes <- df$gene_symbol[keep]
    score <- score[keep]
    genes[order(score, decreasing = TRUE)][seq_len(min(n, length(genes)))]
  }

  fc_total$sender_fc <- fc_total$Sender.2 / fc_total$Sender.1
  fc_total$receiver_fc <- fc_total$Receiver.2 / fc_total$Receiver.1

  fc_total$sender_fc2 <- fold_magnitude(fc_total$Sender.2, fc_total$Sender.1)
  fc_total$receiver_fc2 <- fold_magnitude(fc_total$Receiver.2, fc_total$Receiver.1)

  DG.Sender_total <- top_omics_genes(fc_total, "sender_fc2", 500L)
  DG.Sender_total <- DG.Sender_total[!is.na(DG.Sender_total)]
  DG.Sender_total <- intersect(DG.Sender_total, rownames(Data.input))

  DG.Receiver_total <- top_omics_genes(fc_total, "receiver_fc2", 500L)
  DG.Receiver_total <- DG.Receiver_total[!is.na(DG.Receiver_total)]
  DG.Receiver_total <- intersect(DG.Receiver_total, rownames(Data.input))

  DG.Sender <- unique(c(DG.Sender, DG.Sender_total))
  DG.Receiver <- unique(c(DG.Receiver, DG.Receiver_total))

  DG.Sender <- intersect(DG.Sender, rownames(Data.input))
  DG.Receiver <- intersect(DG.Receiver, rownames(Data.input))

  Xobject <- create_Incytr(object = Data.input@assays$originalexp@data,
                           meta = Data.input@meta.data,
                           sender = Sender.group,
                           receiver = Receiver.group,
                           group.by = "Type",
                           condition = c(condition1, condition2),
                           assay = NULL,
                           do.sparse = T)

  Xobject <- pathway_inference(Xobject,
                               DB = DB.M,
                               gene.use_Sender = DG.Sender,
                               gene.use_Receiver = DG.Receiver,
                               ligand = NULL,
                               receptor = NULL,
                               em = NULL,
                               target = NULL)

  Xobject <- Expr_bygroup(Xobject, mean_method = "mean")

  Xobject <- Cal_SigProb(Xobject, K = 0.5, N = 2, cutoff_SigProb = cutoff_SigProb,
                         correction = 0.001, q = NULL)

  Xobject <- Integr_multiomics(Xobject,
                               pr.data_condition1 = pr_1,
                               pr.data_condition2 = pr_2,
                               pr.correction = 0.001,
                               pr.q = NULL,
                               ps.data_condition1 = ps_1,
                               ps.data_condition2 = ps_2,
                               ps.correction = 0.001,
                               ps.q = NULL,
                               py.data_condition1 = py_1,
                               py.data_condition2 = py_2,
                               py.correction = 0.001,
                               py.q = NULL)

  Xobject <- Pathway_evaluation(Xobject, score.weight = NULL, k_logi = 2, style = NULL, abs.value = NULL)

  Xobject <- Integr_kinasedata(Xobject,
                               kldata = kldata,
                               mean_method = "mean",
                               cell_group = levels(Xobject@meta$Type),
                               fold_threshold = 10)

  Xobject <- Cal_PDS(Xobject, KPDS.weight = 0.5,
                     cutoff_PDS = cutoff_PDS)

  # IMPORTANT: do NOT drop the explicit `n.cores = N_PERM_WORKERS`.
  # The upstream Incytr package default is `min(8L, detectCores() - 2L)`
  # (R/analysis.R post Item 9c). Letting the default through here would
  # break the multiplicative core budget guarded at the top of this file —
  # at CHUNK_PARALLEL=8 × NPAIR=3 × default 8 = 192 fork requests per
  # contrast, on top of any contrast-level wrapper parallelism.
  Xobject <- Permutation_test(Xobject,
                              K = 0.5,
                              N = 2,
                              nboot = nboot,
                              seed.use = 1L,
                              mean_method = "mean",
                              n.cores = N_PERM_WORKERS)

  # Populate per-node single-cell log2FC (Ligand/Receptor/EM/Target_sclog2FC).
  # Without this call object@sc_FC stays empty and Export_results silently
  # skips the sc fold-change block — the viewer would see NULL for all 4 sc
  # cells per row. Cheap (one Cal_foldchange on the union of pathway genes,
  # reusing the expr.bygroup already populated by Expr_bygroup above).
  Xobject <- Cal_scFC(Xobject)

  output <- Export_results(Xobject, indicator = TRUE)

  # Per-node DEG/prG labels (factorial convention: "DEG" if in input gene list,
  # "prG" if only in top-500 proteomics fold-magnitude). Computed here because
  # pair-mode Export_results does not track per-node provenance — the upstream
  # factorial path emits these via assign_path_labels().
  sender_deg <- intersect(
    unique(input_gene_list$gene[input_gene_list$cluster == Sender.group]),
    rownames(Data.input)
  )
  receiver_deg <- intersect(
    unique(input_gene_list$gene[input_gene_list$cluster == Receiver.group]),
    rownames(Data.input)
  )
  sender_set <- union(sender_deg, DG.Sender_total)
  receiver_set <- union(receiver_deg, DG.Receiver_total)
  sender_lbl <- setNames(rep("prG", length(sender_set)), sender_set)
  sender_lbl[sender_deg] <- "DEG"
  receiver_lbl <- setNames(rep("prG", length(receiver_set)), receiver_set)
  receiver_lbl[receiver_deg] <- "DEG"
  output$Ligand.label   <- unname(sender_lbl[output$Ligand])
  output$Receptor.label <- unname(receiver_lbl[output$Receptor])
  output$EM.label       <- unname(receiver_lbl[output$EM])
  output$Target.label   <- unname(receiver_lbl[output$Target])

  return(output)
}

safely_runIncytr <- safely(runIncytr, otherwise = data.frame())

groups <- as.character(unique(Data.input@meta.data$Type))
pairs  <- expand.grid(Sender = groups, Receiver = groups,
                      stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
if (PAIR_LIMIT > 0L && PAIR_LIMIT < nrow(pairs)) {
  pairs <- pairs[seq_len(PAIR_LIMIT), , drop = FALSE]
  cat("[pair-driver] PAIR_LIMIT active; running first", nrow(pairs), "pairs\n")
}

use_fork <- .Platform$OS.type == "unix" && parallelly::supportsMulticore()

# Chunked dispatch: split the 961 pairs into ~N_PAIR_WORKERS × 4 chunks and
# let mclapply/future_lapply dynamically pull chunks. Each worker accumulates
# pair results in-memory, slims columns, and writes one parquet shard per
# chunk. Per-pair sharding (one parquet per pair) was tried in v9 and added
# ~2 min of I/O overhead vs this chunked layout; LPT pair-ordering was also
# tried and offered no speedup because the cell-count work proxy spans only
# ~125× while actual path count spans ~1000×.
# Chunk count tuned to bound per-subprocess RSS. The dense `cond_mats` from
# upstream Permutation_test (genes × all-cells-per-condition) is allocated
# per pair and not reliably reclaimed by R's gc within a process. With 1
# worker × 8 chunks ≈ 45 pairs/chunk, each chunk subprocess peaks well
# under the 30 GB system ceiling on the heaviest contrasts (2mo Ttau /
# ApTt) where in-process 90-pair chunks reach 18+ GB by chunk 2.
N_CHUNK_MULT <- as.integer(Sys.getenv("N_CHUNK_MULT", unset = "8"))
stopifnot(!is.na(N_CHUNK_MULT), N_CHUNK_MULT >= 1L)
n_chunks <- N_PAIR_WORKERS * N_CHUNK_MULT
chunk_ids <- ((seq_len(nrow(pairs)) - 1L) %% n_chunks) + 1L
chunks <- split(seq_len(nrow(pairs)), chunk_ids)

mode_label <- if (CHUNK_INDEX > 0L) {
  sprintf("worker chunk=%d/%d", CHUNK_INDEX, length(chunks))
} else if (SUBPROCESS_CHUNKS == 1L) {
  sprintf("orchestrator (subprocess-per-chunk, %d chunks)", length(chunks))
} else if (use_fork) {
  "in-process multicore/fork"
} else {
  "in-process multisession"
}
cat(sprintf("[pair-driver] %d pairs / %s\n", nrow(pairs), mode_label))
t_loop <- proc.time()[["elapsed"]]

# Columns to keep in the final output. The drop pattern removes the redundant
# per-(node, omics) `_aFC` mirrors (raw fold-change, redundant with log2FC)
# and SiK sub-score / EI breakdowns. We keep `<Node>_<omics>_log2FC` because
# the unified viewer surfaces them per pathway row. The trim is applied
# INSIDE each worker so the per-chunk accumulator and IPC stay small.
drop_pat <- "^(Ligand|Receptor|EM|Target)_(pr|ps|py)_aFC$|^SiK_(R|EM|T)_of_(EM|T|R)(_EI_.*)?$"
num_pat  <- "^(SigProb|p_value|SiK|log2FC|aFC|PDS|TPDS|PPDS|PhPDS|Ack_score|KGG_score|Rme1_score|multimodel_score|pr_|ps_|py_)"

shard_dir <- file.path(OUTPUT_DIR, ".shards",
                       paste0(condition1, "_", condition2))
# Worker invocations must NOT wipe sibling chunks. Only the orchestrator
# (or in-process master) clears the shard dir on entry.
if (CHUNK_INDEX == 0L) {
  unlink(shard_dir, recursive = TRUE, force = TRUE)
}
dir.create(shard_dir, recursive = TRUE, showWarnings = FALSE)

chunk_fn <- function(chunk_idx) {
  pair_ks <- chunks[[chunk_idx]]
  acc <- vector("list", length(pair_ks))
  for (i in seq_along(pair_ks)) {
    k <- pair_ks[i]
    res <- safely_runIncytr(pairs$Sender[k], pairs$Receiver[k])$result
    if (is.null(res) || nrow(res) == 0) next
    res$Sender   <- pairs$Sender[k]
    res$Receiver <- pairs$Receiver[k]
    res <- res[, !grepl(drop_pat, colnames(res)), drop = FALSE]
    acc[[i]] <- res
  }
  acc <- acc[!vapply(acc, is.null, logical(1))]
  if (length(acc) == 0L) return(NULL)
  combined <- data.table::rbindlist(acc, use.names = TRUE, fill = TRUE)
  for (col in grep(num_pat, names(combined), value = TRUE)) {
    if (is.character(combined[[col]])) {
      data.table::set(combined, j = col,
                      value = suppressWarnings(as.numeric(combined[[col]])))
    }
  }
  shard_path <- file.path(shard_dir, sprintf("chunk_%03d.parquet", chunk_idx))
  arrow::write_parquet(as.data.frame(combined), shard_path, compression = "zstd")
  shard_path
}

if (CHUNK_INDEX > 0L) {
  # Worker mode: process exactly one chunk and exit. The OS reclaims our
  # heap when the process exits, defeating R's heap-fragmentation pin on
  # the dense `cond_mats` allocations from upstream Permutation_test.
  if (CHUNK_INDEX > length(chunks)) {
    stop(sprintf("[pair-driver] CHUNK_INDEX=%d exceeds n_chunks=%d",
                 CHUNK_INDEX, length(chunks)), call. = FALSE)
  }
  t_chunk <- proc.time()[["elapsed"]]
  shard <- chunk_fn(CHUNK_INDEX)
  cat(sprintf("[pair-driver] worker done in %.1f min (shard=%s)\n",
              (proc.time()[["elapsed"]] - t_chunk) / 60,
              if (is.null(shard)) "<empty>" else shard))
  quit(status = 0L)
} else if (SUBPROCESS_CHUNKS == 1L) {
  # Orchestrator mode: spawn `Rscript $0 ...` once per chunk, sequentially,
  # each with CHUNK_INDEX=i. Sequential (not parallel) because the workers
  # individually peak at ~13-15 GB on the heaviest contrasts (2mo Ttau /
  # ApTt) — two concurrent workers exceed the 30 GB system ceiling.
  rscript_bin <- file.path(R.home("bin"), "Rscript")
  driver_path <- normalizePath(sub("^--file=", "",
    grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1]))
  shard_paths <- vector("list", length(chunks))
  for (i in seq_along(chunks)) {
    cat(sprintf("[pair-driver] spawn chunk %d/%d\n", i, length(chunks)))
    t_chunk <- proc.time()[["elapsed"]]
    # Use /usr/bin/env to inherit the parent environment and override only
    # the keys we care about. system2(..., env=) replaces the env entirely
    # and would lose PATH / R_HOME / R_LIBS — Rscript would fail to start.
    rc <- system2(
      "/usr/bin/env",
      args = shQuote(c(
        sprintf("CHUNK_INDEX=%d", i),
        sprintf("NPAIR_WORKERS=%d", N_PAIR_WORKERS),
        sprintf("NPERM_WORKERS=%d", N_PERM_WORKERS),
        sprintf("NBOOT=%d", nboot),
        sprintf("PAIR_LIMIT=%d", PAIR_LIMIT),
        sprintf("N_CHUNK_MULT=%d", N_CHUNK_MULT),
        "SUBPROCESS_CHUNKS=0",
        rscript_bin,
        driver_path, condition1, condition2, input_gene_path
      )),
      wait = TRUE
    )
    cat(sprintf("[pair-driver] chunk %d/%d rc=%d in %.1f min\n",
                i, length(chunks), rc,
                (proc.time()[["elapsed"]] - t_chunk) / 60))
    expected <- file.path(shard_dir, sprintf("chunk_%03d.parquet", i))
    if (rc == 0L && file.exists(expected)) {
      shard_paths[[i]] <- expected
    } else {
      warning(sprintf("[pair-driver] chunk %d failed (rc=%d, shard=%s)",
                      i, rc, if (file.exists(expected)) "present" else "missing"),
              call. = FALSE)
      shard_paths[[i]] <- NA_character_
    }
  }
} else if (use_fork) {
  shard_paths <- parallel::mclapply(
    seq_along(chunks), chunk_fn,
    mc.cores = N_PAIR_WORKERS, mc.preschedule = FALSE
  )
} else {
  shard_paths <- future_lapply(
    seq_along(chunks), chunk_fn,
    future.seed = 1L,
    future.packages = c("Incytr", "Seurat", "dplyr", "purrr", "data.table", "arrow")
  )
}
# mclapply with mc.preschedule=FALSE returns a try-error STRING (not NULL)
# when fork() itself fails with EAGAIN under memory/cgroup pressure. That
# string would leak into open_dataset() below as a bogus path. Filter to
# only entries that exist on disk so a partial chunk failure cannot
# destroy the whole comparison's output.
raw_results <- shard_paths
valid_by_chunk <- vapply(raw_results, function(x) {
  any(is.character(x) & file.exists(x))
}, logical(1))
shard_paths <- vapply(shard_paths, function(x) if (is.character(x) && length(x) == 1L) x else NA_character_,
                      character(1))
shard_paths <- shard_paths[!is.na(shard_paths)]
shard_paths <- shard_paths[file.exists(shard_paths)]
n_lost <- sum(!valid_by_chunk)
if (n_lost > 0L) {
  warning(sprintf("[pair-driver] %d/%d chunks lost to fork/exec failure (continuing with %d good shards)",
                  n_lost, length(raw_results), length(shard_paths)), call. = FALSE)
}
cat(sprintf("[pair-driver] pair loop done in %.1f min (%d shards)\n",
            (proc.time()[["elapsed"]] - t_loop) / 60, length(shard_paths)))
if (length(shard_paths) == 0L) stop("[pair-driver] no shards produced", call. = FALSE)

# Master-side concat via streaming arrow: open the per-chunk parquets as a
# single dataset and write to one parquet without ever holding all 6M rows
# in memory.
out_path <- file.path(OUTPUT_DIR, paste0(condition1, "_", condition2, "_incytr_output.parquet"))
ds <- arrow::open_dataset(shard_paths, format = "parquet")
arrow::write_parquet(dplyr::collect(ds), out_path, compression = "zstd")
sz <- file.info(out_path)$size
cat(sprintf("[pair-driver] wrote %s (%.1f MB)\n", out_path, sz / 1e6))
unlink(shard_dir, recursive = TRUE)
