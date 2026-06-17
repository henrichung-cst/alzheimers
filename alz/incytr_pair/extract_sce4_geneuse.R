#!/usr/bin/env Rscript
# Extract sce4's effective PER-PAIR, per-role gene.use from its Allpathway
# pairwise pathway table (the `.rds`).
#
# WHY PER-PAIR, NOT PER-CLUSTER: the Incytr engine gates ALL receiver positions
# (Receptor, EM, Target) by ONE flat `gene.use_Receiver` (grid.R / analysis.R
# `pathway_inference`). sce4's gene.use is not recoverable as a per-cluster INPUT
# set from its artifacts — the `.rds` records the GATED output (Allpathway), whose
# nodes are a per-pair consequence. Reconstructing a per-cluster receiver set by
# unioning a cluster's appearing nodes across all its pairs is strictly broader
# than any single pair used (Cholinergic: 1868 union vs 1232 in the Micro→Cholin
# pair), and feeding that flat union lets the engine recombine cross-pair nodes
# into chains sce4 never enumerated. The gate does NOT remove them — cell-sparse
# receivers (Cholinergic = 1 cell) give every chain extreme SigProb/PDS, so our
# gated set ballooned to 2952 vs sce4's 1283. Proven: feeding the per-pair node
# set reproduces the pair EXACTLY (1283, 0 extra / 0 missing — bench/perf/
# sce4_identity_test.R, archive/sce4_reproduction_2026-06-08/README.md §6).
#
# So reconstruct, per (sender, receiver) PAIR, directly from that pair's element
# of the `.rds` (a data.frame of every gated chain the pair produced):
#
#     gene.use_Sender(s,r)   = { Ligand               : in rds[[s_to_r]] }
#     gene.use_Receiver(s,r) = { Receptor, EM, Target : in rds[[s_to_r]] }
#
# This is exactly the node set sce4's engine produced for that pair, so feeding it
# back reproduces sce4's Allpathway pair-for-pair (the clean pre-cap row-count
# invariant the user requires), and the downstream gate + top-300 cap then yield
# sce4's Top300. Pairs absent from the `.rds` produced no gated chains; they get
# an empty gene.use and emit 0 paths, matching sce4.
#
# SOURCE = the Allpathway `.rds` (gated pairwise table, list keyed
# "Sender_to_Receiver"). The post-cap Top300 table is NOT a valid source (it is a
# strict subset whose re-fed nodes recombine into spurious chains).
#
# EMITS one CSV per contrast: data/incytr_frozen/sce4_geneuse/<c1>_<c2>.csv with
# columns [sender, receiver, gene, role, label] — sender/receiver in OUR spine
# `Type` vocabulary (separator-insensitive crosswalk onto the driver's bare Type
# labels), role in {S, R}, label from sce4's own *.label columns (DEG/prG,
# DEG-priority per (pair, gene, role)).
#
# Consumed by incytr_commandline.R for the AD cohort (REPLACES the DEG∪prG
# derivation there). The T-cell cohort has no sce4 reference and keeps deriving.
# Record: archive/sce4_reproduction_2026-06-08/README.md §6.7
suppressPackageStartupMessages({
  library(data.table)
})

REPO <- normalizePath(file.path(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE)[1])), "..", ".."))
if (is.na(REPO) || !nzchar(REPO)) REPO <- normalizePath(getwd())
REF_DIR    <- file.path(REPO, "data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1")
OUT_DIR    <- file.path(REPO, "data/incytr_frozen/sce4_geneuse")
ALLMARKERS <- file.path(REPO, "data/derived/incytr_inputs/allmarkers.csv")

# Separator-insensitive cluster key. sce4 uses dots; our spine mixes hyphens AND
# spaces (e.g. "Excitatory principal neurons in the hippocampal dentate gyrus"),
# so a dot->hyphen crosswalk is insufficient — collapse every separator class to
# a single space and lowercase.
norm_key <- function(s) tolower(trimws(gsub("[._ -]+", " ", s)))

build_crosswalk <- function() {
  # norm(label) -> our bare Type label, from the driver's own spine. allmarkers
  # idents are "<Type>_ma_<age>_<geno>"; strip that suffix to recover the bare
  # Type labels (== unique(Data.input@meta.data$Type)). Fail loud on a norm()
  # collision so every sce4 cluster resolves to exactly one spine cluster.
  cl <- unique(fread(ALLMARKERS, select = "cluster")$cluster)
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

extract_one <- function(rds_path, xwalk) {
  x <- readRDS(rds_path)
  cols <- c("Ligand", "Receptor", "EM", "Target",
            "Ligand.label", "Receptor.label", "EM.label", "Target.label",
            "Sender.group", "Receiver.group")
  big <- rbindlist(lapply(x, function(e) {
    d <- as.data.table(e)
    d[, ..cols]
  }), use.names = TRUE)

  # Role S: ligands of the sender, keyed by the (sender, receiver) PAIR.
  # Role R: receptor/EM/target of the receiver, same pair key.
  sender <- big[, .(sender = Sender.group, receiver = Receiver.group,
                    gene = Ligand, label = Ligand.label, role = "S")]
  recv <- rbindlist(list(
    big[, .(sender = Sender.group, receiver = Receiver.group, gene = Receptor, label = Receptor.label)],
    big[, .(sender = Sender.group, receiver = Receiver.group, gene = EM,       label = EM.label)],
    big[, .(sender = Sender.group, receiver = Receiver.group, gene = Target,   label = Target.label)]
  ), use.names = TRUE)[, role := "R"]

  gu <- rbindlist(list(sender, recv), use.names = TRUE)
  gu <- gu[!is.na(gene) & gene != "" & gene != "NA"]
  where <- basename(dirname(rds_path))
  gu[, sender   := map_clusters(sender,   xwalk, where)]
  gu[, receiver := map_clusters(receiver, xwalk, where)]

  # One label per (pair, gene, role), DEG-priority on conflict (matches
  # incytr_commandline.R::label_node).
  gu[, is_deg := as.integer(label == "DEG")]
  gu <- gu[, .(label = if (any(is_deg == 1L)) "DEG" else "prG"),
           by = .(sender, receiver, gene, role)]
  setorder(gu, sender, receiver, role, gene)
  gu[]
}

main <- function() {
  dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
  xwalk <- build_crosswalk()
  cat(sprintf("[extract] cluster crosswalk built (%d spine labels)\n", length(xwalk)))

  dirs <- sort(Sys.glob(file.path(REF_DIR, "DEG_PRG_ma_*_10302025")))
  stopifnot(length(dirs) == 9L)

  missing <- character(0)
  for (d in dirs) {
    token <- sub("^DEG_PRG_(ma_.*)_10302025$", "\\1", basename(d))  # ma_2mo_AppP_WTyp
    stopifnot(grepl("_WTyp$", token))
    c1  <- sub("_WTyp$", "", token)                       # ma_2mo_AppP
    age <- sub("^ma_([0-9]+mo)_.*$", "\\1", c1)           # 2mo
    c2  <- sprintf("ma_%s_WTyp", age)                     # ma_2mo_WTyp

    rds <- file.path(d, "sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds")
    if (!file.exists(rds)) {
      # The faithful source is the PRE-CAP rds. It is on box for 5/9 contrasts;
      # the rest have only partial per-pair CSVs (a strict subset of the 418
      # pairs) or the post-cap Top300 — neither reproduces pre-cap counts. Surface
      # and stop; drop the Drive's pre-cap rds here. Do not fabricate from Top300.
      missing <- c(missing, token)
      cat(sprintf("[extract] %-22s -- MISSING pre-cap rds, skipped (place the Drive file at %s)\n",
                  token, rds))
      next
    }

    gu <- extract_one(rds, xwalk)

    # sanity: the genotype's human transgene present as a node. AppP carries App
    # (+Psen1); Ttau carries Mapt; ApTt carries both.
    geno <- sub("^ma_[0-9]+mo_", "", c1)
    expected <- switch(geno, AppP = "App", Ttau = "Mapt", ApTt = c("App", "Mapt"))
    if (!any(expected %in% gu$gene)) {
      stop(sprintf("[extract] %s: no %s transgene (%s) in node set",
                   token, geno, paste(expected, collapse = "/")), call. = FALSE)
    }

    out_csv <- file.path(OUT_DIR, sprintf("%s_%s.csv", c1, c2))
    fwrite(gu, out_csv)
    npair <- uniqueN(gu[, .(sender, receiver)])
    nS <- gu[role == "S", uniqueN(gene)]
    nR <- gu[role == "R", uniqueN(gene)]
    cat(sprintf("[extract] %-22s -> %s  (%d pairs; distinct ligand-genes=%d receiver-genes=%d; %d rows)\n",
                token, basename(out_csv), npair, nS, nR, nrow(gu)))
  }

  n_done <- 9L - length(missing)
  cat(sprintf("[extract] %d/9 contrasts written to %s\n", n_done, OUT_DIR))
  if (length(missing)) {
    cat("[extract] INCOMPLETE — sce4's pre-cap rds is not on box for:\n")
    for (token in missing) cat(sprintf("          %s\n", token))
    quit(status = 1L)
  }
}

main()
