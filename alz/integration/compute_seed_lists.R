#!/usr/bin/env Rscript
# Compute per-cell-type DEGs (pseudobulk + DESeq2) and bulk DEPs (limma)
# from the Incytr factorial fixture, emitting deg_lists.json + prg_list.csv
# for upstream construct_factorial_paths label assignment.
#
# Strict-partition (DEG-first precedence; prG = DEP \ DEG) happens upstream
# at label-assignment time. This script emits the *full* DEP set as prg_list;
# upstream removes overlap with deg_lists.
#
# See docs/integrations/seed_list_labels.md for the design.

suppressPackageStartupMessages({
  library(Matrix)
  library(DESeq2)
  library(limma)
  library(jsonlite)
})

parse_args <- function(argv) {
  args <- list(
    input_dir = "data/incytr_factorial_inputs",
    out_dir = NULL,
    pvalue = 0.05,
    log2fc = 0.5,
    min_cells = 10,
    min_reps = 1
  )
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    val <- if (i < length(argv)) argv[[i + 1L]] else NA_character_
    if (key %in% c("--input-dir", "--out-dir")) {
      args[[gsub("-", "_", sub("^--", "", key))]] <- val
      i <- i + 2L
    } else if (key %in% c("--pvalue", "--log2fc", "--min-cells", "--min-reps")) {
      args[[gsub("-", "_", sub("^--", "", key))]] <- as.numeric(val)
      i <- i + 2L
    } else {
      stop("Unknown argument: ", key)
    }
  }
  if (is.null(args$out_dir)) args$out_dir <- args$input_dir
  args
}

SENTINEL_GENES <- c("Apoe", "App", "Trem2", "Bin1", "Clu")

make_group_key <- function(genotype, timepoint) {
  paste(genotype, timepoint, sep = "_")
}

# Read MANIFEST.contrast_conditions and marginalize to genotype-only pairs
# (alt_genotype, ref_genotype). The factorial design has 12 (genotype × timepoint)
# cells with n=1-2 each — too thin for per-cell DESeq2 fitting. Pooling across
# timepoints gives ~3-4 reps per genotype, which is enough for stable DE calling.
# The marginal seed list is applied uniformly to all 9 factorial contrasts at
# upstream label-assignment time (labels are per-node, not per-contrast).
load_contrast_pairs <- function(input_dir) {
  manifest_path <- file.path(input_dir, "MANIFEST.json")
  if (!file.exists(manifest_path)) {
    stop("MANIFEST.json not found in ", input_dir,
         "; cannot derive contrast pairs. Re-run export-factorial-inputs.")
  }
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
  cc <- manifest$contrast_conditions
  if (is.null(cc) || length(cc) == 0) {
    stop("MANIFEST.json has no contrast_conditions; refusing to fall back ",
         "to a hardcoded list. Regenerate the fixture.")
  }
  # Extract genotype tokens from condition labels like "WTyp_2mo" / "AppP_4mo".
  # Drop the trailing "_<age>" segment.
  extract_geno <- function(label) sub("_[^_]+$", "", label)
  geno_pairs <- lapply(cc, function(pair) {
    c(extract_geno(pair[[2]]), extract_geno(pair[[1]]))
  })
  # Deduplicate; name each pair "<alt>_vs_<ref>".
  uniq <- unique(geno_pairs)
  names(uniq) <- vapply(uniq, function(p) paste0(p[1], "_vs_", p[2]),
                        character(1))
  uniq
}

load_pseudobulk <- function(input_dir) {
  mtx <- Matrix::readMM(file.path(input_dir, "pseudobulk_counts.mtx"))
  genes <- read.csv(file.path(input_dir, "pseudobulk_genes.csv"),
                    stringsAsFactors = FALSE)$gene
  samples <- read.csv(file.path(input_dir, "pseudobulk_pseudosamples.csv"),
                      stringsAsFactors = FALSE)$pseudosample
  meta <- read.csv(file.path(input_dir, "pseudobulk_metadata.csv"),
                   stringsAsFactors = FALSE, check.names = FALSE)
  rownames(mtx) <- genes
  colnames(mtx) <- samples
  if (!identical(samples, meta$pseudosample)) {
    stop("pseudobulk_metadata.csv pseudosample order does not match ",
         "pseudobulk_pseudosamples.csv")
  }
  list(counts = mtx, meta = meta)
}

deseq_one_celltype <- function(ct, pb, contrast_pairs, p_threshold, lfc_threshold, min_reps) {
  fail <- function(status) {
    list(genes = character(0), skipped = names(contrast_pairs), status = status)
  }
  sub_meta <- pb$meta[pb$meta$celltype == ct, , drop = FALSE]
  sub_counts <- as.matrix(pb$counts[, sub_meta$pseudosample, drop = FALSE])
  storage.mode(sub_counts) <- "integer"

  # Marginalize over timepoint: group on genotype only. Per-(genotype × timepoint)
  # cells have n=1-2 in the Song fixture, which leaves DESeq2 with ~0 residual df
  # and no significance signal. Pooling across timepoints gives ~3-4 reps per
  # genotype, which is enough for stable fits.
  sub_meta$group <- factor(sub_meta$genotype)
  group_counts <- table(sub_meta$group)

  contrasts_skipped <- character(0)
  estimable <- list()
  for (cn in names(contrast_pairs)) {
    pair <- contrast_pairs[[cn]]
    alt_n <- as.integer(group_counts[pair[1]])
    ref_n <- as.integer(group_counts[pair[2]])
    if (is.na(alt_n) || is.na(ref_n) || alt_n < min_reps || ref_n < min_reps) {
      contrasts_skipped <- c(contrasts_skipped, cn)
    } else {
      estimable[[cn]] <- pair
    }
  }
  if (length(estimable) == 0) return(fail("no_estimable_contrasts"))

  # Drop groups absent from every estimable contrast — DESeq2 refuses
  # zero-rep coefficients in `~ 0 + group`.
  used_groups <- unique(unlist(estimable))
  keep_cols <- sub_meta$group %in% used_groups
  sub_meta <- sub_meta[keep_cols, , drop = FALSE]
  sub_counts <- sub_counts[, keep_cols, drop = FALSE]
  sub_meta$group <- droplevels(sub_meta$group)

  dds <- tryCatch(
    DESeqDataSetFromMatrix(
      countData = sub_counts,
      colData = sub_meta,
      design = ~ 0 + group
    ),
    error = function(e) NULL
  )
  if (is.null(dds)) return(fail("DESeqDataSet_failed"))
  dds <- tryCatch(DESeq(dds, quiet = TRUE), error = function(e) NULL)
  if (is.null(dds)) return(fail("DESeq_fit_failed"))

  union_genes <- character(0)
  for (cn in names(estimable)) {
    pair <- estimable[[cn]]
    res <- tryCatch(
      results(dds, contrast = c("group", pair[1], pair[2])),
      error = function(e) NULL
    )
    if (is.null(res)) {
      contrasts_skipped <- c(contrasts_skipped, cn)
      next
    }
    # Raw p-value × |log2FC| filter. The Song factorial design is too thin for
    # BH-FDR at any threshold (n=3-4 per genotype after marginalization, raw
    # p-value distribution depleted of small values). |log2FC|>=0.5 (~1.4×)
    # enforces biological-magnitude relevance to compensate for the missing
    # FDR control. See docs/integrations/seed_list_labels.md for the analysis.
    mask <- !is.na(res$pvalue) & res$pvalue < p_threshold &
            !is.na(res$log2FoldChange) & abs(res$log2FoldChange) >= lfc_threshold
    union_genes <- union(union_genes, rownames(res)[mask])
  }
  list(genes = sort(unique(union_genes)),
       skipped = sort(unique(contrasts_skipped)),
       status = "ok")
}

run_deseq <- function(pb, contrast_pairs, p_threshold, lfc_threshold, min_cells, min_reps) {
  # Preserve every cell type present in the un-filtered metadata so the
  # emitted deg_lists.json covers all 22 cell types. Upstream
  # validate_seed_lists() errors on missing coverage; emit an empty
  # character vector and a status flag for cell types that get filtered
  # out by min_cells or fail to fit.
  all_cell_types <- sort(unique(pb$meta$celltype))
  pb$meta <- pb$meta[pb$meta$n_cells >= min_cells, , drop = FALSE]
  pb$counts <- pb$counts[, pb$meta$pseudosample, drop = FALSE]

  surviving <- sort(unique(pb$meta$celltype))
  dropped_by_min_cells <- setdiff(all_cell_types, surviving)
  message(sprintf("DESeq2 over %d cell types (raw p < %g, |log2FC| >= %g, min_cells=%d, min_reps=%d)",
                  length(surviving), p_threshold, lfc_threshold, min_cells, min_reps))
  if (length(dropped_by_min_cells) > 0) {
    message(sprintf("  Skipped by min_cells=%d filter: %s",
                    min_cells, paste(dropped_by_min_cells, collapse = ", ")))
  }

  deg_lists <- list()
  ct_skipped_contrasts <- list()
  ct_status <- list()
  for (ct in all_cell_types) {
    if (ct %in% dropped_by_min_cells) {
      deg_lists[[ct]] <- character(0)
      ct_status[[ct]] <- "filtered_by_min_cells"
      next
    }
    message(sprintf("  %s ...", ct))
    res <- deseq_one_celltype(ct, pb, contrast_pairs, p_threshold, lfc_threshold, min_reps)
    deg_lists[[ct]] <- res$genes
    if (length(res$skipped) > 0) ct_skipped_contrasts[[ct]] <- res$skipped
    ct_status[[ct]] <- res$status
    message(sprintf("    -> %d DEGs (status=%s, skipped contrasts=%d)",
                    length(res$genes), res$status, length(res$skipped)))
  }
  list(deg_lists = deg_lists,
       skipped = ct_skipped_contrasts,
       status = ct_status)
}

run_limma <- function(input_dir, contrast_pairs, p_threshold, lfc_threshold, min_reps) {
  pr <- read.csv(file.path(input_dir, "pr_matrix.csv"),
                 row.names = 1, check.names = FALSE)
  pr_mat <- as.matrix(pr)
  # Bulk proteomics lives in its own animal_id namespace (separate from
  # transcript animal_metadata.csv consumed by factorial.R). The export
  # writes pr_animal_metadata.csv with genotype normalized to transcript
  # convention so contrast_pairs entries like "WTyp_2mo" resolve.
  pr_meta <- read.csv(file.path(input_dir, "pr_animal_metadata.csv"),
                      check.names = FALSE, stringsAsFactors = FALSE)
  matched <- pr_meta[match(colnames(pr_mat), pr_meta$animal_id), , drop = FALSE]
  if (any(is.na(matched$animal_id))) {
    stop("pr_matrix columns do not all appear in pr_animal_metadata.csv")
  }

  # Marginalize over timepoint, same rationale as DESeq2 above.
  # load_contrast_pairs collapses MANIFEST.contrast_conditions to bare
  # genotype pairs (e.g. c("AppP","WTyp")), so the factor must match.
  group <- factor(matched$genotype)
  design <- model.matrix(~ 0 + group)
  colnames(design) <- levels(group)
  group_counts <- table(group)
  message(sprintf("limma over %d animals across %d condition groups",
                  ncol(pr_mat), length(levels(group))))

  estimable <- list()
  skipped <- character(0)
  for (cn in names(contrast_pairs)) {
    pair <- contrast_pairs[[cn]]
    alt_n <- as.integer(group_counts[pair[1]])
    ref_n <- as.integer(group_counts[pair[2]])
    # Permissive gate: limma tolerates n=1 per arm via shared variance,
    # but skip contrasts whose total replicate count is below min_reps.
    if (is.na(alt_n) || is.na(ref_n) || alt_n < 1 || ref_n < 1 ||
        alt_n + ref_n < min_reps) {
      skipped <- c(skipped, cn)
    } else {
      estimable[[cn]] <- pair
    }
  }
  if (length(estimable) == 0) {
    return(list(genes = character(0), skipped = names(contrast_pairs)))
  }

  fit <- lmFit(pr_mat, design)
  contrast_strs <- vapply(estimable,
                          function(p) paste(p[1], "-", p[2]),
                          character(1))
  cm <- makeContrasts(contrasts = unname(contrast_strs), levels = design)
  colnames(cm) <- names(estimable)
  fit2 <- contrasts.fit(fit, cm)
  fit2 <- eBayes(fit2, trend = FALSE, robust = FALSE)

  union_genes <- character(0)
  for (cn in colnames(cm)) {
    tt <- topTable(fit2, coef = cn, number = Inf, sort.by = "none",
                   adjust.method = "BH")
    mask <- !is.na(tt$P.Value) & tt$P.Value < p_threshold &
            !is.na(tt$logFC) & abs(tt$logFC) >= lfc_threshold
    union_genes <- union(union_genes, rownames(tt)[mask])
  }
  list(genes = sort(unique(union_genes)), skipped = sort(unique(skipped)))
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  message(sprintf("Reading fixture: %s", args$input_dir))
  pb <- load_pseudobulk(args$input_dir)
  message(sprintf("  pseudobulk: %d genes x %d pseudosamples (%d cell types)",
                  nrow(pb$counts), ncol(pb$counts),
                  length(unique(pb$meta$celltype))))

  contrast_pairs <- load_contrast_pairs(args$input_dir)
  vocab <- read.csv(file.path(args$input_dir, "expression_genes.csv"),
                    stringsAsFactors = FALSE)$gene
  vocab_set <- unique(vocab)

  deseq_out <- run_deseq(pb, contrast_pairs, args$pvalue, args$log2fc,
                         args$min_cells, args$min_reps)
  deg_lists <- deseq_out$deg_lists

  all_degs <- unique(unlist(deg_lists, use.names = FALSE))
  deg_missing <- setdiff(all_degs, vocab_set)
  if (length(deg_missing) > 0) {
    stop("DEG genes not in expression_genes.csv vocabulary: ",
         paste(head(deg_missing, 10), collapse = ", "),
         if (length(deg_missing) > 10) sprintf(" (+%d more)", length(deg_missing) - 10))
  }

  limma_out <- run_limma(args$input_dir, contrast_pairs, args$pvalue, args$log2fc,
                         args$min_reps)
  prg_full <- limma_out$genes
  prg_kept <- intersect(prg_full, vocab_set)
  prg_dropped <- length(prg_full) - length(prg_kept)
  if (prg_dropped > 0) {
    message(sprintf("Dropped %d prG genes not in expression vocabulary",
                    prg_dropped))
  }
  prg <- sort(prg_kept)

  message("\nDEG cardinalities:")
  for (ct in names(deg_lists)) {
    n <- length(deg_lists[[ct]])
    overlap <- if (length(prg) > 0 && n > 0) {
      ov <- length(intersect(deg_lists[[ct]], prg))
      sprintf(" (overlap with prG: %d, %.1f%%)", ov, 100 * ov / n)
    } else ""
    message(sprintf("  %-30s %5d DEGs%s", ct, n, overlap))
  }
  message(sprintf("\nprG total: %d genes", length(prg)))

  found_in_deg <- SENTINEL_GENES[SENTINEL_GENES %in% all_degs]
  found_in_prg <- SENTINEL_GENES[SENTINEL_GENES %in% prg]
  message(sprintf("Sentinel genes in DEG union: %s",
                  if (length(found_in_deg)) paste(found_in_deg, collapse = ", ") else "(none)"))
  message(sprintf("Sentinel genes in prG: %s",
                  if (length(found_in_prg)) paste(found_in_prg, collapse = ", ") else "(none)"))
  if (length(found_in_deg) == 0 && length(found_in_prg) == 0) {
    warning("No 5XFAD sentinel genes (", paste(SENTINEL_GENES, collapse = "/"),
            ") appear in either DEG or prG seed lists; investigate before shipping.")
  }

  if (!dir.exists(args$out_dir)) {
    dir.create(args$out_dir, recursive = TRUE, showWarnings = FALSE)
  }
  deg_path <- file.path(args$out_dir, "deg_lists.json")
  prg_path <- file.path(args$out_dir, "prg_list.csv")
  jsonlite::write_json(deg_lists, deg_path, pretty = TRUE, auto_unbox = FALSE)
  write.csv(data.frame(gene_symbol = prg), prg_path, row.names = FALSE,
            quote = FALSE)
  message(sprintf("\nWrote %s", deg_path))
  message(sprintf("Wrote %s", prg_path))

  manifest_path <- file.path(args$out_dir, "MANIFEST.json")
  if (file.exists(manifest_path)) {
    manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)
    manifest$seed_lists <- list(
      generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
      generator = "alz/integration/compute_seed_lists.R",
      deg_method = "DESeq2",
      deg_design = "~ 0 + genotype  (timepoint marginalized)",
      deg_contrasts = "AppP_vs_WTyp, Ttau_vs_WTyp, ApTt_vs_WTyp (union)",
      deg_filter = sprintf("raw_pvalue < %g AND |log2FC| >= %g",
                           args$pvalue, args$log2fc),
      deg_p_threshold = args$pvalue,
      deg_log2fc_threshold = args$log2fc,
      deg_min_cells_per_animal_celltype = args$min_cells,
      deg_min_reps_per_group = args$min_reps,
      deg_cell_types_skipped_contrasts = deseq_out$skipped,
      deg_cell_type_status = deseq_out$status,
      prg_method = "limma",
      prg_design = "~ 0 + genotype  (timepoint marginalized)",
      prg_contrasts = "AppP_vs_WTyp, Ttau_vs_WTyp, ApTt_vs_WTyp (union)",
      prg_filter = sprintf("raw_pvalue < %g AND |log2FC| >= %g",
                           args$pvalue, args$log2fc),
      prg_p_threshold = args$pvalue,
      prg_log2fc_threshold = args$log2fc,
      prg_genes_dropped_for_vocab_mismatch = prg_dropped,
      prg_contrasts_skipped = limma_out$skipped,
      methodology_note = paste(
        "BH-FDR not used: marginalized DESeq2/limma on n=3-4 per genotype",
        "produces a depleted raw p-value distribution where padj<0.05",
        "yields ~16 DEGs total across all cell types. Raw p + |log2FC|>=0.5",
        "is the exploratory fallback that gives uniform coverage."
      ),
      partition_note = paste(
        "alz emits the full DEP set; upstream construct_factorial_paths",
        "applies DEG-first precedence at label-assignment time."
      )
    )
    jsonlite::write_json(manifest, manifest_path, pretty = TRUE,
                         auto_unbox = TRUE, null = "null")
    message(sprintf("Updated %s with seed_lists section", manifest_path))
  } else {
    message("No MANIFEST.json found in out_dir; skipping manifest update.")
  }
}

if (!interactive()) main()
