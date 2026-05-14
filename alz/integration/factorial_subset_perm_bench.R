#!/usr/bin/env Rscript
# Bench/diagnostic: run the factorial pipeline on the immune+astro subset to
# characterize permutation-test time cost and p-value distribution. Uses
# multimodel = FALSE (transcript-only) because the permutation test lives in
# the transcript step; this keeps memory bounded and isolates the new code.
#
# Reports:
#   - candidate path counts per receiver
#   - wall time and peak RSS for n_perm = 0 (t-test only) and n_perm = 100
#   - p-value distribution (quantiles + count <= 0.05) for both p-values

suppressPackageStartupMessages({
  library(Incytr)
  library(arrow)
})

INPUT_DIR <- "data/incytr_factorial_inputs/subset_immune_astro"
OUT_DIR   <- "outputs/reports/incytr_factorial_subset_perm_bench"

source("alz/integration/load.R")

peak_rss_mb <- function() {
  gc_stats <- gc(reset = FALSE)
  used_mb <- sum(gc_stats[, "max used"] * c(0.056, 0.008))  # rough; gc is the
  # cheapest cross-platform proxy. Real RSS via /proc:
  stat <- readLines(sprintf("/proc/%d/status", Sys.getpid()))
  vmhwm <- as.numeric(sub("[^0-9]+([0-9]+).*", "\\1",
                          grep("^VmHWM:", stat, value = TRUE)))
  vmhwm / 1024  # KB -> MB
}

cat(sprintf("==> Loading subset inputs from %s\n", INPUT_DIR))
inputs <- load_ad_factorial_inputs(INPUT_DIR)
cat(sprintf("    cells=%d genes=%d animals=%d cell_types=%s\n",
            ncol(inputs$expr), nrow(inputs$expr),
            length(rownames(inputs$design)),
            paste(inputs$senders, collapse = ", ")))

contrasts <- build_factorial_contrasts(inputs$animal_meta)
cat(sprintf("    contrasts: %d  -> %s\n", length(contrasts),
            paste(names(contrasts), collapse = ", ")))

cat("\n==> Building candidate paths\n")
t0 <- Sys.time()
paths <- construct_factorial_paths(
  expression = inputs$expr, metadata = inputs$meta,
  senders = inputs$senders, receivers = inputs$receivers,
  group.by = "labels",
  deg_lists = inputs$deg_lists, prg_list = inputs$prg_list
)
cat(sprintf("    paths constructed in %.1fs: %d total\n",
            as.numeric(Sys.time() - t0, units = "secs"), nrow(paths)))
print(table(paths$sender, paths$receiver))
cat(sprintf("    peak RSS so far: %.0f MB\n", peak_rss_mb()))

run_one <- function(label, n_perm, perm_chunk) {
  out <- file.path(OUT_DIR, label)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  cat(sprintf("\n==> Scoring [%s] n_perm=%d perm_chunk=%d\n",
              label, n_perm, perm_chunk))
  gc(reset = TRUE, full = TRUE)
  t0 <- Sys.time()
  summary <- score_factorial_paths(
    expression = inputs$expr, metadata = inputs$meta,
    paths = paths, contrasts = contrasts, design = inputs$design,
    animal_id = "animal_id", condition_col = "condition",
    cond_pairs = inputs$cond_pairs,
    min_cells = 5,
    multimodel = FALSE,
    n_perm = n_perm, perm_chunk = perm_chunk, perm_seed = 42,
    output_dir = out, force_rerun = TRUE
  )
  dt <- as.numeric(Sys.time() - t0, units = "secs")
  cat(sprintf("    wall=%.1fs peak_RSS=%.0f MB\n", dt, peak_rss_mb()))
  print(summary[, c("receiver", "n_paths", "time_sec", "status")])
  list(label = label, dt = dt, n_perm = n_perm, out = out, summary = summary)
}

bench_t <- run_one("nperm0",  n_perm = 0,   perm_chunk = 100)
bench_p <- run_one("nperm100", n_perm = 100, perm_chunk = 20)

cat("\n==> Cost summary\n")
cat(sprintf("    t-test only   : %.1fs\n", bench_t$dt))
cat(sprintf("    + 100 perms   : %.1fs  (delta = %.1fs, %.1fx)\n",
            bench_p$dt, bench_p$dt - bench_t$dt,
            bench_p$dt / max(bench_t$dt, 1e-6)))

cat("\n==> P-value distribution (n_perm = 100 run)\n")
# Read the per-receiver parquets and concatenate the per-contrast pvalue cols.
files <- list.files(bench_p$out, pattern = "\\.parquet$", full.names = TRUE)
all_df <- do.call(rbind, lapply(files, function(f) {
  as.data.frame(arrow::read_parquet(f))
}))
cat(sprintf("    n_rows = %d (paths x receivers, post-scoring)\n", nrow(all_df)))

for (cn in names(contrasts)) {
  pv_t  <- all_df[[paste0("pvalue_", cn)]]
  pv_p  <- all_df[[paste0("perm_pvalue_", cn)]]
  nn <- sum(!is.na(pv_p))
  if (nn == 0) next
  q_t <- quantile(pv_t, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
  q_p <- quantile(pv_p, probs = c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
  cat(sprintf("\n  contrast %s (n=%d non-NA):\n", cn, nn))
  cat(sprintf("    t-test     pct[5,25,50,75,95] = [%.3f, %.3f, %.3f, %.3f, %.3f]   sig(<=.05) = %d (%.1f%%)\n",
              q_t[1], q_t[2], q_t[3], q_t[4], q_t[5],
              sum(pv_t <= 0.05, na.rm = TRUE),
              100 * mean(pv_t <= 0.05, na.rm = TRUE)))
  cat(sprintf("    perm  pct[5,25,50,75,95] = [%.3f, %.3f, %.3f, %.3f, %.3f]   sig(<=.05) = %d (%.1f%%)\n",
              q_p[1], q_p[2], q_p[3], q_p[4], q_p[5],
              sum(pv_p <= 0.05, na.rm = TRUE),
              100 * mean(pv_p <= 0.05, na.rm = TRUE)))
  # Disagreement: t-test sig but perm not, and vice versa.
  t_sig <- !is.na(pv_t) & pv_t <= 0.05
  p_sig <- !is.na(pv_p) & pv_p <= 0.05
  cat(sprintf("    disagreement: t-sig & !perm-sig = %d ; perm-sig & !t-sig = %d ; both = %d\n",
              sum(t_sig & !p_sig), sum(!t_sig & p_sig), sum(t_sig & p_sig)))
}

cat(sprintf("\n==> Done. Final peak RSS: %.0f MB\n", peak_rss_mb()))
