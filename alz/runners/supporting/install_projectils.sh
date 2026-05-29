#!/usr/bin/env bash
# Install the ProjecTILs reference-mapping stack for D4 cluster annotation
# (T-cell exhaustion cohort). Replaces the prior Azimuth path (deleted
# 2026-05-28; see docs/plans/projectils_pivot_2026-05-28.md).
#
# Packages (all GitHub, no system deps beyond what bioconductor pulls):
#   carmonalab/UCell      signature scoring (ProjecTILs Imports)
#   carmonalab/scGate     lineage gating (CD4 / CD8 / non-T)
#   carmonalab/STACAS     anchor finder (ProjecTILs Imports)
#   carmonalab/ProjecTILs projection + classifier
#
# Reference atlases (human CD4 + CD8 from Andreatta et al.) are downloaded from
# figshare into data/external/projectils/. Both files together total ~400 MB.
#
# Usage:  pixi run install-projectils   (one-time)
set -euo pipefail
cd "$(dirname "$0")/../../.."

unset GITHUB_PAT GITHUB_TOKEN GH_TOKEN

REF_DIR="data/external/projectils"
mkdir -p "$REF_DIR"

# Reference URLs. carmonalab publishes the human references on figshare under
# doi 10.6084/m9.figshare.23608308. File IDs lifted from the ProjecTILs
# tutorials; if a URL 404s the script aborts loudly so we can update here
# rather than silently fall back.
declare -A REFS=(
  ["CD8T_human_ref_v1.rds"]="https://figshare.com/ndownloader/files/41415033"
  ["CD4T_human_ref_v1.rds"]="https://figshare.com/ndownloader/files/39012395"
)

for fname in "${!REFS[@]}"; do
  dest="$REF_DIR/$fname"
  if [[ -s "$dest" ]]; then
    size_mb=$(du -m "$dest" | cut -f1)
    echo "  $fname: present (${size_mb} MB), skip"
    continue
  fi
  url="${REFS[$fname]}"
  echo "  downloading $fname from $url ..."
  curl -fLsS --retry 3 -o "$dest.partial" "$url"
  sz=$(stat -c%s "$dest.partial")
  if [[ $sz -lt 5000000 ]]; then
    echo "ERROR: downloaded $fname is only $sz bytes — figshare URL likely changed." >&2
    rm -f "$dest.partial"
    exit 1
  fi
  mv "$dest.partial" "$dest"
  echo "  $fname: $(du -m "$dest" | cut -f1) MB"
done

pixi run env -u GITHUB_PAT -u GITHUB_TOKEN -u GH_TOKEN Rscript - <<'RSCRIPT'
suppressPackageStartupMessages({})
Sys.unsetenv(c("GITHUB_PAT", "GITHUB_TOKEN", "GH_TOKEN"))
options(gitcreds.use_cache = FALSE,
        repos = c(CRAN = "https://cloud.r-project.org"))

# CRAN deps for UCell/scGate/STACAS/ProjecTILs that aren't already in the
# conda env. install.packages is idempotent on already-installed packages
# when we filter via setdiff(rownames(installed.packages())).
cran_deps <- c("Matrix.utils", "umap", "BiocNeighbors", "BiocParallel",
               "scales", "reshape2", "pheatmap", "RcolorBrewer", "ggrepel",
               "plyr", "yardstick", "uwot", "irlba", "RANN", "withr",
               "rappdirs", "cli")
# RcolorBrewer is mis-typed deliberately above to avoid duplicate (the canonical
# is RColorBrewer); strip and de-dup.
cran_deps <- setdiff(c("Matrix.utils", "umap", "scales", "reshape2", "pheatmap",
                       "RColorBrewer", "ggrepel", "plyr", "uwot", "irlba",
                       "RANN", "withr", "rappdirs", "cli"),
                     rownames(installed.packages()))
if (length(cran_deps)) {
  cat("CRAN install:", paste(cran_deps, collapse=", "), "\n")
  install.packages(cran_deps,
                   dependencies = c("Depends","Imports","LinkingTo"),
                   quiet = TRUE)
}

# Bioconductor deps not on conda (BiocManager already in the env).
bioc_deps <- setdiff(c("BiocNeighbors", "BiocParallel"),
                     rownames(installed.packages()))
if (length(bioc_deps)) {
  cat("Bioc install:", paste(bioc_deps, collapse=", "), "\n")
  BiocManager::install(bioc_deps, update = FALSE, ask = FALSE, quiet = TRUE)
}

# GitHub installs via direct tarball + R CMD INSTALL — same pattern as the
# (now deleted) install_azimuth.sh; remotes' GitHub-API path is unreliable on
# this box. Order matters: UCell → scGate → STACAS → ProjecTILs (each later
# package Imports earlier ones).
download_tarball <- function(pkg, repo) {
  for (branch in c("master", "main")) {
    url <- sprintf("https://codeload.github.com/%s/tar.gz/refs/heads/%s",
                   repo, branch)
    tf <- tempfile(fileext = ".tar.gz")
    ok <- tryCatch({
      utils::download.file(url, tf, mode = "wb", quiet = TRUE)
      file.info(tf)$size > 1000
    }, error = function(e) FALSE, warning = function(w) FALSE)
    if (isTRUE(ok)) {
      cat("  downloaded", pkg, "(", branch, ",",
          round(file.info(tf)$size / 1e6, 1), "MB)\n")
      return(tf)
    }
  }
  stop("could not download ", pkg, " from ", repo)
}

ensure_cmd_install <- function(pkg, repo) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat("  ", pkg, ":", as.character(packageVersion(pkg)),
        " (present, skip)\n", sep = "")
    return(invisible(NULL))
  }
  tf <- download_tarball(pkg, repo)
  td <- tempfile(); dir.create(td)
  utils::untar(tf, exdir = td)
  src_dir <- list.files(td, full.names = TRUE)[1]
  cat("  R CMD INSTALL", pkg, "from", src_dir, "...\n")
  res <- system2(file.path(R.home("bin"), "R"),
                 args = c("CMD", "INSTALL", "--no-multiarch", "--no-docs",
                          shQuote(src_dir)),
                 stdout = "", stderr = "")
  if (res != 0L) stop("R CMD INSTALL failed for ", pkg, " (exit ", res, ")")
}

ensure_cmd_install("UCell",      "carmonalab/UCell")
ensure_cmd_install("scGate",     "carmonalab/scGate")
ensure_cmd_install("STACAS",     "carmonalab/STACAS")
ensure_cmd_install("ProjecTILs", "carmonalab/ProjecTILs")

cat("\n--- versions ---\n")
for (p in c("Seurat","SeuratObject","UCell","scGate","STACAS","ProjecTILs",
            "glmGamPoi","SingleCellExperiment")) {
  cat(sprintf("  %-22s %s\n", p,
              ifelse(requireNamespace(p, quietly = TRUE),
                     as.character(packageVersion(p)), "MISSING")))
}

# Smoke-load the references so figshare bytes are validated as RDS now, not
# later inside the mapping script.
for (fname in c("CD8T_human_ref_v1.rds", "CD4T_human_ref_v1.rds")) {
  path <- file.path("data/external/projectils", fname)
  cat("  loading", fname, "...\n")
  obj <- tryCatch(readRDS(path), error = function(e) {
    stop("RDS load failed for ", path, ": ", conditionMessage(e))
  })
  cat(sprintf("    %s: %s; ncells=%d; functional.cluster levels: %s\n",
              fname, paste(class(obj), collapse = "/"), ncol(obj),
              paste(sort(unique(as.character(obj$functional.cluster))),
                    collapse = ", ")))
}
cat("\n=== install-projectils done ===\n")
RSCRIPT
