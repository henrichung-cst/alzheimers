#!/usr/bin/env bash
# Sprint 2 D2: DuckDB enumeration equivalence runner.
#
# Verifies that code/integration/wrappers/duckdb_enumeration.R produces the
# same pathway set as native Incytr pathway_inference() on the synthetic
# 12-gene × 40-cell × 2-condition fixture (helper-golden.R), with all
# pre-prune cutoffs disabled.
#
# Equivalence claim: with cutoff_SigProb = 0 and em_promiscuity_weight = FALSE,
# the (Ligand, Receptor, EM, Target) tuples returned by
# duckdb_enumerate_pathways() match the @pathways slot of an Incytr object
# built via pathway_inference() on the same DB and gene set.
#
# Pre-prune cutoffs themselves (Hill < 0.01, SigProb >= 0.01) are bucket C
# and audited in Sprint 4 — Sprint 2 only verifies that the enumerator is
# bitwise-equivalent when those filters are off.
#
# Usage: bash code/integration/tests/run_duckdb_enumeration_equiv.sh

set -euo pipefail

INCYTR_DIR="${INCYTR_DIR:-../incytr}"
ALZHEIMERS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

if [[ ! -d "$INCYTR_DIR" ]]; then
  INCYTR_DIR="$ALZHEIMERS_DIR/../incytr"
fi
if [[ ! -d "$INCYTR_DIR" ]]; then
  echo "ERROR: incytr package directory not found at $INCYTR_DIR" >&2
  exit 2
fi

echo "=== Sprint 2 D2: DuckDB enumeration vs native pathway_inference ==="
echo "incytr dir:     $INCYTR_DIR"
echo "alzheimers dir: $ALZHEIMERS_DIR"
echo

# Activate pixi env so R sees duckdb / arrow / Matrix etc.
if command -v pixi >/dev/null 2>&1 && [[ -f "$ALZHEIMERS_DIR/pixi.toml" ]]; then
  eval "$(cd "$ALZHEIMERS_DIR" && pixi shell-hook 2>/dev/null)"
fi

Rscript - "$INCYTR_DIR" "$ALZHEIMERS_DIR" <<'EOF'
options(warn = 1)
args <- commandArgs(trailingOnly = TRUE)
incytr_dir <- args[1]
alz_dir    <- args[2]

suppressMessages(pkgload::load_all(incytr_dir, quiet = TRUE))
source(file.path(incytr_dir, "tests", "testthat", "helper-golden.R"))
source(file.path(alz_dir, "code", "integration", "wrappers", "duckdb_enumeration.R"))

set.seed(42)
n_genes  <- 12
n_cells  <- 40
gene_nm  <- paste0("G", seq_len(n_genes))
cell_nm  <- paste0("cell_", sprintf("%02d", seq_len(n_cells)))

mat <- matrix(abs(rnorm(n_genes * n_cells, mean = 1, sd = 0.5)),
              nrow = n_genes, ncol = n_cells,
              dimnames = list(gene_nm, cell_nm))

meta <- data.frame(
  labels    = rep(c("TypeA", "TypeA", "TypeB", "TypeB"), each = 10),
  condition = factor(rep(c("condA", "condB", "condA", "condB"), each = 10),
                     levels = c("condA", "condB")),
  row.names = cell_nm
)

DB <- list(
  data.frame(from = c("G1", "G2", "G9"),
             to   = c("G3", "G4", "G10"),
             stringsAsFactors = FALSE),
  data.frame(from = c("G3", "G4", "G10"),
             to   = c("G5", "G6", "G11"),
             stringsAsFactors = FALSE),
  data.frame(from   = c("G5", "G6", "G11"),
             to     = c("G7", "G8", "G12"),
             source = c("src1", "src2", "src1"),
             stringsAsFactors = FALSE)
)

# ----- Native pathway_inference path -----
obj_native <- create_Incytr(mat, meta = meta, sender = "TypeA",
                            receiver = "TypeB",
                            conditions = c("condA", "condB"))
obj_native <- pathway_inference(obj_native, DB)
native_pw  <- obj_native@pathways[, c("Ligand", "Receptor", "EM", "Target")]
native_pw  <- native_pw[do.call(order, native_pw), ]
rownames(native_pw) <- NULL

# ----- DuckDB enumeration path with cutoffs DISABLED -----
mat_sparse <- as(mat, "CsparseMatrix")
res_duck <- duckdb_enumerate_pathways(
  mat_sparse, meta, DB,
  sender = "TypeA", receiver = "TypeB",
  conditions = c("condA", "condB"),
  K = 0.5, N = 2,
  cutoff_SigProb = 0,             # disable SigProb filter
  em_promiscuity_weight = FALSE,  # Sprint 4 territory; off for equivalence
  duckdb_memory = "1GB", duckdb_threads = 1L
)
duck_pw <- as.data.frame(res_duck$pathways)[, c("Ligand", "Receptor", "EM", "Target")]
duck_pw <- duck_pw[do.call(order, duck_pw), ]
rownames(duck_pw) <- NULL

cat(sprintf("Native pathway_inference: %d pathways\n", nrow(native_pw)))
cat(sprintf("DuckDB enumeration:       %d pathways\n", nrow(duck_pw)))

ok <- identical(native_pw, duck_pw)
if (!ok) {
  cat("\n--- DIFF: rows in native but not duck ---\n")
  print(setdiff(do.call(paste, c(native_pw, sep = "*")),
                do.call(paste, c(duck_pw,   sep = "*"))))
  cat("--- DIFF: rows in duck but not native ---\n")
  print(setdiff(do.call(paste, c(duck_pw,   sep = "*")),
                do.call(paste, c(native_pw, sep = "*"))))
  cat("\nFAIL: pathway sets differ\n")
  quit(status = 4)
}

cat("\nOK: DuckDB enumeration produces bitwise-identical pathway set to ")
cat("native pathway_inference() on synthetic 2-condition fixture ")
cat("(cutoffs disabled).\n")
EOF

echo
echo "=== D2 PASS: DuckDB enumerator is equivalence-preserving for ALZ-18. ==="
