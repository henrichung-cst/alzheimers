#!/usr/bin/env Rscript
# Test: DuckDB-based pathway enumeration vs data.table cartesian join.
#
# Compares memory and speed for a single pair (Microglia-PVM -> L5 IT)
# at 0% threshold (all genes). Validates pathway count matches 169,462.
#
# Usage:
#   micromamba run -n incytr Rscript code/integration/tests/edge_pruning/test_duckdb_join.R

suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
  library(duckdb)
  library(DBI)
  library(Incytr)
})

hill <- function(x, K = 0.5, N = 2) x^N / (x^N + K^N)

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  getwd()
}
repo_root <- normalizePath(file.path(get_script_dir(), "..", "..", "..", ".."))
int_dir   <- file.path(repo_root, "code", "integration", "intermediates")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cat("Loading expression data...\n")
mat <- readMM(file.path(int_dir, "expression_matrix.mtx"))
mat <- as(mat, "CsparseMatrix")
genes    <- read.csv(file.path(int_dir, "expression_genes.csv"))$gene
barcodes <- read.csv(file.path(int_dir, "expression_barcodes.csv"))$barcode
meta     <- read.csv(file.path(int_dir, "expression_metadata.csv"), row.names = 1)
rownames(mat) <- genes; colnames(mat) <- barcodes

cat("Loading IncytrDB...\n")
data(DB_Layer1_mouse_filtered, package = "Incytr")
data(DB_Layer2_mouse_filtered, package = "Incytr")
data(DB_Layer3_mouse_filtered, package = "Incytr")
all_genes <- rownames(mat)
l1_raw <- as.data.table(DB_Layer1_mouse_filtered[
  DB_Layer1_mouse_filtered$from %in% all_genes &
  DB_Layer1_mouse_filtered$to %in% all_genes, ])
l2_raw <- as.data.table(DB_Layer2_mouse_filtered[
  DB_Layer2_mouse_filtered$from %in% all_genes &
  DB_Layer2_mouse_filtered$to %in% all_genes, ])
l3_raw <- as.data.table(DB_Layer3_mouse_filtered[
  DB_Layer3_mouse_filtered$from %in% all_genes &
  DB_Layer3_mouse_filtered$to %in% all_genes, ])
rm(DB_Layer1_mouse_filtered, DB_Layer2_mouse_filtered, DB_Layer3_mouse_filtered)
gc(verbose = FALSE)
cat(sprintf("  DB: L1=%d, L2=%d, L3=%d edges\n\n",
            nrow(l1_raw), nrow(l2_raw), nrow(l3_raw)))

sender   <- "Microglia-PVM"
receiver <- "L5 IT"
conditions <- c("WT", "App")
K <- 0.5; N <- 2; CUTOFF <- 0.01

# ---------------------------------------------------------------------------
# Compute per-condition mean expression
# ---------------------------------------------------------------------------
cat("Computing per-condition mean expression...\n")
s_c1 <- setNames(Matrix::rowMeans(mat[, meta$labels == sender & meta$condition == "WT"]), all_genes)
s_c2 <- setNames(Matrix::rowMeans(mat[, meta$labels == sender & meta$condition == "App"]), all_genes)
r_c1 <- setNames(Matrix::rowMeans(mat[, meta$labels == receiver & meta$condition == "WT"]), all_genes)
r_c2 <- setNames(Matrix::rowMeans(mat[, meta$labels == receiver & meta$condition == "App"]), all_genes)

# Free expression matrix
rm(mat, genes, barcodes)
gc(verbose = FALSE)

# ---------------------------------------------------------------------------
# Edge pruning (shared by both methods)
# ---------------------------------------------------------------------------
cat("Pruning edges (0% threshold, all genes)...\n")

prune <- function(dt, from_c1, from_c2, to_c1, to_c2) {
  dt <- copy(dt)
  h1 <- hill(from_c1[dt$from] * to_c1[dt$to], K, N)
  h2 <- hill(from_c2[dt$from] * to_c2[dt$to], K, N)
  dt[(h1 >= CUTOFF) | (h2 >= CUTOFF)]
}

l1 <- prune(l1_raw, s_c1, s_c2, r_c1, r_c2)
l2 <- prune(l2_raw, r_c1, r_c2, r_c1, r_c2)
l3 <- prune(l3_raw, r_c1, r_c2, r_c1, r_c2)
cat(sprintf("  Pruned: L1=%d, L2=%d, L3=%d\n\n", nrow(l1), nrow(l2), nrow(l3)))

# Free raw tables
rm(l1_raw, l2_raw, l3_raw)
gc(verbose = FALSE)

# ===================================================================
# Method 1: DuckDB join
# ===================================================================
cat("====== Method 1: DuckDB ======\n")
gc(verbose = FALSE)
mem_before_duck <- gc(verbose = FALSE)[2, 2]  # Mb used

con <- dbConnect(duckdb(), dbdir = ":memory:")
# Cap DuckDB memory to observe spilling behavior
dbExecute(con, "SET memory_limit='4GB'")
dbExecute(con, "SET threads=4")

# Register pruned edge tables (zero-copy from R)
duckdb_register(con, "L1", as.data.frame(l1))
duckdb_register(con, "L2", as.data.frame(l2))
duckdb_register(con, "L3", as.data.frame(l3))

t0_duck <- proc.time()
pathways_duck <- dbGetQuery(con, "
  SELECT DISTINCT
    L1.\"from\" AS Ligand,
    L1.\"to\"   AS Receptor,
    L2.\"to\"   AS EM,
    L3.\"to\"   AS Target
  FROM L1
  JOIN L2 ON L1.\"to\" = L2.\"from\"
  JOIN L3 ON L2.\"to\" = L3.\"from\"
  WHERE L1.\"from\" != L1.\"to\"
    AND L1.\"from\" != L2.\"to\"
    AND L1.\"from\" != L3.\"to\"
    AND L1.\"to\"   != L2.\"to\"
    AND L1.\"to\"   != L3.\"to\"
    AND L2.\"to\"   != L3.\"to\"
")
t_duck <- (proc.time() - t0_duck)["elapsed"]

mem_after_duck <- gc(verbose = FALSE)[2, 2]
n_duck <- nrow(pathways_duck)

cat(sprintf("  Pathways: %s\n", format(n_duck, big.mark = ",")))
cat(sprintf("  Time:     %.1f sec\n", t_duck))
cat(sprintf("  R memory delta: %.0f MB\n", mem_after_duck - mem_before_duck))

# Cleanup DuckDB
duckdb_unregister(con, "L1")
duckdb_unregister(con, "L2")
duckdb_unregister(con, "L3")
dbDisconnect(con, shutdown = TRUE)
rm(con)
gc(verbose = FALSE)

# ===================================================================
# Method 2: data.table join (the approach that OOM'd at scale)
# ===================================================================
cat("\n====== Method 2: data.table cartesian join ======\n")
cat("  (Single pair only — this is what OOM'd across all receivers)\n")
gc(verbose = FALSE)
mem_before_dt <- gc(verbose = FALSE)[2, 2]

t0_dt <- proc.time()

# Rename for join
dt1 <- copy(l1)[, .(Ligand = from, Receptor = to)]
dt2 <- copy(l2)[, .(Receptor = from, EM = to)]
dt3 <- copy(l3)[, .(EM = from, Target = to)]

# L2 ⋈ L3 on EM
setkey(dt2, EM); setkey(dt3, EM)
chains <- dt3[dt2, allow.cartesian = TRUE, nomatch = 0]
chains <- chains[Receptor != EM & Receptor != Target & EM != Target]
chains <- unique(chains[, .(Receptor, EM, Target)])
n_chains <- nrow(chains)

# L1 ⋈ chains on Receptor
setkey(dt1, Receptor); setkey(chains, Receptor)
pathways_dt <- chains[dt1, allow.cartesian = TRUE, nomatch = 0]
pathways_dt <- pathways_dt[Ligand != Receptor & Ligand != EM & Ligand != Target]
pathways_dt <- unique(pathways_dt, by = c("Ligand", "Receptor", "EM", "Target"))

t_dt <- (proc.time() - t0_dt)["elapsed"]
mem_after_dt <- gc(verbose = FALSE)[2, 2]
n_dt <- nrow(pathways_dt)

cat(sprintf("  Intermediate chains (L2⋈L3): %s\n", format(n_chains, big.mark = ",")))
cat(sprintf("  Pathways: %s\n", format(n_dt, big.mark = ",")))
cat(sprintf("  Time:     %.1f sec\n", t_dt))
cat(sprintf("  R memory delta: %.0f MB\n", mem_after_dt - mem_before_dt))

# ===================================================================
# Comparison
# ===================================================================
cat("\n====== Comparison ======\n")
cat(sprintf("  DuckDB:     %d pathways in %.1f sec\n", n_duck, t_duck))
cat(sprintf("  data.table: %d pathways in %.1f sec\n", n_dt, t_dt))
cat(sprintf("  Counts match: %s\n", ifelse(n_duck == n_dt, "YES", "NO")))

if (n_duck == n_dt) {
  # Verify same pathway set
  duck_keys <- paste(pathways_duck$Ligand, pathways_duck$Receptor,
                     pathways_duck$EM, pathways_duck$Target, sep = "|")
  dt_keys   <- paste(pathways_dt$Ligand, pathways_dt$Receptor,
                     pathways_dt$EM, pathways_dt$Target, sep = "|")
  sets_match <- setequal(duck_keys, dt_keys)
  cat(sprintf("  Pathway sets identical: %s\n", ifelse(sets_match, "YES", "NO")))
}

expected <- 169462
cat(sprintf("\n  Expected: %d\n", expected))
cat(sprintf("  DuckDB matches expected: %s\n",
            ifelse(n_duck == expected, "YES", sprintf("NO (%d)", n_duck))))
cat(sprintf("  data.table matches expected: %s\n",
            ifelse(n_dt == expected, "YES", sprintf("NO (%d)", n_dt))))
