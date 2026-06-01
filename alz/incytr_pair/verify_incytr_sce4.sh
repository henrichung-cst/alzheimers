#!/usr/bin/env bash
# sce4 parity gate — self-contained.
#
# Regenerates the two sce4 reference pairs UNFILTERED and verifies them against
# sce4's Top300 reference. It deliberately does NOT read
# outputs/reports/incytr_pair_mode/wide/: run_pair_mode.sh filters wide/ in
# place with sce4's gate (SigProb > 0.1 AND |PDS| >= 0.2, then per-pair top-300
# PDS up ∪ down). That cap keeps each pair to <=600 rows, so verifying against
# the filtered file would conflate the cap's tie-breaking at the rank-300
# boundary with engine parity. Parity is a property of the ENGINE on the
# unfiltered superset, so we regenerate the two pairs and check those.
#
# Deterministic (seed.use=1L) and nboot-independent for every column the gate
# checks (path recall + the four *_sclog2FC), so NBOOT=0 (skip the permutation)
# is sufficient and fast (~40 s for 2 pairs: rds load + substrate + 2 scorings).
#
#   pixi run verify-incytr-sce4
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DRIVER="alz/incytr_pair/incytr_commandline.R"
INPUTS_DIR="data/derived/incytr_inputs"
OUT="outputs/reports/incytr_pair_mode/_sce4_parity"
NDNF="Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic"

# AD reproduction sources gene.use from sce4's own reconstructed node sets (the
# driver consumes it via SCE4_GENEUSE_DIR). The gate MUST regenerate with that
# same source — verifying the deliverable, not the retired DEG∪prG derivation.
GENEUSE_DIR="data/incytr_frozen/sce4_geneuse"
export SCE4_GENEUSE_DIR="$GENEUSE_DIR"
if [[ ! -f "$GENEUSE_DIR/ma_2mo_AppP_ma_2mo_WTyp.csv" ]]; then
  echo "=== Building sce4 gene.use artifacts ($GENEUSE_DIR) ==="
  pixi run Rscript alz/incytr_pair/extract_sce4_geneuse.R || true  # ma_2mo_AppP is enough here
fi

# Fresh regen so we never verify a stale pair: clear prior shards + parquet.
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== regenerating sce4 reference pairs (UNFILTERED, nboot=0) ==="
NBOOT=0 NPAIR_WORKERS=1 \
OUTPUT_DIR_OVERRIDE="$OUT" \
PAIR_SUBSET="Microglia:Cholinergic-Neurons,${NDNF}:${NDNF}" \
  pixi run Rscript "$DRIVER" ma_2mo_AppP ma_2mo_WTyp

# Dump sce4's gated Allpathway tuples for the two benchmark pairs from the
# pre-cap rds — this is the gate REFERENCE (the user's pre-cap path-set invariant).
# The deliverable Top300 cap is NOT the gate: it is ranked by PDS, which carries
# the documented phospho-substrate + App-value residuals, so cap membership can't
# match sce4 regardless of gene.use correctness. The gate is gated-path-set
# identity (symmetric diff must be transgene-only); cap fidelity is informational.
ALLREF="$OUT/sce4_allpathway_ref.csv"
RDS="data/incytr_frozen/outputs/Analysis_new cluster labels_cutoff_0.1/DEG_PRG_ma_2mo_AppP_WTyp_10302025/sce4_DEG_PRG_Pairwise_pathway_table_10302025.rds"
ALLREF="$ALLREF" RDS="$RDS" NDNF="$NDNF" pixi run Rscript -e '
  suppressPackageStartupMessages(library(data.table))
  x <- readRDS(Sys.getenv("RDS"))
  ND <- "Ndnf.positive.neurogliaform.inhibitory.interneurons.GABAergic"
  keys <- c("Microglia_to_Cholinergic.Neurons", paste0(ND, "_to_", ND))
  cols <- c("Sender.group","Receiver.group","Ligand","Receptor","EM","Target",
            "Ligand_sclog2FC","Receptor_sclog2FC","EM_sclog2FC","Target_sclog2FC")
  out <- rbindlist(lapply(keys, function(k) as.data.table(x[[k]])[, ..cols]))
  fwrite(out, Sys.getenv("ALLREF"))
  cat(sprintf("[allref] wrote %d Allpathway rows for %d pairs\n", nrow(out), length(keys)))
'

echo "=== sce4 parity gate (--all-known-pairs) ==="
pixi run python alz/incytr_pair/verify_sce4_parity.py \
  --all-known-pairs --wide-dir "$OUT" --sce4-allpathway "$ALLREF"
