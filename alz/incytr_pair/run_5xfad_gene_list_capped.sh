#!/usr/bin/env bash
# Memory-capped, logged runner for the 5xFAD gene-list build (FindAllMarkers).
#
# Why this wrapper: the gene-list R job legitimately reaches ~7.5 GB RSS per
# tissue. That is survivable ALONE on this 30 GB shared box, but it OOM-killed
# the box on 2026-06-18 when it ran concurrently with a `pixi run viewer` build.
# This wrapper (a) runs the two tissues strictly sequentially, (b) caps each
# tissue's memory via a systemd user scope so a runaway can only kill itself,
# not the box, and (c) logs to a file so no live process monitoring is needed.
#
# Run this ALONE — do not start other heavy jobs while it runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOG="/tmp/5xfad_gene_list_$(date +%H%M%S).log"
CAP="${MEM_CAP:-20G}"   # hard ceiling per tissue; box has ~17G free, 7.5G expected

echo "[capped] log -> $LOG ; per-tissue MemoryMax=$CAP" | tee "$LOG"

for t in cortex hippocampus; do
  echo "[capped] === $t === $(date '+%H:%M:%S')" | tee -a "$LOG"
  # systemd-run --user --scope enforces an RSS ceiling via cgroup v2.
  systemd-run --user --scope -p MemoryMax="$CAP" -p MemorySwapMax=0 \
    --unit "gene-list-$t-$(date +%s)" \
    pixi run Rscript alz/incytr_pair/build_5xfad_input_gene_list.R "$t" \
    >>"$LOG" 2>&1
  echo "[capped] $t done rc=$? $(date '+%H:%M:%S')" | tee -a "$LOG"
done

echo "[capped] ALL DONE" | tee -a "$LOG"
ls -lh data/derived/5xfad_incytr_inputs/*/allmarkers.csv | tee -a "$LOG"
