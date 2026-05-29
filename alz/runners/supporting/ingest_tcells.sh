#!/usr/bin/env bash
# Ingest the T-cell exhaustion cohort (Donor 1 + Donor 2) from Google Drive.
#
# In scope (per 2026-05-27 meeting notes / docs/plans/meeting_notes_triage_2026-05-27.md):
#   * Total proteome  — ForPerseus (both donors)
#   * pY phospho      — ForPerseus (both donors)
#   * IMAC global phospho — Donor 1 ONLY (Donor 2 has no IMAC; kinase MEA skipped there)
#   * scRNA (.rds)    — both donors, ~10 GB total (only with --scrna)
#
# Representation: ForPerseus = site-level (carries PG.Genes, PTM.SiteAA/SiteLocation,
# PTM.FlankingRegion motif), linear intensities. All normalization (log2 + per-run
# median-centering) is done downstream in alz/ingest/tcells.py so both donors and the
# IMAC track share one normalization basis. Donor 1 IMAC has no ForPerseus form, so the
# 18May "Normalized" site report is ingested as-is and re-normalized with the rest.
#
# Out of scope: KGG / AcK / MME enrichments, Flow cytometry, the big "NotParsed"
# May reports, and the collaborator Log2FC files.
#
# Usage:
#   bash alz/runners/supporting/ingest_tcells.sh            # proteomics only
#   bash alz/runners/supporting/ingest_tcells.sh --scrna    # proteomics + scRNA (~10 GB)
set -euo pipefail

DRIVE_FOLDER_ID="1YE_h1jIyBajtm6ArxJqevJ0rt0xLKQgX"
REMOTE="gdrive_shared:"
DEST="data/datasets/tcells"

rc() {  # rc <src-relative-to-folder-id> <dest-dir>
  rclone copy "${REMOTE}$1" --drive-root-folder-id "$DRIVE_FOLDER_ID" "$2" \
    --retries 3 --progress
}

mkdir -p "$DEST/donor1/proteomics" "$DEST/donor2/proteomics"

echo "[tcells] proteomics — Donor 1 (Total, pY, IMAC)"
rc "T Cell Exhaustion Donor 1/Proteomics Data/Total with Ensembl/10Feb2026_Donor1_TotalProteome_ForPerseus.txt" "$DEST/donor1/proteomics/"
rc "T Cell Exhaustion Donor 1/Proteomics Data/pY with Ensembl/10Feb2026_Donor1_pY_ForPerseus.txt" "$DEST/donor1/proteomics/"
rc "T Cell Exhaustion Donor 1/Proteomics Data/IMAC with Ensembl/18May2026_TCellDonor1_Normalized_IMACSiteReporttsv.tsv" "$DEST/donor1/proteomics/"

echo "[tcells] proteomics — Donor 2 (Total, pY; no IMAC)"
rc "T Cell Exhaustion Donor 2/Proteomics Data/Total/10Feb2026_Donor2_TotalProteome_ForPerseus.txt" "$DEST/donor2/proteomics/"
rc "T Cell Exhaustion Donor 2/Proteomics Data/pY/10Feb2026_Donor2_pY_ForPerseus.txt" "$DEST/donor2/proteomics/"

if [[ "${1:-}" == "--scrna" ]]; then
  mkdir -p "$DEST/donor1/scrna" "$DEST/donor2/scrna"
  echo "[tcells] scRNA — ~10 GB, this takes a while"
  rc "T Cell Exhaustion Donor 1/Single Cell Data/Tcells.singlet.rds" "$DEST/donor1/scrna/"
  rc "T Cell Exhaustion Donor 2/Single Cell Data/Tcells_d2.singlet (1).rds" "$DEST/donor2/scrna/"
fi

echo "[tcells] done. Landed files:"
find "$DEST" -type f -name '*.txt' -o -type f -name '*.tsv' -o -type f -name '*.rds' | sort
