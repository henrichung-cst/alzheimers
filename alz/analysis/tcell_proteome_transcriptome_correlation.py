#!/usr/bin/env python3
"""Correlate donor2 day2 bulk protein with raw-Seurat RNA pseudobulk.

The RNA pseudobulk is deliberately independent of cell-state assignments: it
sums the RNA ``counts`` layer across every raw donor2 cell whose
``Sample_Label`` parses as day 2.  Reading the 4.7 GB Seurat object always
happens inside a 40 GB, no-swap systemd scope.

Usage:
    pixi run python alz/analysis/tcell_proteome_transcriptome_correlation.py
"""
from __future__ import annotations

import csv
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from scipy.stats import rankdata, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTEOME_PATH = REPO_ROOT / (
    "data/datasets/tcells/donor2/proteomics/"
    "10Feb2026_Donor2_TotalProteome_ForPerseus.txt"
)
RDS_PATH = REPO_ROOT / "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds"
OUTPUT_PATH = REPO_ROOT / "outputs/reports/tcell_donor2_day2_protein_rna_correlation.csv"

GENE_COLUMN = "PG.Genes"
PRECURSOR_COLUMN = "PG.NrOfPrecursorsIdentified (Experiment-wide)"
PROTEIN_COLUMN = "Day 2 Total Quantity"
PSEUDOBULK_COLUMN = "pseudobulk_transcript_abundance"


def _pixi_executable() -> str:
    """Return the project Pixi executable or fail before loading the RDS."""
    return shutil.which("pixi") or str(Path.home() / ".pixi/bin/pixi")


def preflight_inputs() -> None:
    """Confirm source files and report their on-disk sizes before reading them."""
    for label, path in (("proteome", PROTEOME_PATH), ("raw Seurat RDS", RDS_PATH)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} source is missing: {path}")
        print(f"{label} input: {path.relative_to(REPO_ROOT)} ({path.stat().st_size:,} bytes)")


def read_proteome(path: Path) -> dict[str, float]:
    """Read one day-2 bulk abundance per direct protein-group gene symbol.

    A duplicate single-gene protein group is resolved to the group supported by
    the most experiment-wide identified precursors.  This preserves one
    measured abundance for every direct symbol without summing potentially
    overlapping protein groups.
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"proteome file has no header: {path}")
        missing = {GENE_COLUMN, PRECURSOR_COLUMN, PROTEIN_COLUMN} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"proteome file lacks required columns: {sorted(missing)}")

        values: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for line_number, row in enumerate(reader, start=2):
            gene = row[GENE_COLUMN].strip()
            if not gene or ";" in gene:
                continue
            try:
                precursors = int(row[PRECURSOR_COLUMN])
                abundance = float(row[PROTEIN_COLUMN])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid precursor count or {PROTEIN_COLUMN!r} at {path}:{line_number}"
                ) from exc
            if precursors < 0 or not math.isfinite(abundance):
                raise ValueError(
                    f"invalid precursor count or non-finite {PROTEIN_COLUMN!r} at {path}:{line_number}"
                )
            values[gene].append((precursors, abundance))

    selected: dict[str, float] = {}
    for gene, candidates in values.items():
        highest_precursors = max(precursors for precursors, _ in candidates)
        best = [abundance for precursors, abundance in candidates if precursors == highest_precursors]
        if len(best) != 1:
            raise ValueError(
                f"multiple {gene!r} protein groups share the highest precursor count"
            )
        selected[gene] = best[0]
    return selected


def _r_pseudobulk_program(output_path: Path) -> str:
    """Return the capped worker program that extracts raw day-2 RNA counts."""
    rds = str(RDS_PATH).replace("'", "\\\\'")
    output = str(output_path).replace("'", "\\\\'")
    return f"""
suppressPackageStartupMessages({{ library(Seurat); library(Matrix) }})
obj <- readRDS('{rds}')
DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = "RNA", dimreducs = NULL, graphs = NULL)
obj[["RNA"]]$scale.data <- NULL
day_raw <- as.character(obj@meta.data[["Sample_Label"]])
day <- suppressWarnings(as.integer(sub(".*[Dd]ay[_ ]?([0-9]+).*", "\\\\1", day_raw)))
if (any(is.na(day))) stop("unparsed Sample_Label values: ", paste(sort(unique(day_raw[is.na(day)])), collapse = " | "))
day2_cells <- colnames(obj)[day == 2L]
if (!length(day2_cells)) stop("no day2 cells in raw donor2 Seurat object")
counts <- SeuratObject::GetAssayData(obj, assay = "RNA", layer = "counts")
if (is.null(counts) || !nrow(counts)) stop("RNA counts layer is empty")
if (anyDuplicated(rownames(counts))) stop("RNA counts layer has duplicate gene symbols")
pseudobulk <- Matrix::rowSums(counts[, day2_cells, drop = FALSE])
write.csv(data.frame(gene = rownames(counts), {PSEUDOBULK_COLUMN} = as.numeric(pseudobulk)),
          '{output}', row.names = FALSE, quote = TRUE)
"""


def extract_day2_pseudobulk(output_path: Path) -> None:
    """Run the only RDS-reading step under the mandatory hard memory cap."""
    pixi = _pixi_executable()
    if not Path(pixi).is_file() and shutil.which(pixi) is None:
        raise FileNotFoundError("pixi executable not found")
    command = [
        "systemd-run", "--user", "--scope",
        "-p", "MemoryMax=40G", "-p", "MemorySwapMax=0",
        pixi, "run", "Rscript", "-e", _r_pseudobulk_program(output_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_pseudobulk(path: Path) -> dict[str, float]:
    """Read the bounded day-2 pseudobulk emitted by the capped R worker."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"gene", PSEUDOBULK_COLUMN}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"pseudobulk file lacks required columns: {path}")
        values: dict[str, float] = {}
        for row in reader:
            gene = row["gene"].strip()
            if not gene or gene in values:
                raise ValueError(f"pseudobulk genes must be non-empty and unique: {path}")
            abundance = float(row[PSEUDOBULK_COLUMN])
            if not math.isfinite(abundance):
                raise ValueError(f"non-finite pseudobulk abundance for {gene!r}: {path}")
            values[gene] = abundance
    return values


def matched_rows(proteome: dict[str, float], pseudobulk: dict[str, float]) -> list[dict[str, float | str]]:
    """Build the direct, one-to-one human-symbol intersection for Spearman."""
    genes = sorted(set(proteome) & set(pseudobulk))
    if not genes:
        raise ValueError("no direct protein/RNA gene-symbol matches")
    protein_ranks = rankdata([proteome[gene] for gene in genes], method="average")
    return [
        {
            "gene": gene,
            "protein_abundance": proteome[gene],
            PSEUDOBULK_COLUMN: pseudobulk[gene],
            "rank": float(rank),
        }
        for gene, rank in zip(genes, protein_ranks, strict=True)
    ]


def write_output(rows: list[dict[str, float | str]], output_path: Path) -> tuple[float, float, int]:
    """Write per-gene evidence and return raw-value Spearman statistics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    protein = [float(row["protein_abundance"]) for row in rows]
    transcript = [float(row[PSEUDOBULK_COLUMN]) for row in rows]
    result = spearmanr(protein, transcript)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gene", "protein_abundance", PSEUDOBULK_COLUMN, "rank"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return float(result.statistic), float(result.pvalue), len(rows)


def main() -> None:
    preflight_inputs()
    proteome = read_proteome(PROTEOME_PATH)
    with tempfile.TemporaryDirectory(prefix="tcell_donor2_day2_", dir=OUTPUT_PATH.parent) as temp_dir:
        pseudobulk_path = Path(temp_dir) / "pseudobulk.csv"
        extract_day2_pseudobulk(pseudobulk_path)
        pseudobulk = read_pseudobulk(pseudobulk_path)
    rows = matched_rows(proteome, pseudobulk)
    rho, p_value, n = write_output(rows, OUTPUT_PATH)
    print(f"Spearman rho={rho:.6g} p_value={p_value:.6g} n={n}")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
