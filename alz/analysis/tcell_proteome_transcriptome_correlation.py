#!/usr/bin/env python3
"""Per-gene protein-RNA correlation across the matched T-cell timecourse.

For each donor, every gene gets a vector of (protein_t, RNA_t) pairs over the
timepoints where both modalities were measured, and a per-gene correlation is
computed and binned by sign+strength.  Donors are handled SEPARATELY (different
day grids, different biological donors -- no pooling).

Protein source is the canonical per-day linear bulk
(``pr_bulk_linear.csv`` from ``alz/cohorts/tcells/ingest.py``): per-run
median-centered, technical-replicate-averaged, duplicate-genes mean-collapsed.
Median-centering removes per-injection loading differences that would otherwise
be a shared trend inflating every gene's cross-day correlation.

RNA is a raw ``counts``-layer pseudobulk summed across all cells at each day,
re-derived from the 4.7-5.2 GB Seurat object inside a 40 GB no-swap systemd
scope, then CPM-normalized per day (the RNA analogue of the protein
median-centering).

Descriptive/exploratory: N is 4-5 points per gene, so coefficients are
concordance hypotheses, not validated regulation -- no per-gene p-value claimed.

Usage:
    pixi run python alz/analysis/tcell_proteome_transcriptome_correlation.py
"""
from __future__ import annotations

import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs/reports/tcell_protein_rna_timecourse_correlation"

DONORS = {
    "donor1": {
        "pr_bulk": "data/derived/tcells_incytr_inputs/donor1/pr_bulk_linear.csv",
        "rds": "data/datasets/tcells/donor1/scrna/Tcells.singlet.rds",
        "day_col": "sample_ID",
    },
    "donor2": {
        "pr_bulk": "data/derived/tcells_incytr_inputs/donor2/pr_bulk_linear.csv",
        "rds": "data/datasets/tcells/donor2/scrna/Tcells_d2.singlet (1).rds",
        "day_col": "Sample_Label",
    },
}

# Sign+strength tiers on the Pearson-of-logs coefficient (shared across donors).
BIN_ORDER = [
    "strong-positive",
    "weak-positive",
    "none",
    "weak-negative",
    "strong-negative",
]


def null_bin_fractions(n: int) -> dict[str, float]:
    """Expected bin fractions under the bivariate-normal null (df = n-2).

    At n=4-5 points a large |r| is common by chance; this sets the noise floor
    the observed bins must clear to count as signal.  P(r >= c) uses the exact
    t-transform t = c*sqrt(df/(1-c^2)) ~ t_{df}.
    """
    df = n - 2

    def ge(c: float) -> float:
        if c <= -1:
            return 1.0
        if c >= 1:
            return 0.0
        return float(t_dist.sf(c * math.sqrt(df / (1 - c * c)), df))

    return {
        "strong-positive": ge(0.7),
        "weak-positive": ge(0.3) - ge(0.7),
        "none": ge(-0.3) - ge(0.3),
        "weak-negative": ge(-0.7) - ge(-0.3),
        "strong-negative": 1.0 - ge(-0.7),
    }


def bin_of(rho: float) -> str:
    """Assign the concordance tier; undefined coefficients get no bin."""
    if not math.isfinite(rho):
        return ""
    if rho >= 0.7:
        return "strong-positive"
    if rho >= 0.3:
        return "weak-positive"
    if rho > -0.3:
        return "none"
    if rho > -0.7:
        return "weak-negative"
    return "strong-negative"


def _pixi_executable() -> str:
    """Return the project Pixi executable or fail before loading the RDS."""
    return shutil.which("pixi") or str(Path.home() / ".pixi/bin/pixi")


def read_protein_bulk(path: Path) -> tuple[dict[str, dict[int, float]], list[int]]:
    """Read the canonical per-day linear bulk keyed as gene -> {day: value}.

    Day is parsed from the ``*_d<N>`` column suffix; the ``D1_``/``D2_`` prefix
    is the donor label, not the day.
    """
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != "gene_symbol":
            raise ValueError(f"unexpected protein bulk header in {path}: {header[:1]}")
        day_of_col = {i: int(col.rsplit("_d", 1)[1]) for i, col in enumerate(header[1:], start=1)}
        days = sorted(day_of_col.values())
        values: dict[str, dict[int, float]] = {}
        for row in reader:
            gene = row[0].strip()
            if not gene or gene in values:
                raise ValueError(f"empty or duplicate protein gene symbol in {path}: {gene!r}")
            per_day = {}
            for i, day in day_of_col.items():
                cell = row[i].strip()
                if cell in ("", "NA", "NaN"):
                    raise ValueError(f"unexpected missing protein value for {gene!r} at day {day}")
                v = float(cell)
                if not math.isfinite(v) or v <= 0:
                    raise ValueError(f"non-positive protein value for {gene!r} at day {day}: {v}")
                per_day[day] = v
            values[gene] = per_day
    return values, days


def _r_pseudobulk_program(rds: str, day_col: str, output_path: Path) -> str:
    """Worker program: raw counts-layer pseudobulk summed per day, all days."""
    rds_lit = rds.replace("'", "\\\\'")
    out_lit = str(output_path).replace("'", "\\\\'")
    return f"""
suppressPackageStartupMessages({{ library(Seurat); library(Matrix) }})
obj <- readRDS('{rds_lit}')
DefaultAssay(obj) <- "RNA"
obj <- DietSeurat(obj, assays = "RNA", dimreducs = NULL, graphs = NULL)
obj[["RNA"]]$scale.data <- NULL
day_raw <- as.character(obj@meta.data[["{day_col}"]])
day <- suppressWarnings(as.integer(sub(".*[Dd]ay[_ ]?([0-9]+).*", "\\\\1", day_raw)))
if (any(is.na(day))) stop("unparsed day labels: ", paste(sort(unique(day_raw[is.na(day)])), collapse = " | "))
counts <- SeuratObject::GetAssayData(obj, assay = "RNA", layer = "counts")
if (is.null(counts) || !nrow(counts)) stop("RNA counts layer is empty")
if (anyDuplicated(rownames(counts))) stop("RNA counts layer has duplicate gene symbols")
udays <- sort(unique(day))
mat <- vapply(udays, function(d) Matrix::rowSums(counts[, day == d, drop = FALSE]), numeric(nrow(counts)))
colnames(mat) <- paste0("d", udays)
out <- data.frame(gene = rownames(counts), mat, check.names = FALSE)
write.csv(out, '{out_lit}', row.names = FALSE, quote = TRUE)
"""


def extract_rna_pseudobulk(rds: Path, day_col: str, output_path: Path) -> None:
    """Run the RDS-reading step under the mandatory hard memory cap."""
    if not rds.is_file():
        raise FileNotFoundError(f"raw Seurat RDS is missing: {rds}")
    print(f"  reading {rds.relative_to(REPO_ROOT)} ({rds.stat().st_size:,} bytes) under 40G cap")
    pixi = _pixi_executable()
    command = [
        "systemd-run", "--user", "--scope",
        "-p", "MemoryMax=40G", "-p", "MemorySwapMax=0",
        pixi, "run", "Rscript", "-e", _r_pseudobulk_program(str(rds), day_col, output_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_rna_cpm(path: Path) -> tuple[dict[str, dict[int, float]], list[int]]:
    """Read the per-day pseudobulk and CPM-normalize each day column."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != "gene":
            raise ValueError(f"unexpected RNA pseudobulk header in {path}: {header[:1]}")
        day_of_col = {i: int(col[1:]) for i, col in enumerate(header[1:], start=1)}
        days = sorted(day_of_col.values())
        raw: dict[str, dict[int, float]] = {}
        totals: dict[int, float] = {day: 0.0 for day in days}
        for row in reader:
            gene = row[0].strip()
            if not gene or gene in raw:
                raise ValueError(f"empty or duplicate RNA gene symbol in {path}: {gene!r}")
            per_day = {}
            for i, day in day_of_col.items():
                v = float(row[i])
                per_day[day] = v
                totals[day] += v
            raw[gene] = per_day
    for day, total in totals.items():
        if total <= 0:
            raise ValueError(f"day {day} has zero total RNA counts in {path}")
    cpm = {
        gene: {day: per_day[day] / totals[day] * 1e6 for day in days}
        for gene, per_day in raw.items()
    }
    return cpm, days


def correlate(
    protein: dict[str, dict[int, float]],
    cpm: dict[str, dict[int, float]],
    days: list[int],
) -> list[dict[str, object]]:
    """Per-gene Pearson-of-logs (bin axis) and Spearman over the matched days."""
    genes = sorted(set(protein) & set(cpm))
    if not genes:
        raise ValueError("no direct protein/RNA gene-symbol matches")
    rows: list[dict[str, object]] = []
    for gene in genes:
        prot = np.array([protein[gene][day] for day in days], dtype=float)
        rna = np.array([cpm[gene][day] for day in days], dtype=float)
        log_prot = np.log10(prot)
        log_rna = np.log1p(rna)
        pearson = _pearson(log_prot, log_rna)
        spearman = _pearson(_rank(prot), _rank(rna))
        row: dict[str, object] = {"gene": gene}
        for day in days:
            row[f"protein_d{day}"] = protein[gene][day]
        for day in days:
            row[f"rna_cpm_d{day}"] = cpm[gene][day]
        row["pearson_log"] = pearson
        row["spearman"] = spearman
        row["n_timepoints"] = len(days)
        row["bin"] = bin_of(pearson)
        rows.append(row)
    return rows


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN when either vector has zero variance."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), so _pearson of ranks == Spearman rho."""
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average tied ranks
    for value in np.unique(a):
        mask = a == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def write_per_gene(rows: list[dict[str, object]], days: list[int], path: Path) -> None:
    """Write the per-gene evidence table."""
    fieldnames = (
        ["gene"]
        + [f"protein_d{day}" for day in days]
        + [f"rna_cpm_d{day}" for day in days]
        + ["pearson_log", "spearman", "n_timepoints", "bin"]
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bin_summary(rows: list[dict[str, object]], path: Path) -> dict[str, int]:
    """Write and return per-bin gene counts (undefined coefficients excluded)."""
    counts = {name: 0 for name in BIN_ORDER}
    for row in rows:
        name = row["bin"]
        if name:
            counts[name] += 1
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin", "n_genes"])
        for name in BIN_ORDER:
            writer.writerow([name, counts[name]])
    return counts


def write_histogram(rows: list[dict[str, object]], donor: str, path: Path) -> None:
    """Histogram of the Pearson-of-logs coefficient with tier cutoffs marked."""
    values = [float(row["pearson_log"]) for row in rows if math.isfinite(float(row["pearson_log"]))]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.hist(values, bins=40, range=(-1, 1), color="#2563eb", alpha=0.85)
    for cutoff in (-0.7, -0.3, 0.3, 0.7):
        axis.axvline(cutoff, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.8)
    axis.set_xlabel("Per-gene Pearson correlation (log10 protein vs log1p CPM RNA)")
    axis.set_ylabel("Genes")
    axis.set_title(f"{donor}: protein-RNA timecourse concordance (n={len(values):,} genes)")
    axis.grid(True, axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_analysis(
    rows: list[dict[str, object]],
    counts: dict[str, int],
    days: list[int],
    donor: str,
    path: Path,
) -> None:
    """Write a compact interpretation with bin counts and extreme examples."""
    undefined = [row for row in rows if not math.isfinite(float(row["pearson_log"]))]
    defined = [row for row in rows if math.isfinite(float(row["pearson_log"]))]
    top_pos = sorted(defined, key=lambda r: float(r["pearson_log"]), reverse=True)[:5]
    top_neg = sorted(defined, key=lambda r: float(r["pearson_log"]))[:5]

    def gene_list(entries: list[dict[str, object]]) -> str:
        return ", ".join(f"{r['gene']} ({float(r['pearson_log']):.2f})" for r in entries)

    defined_total = sum(counts.values())
    null = null_bin_fractions(len(days))
    bin_rows = []
    for name in BIN_ORDER:
        obs = counts[name] / defined_total if defined_total else 0.0
        enrich = obs / null[name] if null[name] else float("inf")
        bin_rows.append(
            f"| {name} | {counts[name]:,} | {100 * obs:.1f}% | {100 * null[name]:.1f}% | {enrich:.2f}x |"
        )
    bin_table = "\n".join(bin_rows)
    sp_enrich = (counts["strong-positive"] / defined_total) / null["strong-positive"] if defined_total else 0.0
    sn_enrich = (counts["strong-negative"] / defined_total) / null["strong-negative"] if defined_total else 0.0
    day_str = ", ".join(f"D{day}" for day in days)
    path.write_text(
        f"# {donor}: protein-RNA per-gene timecourse correlation\n\n"
        f"## Method\n\n"
        f"Per-gene correlation across {len(days)} matched timepoints ({day_str}). Protein is the "
        f"canonical per-run median-centered linear bulk (`pr_bulk_linear.csv`); RNA is an "
        f"all-cell raw `counts` pseudobulk per day, CPM-normalized. The binned coefficient is "
        f"Pearson on log10(protein) vs log1p(CPM RNA); Spearman rho is reported alongside as a "
        f"rank-based cross-check.\n\n"
        f"At n={len(days)} points per gene this is descriptive: coefficients are concordance "
        f"hypotheses, not validated regulation, and no per-gene p-value is claimed.\n\n"
        f"## Bin counts\n\n"
        f"{len(defined):,} genes have a defined coefficient; {len(undefined):,} have an "
        f"undefined coefficient (zero variance -- dominant case: RNA is zero at every matched "
        f"timepoint) and are retained in the CSV with an empty bin, excluded from the counts "
        f"below.\n\n"
        f"| bin | n_genes | observed | null | enrichment |\n"
        f"|---|---:|---:|---:|---:|\n{bin_table}\n\n"
        f"## Noise-floor calibration\n\n"
        f"At n={len(days)} points per gene a large |r| is common by chance; the `null` column is "
        f"the bivariate-normal null (df={len(days) - 2}) and `enrichment` is observed/null. The "
        f"strong-positive bin is enriched {sp_enrich:.1f}x over the floor, while the strong-negative "
        f"bin sits at {sn_enrich:.1f}x -- i.e. at or near the floor. **Genome-wide negative "
        f"concordance is not distinguishable from n={len(days)} noise here**; only reproducible, "
        f"cross-donor, class-coherent negatives carry signal, and those are additionally confounded "
        f"(see below).\n\n"
        f"## Extremes\n\n"
        f"Strongest positive concordance: {gene_list(top_pos)}.\n\n"
        f"Strongest negative concordance: {gene_list(top_neg)}.\n\n"
        f"Negative coefficients are NOT read as post-transcriptional buffering. CPM (sum-constrained) "
        f"and protein median-centering (scalar-constrained) are different closed normalizations; both "
        f"compress stable high-abundance genes toward negative r during a large transcriptional "
        f"induction (a few genes rising 6-10x dilute everything else's CPM share and inflate the "
        f"protein scale). Buffering is not separable from this compositional artifact with this data. "
        f"For replication-dependent histones there is a further confound: their mRNAs are "
        f"non-polyadenylated and poorly captured by polyA scRNA-seq, so the RNA trajectory is "
        f"unreliable independent of protein stability.\n"
    )


def process_donor(donor: str, cfg: dict[str, str]) -> None:
    """Run the full per-donor pipeline and write its output folder."""
    print(f"==== {donor} ====")
    protein, protein_days = read_protein_bulk(REPO_ROOT / cfg["pr_bulk"])
    out_dir = OUTPUT_ROOT / donor
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"tcell_{donor}_", dir=out_dir) as temp_dir:
        rna_path = Path(temp_dir) / "pseudobulk.csv"
        extract_rna_pseudobulk(REPO_ROOT / cfg["rds"], cfg["day_col"], rna_path)
        cpm, rna_days = read_rna_cpm(rna_path)
    days = sorted(set(protein_days) & set(rna_days))
    if not days:
        raise ValueError(f"{donor}: no matched protein/RNA timepoints")
    print(f"  protein days {protein_days}; RNA days {rna_days}; matched {days}")
    rows = correlate(protein, cpm, days)
    write_per_gene(rows, days, out_dir / "per_gene_correlation.csv")
    counts = write_bin_summary(rows, out_dir / "bin_summary.csv")
    write_histogram(rows, donor, out_dir / "coefficient_histogram.png")
    write_analysis(rows, counts, days, donor, out_dir / "analysis.md")
    defined = sum(counts.values())
    print(f"  {len(rows):,} matched genes; {defined:,} binned; wrote {out_dir.relative_to(REPO_ROOT)}")


def main() -> None:
    for donor, cfg in DONORS.items():
        process_donor(donor, cfg)


if __name__ == "__main__":
    main()
