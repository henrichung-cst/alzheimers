import csv
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "alz/analysis/tcell_proteome_transcriptome_correlation.py"
SPEC = importlib.util.spec_from_file_location("tcell_protein_rna_correlation", MODULE_PATH)
assert SPEC and SPEC.loader
correlation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(correlation)


def write_proteome(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "PG.Genes",
                "PG.NrOfPrecursorsIdentified (Experiment-wide)",
                "Day 2 Total Quantity",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_read_proteome_drops_multigene_groups(tmp_path: Path) -> None:
    path = tmp_path / "protein.tsv"
    write_proteome(path, [
        {"PG.Genes": "ARF5", "PG.NrOfPrecursorsIdentified (Experiment-wide)": "4", "Day 2 Total Quantity": "12.5"},
        {"PG.Genes": "MAP1LC3B2;MAP1LC3B", "PG.NrOfPrecursorsIdentified (Experiment-wide)": "10", "Day 2 Total Quantity": "99"},
    ])

    assert correlation.read_proteome(path) == {"ARF5": 12.5}


def test_read_proteome_uses_best_supported_single_gene_group(tmp_path: Path) -> None:
    path = tmp_path / "protein.tsv"
    write_proteome(path, [
        {"PG.Genes": "ARF5", "PG.NrOfPrecursorsIdentified (Experiment-wide)": "4", "Day 2 Total Quantity": "12.5"},
        {"PG.Genes": "ARF5", "PG.NrOfPrecursorsIdentified (Experiment-wide)": "7", "Day 2 Total Quantity": "15.5"},
    ])

    assert correlation.read_proteome(path) == {"ARF5": 15.5}


def test_read_proteome_rejects_tied_best_supported_groups(tmp_path: Path) -> None:
    path = tmp_path / "protein.tsv"
    write_proteome(path, [
        {"PG.Genes": "ARF5", "PG.NrOfPrecursorsIdentified (Experiment-wide)": "7", "Day 2 Total Quantity": "12.5"},
        {"PG.Genes": "ARF5", "PG.NrOfPrecursorsIdentified (Experiment-wide)": "7", "Day 2 Total Quantity": "15.5"},
    ])

    try:
        correlation.read_proteome(path)
    except ValueError as error:
        assert "share the highest precursor count" in str(error)
    else:
        raise AssertionError("expected tied protein-group evidence to fail")


def test_matched_rows_is_direct_intersection_with_protein_rank() -> None:
    rows = correlation.matched_rows(
        {"B": 4.0, "A": 2.0, "NOT_RNA": 6.0},
        {"A": 8.0, "B": 3.0, "NOT_PROTEIN": 5.0},
    )

    assert rows == [
        {"gene": "A", "protein_abundance": 2.0, "pseudobulk_transcript_abundance": 8.0, "rank": 1.0},
        {"gene": "B", "protein_abundance": 4.0, "pseudobulk_transcript_abundance": 3.0, "rank": 2.0},
    ]


def test_write_output_uses_spearman_on_raw_values(tmp_path: Path) -> None:
    rows = correlation.matched_rows(
        {"A": 1.0, "B": 2.0, "C": 3.0},
        {"A": 30.0, "B": 20.0, "C": 10.0},
    )

    rho, p_value, n = correlation.write_output(rows, tmp_path / "result.csv")

    assert (rho, n) == (-1.0, 3)
    assert p_value >= 0
    assert (tmp_path / "result.csv").read_text().splitlines()[0] == (
        "gene,protein_abundance,pseudobulk_transcript_abundance,rank"
    )
