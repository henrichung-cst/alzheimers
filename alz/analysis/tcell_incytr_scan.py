"""Build bounded, anchor-calibrated evidence from T-cell pair-mode Incytr.

The seven wide parquet files are scanned by DuckDB.  No wide file is
materialized in pandas; only edge-level aggregates are exported.  The anchor
set is committed inline from docs/plans/tcell-incytr-apriori-novelty-report.md.
Tier assignment follows the plan's data-driven rule across the complete
surface: an observed ligand-receptor edge is Tier 1, an absent edge whose
receptor appears as a downstream Target is Tier 2, and everything else is
reported as absent without target evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "incytr_pair_mode_tcells"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "derived_evidence"

CONTRASTS = {
    "donor1": ("d13_d2", "d17_d2", "d20_d2"),
    "donor2": ("d5_d2", "d7_d2", "d9_d2", "d11_d2"),
}

# These quantile bands are analysis definitions from the approved plan, not
# hidden gates or exported synthetic scores.
TOP_QUARTILE_PERCENTILE = 0.75
TOP_DECILE_PERCENTILE = 0.90


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    axis: str
    ligand: str
    receptor: str
    prior_design: str


# Exact L-R pairs named by the approved plan.  "MHC-derived" is not expanded:
# the plan does not pin concrete MHC ligand-receptor pairs, and inventing them
# after seeing the surface would violate blind-anchor discipline.
ANCHORS = {anchor.anchor_id: anchor for anchor in (
    Anchor("A01", "IL2", "IL2", "IL2RA", "reconstructable_candidate"),
    Anchor("A02", "IL2", "IL2", "IL2RG", "reconstructable_candidate"),
    Anchor("A03", "IFNG", "IFNG", "IFNGR1", "reconstructable_candidate"),
    Anchor("A04", "IFNG", "IFNG", "IFNGR2", "reconstructable_candidate"),
    Anchor("A05", "TNF", "TNF", "TNFRSF1A", "reconstructable_candidate"),
    Anchor("A06", "TNF", "TNF", "TNFRSF1B", "reconstructable_candidate"),
    Anchor("A07", "FASLG", "FASLG", "FAS", "reconstructable_candidate"),
    Anchor("A08", "IL15", "IL15", "IL15RA", "reconstructable_candidate"),
    Anchor("A09", "lymphotoxin", "LTA", "LTBR", "reconstructable_candidate"),
    Anchor("A10", "lymphotoxin", "LTB", "LTBR", "reconstructable_candidate"),
    Anchor(
        "A11", "TNFSF costimulation", "TNFSF9", "TNFRSF9", "reconstructable_candidate"
    ),
    Anchor("A12", "checkpoint", "CD274", "PDCD1", "structurally_absent_reference"),
    Anchor("A13", "costimulation", "CD80", "CD28", "structurally_absent_reference"),
    Anchor("A14", "costimulation", "CD86", "CD28", "structurally_absent_reference"),
    Anchor("A15", "checkpoint", "CD86", "CTLA4", "structurally_absent_reference"),
)}

# Explicit biological-family membership used only to prevent a paralog or
# shared-receptor echo from being presented as an independent novel edge.
# The exported evidence retains the categorical family name and raw PDS fields.
ANCHOR_GENE_FAMILIES = {
    "IL2": "common_gamma_cytokine",
    "IL4": "common_gamma_cytokine",
    "IL7": "common_gamma_cytokine",
    "IL9": "common_gamma_cytokine",
    "IL15": "common_gamma_cytokine",
    "IL21": "common_gamma_cytokine",
    "IL2RA": "common_gamma_cytokine",
    "IL2RB": "common_gamma_cytokine",
    "IL2RG": "common_gamma_cytokine",
    "IL4R": "common_gamma_cytokine",
    "IL7R": "common_gamma_cytokine",
    "IL9R": "common_gamma_cytokine",
    "IL15RA": "common_gamma_cytokine",
    "IL21R": "common_gamma_cytokine",
    "IFNG": "interferon",
    "IFNGR1": "interferon",
    "IFNGR2": "interferon",
    "TNF": "tnf_superfamily",
    "LTA": "tnf_superfamily",
    "LTB": "tnf_superfamily",
    "FASLG": "tnf_superfamily",
    "TNFSF4": "tnf_superfamily",
    "TNFSF9": "tnf_superfamily",
    "TNFRSF1A": "tnf_superfamily",
    "TNFRSF1B": "tnf_superfamily",
    "TNFRSF4": "tnf_superfamily",
    "TNFRSF9": "tnf_superfamily",
    "FAS": "tnf_superfamily",
    "LTBR": "tnf_superfamily",
    "CD274": "checkpoint_costimulation",
    "PDCD1": "checkpoint_costimulation",
    "CD80": "checkpoint_costimulation",
    "CD86": "checkpoint_costimulation",
    "CD28": "checkpoint_costimulation",
    "CTLA4": "checkpoint_costimulation",
}

# Filled only for post-ranking manual biological reads.  Absence from this map
# remains explicit in the CSV rather than receiving an invented score.
CURATED_NOVEL_ANNOTATIONS = {
    ("VIM", "CD44"): (
        "Cytoskeletal/adhesion edge; very strong but donor trends oppose, so treat as "
        "a state/stress-associated signal rather than an exhaustion-specific mechanism."
    ),
    ("PECAM1", "CD38"): (
        "Adhesion-to-ectoenzyme edge; recurrent and rising in both donors, but the "
        "surface alone does not establish a causal PECAM1→CD38 program."
    ),
    ("SELPLG", "SELL"): (
        "Leukocyte homing/adhesion axis; high-ranked but too sparsely reconstructed "
        "for a cross-donor trend claim."
    ),
    ("GZMB", "IGF2R"): (
        "Cytotoxic-effector-associated edge; opposing donor trends argue against a "
        "shared exhaustion trajectory."
    ),
    ("COPA", "CD74"): (
        "Antigen-presentation trafficking edge; only two top-band contrasts and "
        "opposing donor directions."
    ),
    ("PTPRC", "CD247"): (
        "TCR-complex-associated structural edge; donor-1 dominant and biologically "
        "adjacent to known activation rather than an independent checkpoint."
    ),
    ("FURIN", "ADAM19"): (
        "Protease-processing edge; donor-1 only, so retain as exploratory."
    ),
    ("LGALS3", "LAG3"): (
        "Plausible LAG3-associated coinhibitory edge; below the top decile here but "
        "directionally rising in both donors."
    ),
    ("CCL3", "CCR1"): (
        "Canonical chemokine-receptor biology, but reconstructed in only one contrast."
    ),
    ("TGFB1", "TGFBR2"): (
        "Canonical immunoregulatory axis, but the top-band evidence is limited to one contrast."
    ),
    ("RPS27A", "TGFBR2"): (
        "Ribosomal/ubiquitin-fusion ligand assignment; likely database-role or "
        "abundance-driven rather than a specific sender signal."
    ),
    ("RPS27A", "SMAD3"): (
        "Ribosomal/ubiquitin-fusion ligand assignment; interpret as a role-assignment caveat."
    ),
    ("UBA52", "TGFBR1"): (
        "Ribosomal/ubiquitin-fusion ligand assignment; interpret as a role-assignment caveat."
    ),
    ("UBA52", "TGFBR2"): (
        "Ribosomal/ubiquitin-fusion ligand assignment; interpret as a role-assignment caveat."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _surface_files(input_dir: Path) -> list[Path]:
    files = [
        input_dir / donor / "wide" / f"{contrast}_incytr_output.parquet"
        for donor, contrasts in CONTRASTS.items()
        for contrast in contrasts
    ]
    missing = [path for path in files if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing canonical Incytr parquet files:\n{formatted}")
    return files


def _register_inputs(connection: duckdb.DuckDBPyConnection, files: list[Path]) -> None:
    connection.read_parquet(
        [str(path) for path in files], filename=True, union_by_name=True
    ).create_view("wide_surface")
    connection.execute(
        """
        CREATE TEMP VIEW surface AS
        SELECT
            regexp_extract(filename, '/(donor[12])/', 1) AS donor,
            regexp_extract(filename, '/([^/]+)_incytr_output[.]parquet$', 1) AS contrast,
            CAST(regexp_extract(filename, '/d([0-9]+)_d2_incytr_output[.]parquet$', 1)
                 AS INTEGER) AS day,
            Ligand AS ligand,
            Receptor AS receptor,
            EM AS em,
            Target AS target,
            "Sender.group" AS sender_group,
            "Receiver.group" AS receiver_group,
            PDS AS pds
        FROM wide_surface
        WHERE PDS IS NOT NULL AND isfinite(PDS)
        """
    )


def _register_reference_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE anchors (
            anchor_id VARCHAR,
            axis VARCHAR,
            ligand VARCHAR,
            receptor VARCHAR,
            prior_design VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO anchors VALUES (?, ?, ?, ?, ?)",
        [
            (a.anchor_id, a.axis, a.ligand, a.receptor, a.prior_design)
            for a in ANCHORS.values()
        ],
    )
    connection.execute(
        "CREATE TEMP TABLE anchor_gene_families (gene VARCHAR, family VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO anchor_gene_families VALUES (?, ?)",
        list(ANCHOR_GENE_FAMILIES.items()),
    )
    connection.execute(
        """
        CREATE TEMP VIEW effective_anchor_gene_families AS
        SELECT gene, family FROM anchor_gene_families
        UNION
        SELECT DISTINCT ligand AS gene, 'mhc_derived' AS family
        FROM surface
        WHERE ligand LIKE 'HLA-%' OR ligand = 'B2M'
        UNION
        SELECT DISTINCT ligand AS gene, 'tnf_superfamily' AS family
        FROM surface
        WHERE ligand LIKE 'TNFSF%'
           OR ligand IN ('TNF', 'LTA', 'LTB', 'FASLG', 'CD40LG')
        UNION
        SELECT DISTINCT receptor AS gene, 'tnf_superfamily' AS family
        FROM surface
        WHERE receptor LIKE 'TNFRSF%'
           OR receptor IN ('FAS', 'LTBR', 'CD40')
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE curated_novel_annotations (
            ligand VARCHAR,
            receptor VARCHAR,
            curated_annotation VARCHAR
        )
        """
    )
    if CURATED_NOVEL_ANNOTATIONS:
        connection.executemany(
            "INSERT INTO curated_novel_annotations VALUES (?, ?, ?)",
            [
                (ligand, receptor, annotation)
                for (ligand, receptor), annotation in CURATED_NOVEL_ANNOTATIONS.items()
            ],
        )


def _build_evidence_views(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP VIEW surface_summary AS
        SELECT
            donor,
            contrast,
            day,
            count(*) AS path_count,
            count(DISTINCT (ligand, receptor)) AS edge_count,
            count(DISTINCT sender_group) AS sender_count,
            count(DISTINCT receiver_group) AS receiver_count
        FROM surface
        GROUP BY donor, contrast, day
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW edge_evidence AS
        WITH aggregated AS (
            SELECT
                donor,
                contrast,
                day,
                ligand,
                receptor,
                count(*) AS path_count,
                count(DISTINCT (sender_group, receiver_group)) AS sender_receiver_pairs,
                count(DISTINCT em) AS em_count,
                count(DISTINCT target) AS target_count,
                max(abs(pds)) AS best_abs_pds,
                avg(abs(pds)) AS mean_abs_pds,
                avg(pds) AS mean_pds
            FROM surface
            GROUP BY donor, contrast, day, ligand, receptor
        ), ranked AS (
            SELECT
                *,
                rank() OVER (
                    PARTITION BY donor, contrast ORDER BY best_abs_pds DESC
                ) AS edge_rank,
                count(*) OVER (PARTITION BY donor, contrast) AS edge_count,
                percent_rank() OVER (
                    PARTITION BY donor, contrast ORDER BY best_abs_pds
                ) AS abs_pds_percentile
            FROM aggregated
        )
        SELECT
            *,
            CASE
                WHEN abs_pds_percentile >= {TOP_DECILE_PERCENTILE}
                    THEN 'top_decile'
                WHEN abs_pds_percentile >= {TOP_QUARTILE_PERCENTILE}
                    THEN 'top_quartile'
                ELSE 'below_top_quartile'
            END AS rank_band
        FROM ranked
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW anchor_status AS
        WITH edge_presence AS (
            SELECT ligand, receptor, count(*) AS observed_paths
            FROM surface
            GROUP BY ligand, receptor
        ), target_presence AS (
            SELECT target AS receptor, count(*) AS receptor_as_target_paths
            FROM surface
            GROUP BY target
        ), receptor_presence AS (
            SELECT receptor, count(*) AS receptor_role_paths
            FROM surface
            GROUP BY receptor
        )
        SELECT
            a.*,
            coalesce(e.observed_paths, 0) AS observed_paths,
            coalesce(t.receptor_as_target_paths, 0) AS receptor_as_target_paths,
            coalesce(r.receptor_role_paths, 0) AS receptor_role_paths,
            CASE
                WHEN t.receptor_as_target_paths IS NOT NULL
                 AND r.receptor_role_paths IS NOT NULL THEN 'target_and_receptor'
                WHEN t.receptor_as_target_paths IS NOT NULL THEN 'target_only'
                WHEN r.receptor_role_paths IS NOT NULL THEN 'receptor_only'
                ELSE 'absent_both_roles'
            END AS receptor_role_status,
            e.observed_paths IS NOT NULL AS scored,
            CASE
                WHEN e.observed_paths IS NOT NULL THEN 'tier1_reconstructable'
                WHEN t.receptor_as_target_paths IS NOT NULL THEN 'tier2_target_observed'
                ELSE 'absent_no_target_evidence'
            END AS tier,
            CASE
                WHEN e.observed_paths IS NOT NULL
                    THEN 'ligand-receptor edge appears in the filtered surface'
                WHEN t.receptor_as_target_paths IS NOT NULL
                 AND r.receptor_role_paths IS NOT NULL
                    THEN 'edge absent; receptor appears as Target and as Receptor in other edges'
                WHEN t.receptor_as_target_paths IS NOT NULL
                    THEN 'edge absent; receptor appears only as a downstream Target'
                WHEN r.receptor_role_paths IS NOT NULL
                    THEN 'edge absent; receptor appears as Receptor in other edges but not as Target'
                ELSE 'edge absent; receptor not observed as a downstream Target'
            END AS evidence_reason
        FROM anchors a
        LEFT JOIN edge_presence e USING (ligand, receptor)
        LEFT JOIN target_presence t USING (receptor)
        LEFT JOIN receptor_presence r USING (receptor)
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW anchor_evidence AS
        WITH contrast_grid AS (
            SELECT DISTINCT donor, contrast, day FROM surface
        )
        SELECT
            g.donor,
            g.contrast,
            g.day,
            s.anchor_id,
            s.axis,
            s.ligand,
            s.receptor,
            s.prior_design,
            s.tier,
            s.scored,
            e.ligand IS NOT NULL AS formed,
            e.path_count,
            e.sender_receiver_pairs,
            e.em_count,
            e.target_count,
            e.best_abs_pds,
            e.mean_abs_pds,
            e.mean_pds,
            e.edge_rank,
            e.edge_count,
            e.abs_pds_percentile,
            e.rank_band
        FROM anchor_status s
        CROSS JOIN contrast_grid g
        LEFT JOIN edge_evidence e
          ON e.donor = g.donor
         AND e.contrast = g.contrast
         AND e.ligand = s.ligand
         AND e.receptor = s.receptor
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW edge_trends AS
        SELECT
            donor,
            ligand,
            receptor,
            count(*) AS contrasts_observed,
            min(day) AS first_day,
            max(day) AS last_day,
            arg_min(mean_pds, day) AS first_mean_pds,
            arg_max(mean_pds, day) AS last_mean_pds,
            regr_slope(mean_pds, day) AS pds_per_day,
            CASE
                WHEN count(*) < 2 THEN 'insufficient_timepoints'
                WHEN regr_slope(mean_pds, day) > 0 THEN 'rising'
                WHEN regr_slope(mean_pds, day) < 0 THEN 'falling'
                ELSE 'flat'
            END AS direction
        FROM edge_evidence e
        WHERE NOT EXISTS (
            SELECT 1
            FROM effective_anchor_gene_families f
            WHERE f.gene = e.ligand AND f.family = 'mhc_derived'
        )
        GROUP BY donor, ligand, receptor
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW cross_donor_concordance AS
        SELECT
            coalesce(d1.ligand, d2.ligand) AS ligand,
            coalesce(d1.receptor, d2.receptor) AS receptor,
            d1.contrasts_observed AS donor1_contrasts,
            d1.pds_per_day AS donor1_pds_per_day,
            d1.direction AS donor1_direction,
            d2.contrasts_observed AS donor2_contrasts,
            d2.pds_per_day AS donor2_pds_per_day,
            d2.direction AS donor2_direction,
            CASE
                WHEN d1.ligand IS NULL OR d2.ligand IS NULL THEN 'missing_donor'
                WHEN d1.direction = 'insufficient_timepoints'
                  OR d2.direction = 'insufficient_timepoints' THEN 'insufficient_timepoints'
                WHEN d1.direction = 'flat' OR d2.direction = 'flat' THEN 'flat_in_one_or_both'
                WHEN d1.direction = d2.direction THEN 'same_direction'
                ELSE 'opposite_direction'
            END AS concordance
        FROM (SELECT * FROM edge_trends WHERE donor = 'donor1') d1
        FULL OUTER JOIN (SELECT * FROM edge_trends WHERE donor = 'donor2') d2
          USING (ligand, receptor)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW novel_candidates AS
        WITH classified AS (
            SELECT
                e.*,
                lf.family AS ligand_anchor_family,
                rf.family AS receptor_anchor_family,
                a.anchor_id,
                n.curated_annotation
            FROM edge_evidence e
            LEFT JOIN anchors a USING (ligand, receptor)
            LEFT JOIN effective_anchor_gene_families lf ON lf.gene = e.ligand
            LEFT JOIN effective_anchor_gene_families rf ON rf.gene = e.receptor
            LEFT JOIN curated_novel_annotations n USING (ligand, receptor)
        )
        SELECT
            donor,
            contrast,
            day,
            ligand,
            receptor,
            path_count,
            sender_receiver_pairs,
            em_count,
            target_count,
            best_abs_pds,
            mean_abs_pds,
            mean_pds,
            edge_rank,
            edge_count,
            abs_pds_percentile,
            rank_band,
            ligand_anchor_family,
            receptor_anchor_family,
            coalesce(curated_annotation, '') AS curated_annotation
        FROM classified
        WHERE anchor_id IS NULL
          AND ligand_anchor_family IS NULL
          AND receptor_anchor_family IS NULL
          AND abs_pds_percentile >= {TOP_QUARTILE_PERCENTILE}
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW novel_candidate_summary AS
        SELECT
            n.ligand,
            n.receptor,
            count(DISTINCT n.donor) AS donors_observed,
            count(*) AS top_band_contrasts,
            count(*) FILTER (WHERE n.rank_band = 'top_decile') AS top_decile_contrasts,
            max(n.abs_pds_percentile) AS max_abs_pds_percentile,
            avg(n.abs_pds_percentile) AS mean_abs_pds_percentile,
            max(n.best_abs_pds) AS max_best_abs_pds,
            avg(n.mean_pds) AS mean_signed_pds,
            max(n.curated_annotation) AS curated_annotation,
            c.donor1_pds_per_day,
            c.donor1_direction,
            c.donor2_pds_per_day,
            c.donor2_direction,
            c.concordance
        FROM novel_candidates n
        LEFT JOIN cross_donor_concordance c USING (ligand, receptor)
        GROUP BY
            n.ligand,
            n.receptor,
            c.donor1_pds_per_day,
            c.donor1_direction,
            c.donor2_pds_per_day,
            c.donor2_direction,
            c.concordance
        """
    )


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _export_view(
    connection: duckdb.DuckDBPyConnection,
    view: str,
    output_dir: Path,
    order_by: str,
) -> None:
    destination = _sql_path(output_dir / f"{view}.csv")
    connection.execute(
        f"COPY (SELECT * FROM {view} ORDER BY {order_by}) "
        f"TO '{destination}' (HEADER, DELIMITER ',')"
    )


def _print_summary(connection: duckdb.DuckDBPyConnection) -> None:
    print("\n=== Anchor tier assignment ===")
    for row in connection.execute(
        """
        SELECT ligand, receptor, tier, observed_paths, receptor_as_target_paths,
               receptor_role_paths, receptor_role_status
        FROM anchor_status ORDER BY anchor_id
        """
    ).fetchall():
        ligand, receptor, tier, observed_paths, target_paths, receptor_paths, role_status = row
        print(
            f"{ligand} → {receptor}: {tier}; "
            f"paths={observed_paths}, receptor-as-Target={target_paths}, "
            f"receptor-role paths={receptor_paths} ({role_status})"
        )

    print("\n=== Formed anchor landing ===")
    for row in connection.execute(
        """
        SELECT donor, contrast, ligand, receptor, best_abs_pds,
               abs_pds_percentile, edge_rank, edge_count
        FROM anchor_evidence
        WHERE formed
        ORDER BY anchor_id, donor, day
        """
    ).fetchall():
        donor, contrast, ligand, receptor, best_abs, percentile, rank, edge_count = row
        print(
            f"{donor} {contrast} {ligand} → {receptor}: "
            f"best |PDS|={best_abs:.3f}, percentile={100 * percentile:.1f}, "
            f"rank={rank}/{edge_count}"
        )

    print("\n=== Novel top-band edges (anchor-family filtered) ===")
    candidates = connection.execute(
        """
        SELECT ligand, receptor, donors_observed, top_band_contrasts,
               top_decile_contrasts, max_best_abs_pds, concordance
        FROM novel_candidate_summary
        ORDER BY donors_observed DESC, top_decile_contrasts DESC,
                 top_band_contrasts DESC, max_best_abs_pds DESC,
                 ligand, receptor
        """
    ).fetchall()
    if not candidates:
        print("No non-anchor edges remain after anchor-family filtering.")
        return
    for ligand, receptor, donors, top_band, top_decile, best_abs, concordance in candidates:
        print(
            f"{ligand} → {receptor}: donors={donors}, "
            f"top-band contrasts={top_band}, top-decile contrasts={top_decile}, "
            f"max |PDS|={best_abs:.3f}, concordance={concordance}"
        )


def main() -> None:
    args = _parse_args()
    files = _surface_files(args.input_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    try:
        _register_inputs(connection, files)
        _register_reference_tables(connection)
        _build_evidence_views(connection)

        exports = {
            "surface_summary": "donor, day",
            "edge_evidence": "donor, day, edge_rank, ligand, receptor",
            "anchor_status": "anchor_id",
            "anchor_evidence": "anchor_id, donor, day",
            "edge_trends": "ligand, receptor, donor",
            "cross_donor_concordance": "ligand, receptor",
            "novel_candidates": "ligand, receptor, donor, day",
            "novel_candidate_summary": "donors_observed DESC, top_decile_contrasts DESC, ligand, receptor",
        }
        for view, order_by in exports.items():
            _export_view(connection, view, output_dir, order_by)
        _print_summary(connection)
    finally:
        connection.close()

    print(f"\nDerived evidence written to {output_dir}")


if __name__ == "__main__":
    main()
