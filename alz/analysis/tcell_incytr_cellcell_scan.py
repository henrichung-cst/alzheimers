"""Build bounded cell-to-cell signaling evidence from T-cell pair-mode Incytr.

The seven canonical wide parquet files are scanned with DuckDB. Wide pathway
tables are never materialized in pandas; only donor/day/state/channel
aggregates and the leading molecular backbones are exported. Cell fractions
come from the per-cell state-label CSVs and retain stateless CD4 and CD8 as
their own categories.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "incytr_pair_mode_tcells_percell_posneg"
)
DEFAULT_LABELS_DIR = PROJECT_ROOT / "outputs" / "reports" / "tcell_labeling" / "cells"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "derived_cellcell"

CONTRASTS = {
    "donor1": ("d13_d2", "d17_d2", "d20_d2"),
    "donor2": ("d5_d2", "d7_d2", "d9_d2", "d11_d2"),
}

# Display bounds requested by the approved report plan. They limit the bounded
# molecular evidence table; they are not analysis scores or pathway gates.
TOP_CHANNELS_PER_DONOR = 8
TOP_BACKBONES_PER_CHANNEL = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
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


def _label_files(labels_dir: Path) -> list[Path]:
    files = [labels_dir / f"{donor}_state_labels.csv" for donor in CONTRASTS]
    missing = [path for path in files if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing T-cell state-label files:\n{formatted}")
    return files


def _configure_connection(connection: duckdb.DuckDBPyConnection) -> None:
    spill_dir = Path(
        os.environ.get("DUCKDB_TEMP_DIR", Path.home() / ".cache" / "duckdb")
    )
    spill_dir.mkdir(parents=True, exist_ok=True)
    escaped = str(spill_dir).replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped}'")


def _register_inputs(
    connection: duckdb.DuckDBPyConnection,
    surface_files: list[Path],
    label_files: list[Path],
) -> None:
    connection.read_parquet(
        [str(path) for path in surface_files], filename=True, union_by_name=True
    ).create_view("wide_surface")
    connection.execute(
        """
        CREATE TEMP VIEW surface AS
        SELECT
            regexp_extract(filename, '/(donor[12])/', 1) AS donor,
            regexp_extract(filename, '/([^/]+)_incytr_output[.]parquet$', 1) AS contrast,
            CAST(regexp_extract(filename, '/d([0-9]+)_d2_incytr_output[.]parquet$', 1)
                 AS INTEGER) AS day,
            "Sender.group" AS sender_state,
            "Receiver.group" AS receiver_state,
            Ligand AS ligand,
            Receptor AS receptor,
            EM AS em,
            Target AS target,
            PDS AS pds
        FROM wide_surface
        WHERE PDS IS NOT NULL AND isfinite(PDS)
        """
    )

    connection.read_csv(
        [str(path) for path in label_files],
        filename=True,
        header=True,
        union_by_name=True,
    ).create_view("raw_state_labels")
    connection.execute(
        """
        CREATE TEMP VIEW retained_state_labels AS
        SELECT donor, CAST(day AS INTEGER) AS day, type AS state
        FROM raw_state_labels
        WHERE type IS NOT NULL AND type <> 'contaminant'
        """
    )


def _build_summary_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE macro_summary AS
        SELECT
            donor,
            contrast,
            day,
            count(*) AS path_count,
            count(DISTINCT (sender_state, receiver_state)) AS channel_count,
            count(*) FILTER (WHERE pds > 0) AS up_path_count,
            count(*) FILTER (WHERE pds < 0) AS down_path_count,
            coalesce(sum(pds) FILTER (WHERE pds > 0), 0.0) AS up_pds_mass,
            coalesce(sum(pds) FILTER (WHERE pds < 0), 0.0) AS down_pds_mass,
            sum(pds) AS net_pds_mass,
            avg(pds) AS mean_pds,
            avg(abs(pds)) AS mean_abs_pds
        FROM surface
        GROUP BY donor, contrast, day
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE cell_fractions AS
        WITH relevant_days AS (
            SELECT DISTINCT donor, day FROM surface
            UNION
            SELECT DISTINCT donor, 2 AS day FROM surface
        ), state_counts AS (
            SELECT l.donor, l.day, l.state, count(*) AS state_cell_count
            FROM retained_state_labels l
            INNER JOIN relevant_days d USING (donor, day)
            GROUP BY l.donor, l.day, l.state
        )
        SELECT
            donor,
            day,
            state,
            state_cell_count,
            sum(state_cell_count) OVER (PARTITION BY donor, day) AS retained_cell_count,
            state_cell_count::DOUBLE
                / sum(state_cell_count) OVER (PARTITION BY donor, day) AS cell_fraction
        FROM state_counts
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE state_role_paths AS
        SELECT
            donor,
            contrast,
            day,
            r.state,
            r.role,
            count(*) AS path_count,
            count(DISTINCT r.partner_state) AS partner_state_count,
            count(*) FILTER (WHERE pds > 0) AS up_path_count,
            count(*) FILTER (WHERE pds < 0) AS down_path_count,
            coalesce(sum(pds) FILTER (WHERE pds > 0), 0.0) AS up_pds_mass,
            coalesce(sum(pds) FILTER (WHERE pds < 0), 0.0) AS down_pds_mass,
            sum(abs(pds)) AS abs_pds_mass,
            sum(pds) AS net_pds_mass,
            avg(pds) AS mean_pds,
            avg(abs(pds)) AS mean_abs_pds
        FROM surface s,
        LATERAL (
            VALUES
                (s.sender_state, 'sender', s.receiver_state),
                (s.receiver_state, 'receiver', s.sender_state)
        ) AS r(state, role, partner_state)
        GROUP BY donor, contrast, day, r.state, r.role
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE state_role_summary AS
        WITH roles(role) AS (VALUES ('sender'), ('receiver'))
        SELECT
            f.donor,
            p.contrast,
            f.day,
            f.day = 2 AS is_baseline,
            f.state,
            CASE
                WHEN starts_with(f.state, 'CD4') THEN 'CD4'
                WHEN starts_with(f.state, 'CD8') THEN 'CD8'
                ELSE 'other'
            END AS lineage,
            CASE
                WHEN f.state IN ('CD4', 'CD8') THEN 'stateless'
                WHEN f.state LIKE '%Exhaust%' THEN 'exhaustion_associated'
                WHEN f.state LIKE '%ActivatedEffector'
                  OR f.state LIKE '%Cytotoxic%' THEN 'activated_or_cytotoxic'
                WHEN f.state LIKE '%RestingMemory' THEN 'resting_memory'
                WHEN f.state LIKE '%NaiveLike' THEN 'naive_like'
                ELSE 'other'
            END AS state_class,
            r.role,
            p.path_count,
            p.partner_state_count,
            p.up_path_count,
            p.down_path_count,
            p.up_pds_mass,
            p.down_pds_mass,
            p.abs_pds_mass,
            p.net_pds_mass,
            p.mean_pds,
            p.mean_abs_pds,
            f.state_cell_count,
            f.retained_cell_count,
            f.cell_fraction
        FROM cell_fractions f
        CROSS JOIN roles r
        LEFT JOIN state_role_paths p
          ON p.donor = f.donor
         AND p.day = f.day
         AND p.state = f.state
         AND p.role = r.role
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE channel_summary AS
        SELECT
            donor,
            contrast,
            day,
            sender_state,
            receiver_state,
            count(*) AS path_count,
            count(DISTINCT (ligand, receptor, em)) AS backbone_count,
            count(DISTINCT target) AS target_count,
            count(*) FILTER (WHERE pds > 0) AS up_path_count,
            count(*) FILTER (WHERE pds < 0) AS down_path_count,
            coalesce(sum(pds) FILTER (WHERE pds > 0), 0.0) AS up_pds_mass,
            coalesce(sum(pds) FILTER (WHERE pds < 0), 0.0) AS down_pds_mass,
            sum(pds) AS net_pds_mass,
            avg(pds) AS mean_pds,
            avg(abs(pds)) AS mean_abs_pds,
            max(abs(pds)) AS max_abs_pds
        FROM surface
        GROUP BY donor, contrast, day, sender_state, receiver_state
        """
    )


def _build_channel_trends(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TEMP TABLE channel_trend_base AS
        WITH ordered AS (
            SELECT
                *,
                lag(mean_pds) OVER (
                    PARTITION BY donor, sender_state, receiver_state ORDER BY day
                ) AS previous_mean_pds
            FROM channel_summary
        ), donor_coverage AS (
            SELECT donor, count(*) AS donor_day_count
            FROM macro_summary
            GROUP BY donor
        ), trend AS (
            SELECT
                c.donor,
                sender_state,
                receiver_state,
                count(*) AS days_observed,
                max(d.donor_day_count) AS donor_day_count,
                min(day) AS first_day,
                max(day) AS last_day,
                arg_min(mean_pds, day) AS first_mean_pds,
                arg_max(mean_pds, day) AS last_mean_pds,
                regr_slope(mean_pds, day) AS pds_per_day,
                avg(mean_abs_pds) AS mean_daily_abs_pds,
                avg(max_abs_pds) AS mean_daily_max_abs_pds
            FROM ordered c
            INNER JOIN donor_coverage d USING (donor)
            GROUP BY c.donor, sender_state, receiver_state
        ), steps AS (
            SELECT
                o.donor,
                o.sender_state,
                o.receiver_state,
                count(*) FILTER (WHERE o.previous_mean_pds IS NOT NULL) AS step_count,
                count(*) FILTER (
                    WHERE o.previous_mean_pds IS NOT NULL
                      AND (o.mean_pds - o.previous_mean_pds) * t.pds_per_day > 0
                ) AS aligned_step_count
            FROM ordered o
            INNER JOIN trend t USING (donor, sender_state, receiver_state)
            GROUP BY o.donor, o.sender_state, o.receiver_state
        )
        SELECT
            t.*,
            s.step_count,
            s.aligned_step_count,
            CASE
                WHEN t.days_observed < 2 THEN 'insufficient_days'
                WHEN t.days_observed < t.donor_day_count THEN 'partial_coverage'
                WHEN t.pds_per_day > 0 THEN 'rising'
                WHEN t.pds_per_day < 0 THEN 'falling'
                ELSE 'flat'
            END AS direction,
            CASE
                WHEN t.days_observed < 2 THEN 'insufficient_days'
                WHEN t.days_observed < t.donor_day_count THEN 'partial_coverage'
                WHEN t.pds_per_day = 0 THEN 'flat'
                WHEN s.aligned_step_count = s.step_count THEN 'consistent'
                ELSE 'mixed'
            END AS trend_consistency
        FROM trend t
        INNER JOIN steps s USING (donor, sender_state, receiver_state)
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE channel_trend_ranked AS
        SELECT
            *,
            CASE
                WHEN trend_consistency = 'consistent' THEN
                    rank() OVER (
                        PARTITION BY donor, trend_consistency
                        ORDER BY mean_daily_abs_pds DESC,
                                 abs(pds_per_day) DESC,
                                 sender_state,
                                 receiver_state
                    )
            END AS driver_rank
        FROM channel_trend_base
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE channel_agreement AS
        SELECT
            coalesce(d1.sender_state, d2.sender_state) AS sender_state,
            coalesce(d1.receiver_state, d2.receiver_state) AS receiver_state,
            d1.direction AS donor1_direction,
            d2.direction AS donor2_direction,
            CASE
                WHEN d1.sender_state IS NULL OR d2.sender_state IS NULL THEN 'missing_donor'
                WHEN d1.trend_consistency <> 'consistent'
                  OR d2.trend_consistency <> 'consistent' THEN 'not_consistent_both'
                WHEN d1.direction = d2.direction THEN 'same_direction'
                ELSE 'opposite_direction'
            END AS cross_donor_agreement
        FROM (SELECT * FROM channel_trend_ranked WHERE donor = 'donor1') d1
        FULL OUTER JOIN (SELECT * FROM channel_trend_ranked WHERE donor = 'donor2') d2
          USING (sender_state, receiver_state)
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE channel_trends AS
        SELECT
            t.*,
            a.donor1_direction,
            a.donor2_direction,
            a.cross_donor_agreement
        FROM channel_trend_ranked t
        LEFT JOIN channel_agreement a USING (sender_state, receiver_state)
        """
    )


def _build_backbone_table(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE channel_top_backbones AS
        WITH top_channels AS (
            SELECT donor, sender_state, receiver_state, driver_rank
            FROM channel_trends
            WHERE driver_rank <= {TOP_CHANNELS_PER_DONOR}
        ), backbone AS (
            SELECT
                s.donor,
                t.driver_rank AS channel_driver_rank,
                s.sender_state,
                s.receiver_state,
                s.ligand,
                s.receptor,
                s.em,
                count(DISTINCT s.contrast) AS days_observed,
                count(*) AS path_count,
                count(DISTINCT s.target) AS target_fan_count,
                coalesce(sum(s.pds) FILTER (WHERE s.pds > 0), 0.0) AS up_pds_mass,
                coalesce(sum(s.pds) FILTER (WHERE s.pds < 0), 0.0) AS down_pds_mass,
                avg(s.pds) AS mean_pds,
                avg(abs(s.pds)) AS mean_abs_pds,
                max(abs(s.pds)) AS max_abs_pds
            FROM surface s
            INNER JOIN top_channels t
              ON t.donor = s.donor
             AND t.sender_state = s.sender_state
             AND t.receiver_state = s.receiver_state
            GROUP BY
                s.donor,
                t.driver_rank,
                s.sender_state,
                s.receiver_state,
                s.ligand,
                s.receptor,
                s.em
        ), ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY donor, sender_state, receiver_state
                    ORDER BY mean_abs_pds DESC, max_abs_pds DESC,
                             ligand, receptor, em
                ) AS backbone_rank
            FROM backbone
        )
        SELECT *
        FROM ranked
        WHERE backbone_rank <= {TOP_BACKBONES_PER_CHANNEL}
        """
    )


def _validate(connection: duckdb.DuckDBPyConnection) -> None:
    missing_fractions = connection.execute(
        """
        SELECT count(*)
        FROM state_role_paths p
        LEFT JOIN cell_fractions f USING (donor, day, state)
        WHERE f.state IS NULL
        """
    ).fetchone()[0]
    if missing_fractions:
        raise ValueError(
            f"{missing_fractions} donor/day/state signaling summaries lack cell fractions"
        )

    bad_fraction_sums = connection.execute(
        """
        SELECT count(*)
        FROM (
            SELECT donor, day, sum(cell_fraction) AS total_fraction
            FROM cell_fractions
            GROUP BY donor, day
            HAVING abs(total_fraction - 1.0) > 1e-9
        )
        """
    ).fetchone()[0]
    if bad_fraction_sums:
        raise ValueError("Cell fractions do not sum to one within donor and day")


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _export_table(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    output_dir: Path,
    order_by: str,
) -> None:
    destination = _sql_path(output_dir / f"{table}.csv")
    connection.execute(
        f"COPY (SELECT * FROM {table} ORDER BY {order_by}) "
        f"TO '{destination}' (HEADER, DELIMITER ',')"
    )


def _print_summary(connection: duckdb.DuckDBPyConnection) -> None:
    print("\n=== Cell-to-cell signaling surface ===")
    for donor, day, paths, channels in connection.execute(
        """
        SELECT donor, day, path_count, channel_count
        FROM macro_summary ORDER BY donor, day
        """
    ).fetchall():
        print(f"{donor} day {day}: {paths:,} paths across {channels} channels")

    print("\n=== Leading consistent channel trends ===")
    for row in connection.execute(
        f"""
        SELECT donor, driver_rank, sender_state, receiver_state, direction,
               mean_daily_abs_pds, pds_per_day, cross_donor_agreement
        FROM channel_trends
        WHERE driver_rank <= {TOP_CHANNELS_PER_DONOR}
        ORDER BY donor, driver_rank
        """
    ).fetchall():
        donor, rank, sender, receiver, direction, magnitude, slope, agreement = row
        print(
            f"{donor} #{rank}: {sender} → {receiver}; {direction}; "
            f"mean daily |PDS|={magnitude:.3f}; slope={slope:+.4f} PDS/day; "
            f"{agreement}"
        )


def main() -> None:
    args = _parse_args()
    surface_files = _surface_files(args.input_dir.resolve())
    label_files = _label_files(args.labels_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    try:
        _configure_connection(connection)
        _register_inputs(connection, surface_files, label_files)
        _build_summary_tables(connection)
        _build_channel_trends(connection)
        _build_backbone_table(connection)
        _validate(connection)

        exports = {
            "macro_summary": "donor, day",
            "state_role_summary": "donor, day, role, state",
            "channel_summary": "donor, day, sender_state, receiver_state",
            "channel_trends": "donor, driver_rank NULLS LAST, sender_state, receiver_state",
            "channel_top_backbones": (
                "donor, channel_driver_rank, sender_state, receiver_state, backbone_rank"
            ),
        }
        for table, order_by in exports.items():
            _export_table(connection, table, output_dir, order_by)
        _print_summary(connection)
    finally:
        connection.close()

    print(f"\nDerived cell-to-cell evidence written to {output_dir}")


if __name__ == "__main__":
    main()
