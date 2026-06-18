"""Projected-state T-cell MEA input loading, contrasts, and execution helpers.

Packet 3A/3B added input loading, matrix construction, and manifest logic.
Packet 3C adds runner execution (implemented with a local `_run_mea` wrapper
so per-state projected outputs can use required filenames without changing the
shared runner contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alz.core.mechanism_attribution import classify_mechanisms
from alz.core.mea_outputs import build_nes_fdr_matrices

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


TRACK_FILES = {
    "st": "ps_deconvoluted.csv",
    "py": "py_deconvoluted.csv",
}
_DONOR_CHOICES = ("donor1", "donor2", "both")

_KIND_SPECS = {
    "stoich": {
        "lfc_key": "stoich_lfc",
        "infix": "",
    },
    "raw": {
        "lfc_key": "raw_lfc",
        "infix": "_raw",
    },
}

STATE_META_COLUMNS = ("site_id", "gene_symbol", "motif")
CELL_COUNTS_RELATIVE = Path("scrna/cell_counts.csv")
PROJECTED_ROOT = Path("data") / "derived" / "tcells_incytr_inputs"
PROTEIN_FILENAME = "pr_deconvoluted.csv"
_MECHANISM_ATTR_PATH = "mechanism_attribution_projected_state.csv"

_STATE_DAY_RE = re.compile(r"^d(\d+)_(.+)$")
_DEFAULT_BASELINE_DAY = 2
_MISSING_STATE_LABEL = "_missing_input"


@dataclass(frozen=True)
class StateDayColumn:
    column: str
    day: int
    state: str


@dataclass(frozen=True)
class ProjectedInputs:
    donor: str
    track: str
    root: Path
    projected: pd.DataFrame
    protein: pd.DataFrame
    cell_counts: pd.DataFrame | None
    projected_path: Path
    protein_path: Path
    cell_counts_path: Path | None


def parse_state_day_columns(columns: Iterable[str]) -> list[StateDayColumn]:
    """Extract ``d{day}_{state}`` columns in source order."""
    parsed: list[StateDayColumn] = []
    for column in columns:
        match = _STATE_DAY_RE.match(str(column))
        if match is None:
            continue
        parsed.append(StateDayColumn(
            column=str(column),
            day=int(match.group(1)),
            state=match.group(2),
        ))
    return parsed


def _coerce_day(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        if value.startswith("d") and value[1:].isdigit():
            return int(value[1:])
        if value.isdigit():
            return int(value)
    return None


def _cell_counts_by_day(frame: pd.DataFrame | None, state: str) -> dict[int, int]:
    if frame is None:
        return {}
    required = {"state", "day", "n_cells"}
    if not required.issubset(frame.columns):
        return {}

    subset = frame.loc[frame["state"].astype(str) == state, ["day", "n_cells"]].copy()
    if subset.empty:
        return {}

    subset["day"] = subset["day"].map(_coerce_day)
    subset = subset.dropna(subset=["day"])
    subset["day"] = subset["day"].astype(int)
    subset["n_cells"] = pd.to_numeric(subset["n_cells"], errors="coerce")
    subset = subset.dropna(subset=["n_cells"])
    subset["n_cells"] = subset["n_cells"].astype(int)

    counts = subset.groupby("day", as_index=False)["n_cells"].sum()
    return {int(row.day): int(row.n_cells) for row in counts.itertuples(index=False)}


def _manifest_path(value: Path) -> str:
    try:
        return str(value.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(value)


def should_run_state(
    *,
    n_cells_by_day: dict[int, int],
    n_motif_sites: int,
    baseline_day: int,
    days_available: list[int],
    count_days: set[int] | None = None,
) -> tuple[bool, str | None]:
    """Apply initial Packet 3B gates and return (should_run, skip_reason)."""
    if baseline_day not in days_available:
        return False, "missing_baseline_day"

    target_days = [d for d in days_available if d > baseline_day]
    if len(target_days) == 0:
        return False, "no_post_baseline_days"

    if count_days is not None:
        required_days = {baseline_day, *target_days}
        if not required_days.issubset(count_days):
            return False, "no_cell_count_for_state_day"

    if n_motif_sites <= 0:
        return False, "no_motif_sites"

    if n_cells_by_day.get(baseline_day, 0) <= 0:
        return False, "state_has_no_cells"
    if not any(n_cells_by_day.get(day, 0) > 0 for day in target_days):
        return False, "state_has_no_cells"

    return True, None


def _build_state_qc_records(
    inputs: ProjectedInputs,
    baseline_day: int,
) -> list[dict[str, Any]]:
    parsed = parse_state_day_columns(inputs.projected.columns)
    by_state: dict[str, list[StateDayColumn]] = {}
    for col in parsed:
        by_state.setdefault(col.state, []).append(col)

    n_motif_sites = int(
        (inputs.projected["motif"].notna() & (inputs.projected["motif"].astype(str) != ""))
        .sum()
    )

    input_files = [
        _manifest_path(inputs.projected_path),
        _manifest_path(inputs.protein_path),
    ]
    if inputs.cell_counts_path is not None:
        input_files.append(_manifest_path(inputs.cell_counts_path))

    rows: list[dict[str, Any]] = []
    for state, state_columns in by_state.items():
        days = sorted({col.day for col in state_columns})
        n_cells_by_day = _cell_counts_by_day(inputs.cell_counts, state)
        should_run, skip_reason = should_run_state(
            n_cells_by_day=n_cells_by_day,
            n_motif_sites=n_motif_sites,
            baseline_day=baseline_day,
            days_available=days,
            count_days=set(n_cells_by_day),
        )
        rows.append(
            {
                "donor": inputs.donor,
                "state": state,
                "track": inputs.track,
                "baseline_day": baseline_day,
                "days_available": days,
                "n_cells_by_day": n_cells_by_day,
                "n_sites": int(len(inputs.projected)),
                "n_motif_sites": n_motif_sites,
                "input_files": input_files,
                "skip_reason": skip_reason,
                "_run_state": should_run,
            }
        )

    # keep stable output order for tests and CLI readability
    rows.sort(key=lambda row: row["state"])
    return rows


def summarize_state_qc(inputs: ProjectedInputs) -> pd.DataFrame:
    """Summarize projected-state QC by state for one donor/track input bundle."""
    rows = _build_state_qc_records(inputs, baseline_day=_DEFAULT_BASELINE_DAY)
    for row in rows:
        row.pop("_run_state", None)
    return pd.DataFrame(rows)


def build_manifest_records(inputs: ProjectedInputs, baseline_day: int = 2) -> list[dict]:
    """Build Packet 3B manifest skeleton rows for one donor/track combination."""
    if baseline_day <= 0:
        raise ValueError("baseline_day must be a positive day number")

    rows = _build_state_qc_records(inputs, baseline_day=baseline_day)

    out: list[dict] = []
    for row in rows:
        should_run = row.pop("_run_state", None)
        if should_run is None:
            should_run, skip_reason = should_run_state(
                n_cells_by_day=row["n_cells_by_day"],
                n_motif_sites=row["n_motif_sites"],
                baseline_day=baseline_day,
                days_available=row["days_available"],
            )
            row["skip_reason"] = skip_reason
        days = [d for d in row["days_available"] if d > baseline_day]
        if should_run:
            days_run = [d for d in days if row["n_cells_by_day"].get(d, 0) > 0]
        else:
            days_run = []
        out.append(
            {
                "donor": row["donor"],
                "state": row["state"],
                "track": row["track"],
                "kind": "projected_state",
                "baseline_day": row["baseline_day"],
                "days_available": row["days_available"],
                "days_run": days_run,
                "n_cells_by_day": row["n_cells_by_day"],
                "n_sites": row["n_sites"],
                "n_motif_sites": row["n_motif_sites"],
                "input_files": row["input_files"],
                "skip_reason": row["skip_reason"],
            }
        )

    return out


def _input_paths(
    donor: str,
    track: str,
    root: Path | None = None,
) -> tuple[Path, Path, Path]:
    if track not in TRACK_FILES:
        raise ValueError(f"unsupported track={track!r}; expected one of {sorted(TRACK_FILES)}")
    base = root if root is not None else _PROJECT_ROOT
    donor_dir = Path(base) / PROJECTED_ROOT / donor
    return (
        donor_dir / TRACK_FILES[track],
        donor_dir / PROTEIN_FILENAME,
        donor_dir / CELL_COUNTS_RELATIVE,
    )


def build_missing_input_manifest_records(
    donor: str,
    track: str,
    *,
    reason: str,
    baseline_day: int = _DEFAULT_BASELINE_DAY,
    input_files: list[str] | None = None,
    state: str | None = None,
) -> list[dict[str, Any]]:
    """Build manifest rows when required input files are unavailable."""
    if state is None:
        state = _MISSING_STATE_LABEL
    return [
        {
            "donor": donor,
            "state": state,
            "track": track,
            "kind": "projected_state",
            "baseline_day": baseline_day,
            "days_available": [],
            "days_run": [],
            "n_cells_by_day": {},
            "n_sites": 0,
            "n_motif_sites": 0,
            "input_files": list(input_files or []),
            "skip_reason": reason,
        }
    ]


def _load_inputs_or_missing_records(
    donor: str,
    track: str,
    *,
    root: Path | None = None,
    baseline_day: int = _DEFAULT_BASELINE_DAY,
    state_filter: str | None = None,
) -> tuple[ProjectedInputs | None, list[dict[str, Any]]]:
    projected_path, protein_path, _ = _input_paths(donor, track, root=root)
    if not projected_path.exists():
        return None, build_missing_input_manifest_records(
            donor,
            track,
            reason="missing_projected_phospho_file",
            baseline_day=baseline_day,
            input_files=[_manifest_path(projected_path), _manifest_path(protein_path)],
            state=state_filter,
        )

    if not protein_path.exists():
        return None, build_missing_input_manifest_records(
            donor,
            track,
            reason="missing_projected_protein_file",
            baseline_day=baseline_day,
            input_files=[_manifest_path(projected_path), _manifest_path(protein_path)],
            state=state_filter,
        )

    return load_projected_inputs(donor, track, root=root), []


def write_manifest(records: list[dict], out_path: Path) -> None:
    """Write manifest records to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=2)


def load_projected_inputs(
    donor: str,
    track: str,
    root: Path | None = None,
) -> ProjectedInputs:
    """Load projected-state phospho/protein matrices for one donor and track."""
    if track not in TRACK_FILES:
        raise ValueError(f"unsupported track={track!r}; expected one of {sorted(TRACK_FILES)}")
    base = root if root is not None else _PROJECT_ROOT
    projected_path, protein_path, cell_counts_path = _input_paths(donor, track, root=root)

    if not projected_path.exists():
        raise FileNotFoundError(f"missing projected file: {projected_path}")
    if not protein_path.exists():
        raise FileNotFoundError(f"missing protein file: {protein_path}")

    projected = pd.read_csv(projected_path)
    protein = pd.read_csv(protein_path)

    if not (
        {"gene_symbol"}.issubset(set(projected.columns))
        and {"gene_symbol"}.issubset(set(protein.columns))
    ):
        # keep error shape simple and explicit; caller can decide whether this is
        # a hard skip or a pre-run remediation.
        raise ValueError("both projected and protein inputs must include gene_symbol")

    cell_counts: pd.DataFrame | None = None
    if cell_counts_path.exists():
        cell_counts = pd.read_csv(cell_counts_path)

    return ProjectedInputs(
        donor=donor,
        track=track,
        root=Path(base),
        projected=projected,
        protein=protein,
        cell_counts=cell_counts,
        projected_path=projected_path,
        protein_path=protein_path,
        cell_counts_path=cell_counts_path if cell_counts is not None else None,
    )


def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _positive_log2(series: pd.Series) -> pd.Series:
    return np.log2(series.where(series > 0))


def _build_metadata_frame(projected: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        name: projected[name] if name in projected.columns else pd.NA
        for name in STATE_META_COLUMNS
    })


def build_state_matrices(
    inputs: ProjectedInputs,
    state: str,
    track: str,
) -> dict[str, pd.DataFrame]:
    """Build raw projected matrix and stoich matrix for a single state.

    Returns:
        dict[str, pd.DataFrame] with keys:
        - "raw": metadata + raw projected phospho values (`d{day}`)
        - "stoich": metadata + stoich values (`log2(projected) - log2(protein)`)
    """
    if track != inputs.track:
        raise ValueError(
            f"requested track {track!r} does not match loaded inputs track {inputs.track!r}"
        )

    projected_state_cols = [c for c in parse_state_day_columns(inputs.projected.columns) if c.state == state]
    protein_state_cols = [c for c in parse_state_day_columns(inputs.protein.columns) if c.state == state]

    if not projected_state_cols:
        raise ValueError(f"no projected columns for state={state!r} in {inputs.projected_path}")

    protein_col_by_day = {col.day: col.column for col in protein_state_cols}

    projected_cols = [col.column for col in projected_state_cols]
    projected_state_vals = _to_numeric(inputs.projected[projected_cols])
    renamed_cols = {col.column: f"d{col.day}" for col in projected_state_cols}

    metadata = _build_metadata_frame(inputs.projected)
    raw = pd.concat([metadata, projected_state_vals], axis=1)
    raw = raw.rename(columns=renamed_cols)

    protein_columns = [c.column for c in protein_state_cols]
    proteins = _to_numeric(inputs.protein[protein_columns]) if protein_columns else pd.DataFrame(index=inputs.protein.index)
    proteins["gene_symbol"] = inputs.protein["gene_symbol"].astype(str)
    proteins_by_gene = proteins.set_index("gene_symbol").groupby(level=0).first()

    gene_keys = inputs.projected["gene_symbol"].astype(str).tolist()

    stoich_data: dict[str, pd.Series] = {}
    for rec in projected_state_cols:
        day_label = f"d{rec.day}"
        pcol = protein_col_by_day.get(rec.day)
        projected_log = _positive_log2(projected_state_vals[rec.column].reset_index(drop=True))
        if pcol is None or pcol not in proteins_by_gene.columns:
            stoich_data[day_label] = pd.Series([np.nan] * len(inputs.projected))
            continue

        protein_for_gene = proteins_by_gene[pcol].reindex(gene_keys).reset_index(drop=True)
        protein_log = _positive_log2(protein_for_gene)
        stoich_data[day_label] = projected_log - protein_log

    stoich = pd.concat([metadata, pd.DataFrame(stoich_data)], axis=1)
    return {"raw": raw, "stoich": stoich}


def _add_metadata_columns(
    frame: pd.DataFrame,
    *,
    donor: str,
    state: str,
    track: str,
    kind: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["donor"] = donor
    out["state"] = state
    out["track"] = track
    out["kind"] = kind
    return out


def _add_timepoint(frame: pd.DataFrame, *, baseline_day: int) -> pd.DataFrame:
    if "contrast" not in frame.columns:
        return frame
    out = frame.copy()
    baseline_suffix = f"_vs_d{baseline_day}"
    out["timepoint"] = [
        value[:-len(baseline_suffix)] if value.endswith(baseline_suffix) else value
        for value in out["contrast"].astype(str)
    ]
    return out


def _build_results_by_contrast(
    matrix: pd.DataFrame,
    *,
    lfc_key: str,
    baseline_day: int,
    days: list[int],
) -> dict[str, dict[str, np.ndarray]]:
    baseline_col = f"d{baseline_day}"
    if baseline_col not in matrix.columns:
        raise ValueError(f"missing baseline column {baseline_col}")
    base = matrix[baseline_col]

    if lfc_key == "raw_lfc":
        base = _positive_log2(base).to_numpy(dtype=float)
    else:
        base = base.to_numpy(dtype=float)

    out: dict[str, dict[str, np.ndarray]] = {}
    for day in days:
        day_col = f"d{day}"
        if day_col not in matrix.columns:
            continue
        day_vals = matrix[day_col]
        if lfc_key == "raw_lfc":
            day_vals = _positive_log2(day_vals).to_numpy(dtype=float)
        else:
            day_vals = day_vals.to_numpy(dtype=float)
        out[f"{day_col}_vs_d{baseline_day}"] = {
            lfc_key: day_vals - base,
        }

    return out


def _state_timepoint_order(mea_df: pd.DataFrame) -> list[str]:
    """Return deterministic ``state|timepoint`` column order for aggregate matrices."""
    by_state: dict[str, set[str]] = {}
    for state, timepoint in zip(mea_df["state"], mea_df["timepoint"]):
        if pd.isna(state) or pd.isna(timepoint):
            continue
        state_key = str(state)
        tp_key = str(timepoint)
        by_state.setdefault(state_key, set()).add(tp_key)

    def tp_sort_key(value: str) -> tuple[int, int, str]:
        parsed = _coerce_day(value)
        if parsed is None:
            return (1, 0, value)
        return (0, parsed, value)

    columns: list[str] = []
    for state in sorted(by_state):
        for timepoint in sorted(by_state[state], key=tp_sort_key):
            columns.append(f"{state}|{timepoint}")
    return columns


def build_state_timepoint_matrices(
    mea_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot projected-state long MEA into kinase x (state|timepoint) NES/FDR matrices.

    Returns ``(nes_wide, fdr_wide)`` with one row per kinase.
    """
    if mea_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"kinase", "NES", "FDR", "state", "timepoint"}
    if not required.issubset(set(mea_df.columns)):
        return pd.DataFrame(), pd.DataFrame()

    work = mea_df.loc[:, ["kinase", "NES", "FDR", "state", "timepoint"]].copy()
    work = work.dropna(subset=["kinase", "state", "timepoint"])
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    work["state|timepoint"] = [
        f"{str(state)}|{str(timepoint)}" for state, timepoint in zip(work["state"], work["timepoint"])
    ]
    temp = work.assign(contrast=work["state|timepoint"])
    entity_order = _state_timepoint_order(work)
    if not entity_order:
        return pd.DataFrame(), pd.DataFrame()

    nes_wide, fdr_wide = build_nes_fdr_matrices(
        temp,
        entity_col_name="state|timepoint",
        contrast_suffix="",
        entity_order=entity_order,
    )
    return nes_wide, fdr_wide


def write_state_timepoint_aggregates(
    mea_df: pd.DataFrame,
    out_dir: Path,
    *,
    infix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write projected-state NES/FDR aggregates for one kind into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nes_matrix, fdr_matrix = build_state_timepoint_matrices(mea_df)
    if not isinstance(nes_matrix.index, pd.Index):
        nes_matrix = pd.DataFrame(columns=[])
    if not isinstance(fdr_matrix.index, pd.Index):
        fdr_matrix = pd.DataFrame(columns=[])

    nes_path = out_dir / f"kinase_state_timepoint_nes{infix}.csv"
    fdr_path = out_dir / f"kinase_state_timepoint_fdr{infix}.csv"
    nes_matrix.to_csv(nes_path)
    fdr_matrix.to_csv(fdr_path)
    return nes_matrix, fdr_matrix


def _infer_context_from_output_dir(out_dir: Path) -> tuple[str | None, str | None]:
    """Infer donor/track context from directory structure when explicit columns are missing."""
    donor: str | None = None
    track: str | None = None
    for part in reversed(out_dir.resolve().parts):
        if donor is None and part in {"donor1", "donor2"}:
            donor = part
        if track is None and part in {"st", "py"}:
            track = part
        if donor is not None and track is not None:
            break
    return donor, track


def write_projected_state_mechanism_attribution(
    out_dir: Path,
    *,
    cohort: str = "tcells",
) -> pd.DataFrame | None:
    """Classify paired projected-state stoich/raw MEA outputs for one projection run."""
    out_dir = Path(out_dir)
    stoich_path = out_dir / "mea_projected_state.csv"
    raw_path = out_dir / "mea_projected_state_raw.csv"
    if not stoich_path.exists():
        print(f"mechanism attribution skipped: missing {stoich_path}")
        return None
    if not raw_path.exists():
        print(f"mechanism attribution skipped: missing {raw_path}")
        return None

    stoich_df = pd.read_csv(stoich_path)
    raw_df = pd.read_csv(raw_path)
    if stoich_df.empty or raw_df.empty:
        print(
            "mechanism attribution skipped: stoich/raw projected MEA file empty; "
            f"stoich_rows={len(stoich_df)} raw_rows={len(raw_df)}"
        )
        return None

    inferred_donor, inferred_track = _infer_context_from_output_dir(out_dir)

    for frame in (stoich_df, raw_df):
        if "cohort" not in frame.columns:
            frame["cohort"] = cohort
        if "projection" not in frame.columns:
            frame["projection"] = "projected_state"
        if "donor" not in frame.columns:
            if inferred_donor is not None:
                frame["donor"] = inferred_donor
        if "track" not in frame.columns:
            if inferred_track is not None:
                frame["track"] = inferred_track
        if "timepoint" not in frame.columns:
            if "contrast" in frame.columns:
                frame["timepoint"] = frame["contrast"].astype(str).str.replace(
                    r"_vs_d\d+$", "", regex=True
                )

    context_cols = ["cohort", "donor", "track", "state", "timepoint"]
    for column in context_cols:
        if column not in stoich_df.columns or column not in raw_df.columns:
            print(f"mechanism attribution skipped: missing {column} context column")
            return None

    attributed = classify_mechanisms(
        stoich_df,
        raw_df,
        context_cols=context_cols,
    )
    out_path = out_dir / _MECHANISM_ATTR_PATH
    attributed.to_csv(out_path, index=False)
    print(f"mechanism attribution written: {out_path}  rows={len(attributed)}")
    return attributed


def _append_skip_reason(row: dict[str, Any], kind: str, reason: str) -> None:
    if "skip_reasons" not in row:
        row["skip_reasons"] = {}
    by_kind = row["skip_reasons"]
    if not isinstance(by_kind, dict):
        by_kind = {}
        row["skip_reasons"] = by_kind

    reasons = by_kind.get(kind)
    if reasons is None:
        by_kind[kind] = [reason]
    else:
        if reason not in reasons:
            reasons.append(reason)


def run_projected_state_mea(
    donor: str,
    track: str,
    out_dir: Path,
    *,
    states: list[str] | None = None,
    baseline_day: int = _DEFAULT_BASELINE_DAY,
    mea_caller=None,
) -> dict:
    """Run projected-state MEA for one donor and track.

    The shared runner is intentionally not used because its fixed output naming
    does not encode per-state projected outputs, so this function uses a local
    `_run_mea` wrapper and writes `mea_projected_state*` tables directly.
    """
    if baseline_day <= 0:
        raise ValueError("baseline_day must be a positive day number")
    if mea_caller is None:
        from alz.bulk_mea import enrich as kinase_enrich
        mea_caller = kinase_enrich._run_mea

    inputs = load_projected_inputs(donor, track)
    records = build_manifest_records(inputs, baseline_day=baseline_day)

    if states is not None:
        wanted = {state for state in states}
        records = [row for row in records if row["state"] in wanted]

    out_dir.mkdir(parents=True, exist_ok=True)

    all_mea: list[pd.DataFrame] = []
    all_shift: list[pd.DataFrame] = []
    all_winsorized: list[pd.DataFrame] = []
    all_substrate: list[pd.DataFrame] = []

    for row in records:
        if row.get("skip_reason") is not None:
            continue

        state = row["state"]
        matrices = build_state_matrices(inputs, state=state, track=track)
        days = [day for day in row.get("days_run", []) if isinstance(day, int)]
        if not days:
            continue

        for kind, spec in _KIND_SPECS.items():
            lfc_key = spec["lfc_key"]
            results_by_contrast = _build_results_by_contrast(
                matrices[kind],
                lfc_key=lfc_key,
                baseline_day=baseline_day,
                days=days,
            )
            if not results_by_contrast:
                continue

            motif_series = matrices[kind]["motif"]
            site_ids = matrices[kind]["site_id"].to_numpy()
            gene_symbols = matrices[kind]["gene_symbol"].to_numpy()

            mea_df, shift_df, wins_df, substrate_df = mea_caller(
                motif_series=motif_series,
                results_by_contrast=results_by_contrast,
                lfc_key=lfc_key,
                site_ids=site_ids,
                gene_symbols=gene_symbols,
                track=track,
            )

            if mea_df is None:
                mea_df = pd.DataFrame()
            if shift_df is None:
                shift_df = pd.DataFrame()
            if wins_df is None:
                wins_df = pd.DataFrame()
            if substrate_df is None:
                substrate_df = pd.DataFrame()

            mea_df = _add_metadata_columns(mea_df, donor=donor, state=state, track=track, kind=kind)
            mea_df = _add_timepoint(mea_df, baseline_day=baseline_day)
            shift_df = _add_metadata_columns(shift_df, donor=donor, state=state, track=track, kind=kind)
            wins_df = _add_metadata_columns(wins_df, donor=donor, state=state, track=track, kind=kind)
            substrate_df = _add_metadata_columns(substrate_df, donor=donor, state=state, track=track, kind=kind)

            all_mea.append(mea_df)
            all_shift.append(shift_df)
            all_winsorized.append(wins_df)
            all_substrate.append(substrate_df)

            if mea_df.empty:
                _append_skip_reason(row, kind, "empty_mea_result")

    def _concat(frames: list[pd.DataFrame], required: list[str]) -> pd.DataFrame:
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=required)

    mea_stacked = _concat(all_mea, ["donor", "state", "track", "kind", "timepoint"])
    shift_stacked = _concat(all_shift, ["donor", "state", "track", "kind"])
    wins_stacked = _concat(all_winsorized, ["donor", "state", "track", "kind"])
    substrate_stacked = _concat(all_substrate, ["donor", "state", "track", "kind"])

    stem_pairs = (
        ("mea_projected_state", mea_stacked),
        ("mea_global_shift_projected_state", shift_stacked),
        ("winsorized_sites_projected_state", wins_stacked),
        ("mea_substrate_sets_projected_state", substrate_stacked),
    )
    for stem, frame in stem_pairs:
        for kind, spec in _KIND_SPECS.items():
            infix = spec["infix"]
            path = out_dir / f"{stem}{infix}.csv"
            to_write = frame.loc[frame["kind"] == kind]
            to_write.to_csv(path, index=False)

    for kind, spec in _KIND_SPECS.items():
        infix = spec["infix"]
        kind_df = mea_stacked.loc[mea_stacked["kind"] == kind]
        write_state_timepoint_aggregates(kind_df, out_dir, infix=infix)

    mechanism_attribution = write_projected_state_mechanism_attribution(
        out_dir,
        cohort="tcells",
    )

    recurrence_note = out_dir / "recurrence_projected_state_deferred.txt"
    recurrence_note.write_text(
        "recurrence over (state, timepoint) is deferred because the recurrence axis is ambiguous."
    )

    manifest_path = out_dir / "projected_state_mea_manifest.json"
    write_manifest(records, manifest_path)

    return {
        "manifest_records": records,
        "manifest_path": manifest_path,
        "out_dir": out_dir,
        "mechanism_attribution": mechanism_attribution,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Projected-state T-cell MEA matrix loader/builder helpers. "
            "MEA execution is optional and writes outputs to a scratch directory "
            "instead of canonical project outputs."
        )
    )
    parser.add_argument("--donor", choices=[*_DONOR_CHOICES])
    parser.add_argument("--track", choices=["st", "py", "both"])
    parser.add_argument("--state")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Optional path for manifest JSON output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate requested inputs, then exit without running MEA.",
    )
    parser.add_argument(
        "--runner-scratch-dir",
        type=Path,
        metavar="DIR",
        help=(
            "Run projected-state MEA with a local runner wrapper and write scratch "
            "outputs only (no canonical output paths are touched)."
        ),
    )
    parser.add_argument(
        "--mechanism-attribution",
        action="store_true",
        help=(
            "When set, run projected-state mechanism attribution from paired long outputs "
            "after the MEA write path runs."
        ),
    )
    return parser


def _combo_out_dir(out_root: Path, donor: str, track: str, *, use_subdirs: bool) -> Path:
    if use_subdirs:
        return out_root / donor / track
    return out_root


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.donor is None or args.track is None:
        return

    if not args.dry_run and args.runner_scratch_dir is None:
        raise SystemExit("--runner-scratch-dir is required unless --dry-run is set")

    donors = ["donor1", "donor2"] if args.donor == "both" else [args.donor]
    tracks = ["st", "py"] if args.track == "both" else [args.track]

    dry_run = bool(args.dry_run)
    use_subdirs = (len(donors) * len(tracks)) > 1
    all_records: list[dict[str, Any]] = []

    for donor in donors:
        for track in tracks:
            inputs, missing_records = _load_inputs_or_missing_records(
                donor,
                track,
                baseline_day=_DEFAULT_BASELINE_DAY,
                state_filter=args.state,
            )

            combo_out_dir = (
                _combo_out_dir(
                    args.runner_scratch_dir,
                    donor,
                    track,
                    use_subdirs=use_subdirs,
                )
                if not dry_run
                else Path(".")
            )

            if dry_run:
                if inputs is not None:
                    combo_records = build_manifest_records(inputs, baseline_day=_DEFAULT_BASELINE_DAY)
                else:
                    combo_records = missing_records

                if args.state is not None:
                    combo_records = [r for r in combo_records if r["state"] == args.state]

                all_records.extend(combo_records)
                continue

            if inputs is None:
                if args.state is not None:
                    missing_records = [
                        r for r in missing_records if r["state"] == args.state
                    ]
                combo_out_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = combo_out_dir / "projected_state_mea_manifest.json"
                write_manifest(missing_records, manifest_path)
                continue

            result = run_projected_state_mea(
                donor,
                track,
                out_dir=combo_out_dir,
                states=[args.state] if args.state is not None else None,
                baseline_day=_DEFAULT_BASELINE_DAY,
            )
            if args.mechanism_attribution and result.get("mechanism_attribution") is None:
                write_projected_state_mechanism_attribution(combo_out_dir, cohort="tcells")

    if dry_run:
        qc = pd.DataFrame.from_records(all_records)
        if not qc.empty:
            print(
                qc[
                    [
                        "donor",
                        "state",
                        "track",
                        "baseline_day",
                        "days_available",
                        "skip_reason",
                    ]
                ].to_string(index=False)
            )
        if args.manifest_out is not None:
            write_manifest(all_records, args.manifest_out)
        return



if __name__ == "__main__":
    main()
