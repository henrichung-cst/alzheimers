"""Kinase cell-type attribution by motif-peer resolution.

The MEA assay cannot distinguish kinases with confusable substrate motifs, so a
signal it attributes to one kinase may belong to any of its motif twins.  This
module uses transcript detection to break that tie per cell type: for a detected
kinase in a cell type, how many of its motif-confusable candidates (itself plus
its twins) are also transcribed there?  When the answer is one, the kinase is
the sole plausible source in that cell type — the specific attribution this
analysis exists to surface.  Every detected kinase is reported; a kinase with a
unique motif is sole-source by construction and is not excluded.

The calculation is deliberately kept separate from the MEA and Incytr bridge
artifacts.  In particular, it never materializes ``kinase_node_hits.parquet``;
the small evidence table written here is joined to viewer rows client-side.

Run with::

    pixi run python -m alz.cross_reference.motif_peer_narrowing

Outputs under ``outputs/reports/motif_peer_narrowing/`` are the auditable
evidence table, a JSON payload fragment, a peer roster, and a report of the
sole-source attributions and survivor distribution for every cohort.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from alz.cross_reference.specificity import DETECTION_FRAC_MIN

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "outputs" / "reports"
OUT_DIR = REPORTS / "motif_peer_narrowing"

SONG_EVIDENCE = REPORTS / "attribution_recovery" / "celltype_evidence_table.csv"
FIVEXFAD_EVIDENCE = (
    REPORTS / "kinase_attribution_5xfad" / "fivexfad_expression_specificity.csv"
)
TCELL_EVIDENCE = REPORTS / "kinase_attribution_tcells" / "donor1" / "unified_attribution_tcells.csv"

MOTIF_COSINE_CUT = 0.60
TCELL_COHORT = "tcell"
EXPECTED_SONG_UNDETECTED_KINASES = 59

_EVIDENCE_COLUMNS = (
    "cohort",
    "kinase",
    "cell_type",
    "detection_fraction",
    "motif_peers_detected",
    "motif_peers_informative",
    "motif_peer_roster",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _validate_detection_grain(
    frame: pd.DataFrame,
    *,
    cohort: str,
    fraction_column: str,
    detected_column: str | None = None,
) -> pd.DataFrame:
    """Return one row per kinase × cell type and reject conflicting repeats.

    Song and T-cell source files contain contrast rows, while the measurement
    is explicitly contrast-invariant.  A conflict is a source-data error, not a
    reason to choose an arbitrary first row.
    """
    required = {"kinase", "cell_type", fraction_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{cohort}: missing detection columns {sorted(missing)}")

    out = frame[["kinase", "cell_type", fraction_column] +
               ([detected_column] if detected_column else [])].copy()
    out["kinase"] = out["kinase"].astype(str)
    out["cell_type"] = out["cell_type"].astype(str)
    out["detection_fraction"] = pd.to_numeric(
        out[fraction_column], errors="coerce"
    ).fillna(0.0)
    if (out["detection_fraction"] < 0).any() or (out["detection_fraction"] > 1).any():
        bad = out.loc[
            (out["detection_fraction"] < 0) | (out["detection_fraction"] > 1),
            ["kinase", "cell_type", "detection_fraction"],
        ].head(3).to_dict("records")
        raise ValueError(f"{cohort}: detection fractions must be in [0, 1], examples={bad}")

    key = ["kinase", "cell_type"]
    conflicts = (
        out.groupby(key, dropna=False)["detection_fraction"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicts.any():
        examples = list(conflicts[conflicts].index[:3])
        raise ValueError(
            f"{cohort}: detection fraction is not contrast-invariant for {examples}"
        )

    if detected_column:
        out["detection_call"] = out[detected_column].map(_as_bool)
        call_conflicts = out.groupby(key, dropna=False)["detection_call"].nunique().gt(1)
        if call_conflicts.any():
            examples = list(call_conflicts[call_conflicts].index[:3])
            raise ValueError(f"{cohort}: detected call conflicts for {examples}")
    else:
        out["detection_call"] = True

    return (
        out.drop_duplicates(key, keep="first")
        [["kinase", "cell_type", "detection_fraction", "detection_call"]]
        .sort_values(key)
        .reset_index(drop=True)
    )


def load_detection_tables(
    *,
    song_path: Path = SONG_EVIDENCE,
    fivexfad_path: Path = FIVEXFAD_EVIDENCE,
    tcell_path: Path = TCELL_EVIDENCE,
) -> dict[str, pd.DataFrame]:
    """Load cohort-native detection fractions at the kinase × cell-type grain.

    The two 5xFAD tissues are separate cohort contexts because their snRNA
    detection fractions and Incytr pathway vocabularies are tissue-specific.
    The T-cell artifact currently available for the active MEA donor is donor1;
    it is emitted under the stable viewer cohort key ``tcell``.
    """
    missing = [str(p) for p in (song_path, fivexfad_path, tcell_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing motif-peer detection input(s): " + ", ".join(missing))

    song = _validate_detection_grain(
        pd.read_csv(song_path, usecols=[
            "kinase", "cell_type", "song_fraction_cells_expressing", "song_detected"
        ]),
        cohort="song",
        fraction_column="song_fraction_cells_expressing",
        detected_column="song_detected",
    )

    f5 = pd.read_csv(
        fivexfad_path,
        usecols=[
            "kinase", "tissue", "cell_type",
            "fivexfad_fraction_cells_expressing", "fivexfad_detected",
        ],
    )
    f5 = f5[~f5["cell_type"].astype(str).str.startswith("cluster-")].copy()
    f5_by_tissue: dict[str, pd.DataFrame] = {}
    for tissue, tissue_frame in f5.groupby("tissue", sort=True):
        f5_by_tissue[f"fivexfad_{tissue}"] = _validate_detection_grain(
            tissue_frame,
            cohort=f"fivexfad_{tissue}",
            fraction_column="fivexfad_fraction_cells_expressing",
            detected_column="fivexfad_detected",
        )

    tcell = _validate_detection_grain(
        pd.read_csv(tcell_path, usecols=[
            "kinase", "cell_type", "tcell_fraction_expressing", "tcell_detected"
        ]),
        cohort=TCELL_COHORT,
        fraction_column="tcell_fraction_expressing",
        detected_column="tcell_detected",
    )

    return {"song": song, **f5_by_tissue, TCELL_COHORT: tcell}


def centered_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine after removing each position's uniform amino-acid background."""
    if left.shape != right.shape:
        raise ValueError(f"motif matrices must have the same shape: {left.shape} vs {right.shape}")
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    left_flat = left_centered.ravel()
    right_flat = right_centered.ravel()
    denom = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    return float(left_flat @ right_flat / denom) if denom else float("nan")


def build_peer_sets(
    motifs: dict[str, dict],
    *,
    cosine_cut: float = MOTIF_COSINE_CUT,
) -> dict[str, list[dict[str, Any]]]:
    """Build local, within-kinase-type neighbour sets from viewer PSSMs."""
    by_type: dict[str, list[str]] = {}
    matrices: dict[str, np.ndarray] = {}
    for name, entry in motifs.items():
        kin_type = str(entry.get("kin_type", ""))
        matrix = np.asarray(entry.get("matrix", []), dtype=float)
        if not kin_type or matrix.ndim != 2 or matrix.size == 0:
            continue
        by_type.setdefault(kin_type, []).append(str(name))
        matrices[str(name)] = matrix

    peers: dict[str, list[dict[str, Any]]] = {str(name): [] for name in motifs}
    for names in by_type.values():
        for i, name in enumerate(sorted(names)):
            for other in sorted(names)[i + 1:]:
                cosine = centered_cosine(matrices[name], matrices[other])
                if np.isfinite(cosine) and cosine >= cosine_cut:
                    peers[name].append({"kinase": other, "centered_cosine": cosine})
                    peers[other].append({"kinase": name, "centered_cosine": cosine})
    for name in peers:
        peers[name].sort(key=lambda row: (row["kinase"], -row["centered_cosine"]))
    return peers


def load_kinase_motifs(kinases: Iterable[str]) -> dict[str, dict]:
    """Resolve PSSMs through the same alias-aware viewer motif builder."""
    from alz.viewer.shared.payload_helpers import _build_kinase_motifs

    return _build_kinase_motifs(sorted({str(k) for k in kinases if str(k)}))


def _detection_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], float]:
    return {
        (str(row.kinase), str(row.cell_type)): float(row.detection_fraction)
        for row in frame.itertuples(index=False)
    }


def _detection_call_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], bool]:
    if "detection_call" not in frame:
        return {
            (str(row.kinase), str(row.cell_type)): True
            for row in frame.itertuples(index=False)
        }
    return {
        (str(row.kinase), str(row.cell_type)): bool(row.detection_call)
        for row in frame.itertuples(index=False)
    }


def compute_narrowing(
    cohort: str,
    detections: pd.DataFrame,
    peer_sets: dict[str, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per detected (kinase, cell type); no floor.

    The deliverable is "which kinases are attributable to a specific cell type".
    Every detected kinase gets a row, including those with a unique motif (no
    twins): a kinase that is the *sole* plausible source (survivors k = 1) is the
    strongest result, whether it had 12 twins to rule out or none. Suppressing
    short candidate lists would delete exactly those cleanest attributions.
    """
    fractions = _detection_lookup(detections)
    calls = _detection_call_lookup(detections)
    detected_anywhere = {
        kinase for (kinase, _cell_type), fraction in fractions.items()
        if calls[(kinase, _cell_type)] and fraction >= DETECTION_FRAC_MIN
    }
    by_kinase: dict[str, list[dict[str, Any]]] = {}
    for kinase, rows in peer_sets.items():
        informative = [peer for peer in rows if peer["kinase"] in detected_anywhere]
        by_kinase[kinase] = informative

    rows: list[dict[str, Any]] = []
    unique_motif = 0
    k_distribution: Counter[int] = Counter()
    sole_source_kinases: set[str] = set()
    for kinase in sorted(detected_anywhere):
        informative = by_kinase.get(kinase, [])
        if not peer_sets.get(kinase):
            unique_motif += 1
        n_candidates = 1 + len(informative)
        for cell_type in sorted({cell for candidate, cell in fractions if candidate == kinase}):
            center_fraction = fractions[(kinase, cell_type)]
            if not calls[(kinase, cell_type)] or center_fraction < DETECTION_FRAC_MIN:
                continue
            roster = [
                {
                    "kinase": peer["kinase"],
                    "detection_fraction": fractions.get((peer["kinase"], cell_type), 0.0),
                }
                for peer in informative
            ]
            k_detected = 1 + sum(
                fraction >= DETECTION_FRAC_MIN
                for fraction in (row["detection_fraction"] for row in roster)
            )
            k_distribution[k_detected] += 1
            if k_detected == 1:
                sole_source_kinases.add(kinase)
            rows.append({
                "cohort": cohort,
                "kinase": kinase,
                "cell_type": cell_type,
                "detection_fraction": center_fraction,
                "motif_peers_detected": int(k_detected),
                "motif_peers_informative": int(n_candidates),
                "motif_peer_roster": roster,
            })

    emitted = pd.DataFrame(rows, columns=[*(_EVIDENCE_COLUMNS[:-1]), "motif_peer_roster"])
    sole_rows = int(sum(v for k, v in k_distribution.items() if k == 1))
    stats = {
        "cohort": cohort,
        "kinases_in_detection_table": int(detections["kinase"].nunique()),
        "kinases_detected_anywhere": len(detected_anywhere),
        "kinases_unique_motif": unique_motif,
        "rows_emitted": len(rows),
        "sole_source_rows": sole_rows,
        "sole_source_kinases": len(sole_source_kinases),
        "k_distribution": {str(k): int(v) for k, v in sorted(k_distribution.items())},
    }
    return emitted, stats


def validate_undetected_kinases_excluded(
    detections: pd.DataFrame,
    peer_sets: dict[str, list[dict[str, Any]]],
    *,
    expected_count: int | None = None,
) -> set[str]:
    """Assert that kinases silent across a cohort cannot enter any denominator."""
    detected = {
        str(kinase) for kinase, fraction, call in zip(
            detections["kinase"], detections["detection_fraction"],
            detections.get("detection_call", pd.Series(True, index=detections.index)),
        ) if bool(call) and float(fraction) >= DETECTION_FRAC_MIN
    }
    all_kinases = set(detections["kinase"].astype(str))
    undetected = all_kinases - detected
    if expected_count is not None and len(undetected) != expected_count:
        raise AssertionError(
            f"expected {expected_count} kinases undetected across cohort, found {len(undetected)}"
        )
    included_in_peer_sets = {
        peer["kinase"]
        for center, peers in peer_sets.items()
        if center in detected
        for peer in peers
        if peer["kinase"] in detected
    }
    leaked = undetected & included_in_peer_sets
    if leaked:
        raise AssertionError(
            f"undetected kinases entered motif-peer denominators: {sorted(leaked)[:5]}"
        )
    return undetected


def _json_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def build_artifacts(
    *,
    detections: dict[str, pd.DataFrame] | None = None,
    motifs: dict[str, dict] | None = None,
    out_dir: Path = OUT_DIR,
) -> dict[str, Any]:
    """Build and write all report artifacts; return the JSON payload fragment."""
    detections = detections or load_detection_tables()
    all_kinases = sorted({k for frame in detections.values() for k in frame["kinase"].unique()})
    song_kinases = set(detections.get("song", pd.DataFrame(columns=["kinase"]))["kinase"])
    motifs = motifs if motifs is not None else load_kinase_motifs(all_kinases)
    missing_song_motifs = sorted(song_kinases - set(motifs))
    if len(song_kinases) != 377 or missing_song_motifs:
        raise AssertionError(
            "Song PSSM resolution regression: "
            f"expected 377 kinases, found {len(song_kinases)}; "
            f"missing={missing_song_motifs[:5]}"
        )

    peer_sets = build_peer_sets(motifs)
    if "song" in detections:
        validate_undetected_kinases_excluded(
            detections["song"], peer_sets,
            expected_count=EXPECTED_SONG_UNDETECTED_KINASES,
        )
    all_frames: list[pd.DataFrame] = []
    cohort_payload: dict[str, Any] = {}
    stats: list[dict[str, Any]] = []
    for cohort in sorted(detections):
        frame, cohort_stats = compute_narrowing(cohort, detections[cohort], peer_sets)
        all_frames.append(frame)
        stats.append(cohort_stats)
        cohort_payload[cohort] = {"rows": _json_rows(frame), "stats": cohort_stats}

    # Peer membership is motif-derived and therefore cohort-independent: one copy,
    # not one per cohort. Only the per-row roster (which carries cohort- and
    # cell-type-specific detection fractions) varies.
    peer_roster = {
        kinase: [
            {"kinase": peer["kinase"], "centered_cosine": peer["centered_cosine"]}
            for peer in peer_sets.get(kinase, [])
        ]
        for kinase in sorted(peer_sets)
    }

    all_rows = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_frame = all_rows.copy()
    if not csv_frame.empty:
        csv_frame["motif_peer_roster"] = csv_frame["motif_peer_roster"].map(
            lambda value: json.dumps(value, separators=(",", ":"), sort_keys=True)
        )
    csv_frame.to_csv(out_dir / "kinase_celltype_evidence.csv", index=False)

    payload = {
        "schema_version": 1,
        "motif_cosine_cut": MOTIF_COSINE_CUT,
        "detection_fraction_min": DETECTION_FRAC_MIN,
        "peer_roster": peer_roster,
        "cohorts": cohort_payload,
    }
    (out_dir / "payload.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "peer_roster.json").write_text(
        json.dumps(
            {"schema_version": 1, "motif_cosine_cut": MOTIF_COSINE_CUT,
             "peer_roster": peer_roster},
            indent=2, sort_keys=True, allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, stats, all_rows, len(motifs))
    return payload


def _quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if len(values) else float("nan")


def _specificity_table(rows: pd.DataFrame) -> list[str]:
    """Per-cohort specific attributions, split by whether the analysis did work.

    A sole-source call (survivors = 1) is *resolved* when there were twins to
    rule out (candidates N > 1) — genuine discrimination — versus *unique-motif*
    (N = 1), where MEA was never ambiguous and the row only confirms detection.
    Median candidates (N) exposes candidate-pool thinness: a cohort that detects
    few kinases has small N, so its sole-source rate is inflated by absence, not
    discrimination.
    """
    out = [
        "| cohort | rows | median candidates (N) | sole-source: resolved (N>1) "
        "| kinases resolved | sole-source: unique-motif (N=1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cohort, frame in rows.groupby("cohort", sort=True):
        k = frame["motif_peers_detected"]
        n = frame["motif_peers_informative"]
        resolved = frame[(k == 1) & (n > 1)]
        unique = int(((k == 1) & (n == 1)).sum())
        out.append(
            f"| {cohort} | {len(frame)} | {_quantile(n, 0.5):.0f} | "
            f"{len(resolved)} | {resolved['kinase'].nunique()} | {unique} |"
        )
    return out


def _examples_table(rows: pd.DataFrame, cohort: str, limit: int = 12) -> list[str]:
    """Kinases that are the sole plausible source in a cell type.

    Ordered by how many motif twins were present to rule out (candidates N),
    so the most-resolved ambiguities lead; unique-motif kinases (N = 1, sole by
    construction) are equally valid sole-source calls and sit at the tail.
    """
    frame = rows[(rows["cohort"] == cohort) & (rows["motif_peers_detected"] == 1)]
    frame = frame.nlargest(limit, "motif_peers_informative")
    if frame.empty:
        return ["_No sole-source rows in this cohort._"]
    out = ["| kinase | cell type | motif twins present elsewhere (N−1) |", "|---|---|---:|"]
    for row in frame.itertuples(index=False):
        out.append(
            f"| {row.kinase} | {row.cell_type} | {row.motif_peers_informative - 1} |"
        )
    return out


def _write_report(
    out_dir: Path,
    stats: list[dict[str, Any]],
    rows: pd.DataFrame,
    n_motifs: int,
) -> None:
    sole = rows["motif_peers_detected"] == 1
    resolved = sole & (rows["motif_peers_informative"] > 1)
    resolved_rows = int(resolved.sum())
    resolved_kinases = int(rows.loc[resolved, "kinase"].nunique())
    unique_rows = int((sole & (rows["motif_peers_informative"] == 1)).sum())
    lines = [
        "# Kinase cell-type attribution by motif-peer resolution",
        "",
        "Generated by `alz/cross_reference/motif_peer_narrowing.py`.",
        "",
        "Which kinases can be attributed to a specific cell type? MEA cannot separate kinases",
        "with confusable substrate motifs, so a signal it attributes to one kinase might belong",
        "to any of its motif twins. Transcript detection breaks that tie: a twin that is not",
        "expressed in a cell type cannot be the source there.",
        "",
        "For each detected kinase in a cell type, **survivors** (`motif_peers_detected`) counts",
        "how many of its motif-confusable candidates — the kinase itself plus its twins — are",
        "transcribed there; **candidates** (`motif_peers_informative`) is the full confusable",
        "set. **survivors = 1 is the result of interest**: the kinase is the *sole plausible",
        "source* in that cell type. That holds whether it ruled out a dozen twins or never had",
        "any (a unique motif). Every detected kinase is reported; there is no floor.",
        "",
        f"- Resolved PSSMs: {n_motifs}",
        f"- Kinases with a unique motif (no twins): counted per cohort below",
        f"- Centered cosine cut: {MOTIF_COSINE_CUT:.2f}",
        f"- Detection gate: fraction ≥ {DETECTION_FRAC_MIN:.2f}",
        "",
        "## Headline",
        "",
        f"Across all cohorts, transcript detection **resolved genuine motif ambiguity** in",
        f"**{resolved_rows} cell-type attributions** — a kinase with ≥1 motif twin where none of",
        f"those twins are transcribed in that cell type, leaving it the sole plausible source —",
        f"spanning **{resolved_kinases} distinct kinases**. A further {unique_rows} sole-source",
        "rows come from kinases with a unique motif (no twins), where MEA was never ambiguous and",
        "the row only confirms detection; those are reported but are not discrimination. The rest",
        "quantify how far the candidate set narrows where a single answer is not reached.",
        "",
        "## Specific attributions by cohort",
        "",
        *_specificity_table(rows),
        "",
        "T-cell discriminates least: it detects the fewest kinases, so candidate pools are thin",
        "(median N well below the other cohorts) and most of its sole-source rows are unique-motif",
        "or twins simply absent from the shallow detected set — not resolved ambiguity. Its",
        "resolved count is the smallest and should be read with the depth caveat below.",
        "",
        "## Resolved attributions (Song)",
        "",
        "Kinases pinned to a cell type by ruling out present motif twins — ordered by how many",
        "twins existed to resolve (all have candidates N > 1):",
        "",
        *_examples_table(rows, "song"),
        "",
        "## Coverage",
        "",
        "| cohort | kinases in table | detected anywhere | unique motif | rows emitted | sole-source rows | sole-source kinases |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['cohort']} | {row['kinases_in_detection_table']} | "
            f"{row['kinases_detected_anywhere']} | {row['kinases_unique_motif']} | "
            f"{row['rows_emitted']} | {row['sole_source_rows']} | {row['sole_source_kinases']} |"
        )
    lines += [
        "",
        "Per-row survivors and candidates are in `kinase_celltype_evidence.csv`; the full",
        "survivor distribution is in `payload.json` under each cohort's `stats`.",
        "",
        "## Interpretation and limitations",
        "",
        "This is transcript evidence only: transcript ≠ protein ≠ activity. Absence is the",
        "informative direction — a twin that is not transcribed cannot be the source; presence",
        "of the surviving kinase is not itself evidence of activity. snRNA dropout depends on",
        "cell-type depth, and nuclei counts are not in the gate, so a twin absent from a shallow",
        "cell type may be a depth artifact rather than a true absence. The 0.60 motif cut is a",
        "threshold on a continuum, so a twin just below it is still somewhat confusable and is",
        "not counted against the attribution.",
        "",
        "5xFAD contexts are tissue-specific and have a cell-type-resolved MEA score of their own,",
        "so this resolution corroborates rather than substitutes for it. Song and T-cell have no",
        "cell-type-resolved score, so it is the primary transcript-based check on which kinase",
        "owns a bulk motif signal in a cell type.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_payload(path: Path = OUT_DIR / "payload.json") -> dict[str, Any]:
    """Load the generated payload fragment, or an empty contract when absent."""
    if not path.exists():
        return {
            "schema_version": 1,
            "motif_cosine_cut": MOTIF_COSINE_CUT,
            "detection_fraction_min": DETECTION_FRAC_MIN,
            "cohorts": {},
        }
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cohorts"), dict):
        raise ValueError(f"{path}: unexpected motif-peer payload schema")
    return payload


def evidence_lookup(payload: dict[str, Any], cohort: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Index emitted rows by kinase and cell type for payload builders."""
    rows = payload.get("cohorts", {}).get(cohort, {}).get("rows", [])
    return {
        (str(row.get("kinase", "")), str(row.get("cell_type", ""))): row
        for row in rows
        if row.get("kinase") and row.get("cell_type")
    }


ROSTER_FRACTION_DECIMALS = 3


def _compact_cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Split per-row rosters into one shared name table plus aligned fractions.

    A kinase's informative peer list is fixed within a cohort, so repeating the
    names (and the ``kinase``/``detection_fraction`` JSON keys) on every one of
    its cell-type rows costs an order of magnitude more than the numbers do.
    Names are emitted once per kinase; each row keeps only the detection
    fractions, positionally aligned to that list. ``ViewerPayload.motifPeerRoster``
    rehydrates the pairs client-side.
    """
    peer_names: dict[str, list[str]] = {}
    compact: list[dict[str, Any]] = []
    for row in rows:
        kinase = str(row["kinase"])
        roster = row.get("motif_peer_roster") or []
        names = [str(peer["kinase"]) for peer in roster]
        known = peer_names.setdefault(kinase, names)
        if known != names:
            raise ValueError(
                f"{kinase}: informative peer order is not stable across cell types"
            )
        compact.append({
            "kinase": kinase,
            "cell_type": str(row["cell_type"]),
            "detection_fraction": round(float(row["detection_fraction"]), ROSTER_FRACTION_DECIMALS),
            "motif_peers_detected": int(row["motif_peers_detected"]),
            "motif_peers_informative": int(row["motif_peers_informative"]),
            "motif_peer_fractions": [
                round(float(peer["detection_fraction"]), ROSTER_FRACTION_DECIMALS)
                for peer in roster
            ],
        })
    return {"peer_names": peer_names, "rows": compact}


def narrowing_sections(payload: dict[str, Any], cohorts: Iterable[str]) -> dict[str, Any]:
    """The viewer-bound slice of the payload: emitted rows for ``cohorts`` only.

    A viewer resolves at most its own contexts, so shipping every cohort's rows
    inlines the other cohorts' evidence into its HTML for nothing. ``stats`` and
    the motif ``peer_roster`` are report-side artifacts and are not carried
    either; what the UI renders is the per-cell-type detection fraction, which
    ``_compact_cohort`` keeps.
    """
    available = payload.get("cohorts", {})
    unknown = sorted(set(map(str, cohorts)) - set(available))
    if unknown:
        raise KeyError(f"motif-peer payload has no cohort(s) {unknown}")
    return {
        "schema_version": payload.get("schema_version"),
        "motif_cosine_cut": payload.get("motif_cosine_cut"),
        "detection_fraction_min": payload.get("detection_fraction_min"),
        "cohorts": {
            c: _compact_cohort(available[c].get("rows", [])) for c in map(str, cohorts)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = build_artifacts(out_dir=args.out_dir)
    for cohort, block in payload["cohorts"].items():
        log.info(
            "%s: emitted=%s sole-source rows=%s across %s kinases",
            cohort,
            block["stats"]["rows_emitted"],
            block["stats"]["sole_source_rows"],
            block["stats"]["sole_source_kinases"],
        )


if __name__ == "__main__":
    main()
