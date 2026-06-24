"""Kinase slice builders and projected-state MEA loader for the T-cell viewer."""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from alz.shared import config  # noqa: E402
from alz.tcell_viewer.paths import (  # noqa: E402
    KINASE_ATTRIBUTION_TCELLS_DIR,
    TCELLS_INCYTR_INPUTS_DIR,
    UNIFIED_VIEWER_DIR,
)
from alz.tcell_viewer.common import (  # noqa: E402
    DONORS,
    PROJECTILS_LABEL_MAP,
    _incytr_sanitize,
)

# ---------------------------------------------------------------------------
# Kinase attribution loader (donor MEA)
# ---------------------------------------------------------------------------

def _load_donor_kinase_attribution(donor: str) -> dict | None:
    """Load NES/FDR tables for one donor across {ST, pY} × {stoich, raw}.

    Returns None if no MEA outputs exist for the donor (donor2's expected case).
    """
    mea_dir = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor, "mea")
    manifest_path = os.path.join(mea_dir, "mea_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest.get("mea_ran"):
        return None

    tracks: dict[str, dict] = {}
    for track_suffix, residue_label in (("", "ST"), ("_pY", "Y")):
        for variant in ("", "_raw"):
            nes_path = os.path.join(
                mea_dir, f"kinase_timepoint_nes{variant}{track_suffix}.csv"
            )
            fdr_path = os.path.join(
                mea_dir, f"kinase_timepoint_fdr{variant}{track_suffix}.csv"
            )
            if not (os.path.exists(nes_path) and os.path.exists(fdr_path)):
                continue
            nes_df = pd.read_csv(nes_path)
            fdr_df = pd.read_csv(fdr_path)
            key = f"{residue_label}{variant}"
            tracks[key] = {"nes": nes_df, "fdr": fdr_df}

    if not tracks:
        return None
    return {"manifest": manifest, "tracks": tracks}


def _load_tcell_attribution(donor: str) -> "pd.DataFrame | None":
    """Shipped (concordant) within-cohort attribution rows for one donor, or
    None if the module hasn't been run. Source:
    outputs/reports/kinase_attribution_tcells/<donor>/unified_attribution_tcells.csv
    (alz/cross_reference/tcell_within_cohort.py)."""
    path = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor,
                        "unified_attribution_tcells.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


# ---------------------------------------------------------------------------
# Projected-state MEA (optional side-channel)
# ---------------------------------------------------------------------------

def _projected_state_candidate_dirs(donor: str) -> list[str]:
    """Potential output dirs for optional projected-state MEA.

    The state MEA CLI writes wherever ``--runner-scratch-dir`` points. The
    canonical viewer location, when a producer run is intentionally promoted,
    is ``<donor>/state_mea``. A few audit runs use nested donor/track folders;
    supporting those keeps the loader harmlessly permissive without making the
    viewer depend on them.
    """
    base = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor)
    candidates = [
        os.path.join(base, "state_mea"),
        os.path.join(base, "state_mea", "st"),
        os.path.join(base, "state_mea", "py"),
    ]
    return [p for p in candidates if os.path.isdir(p)]


def _read_projected_state_rows(path: str, *, kind: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    keep = [
        "kinase",
        "track",
        "state",
        "timepoint",
        "contrast",
        "NES",
        "FDR",
        "ES",
        "p-value",
        "Subs fraction",
    ]
    for col in ("kinase", "track", "state", "timepoint", "contrast", "NES", "FDR"):
        if col not in df.columns:
            return []
    rows = []
    for record in df[[c for c in keep if c in df.columns]].to_dict(orient="records"):
        record["kind"] = kind
        rows.append(record)
    return rows


def _read_projected_state_manifest(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            records = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(records, list):
        return []
    keep = [
        "donor",
        "state",
        "track",
        "kind",
        "baseline_day",
        "days_available",
        "days_run",
        "n_cells_by_day",
        "n_sites",
        "n_motif_sites",
        "skip_reason",
    ]
    return [
        {k: row.get(k) for k in keep if k in row}
        for row in records
        if isinstance(row, dict)
    ]


def _read_projected_state_mechanism(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    keep = [
        "cohort",
        "donor",
        "track",
        "state",
        "timepoint",
        "contrast",
        "projection",
        "kinase",
        "stoich_NES",
        "stoich_FDR",
        "raw_NES",
        "raw_FDR",
        "stoich_significant",
        "raw_significant",
        "sign_relation",
        "mechanism_call",
        "skip_reason",
    ]
    if "mechanism_score" in df.columns or "mechanism_call" not in df.columns:
        return []
    return df[[c for c in keep if c in df.columns]].to_dict(orient="records")


def _load_projected_state_mea_payload() -> dict | None:
    """Optional compact projected-state MEA block for the T-cell viewer.

    Returns None when promoted projected-state files are absent. That is the
    expected state until a real run succeeds in an environment with
    kinase_library installed.
    """
    by_context: dict[str, dict] = {}
    for donor in DONORS:
        donor_rows: list[dict] = []
        donor_manifest: list[dict] = []
        donor_mechanism: list[dict] = []
        source_files: list[str] = []

        for out_dir in _projected_state_candidate_dirs(donor):
            for filename, kind in (
                ("mea_projected_state.csv", "stoich"),
                ("mea_projected_state_raw.csv", "raw"),
            ):
                path = os.path.join(out_dir, filename)
                rows = _read_projected_state_rows(path, kind=kind)
                if rows:
                    donor_rows.extend(rows)
                    source_files.append(os.path.relpath(path, UNIFIED_VIEWER_DIR))

            manifest_path = os.path.join(out_dir, "projected_state_mea_manifest.json")
            manifest = _read_projected_state_manifest(manifest_path)
            if manifest:
                donor_manifest.extend(manifest)
                source_files.append(os.path.relpath(manifest_path, UNIFIED_VIEWER_DIR))

            mechanism_path = os.path.join(out_dir, "mechanism_attribution_projected_state.csv")
            mechanism = _read_projected_state_mechanism(mechanism_path)
            if mechanism:
                donor_mechanism.extend(mechanism)
                source_files.append(os.path.relpath(mechanism_path, UNIFIED_VIEWER_DIR))

        if not (donor_rows or donor_manifest or donor_mechanism):
            continue

        by_context[donor] = {
            "tracks": sorted({str(r.get("track")) for r in donor_rows if r.get("track")}),
            "states": sorted({str(r.get("state")) for r in donor_rows if r.get("state")}),
            "timepoints": sorted({str(r.get("timepoint")) for r in donor_rows if r.get("timepoint")}),
            "rows": donor_rows,
            "manifest": donor_manifest,
            "mechanism_attribution": donor_mechanism,
            "source_files": sorted(set(source_files)),
            "interpretation": "projected_state_mea",
        }

    if not by_context:
        return None
    return {"schema_version": 1, "by_context": by_context}


# ---------------------------------------------------------------------------
# Kinase-to-gene map
# ---------------------------------------------------------------------------

def _load_kinase_to_gene_map() -> dict[str, str]:
    """Kinase Library abbreviation -> gene symbol, with identity fallback."""
    try:
        df = pd.read_csv(config.MAPPING_CACHE_FILE)
    except Exception:
        return {}
    if not {"kinase_abbreviation", "gene_symbol"}.issubset(df.columns):
        return {}
    return {
        str(k): str(g)
        for k, g in zip(df["kinase_abbreviation"], df["gene_symbol"])
        if pd.notna(k) and pd.notna(g)
    }


# ---------------------------------------------------------------------------
# Attribution uniform baselines
# ---------------------------------------------------------------------------

def _tcell_attribution_uniform(donor: str) -> float | None:
    """Uniform specificity baseline 1/N_states for the badge tooltip."""
    cc = os.path.join(TCELLS_INCYTR_INPUTS_DIR, donor, "scrna", "cell_counts.csv")
    if not os.path.exists(cc):
        return None
    n_states = pd.read_csv(cc, usecols=["state"])["state"].nunique()
    return (1.0 / n_states) if n_states else None


def _nsclc_attribution_uniform() -> float | None:
    """Uniform baseline 1/N_coarse_groups for the external-reference NSCLC tier.

    The external 10x NSCLC reference is the human cohort's analog of the mouse
    viewer's WMB cross-check: a kinase's transcript share across the coarse TME
    groups (the 14 ProjecTILs T-states collapse to one T_NK group; the non-T
    lineages stay separate). The fold-over-uniform tier in the attribution
    verdict table is share / (1/N_groups), so we ship N_groups here.
    """
    src = config.NSCLC_KINASE_EXPRESSION_FILE
    if not os.path.exists(src):
        return None
    n_groups = pd.read_csv(src, usecols=["spec_group"])["spec_group"].nunique()
    return (1.0 / n_groups) if n_groups else None


# ---------------------------------------------------------------------------
# Attribution index
# ---------------------------------------------------------------------------

def _build_tcell_attribution_index(attr_df, kid: dict, short_contrasts: list) -> dict:
    """Columnar attribution_index consumed by the viewer JS getScopedAttribution.

    Maps kinase→kinase_id (the slice's kid map) and contrast day→contrast_id
    (index into the slice's short_contrasts). Rows whose kinase or day aren't in
    the MEA slice are dropped (shouldn't happen — same MEA source)."""
    c_idx = {c: i for i, c in enumerate(short_contrasts)}
    cols: dict[str, list] = {k: [] for k in (
        "kinase_id", "contrast_id", "cell_type", "tcell_specificity",
        "tcell_tier", "tcell_lfc", "tcell_concordance", "tcell_concordant",
        "tcell_consistency", "nes", "fdr")}

    def _f(v):  # NaN/None → null (the full grid carries NaN for genes absent in scRNA)
        return float(v) if pd.notna(v) else None

    for r in attr_df.itertuples(index=False):
        residue_type = str(getattr(r, "residue_type", "") or "")
        kk = kid.get((str(r.kinase), residue_type)) if residue_type else None
        if kk is None:
            kk = kid.get(str(r.kinase))
        cc = c_idx.get(str(r.contrast))
        if kk is None or cc is None:
            continue
        cols["kinase_id"].append(kk)
        cols["contrast_id"].append(cc)
        cols["cell_type"].append(str(r.cell_type))
        cols["tcell_specificity"].append(_f(r.tcell_specificity))
        cols["tcell_tier"].append(int(r.tcell_tier))
        cols["tcell_lfc"].append(_f(r.tcell_lfc))
        cols["tcell_concordance"].append(_f(r.tcell_concordance))
        cols["tcell_concordant"].append(bool(r.tcell_concordant))
        cols["tcell_consistency"].append(int(r.tcell_consistency))
        cols["nes"].append(_f(r.NES))
        cols["fdr"].append(_f(r.FDR))
    return cols


# ---------------------------------------------------------------------------
# Donor kinase slice
# ---------------------------------------------------------------------------

def _build_donor_kinases_slice(donor: str) -> dict | None:
    """Columnar kinases table for one donor.

    Schema mirrors the mouse-cohort kinases slice: `id`, `name`, `gene_symbol`,
    `residue_type`, `trajectory`, `peak_contrast`, `peak_NES`, `n_sig_contrasts`,
    `NES_<contrast>`, `FDR_<contrast>` — with cell-type attribution fields
    omitted (no per-cluster MEA on T-cells).
    """
    attribution = _load_donor_kinase_attribution(donor)
    if attribution is None:
        return None

    # Build one visible row per primary MEA residue track, mirroring the
    # unified viewer. ST and Y stoichiometry rows appear together; raw phospho
    # remains an audit/sensitivity comparison, not a separate browser row.
    track_frames: list[dict] = []
    short_contrasts: list[str] = []
    for residue_type in ("ST", "Y"):
        track = attribution["tracks"].get(residue_type)
        if track is None:
            continue
        nes = track["nes"].copy()
        fdr = track["fdr"].copy()
        if "kinase" not in nes.columns:
            continue
        contrasts = [c for c in nes.columns if c != "kinase"]
        contrast_to_short = {c: re.sub(r"^D\d+_", "", c) for c in contrasts}
        for c in contrasts:
            sc = contrast_to_short[c]
            if sc not in short_contrasts:
                short_contrasts.append(sc)
        for k in nes["kinase"].astype(str).tolist():
            track_frames.append({
                "kinase": k,
                "residue_type": residue_type,
                "nes": nes,
                "fdr": fdr,
                "contrasts": contrasts,
                "contrast_to_short": contrast_to_short,
            })
    if not track_frames:
        return None

    gene_map = _load_kinase_to_gene_map()
    rows = [(r["kinase"], r["residue_type"]) for r in track_frames]
    kid: dict = {}
    name_counts: dict[str, int] = {}
    for k, _residue_type in rows:
        name_counts[k] = name_counts.get(k, 0) + 1
    for i, (k, residue_type) in enumerate(rows):
        kid[(k, residue_type)] = i
        if name_counts.get(k, 0) == 1:
            kid[k] = i

    cols: dict[str, list] = {
        "id": list(range(len(rows))),
        "name": [k for k, _residue_type in rows],
        "gene_symbol": [gene_map.get(k, k) for k, _residue_type in rows],
        "residue_type": [residue_type for _k, residue_type in rows],
        "trajectory": [""] * len(rows),
        "peak_contrast": [],
        "peak_NES": [],
        "n_sig_contrasts": [],
        "top_celltype_1": [""] * len(rows),
    }
    for sc in short_contrasts:
        cols[f"NES_{sc}"] = []
        cols[f"FDR_{sc}"] = []

    fdr_thresh = float(attribution["manifest"].get("mea_fdr_thresh", 0.25))

    for row in track_frames:
        k = row["kinase"]
        contrasts = row["contrasts"]
        contrast_to_short = row["contrast_to_short"]
        nes_idx = row["nes"].set_index("kinase")
        fdr_idx = row["fdr"].set_index("kinase")
        nes_row = nes_idx.loc[k] if k in nes_idx.index else None
        fdr_row = fdr_idx.loc[k] if k in fdr_idx.index else None
        vals_by_short: dict[str, tuple[float, float]] = {}
        for c in contrasts:
            n_val = float(nes_row[c]) if nes_row is not None and pd.notna(nes_row[c]) else float("nan")
            f_val = float(fdr_row[c]) if fdr_row is not None and pd.notna(fdr_row[c]) else float("nan")
            vals_by_short[contrast_to_short[c]] = (n_val, f_val)
        nes_vec, fdr_vec = [], []
        for scn in short_contrasts:
            n_val, f_val = vals_by_short.get(scn, (float("nan"), float("nan")))
            nes_vec.append(n_val)
            fdr_vec.append(f_val)
            cols[f"NES_{scn}"].append(n_val)
            cols[f"FDR_{scn}"].append(f_val)
        # Peak: largest |NES| among contrasts with finite FDR.
        finite = [(i, nes_vec[i]) for i in range(len(short_contrasts))
                  if not (np.isnan(nes_vec[i]) or np.isnan(fdr_vec[i]))]
        if finite:
            i_peak = max(finite, key=lambda t: abs(t[1]))[0]
            cols["peak_contrast"].append(short_contrasts[i_peak])
            cols["peak_NES"].append(nes_vec[i_peak])
            cols["n_sig_contrasts"].append(
                int(sum(1 for j in range(len(short_contrasts))
                        if not np.isnan(fdr_vec[j]) and fdr_vec[j] < fdr_thresh))
            )
        else:
            cols["peak_contrast"].append("")
            cols["peak_NES"].append(float("nan"))
            cols["n_sig_contrasts"].append(0)

    # Within-cohort attribution: columnar index for the viewer + per-kinase
    # top attributed cell type (highest specificity tier, then concordance).
    attr_df = _load_tcell_attribution(donor)
    attribution_index = None
    attribution_uniform = None
    if attr_df is not None:
        attribution_index = _build_tcell_attribution_index(
            attr_df, kid, short_contrasts)
        attribution_uniform = _tcell_attribution_uniform(donor)
        top = attr_df.sort_values(
            ["tcell_tier", "tcell_concordance"], ascending=False)
        top_by_key = {}
        for r in top.itertuples(index=False):
            residue_type = str(getattr(r, "residue_type", "") or "")
            key = (str(r.kinase), residue_type)
            if key not in top_by_key:
                top_by_key[key] = str(r.cell_type)
        cols["top_celltype_1"] = [
            top_by_key.get((k, residue_type), "")
            for k, residue_type in rows
        ]

    return {
        "kinases_slice": cols,
        "kinase_names": [k for k, _residue_type in rows],
        "contrasts": short_contrasts,
        "kid": kid,
        "fdr_threshold": fdr_thresh,
        "attribution_index": attribution_index,
        "attribution_uniform": attribution_uniform,
    }


# ---------------------------------------------------------------------------
# Celltypes slice (ProjecTILs labels per donor)
# ---------------------------------------------------------------------------

def _load_donor_clusters(donor: str) -> list[str]:
    """Sorted ProjecTILs cluster names for one donor (alphanumeric-sanitized)."""
    pred_path = os.path.join(
        TCELLS_INCYTR_INPUTS_DIR, donor, "scrna", "projectils_predictions.csv"
    )
    if not os.path.exists(pred_path):
        return []
    df = pd.read_csv(pred_path, usecols=["functional.cluster"])
    raw = df["functional.cluster"].dropna().astype(str).unique().tolist()
    # ProjecTILs predictions still carry the dotted source labels. Use the same
    # explicit state map as tcells_scrna_extract.R / build_tcells_seurat.R so the
    # celltypes slice matches pair-mode `Sender.group` / `Receiver.group`.
    return sorted(PROJECTILS_LABEL_MAP.get(c, _incytr_sanitize(c)) for c in raw)


def _build_celltypes_slice(donor_clusters: dict[str, list[str]]) -> dict:
    """Union of per-donor ProjecTILs cluster names; one row per cluster.

    The viewer's existing `celltypes` consumer indexes by id; per-donor
    membership is exposed via `available_donors` so the donor selector can
    hide empty clusters.
    """
    seen: dict[str, set[str]] = {}
    for donor, clusters in donor_clusters.items():
        for c in clusters:
            seen.setdefault(c, set()).add(donor)
    ordered = sorted(seen.keys())
    return {
        "id": list(range(len(ordered))),
        "name": ordered,
        "tissue_category": ["T-cell"] * len(ordered),
        "available_donors": [sorted(seen[c]) for c in ordered],
    }


def _build_celltype_assignment() -> dict:
    """Compact ProjecTILs cell-type count summary for the T-cell viewer."""
    from typing import Any
    by_context: dict[str, dict] = {}
    all_states: set[str] = set()
    for donor in DONORS:
        scrna_dir = os.path.join(TCELLS_INCYTR_INPUTS_DIR, donor, "scrna")
        audit_path = os.path.join(scrna_dir, "state_audit.json")
        pred_path = os.path.join(scrna_dir, "projectils_predictions.csv")
        counts_path = os.path.join(scrna_dir, "cell_counts.csv")
        embedding_path = os.path.join(scrna_dir, "projectils_embeddings.csv")

        audit: dict[str, Any] = {}
        if os.path.exists(audit_path):
            with open(audit_path) as f:
                audit = json.load(f)

        confidence_by_state: dict[str, float | None] = {}
        if os.path.exists(pred_path):
            pred = pd.read_csv(
                pred_path,
                usecols=["functional.cluster", "functional.cluster.conf"],
            )
            resolved = pred.dropna(subset=["functional.cluster"]).copy()
            if not resolved.empty:
                conf = (
                    resolved.groupby("functional.cluster")["functional.cluster.conf"]
                    .median()
                    .dropna()
                )
                confidence_by_state = {
                    PROJECTILS_LABEL_MAP.get(str(k), _incytr_sanitize(str(k))): float(v)
                    for k, v in conf.items()
                }

        state_by_day: list[dict[str, Any]] = []
        if os.path.exists(counts_path):
            cc = pd.read_csv(counts_path)
            if {"state", "day", "n_cells"}.issubset(cc.columns):
                for r in cc.itertuples(index=False):
                    state = str(r.state)
                    all_states.add(state)
                    state_by_day.append({
                        "state": state,
                        "day": f"d{int(r.day)}",
                        "n_cells": int(r.n_cells),
                    })

        state_totals = {
            str(k): int(v) for k, v in (audit.get("state_totals") or {}).items()
        }
        all_states.update(state_totals.keys())
        embedding = {
            "available": False,
            "projection_references": [],
            "reductions": [],
            "points": {
                "x": [], "y": [], "state": [], "day": [],
                "projection_reference": [], "reduction": [], "confidence": [],
            },
        }
        if os.path.exists(embedding_path):
            emb = pd.read_csv(embedding_path)
            required = {
                "axis_1", "axis_2", "projection_reference", "reduction",
                "functional.cluster",
            }
            if required.issubset(emb.columns) and not emb.empty:
                emb = emb.dropna(subset=["axis_1", "axis_2", "functional.cluster"]).copy()
                if not emb.empty:
                    emb["state"] = [
                        PROJECTILS_LABEL_MAP.get(str(x), _incytr_sanitize(str(x)))
                        for x in emb["functional.cluster"]
                    ]
                    day_col = emb["day"] if "day" in emb.columns else pd.Series([pd.NA] * len(emb))
                    conf_col = (
                        emb["functional.cluster.conf"]
                        if "functional.cluster.conf" in emb.columns
                        else pd.Series([pd.NA] * len(emb))
                    )
                    embedding = {
                        "available": True,
                        "projection_references": sorted(
                            emb["projection_reference"].dropna().astype(str).unique().tolist()
                        ),
                        "reductions": sorted(
                            emb["reduction"].dropna().astype(str).unique().tolist()
                        ),
                        "points": {
                            "x": emb["axis_1"].astype(float).tolist(),
                            "y": emb["axis_2"].astype(float).tolist(),
                            "state": emb["state"].astype(str).tolist(),
                            "day": [
                                f"d{int(x)}" if pd.notna(x) else ""
                                for x in day_col
                            ],
                            "projection_reference": emb["projection_reference"].astype(str).tolist(),
                            "reduction": emb["reduction"].astype(str).tolist(),
                            "confidence": [
                                float(x) if pd.notna(x) else None
                                for x in conf_col
                            ],
                        },
                    }
        by_context[donor] = {
            "n_kept": int(audit.get("n_kept", 0) or 0),
            "state_totals": state_totals,
            "confidence_by_state": confidence_by_state,
            "state_by_day": state_by_day,
            "embedding": embedding,
        }

    return {
        "states": sorted(all_states),
        "by_context": by_context,
    }
