"""Kinase slice builders and projected-state MEA loader for the T-cell viewer."""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

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
            # Decomposition NES uses the RAW (non-stoichiometry-corrected)
            # projection. Stoich correction subtracts the per-state protein, and
            # the decomposition share scales both phosphosite and protein, so the
            # share cancels in the subtraction — collapsing per-state NES to the
            # bulk value (a known limitation). The raw projection keeps the share,
            # so the per-state NES is genuinely state-specific.
            path = os.path.join(out_dir, "mea_projected_state_raw.csv")
            rows = _read_projected_state_rows(path, kind="raw")
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

def _upper_gene_map(gene_map: dict) -> dict[str, str]:
    """Kinase -> upper-cased gene symbol, for case-insensitive gene lookups."""
    return {str(k): str(v).upper() for k, v in gene_map.items()}


# Crosswalk: within-cohort sanitized ProjecTILs state → NSCLC reference cell
# type (raw ProjecTILs `functional.cluster`). Both derive from the SAME
# ProjecTILs vocabulary, so this is a 1:1 relabeling — the exact inverse of
# PROJECTILS_LABEL_MAP, derived so a state rename is made in only one place.
_TCELL_STATE_TO_NSCLC = {v: k for k, v in PROJECTILS_LABEL_MAP.items()}


def _load_nsclc_detection() -> dict:
    """Per-(gene, NSCLC cell type) detection from the reference, the independent
    corroborator for within-cohort T-state attribution.
    Returns {(GENE_UPPER, nsclc_cell_type): (frac, detected)}.
    NSCLC detection = ≥10% of the cell type's cells express it — the single
    cross-cohort detection floor (specificity.DETECTION_FRAC_MIN), identical to
    the within-cohort scRNA so "detected" never means two different things.
    Computed in alz/reference/nsclc_expression.py."""
    src = config.NSCLC_KINASE_EXPRESSION_FILE
    if not os.path.exists(src):
        return {}
    usecols = ["gene_symbol", "cell_type", "fraction_cells_expressing", "detected"]
    df = pd.read_csv(src, usecols=usecols)
    out: dict = {}
    for r in df.itertuples(index=False):
        out[(str(r.gene_symbol).upper(), str(r.cell_type))] = (
            float(r.fraction_cells_expressing),
            bool(r.detected) if pd.notna(r.detected) else False)
    return out


def _load_nsclc_expression_full() -> "pd.DataFrame | None":
    """Full nsclc_kinase_expression.csv with per-native-type rows.
    Returns None when the file is absent."""
    src = config.NSCLC_KINASE_EXPRESSION_FILE
    if not os.path.exists(src):
        return None
    return pd.read_csv(src)


def _build_decomposition_index_from_state_mea(
    projected_state_payload: dict,
    kinases_slice: dict,
    meta: dict,
) -> dict | None:
    """Build a `decomposition_index`-compatible payload block from projected-state
    MEA rows, mirroring the structure used by the AD cohort in song.py.

    decomp_nes is the RAW (non-stoichiometry-corrected) per-state NES — the only
    projection that varies across states (stoich correction cancels the per-state
    decomposition share; see _load_projected_state_mea_payload).

    Schema (parallel arrays, one entry per (kinase, contrast, state) row):
      kinase_id[]   → uint16 index into kinases_slice["by_context"][donor]["name"]
      contrast_id[] → uint8  index into meta["contrasts"]
      cell_type[]   → ProjecTILs state label (str)
      decomp_nes[]  → float, 4 dp
      decomp_fdr[]  → float, 4 dp

    Returns None when the projected_state_payload carries no rows (gate
    dependency: mea_projected_state_raw.csv absent until state_mea runs).
    """
    by_context = (projected_state_payload or {}).get("by_context", {})
    # State MEA rows carry their own `donor`; the viewer's kinase identity is the
    # per-donor integer index into kinases_slice["by_context"][donor]["name"]
    # (== attribution_index.kinase_id == ctx.kinase_id). Build one name→id map per
    # donor so the emitted kinase_id matches the index the JS keys on.
    kinase_ctx = (kinases_slice or {}).get("by_context", {})
    kid_by_donor: dict[str, dict[str, int]] = {
        donor: {name: i for i, name in enumerate(block.get("name", []))}
        for donor, block in kinase_ctx.items()
    }
    # Contrast → position map.
    contrasts: list[str] = meta.get("contrasts", [])
    cid = {c: i for i, c in enumerate(contrasts)}

    all_rows: list[dict] = []
    for donor, block in by_context.items():
        for row in block.get("rows", []):
            all_rows.append({**row, "donor": row.get("donor", donor)})
    if not all_rows:
        return None

    kinase_ids: list[int] = []
    contrast_ids: list[int] = []
    cell_types: list[str] = []
    decomp_nes: list[float] = []
    decomp_fdr: list[float] = []
    seen: set = set()

    for r in all_rows:
        donor = str(r.get("donor", ""))
        kid = kid_by_donor.get(donor, {})
        k_name = str(r.get("kinase", ""))
        # The viewer's contrast axis is the bare day (meta["contrasts"] = d13,
        # d17, ...). State MEA emits contrast="d13_vs_d2" but timepoint="d13", so
        # key on timepoint to match `cid`; the raw "_vs_d2" form is never in cid.
        contrast = str(r.get("timepoint") or r.get("contrast", ""))
        state = str(r.get("state", ""))
        nes_val = r.get("NES")
        fdr_val = r.get("FDR")
        if k_name not in kid or contrast not in cid:
            continue
        key = (donor, k_name, contrast, state)
        if key in seen:
            continue
        seen.add(key)
        kinase_ids.append(kid[k_name])
        contrast_ids.append(cid[contrast])
        cell_types.append(state)
        decomp_nes.append(round(float(nes_val), 4) if nes_val is not None and nes_val == nes_val else None)
        decomp_fdr.append(round(float(fdr_val), 4) if fdr_val is not None and fdr_val == fdr_val else None)

    if not kinase_ids:
        return None

    print(f"  decomposition_index (state MEA): {len(kinase_ids):,} rows", flush=True)
    return {
        "kinase_id": kinase_ids,
        "contrast_id": contrast_ids,
        "cell_type": cell_types,
        "decomp_nes": decomp_nes,
        "decomp_fdr": decomp_fdr,
    }


def _tcell_celltype_pill(tier: int, corroborated: bool) -> str:
    """Confidence pill for the cell-type axis (CD8 / CD4 / Treg), driven by the
    N-normalized concentration tier rather than effective_n.

    The cell-type axis has only 3 units, so effective_n is bounded by 3 and the
    brain pill's absolute thresholds (1.5/3.0) make "broad" unreachable. The
    concentration tier (fold of the dominant type's share over the even 1/N) is
    the cardinality-independent exclusivity signal, so the pill reads it directly:

        tier 0  → none      no measurable cell-type expression distribution
        tier 1  → low       broad across CD8/CD4/Treg (not cell-type specific)
        tier ≥2 → moderate  concentrated in one cell type, uncorroborated
                  high      concentrated + NSCLC detects the kinase in T_NK

    (For N=3 the tier saturates at 2 — a share ≥5/3 is impossible — so very_high,
    which would need a stronger fold, is not reachable on this axis.)
    """
    if tier <= 0:
        return "none"
    if tier == 1:
        return "low"
    return "high" if corroborated else "moderate"


def _load_nsclc_tnk_detection() -> dict:
    """Per-gene NSCLC detection in the T_NK cell type — the cell-type-axis
    corroborator ("does the independent reference agree this is a T-cell kinase?").

    Derived from nsclc_kinase_expression.csv: a gene is T_NK-detected when any
    of its ProjecTILs T-state rows has binary_expressed = True (fraction ≥ 10% of
    cells). Returns {GENE_UPPER: bool}; a gene absent from the dict is outside
    the NSCLC probe panel (treated as uncorroborated, not disconfirming).
    """
    src = config.NSCLC_KINASE_EXPRESSION_FILE
    if not os.path.exists(src):
        return {}
    df = pd.read_csv(src, usecols=["gene_symbol", "spec_group", "binary_expressed"])
    t_rows = df[df["spec_group"] == "T_NK"]
    tnk_det = t_rows.groupby("gene_symbol")["binary_expressed"].any()
    return {str(g).upper(): bool(v) for g, v in tnk_det.items()}


def _load_nsclc_specificity_count() -> dict:
    """Per-gene NSCLC cell-type specificity — the SINGLE breadth metric.

    N of the coarse cell-type groups (T_NK + non-T: B_plasma / Myeloid /
    Epithelial / Endothelial / Fibroblast / Mast) whose cell-weighted
    group_fraction ≥ the shared detection floor (specificity.DETECTION_FRAC_MIN).
    Read straight from the canonical nsclc_kinase_expression.csv columns the
    builder writes (`specificity_count` / `expressing_groups` / `top_coarse_group`)
    — no inline recompute, no native-max variant.
    Returns {GENE_UPPER: (count, expressing_groups, n_groups_total, top_group)}.
    """
    src = config.NSCLC_KINASE_EXPRESSION_FILE
    if not os.path.exists(src):
        return {}
    df = pd.read_csv(src, usecols=["gene_symbol", "spec_group", "specificity_count",
                                   "expressing_groups", "top_coarse_group"])
    totals = df.groupby("gene_symbol")["spec_group"].nunique()
    out: dict = {}
    for r in df.drop_duplicates("gene_symbol").itertuples(index=False):
        gene = str(r.gene_symbol)
        groups = ([] if pd.isna(r.expressing_groups)
                  else [g for g in str(r.expressing_groups).split(",") if g])
        out[gene.upper()] = (
            int(r.specificity_count),
            groups,
            int(totals.get(gene, 0)),
            "" if pd.isna(r.top_coarse_group) else str(r.top_coarse_group),
        )
    return out


def _build_tcell_attribution_index(attr_df, kid: dict, short_contrasts: list,
                                   gene_map: dict, nsclc_det: dict,
                                   nsclc_tnk: dict) -> dict:
    """Columnar attribution_index consumed by the viewer JS getScopedAttribution.

    Maps kinase→kinase_id (the slice's kid map) and contrast day→contrast_id
    (index into the slice's short_contrasts). Rows whose kinase or day aren't in
    the MEA slice are dropped (shouldn't happen — same MEA source).

    Each (kinase, within-cohort state) row is joined to the NSCLC reference's
    detection at the crosswalked state (`nsclc_frac`, `nsclc_detected`) as an
    independent corroborator of the within-cohort attribution.

    Confidence tier (cell-type axis — CD8 / CD4 / Treg; see _tcell_celltype_pill):
        tier         = tcell_celltype_concentration_tier (fold of the dominant cell
                       type's share over the even 1/3; per-kinase). N-normalized,
                       so it reads correctly on a 3-unit axis where effective_n
                       (bounded by 3) cannot.
        corroborated = NSCLC reference detects the kinase in its T_NK cell type
                       (`nsclc_tnk`; detection = ≥10% of cells, the single cross-cohort floor). Kinase outside the NSCLC probe
                       panel → uncorroborated (not disconfirming).
        tier 0 → none; tier 1 → low (broad); tier ≥2 → high if corroborated else
        moderate.

    The per-(kinase, within-cohort STATE) NSCLC detection (`nsclc_frac`,
    `nsclc_detected`, via the fine state crosswalk `nsclc_det`) is still shown per
    row in the attribution detail — distinct from the T_NK corroboration above.

    Direction concordance (tcell_concordant / tcell_lfc) is info-only — never gates
    the tier (see TCELL_ATTRIBUTION_CAVEAT: OR≈1 for kinases).
    """
    c_idx = {c: i for i, c in enumerate(short_contrasts)}
    # Normalize the kinase→gene map to upper-case once; the loop below is over the
    # full kinase × state × contrast grid.
    gene_upper_map = _upper_gene_map(gene_map)

    # Pre-compute per-(kinase, residue_type) confidence tier on the CELL-TYPE axis
    # (CD8 / CD4 / Treg). tcell_top_celltype + tcell_celltype_concentration_tier are
    # per-kinase (same across all state×contrast rows). The cell-type axis has only
    # N=3 units, so the brain pill's absolute effective_n thresholds (1.5/3.0) don't
    # apply (eff is bounded by 3, "broad" would be unreachable); the N-normalized
    # concentration tier (fold of the dominant type's share over the even 1/N) IS
    # the right exclusivity signal at any cardinality. Detection fractions are shown
    # separately and never gate the specificity denominator or the tier.
    # Graceful column fallback: pre-A1-rename artifacts have `tcell_tier` (int)
    # and lack `tcell_celltype_concentration_tier` / `tcell_top_celltype`. Use
    # getattr so the builder degrades to tier=0 / top_ct="" on stale artifacts
    # rather than raising AttributeError — the gate re-run of tcell_within_cohort
    # produces the correct schema.
    kinase_meta: dict[tuple[str, str], dict] = {}
    for r in attr_df.itertuples(index=False):
        residue_type = str(getattr(r, "residue_type", "") or "")
        kk_key = (str(r.kinase), residue_type)
        if kk_key not in kinase_meta:
            _tier_raw = getattr(r, "tcell_celltype_concentration_tier", None)
            _top_ct_raw = getattr(r, "tcell_top_celltype", None)
            kinase_meta[kk_key] = {
                "tier": (int(_tier_raw) if _tier_raw is not None and pd.notna(_tier_raw) else 0),
                "top_ct": (str(_top_ct_raw) if _top_ct_raw is not None and pd.notna(_top_ct_raw) else ""),
                "gene_upper": gene_upper_map.get(str(r.kinase), str(r.kinase).upper()),
            }

    # corroborated = the independent NSCLC reference detects the kinase in its T_NK
    # cell type (detection = ≥10% of cells, the single cross-cohort floor). Absence from the NSCLC probe panel → uncorroborated (not
    # disconfirming). Pill: tier 0 (no cell-type expression) → none; tier 1 (broad
    # across cell types) → low; tier ≥2 (concentrated in one cell type) → high if
    # corroborated else moderate (uncorroborated caps at moderate, as in the brain
    # model). very_high is reserved for concentrated + corroborated AND broadly
    # T_NK-detected — see _tcell_celltype_pill.
    kinase_tier: dict[tuple[str, str], tuple[str, str]] = {}
    for kk_key, meta in kinase_meta.items():
        corroborated = bool(nsclc_tnk.get(meta["gene_upper"], False))
        in_panel = meta["gene_upper"] in nsclc_tnk
        tier_name = _tcell_celltype_pill(meta["tier"], corroborated)
        top_ct = meta["top_ct"]
        if tier_name == "none":
            basis = "No measurable cell-type (CD8/CD4/Treg) expression distribution"
        elif tier_name == "low":
            corr_s = ("NSCLC detects kinase in T_NK" if corroborated
                      else "not in NSCLC probe panel" if not in_panel
                      else "NSCLC does not detect kinase in T_NK")
            basis = (f"broadly expressed across CD8/CD4/Treg (not cell-type "
                     f"specific); {corr_s}")
        else:
            if corroborated:
                corr_note = "NSCLC detects kinase in T_NK (corroborated)"
            elif not in_panel:
                corr_note = ("NSCLC probe panel does not include this kinase; "
                             "uncorroborated (caps at moderate)")
            else:
                corr_note = ("NSCLC does not detect kinase in T_NK; "
                             "uncorroborated (caps at moderate)")
            basis = f"concentrated in {top_ct} (≥2× the even 1/3 share); {corr_note}"
        kinase_tier[kk_key] = (tier_name, basis)

    # Per-(kinase, residue) pill-anchor row = the most state-enriched ELIGIBLE state
    # (NaN enrichment never wins); fall back to the first state seen when a kinase
    # has no eligible state. The cell-type confidence pill is rendered once, on this
    # anchor row, in the attribution table (the rows are states, not cell types).
    pill_anchor: dict[tuple[str, str], tuple[str, float]] = {}
    seen_first: dict[tuple[str, str], str] = {}
    for r in attr_df.itertuples(index=False):
        rt = str(getattr(r, "residue_type", "") or "")
        key = (str(r.kinase), rt)
        seen_first.setdefault(key, str(r.cell_type))
        e = r.tcell_state_enrichment
        if pd.isna(e):
            continue
        if key not in pill_anchor or float(e) > pill_anchor[key][1]:
            pill_anchor[key] = (str(r.cell_type), float(e))

    def _anchor(key: tuple[str, str]) -> str:
        return pill_anchor[key][0] if key in pill_anchor else seen_first.get(key, "")

    cols: dict[str, list] = {k: [] for k in (
        "kinase_id", "contrast_id", "cell_type",
        "tcell_detected", "tcell_fraction_expressing", "tcell_state_enrichment",
        "tcell_effective_n", "tcell_top_celltype", "_pill_anchor",
        "tcell_lfc", "tcell_concordance", "tcell_concordant",
        "tcell_consistency", "nes", "fdr",
        "nsclc_frac", "nsclc_detected",
        "confidence_tier", "confidence_basis")}

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
        _kn = str(r.kinase)
        gene_upper = gene_upper_map.get(_kn, _kn.upper())
        nsclc_ct = _TCELL_STATE_TO_NSCLC.get(str(r.cell_type))
        det_entry = nsclc_det.get((gene_upper, nsclc_ct)) if nsclc_ct else None
        nsclc_frac = det_entry[0] if det_entry else None
        nsclc_detected = det_entry[1] if det_entry else None
        detected = bool(r.tcell_detected)
        kk_key = (str(r.kinase), residue_type)
        conf_tier, conf_basis = kinase_tier.get(kk_key, ("none", "no tier computed"))
        cols["kinase_id"].append(kk)
        cols["contrast_id"].append(cc)
        cols["cell_type"].append(str(r.cell_type))
        cols["tcell_detected"].append(detected)
        cols["tcell_fraction_expressing"].append(_f(r.tcell_fraction_expressing))
        cols["tcell_effective_n"].append(_f(r.tcell_effective_n))
        cols["tcell_top_celltype"].append(
            str(r.tcell_top_celltype) if pd.notna(r.tcell_top_celltype) else "")
        cols["_pill_anchor"].append(_anchor(kk_key))
        cols["tcell_state_enrichment"].append(_f(r.tcell_state_enrichment))
        cols["tcell_lfc"].append(_f(r.tcell_lfc))
        cols["tcell_concordance"].append(_f(r.tcell_concordance))
        cols["tcell_concordant"].append(bool(r.tcell_concordant))
        cols["tcell_consistency"].append(int(r.tcell_consistency))
        cols["nes"].append(_f(r.NES))
        cols["fdr"].append(_f(r.FDR))
        cols["nsclc_frac"].append(round(nsclc_frac, 4) if nsclc_frac is not None else None)
        cols["nsclc_detected"].append(nsclc_detected)
        cols["confidence_tier"].append(conf_tier)
        cols["confidence_basis"].append(conf_basis)
    return cols


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
        "tcell_celltype": [""] * len(rows),
        "tcell_celltype_tier": [0] * len(rows),
        # NSCLC specificity: N of the coarse groups with cell-weighted
        # group_fraction ≥ the shared detection floor, + denominator, members,
        # and dominant group. Single metric, read from the canonical CSV columns.
        "nsclc_specificity_count": [None] * len(rows),
        "nsclc_groups_total": [None] * len(rows),
        "nsclc_expressing_groups": [None] * len(rows),
        "nsclc_top_group": [""] * len(rows),
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

    gene_upper_map = _upper_gene_map(gene_map)

    # NSCLC cross-cell-type specificity (independent reference): per-kinase count
    # of coarse cell types whose cell-weighted prevalence ≥ the shared 10% floor,
    # out of those present, + members + dominant group. Gene-keyed and cohort-
    # independent — answers the cell-type-specificity question the within-cohort
    # T-STATE columns cannot.
    # Absent gene (outside the NSCLC probe panel) → left null ("n/a" in the UI).
    nsclc_spec = _load_nsclc_specificity_count()
    for i, (k, _residue_type) in enumerate(rows):
        gene_up = gene_upper_map.get(k, k.upper())
        spec_entry = nsclc_spec.get(gene_up)
        if spec_entry is not None:
            cols["nsclc_specificity_count"][i] = spec_entry[0]
            cols["nsclc_expressing_groups"][i] = spec_entry[1]
            cols["nsclc_groups_total"][i] = spec_entry[2]
            cols["nsclc_top_group"][i] = spec_entry[3]

    # Within-cohort attribution: columnar index for the viewer + per-kinase
    # headline STATE (most state-enriched eligible state) and the cell TYPE.
    attr_df = _load_tcell_attribution(donor)
    attribution_index = None
    if attr_df is not None:
        nsclc_det = _load_nsclc_detection()
        nsclc_tnk = _load_nsclc_tnk_detection()
        attribution_index = _build_tcell_attribution_index(
            attr_df, kid, short_contrasts, gene_map, nsclc_det, nsclc_tnk)
        # Headline STATE: the most state-enriched ELIGIBLE state (guarded
        # tcell_state_enrichment; ineligible/undetected states are NaN). A kinase
        # with no eligible state gets "" — no state headline (e.g. SYK).
        top = attr_df.sort_values(
            ["tcell_state_enrichment", "tcell_concordance"], ascending=False)
        top_by_key = {}
        for r in top.itertuples(index=False):
            residue_type = str(getattr(r, "residue_type", "") or "")
            key = (str(r.kinase), residue_type)
            if key in top_by_key or pd.isna(r.tcell_state_enrichment):
                continue
            top_by_key[key] = str(r.cell_type)
        cols["top_celltype_1"] = [
            top_by_key.get((k, residue_type), "")
            for k, residue_type in rows
        ]
        # Cell TYPE (CD8/CD4/Treg) + concentration tier — per-kinase, the
        # within-cohort cell-type confidence column. Distinct from top_celltype_1
        # (a STATE) and from the NSCLC cross-cell-type breadth columns.
        ct_by_key: dict = {}
        for r in attr_df.drop_duplicates(["kinase", "residue_type"]).itertuples(index=False):
            rt = str(getattr(r, "residue_type", "") or "")
            ct_by_key[(str(r.kinase), rt)] = (
                str(r.tcell_top_celltype) if pd.notna(r.tcell_top_celltype) else "",
                int(r.tcell_celltype_concentration_tier)
                if pd.notna(r.tcell_celltype_concentration_tier) else 0)
        cols["tcell_celltype"] = [ct_by_key.get((k, rt), ("", 0))[0]
                                  for k, rt in rows]
        cols["tcell_celltype_tier"] = [ct_by_key.get((k, rt), ("", 0))[1]
                                       for k, rt in rows]

    return {
        "kinases_slice": cols,
        "kinase_names": [k for k, _residue_type in rows],
        "contrasts": short_contrasts,
        "kid": kid,
        "fdr_threshold": fdr_thresh,
        "attribution_index": attribution_index,
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
