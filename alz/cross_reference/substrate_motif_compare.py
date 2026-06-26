"""alz/cross_reference/substrate_motif_compare.py

Cross-cohort substrate phosphosite comparator — D1 reusable engine.

Compares leading-edge phosphosite substrate profiles between cohorts (Song mouse,
5xFAD mouse, human Mukesh) using center-aligned BLOSUM62 motif similarity.

C5-stable public API
--------------------
    build_profile(kinase, cohort, contrast, track, *, human_mode, human_m)
        → Profile  (dict[motif_upper: ProfileEntry])
    compare(profile_a, profile_b) → Decomposition
    motif_similarity(m_a, m_b)    → SimilarityResult

Profile key: uppercased 15-mer motif string.
ProfileEntry: direction (+1/-1/0), support (int), gene (str),
              site_position (str, '' when not available), track (str).

Decomposition: shared/a_only/b_only motif keys + direction agreement on shared.

Cohort identifiers
------------------
    'song'     — Song mouse (outputs/reports/kinase_attribution/)
    'fivexfad' — 5xFAD mouse; tissue encoded in contrast:
                 contrast = '{tissue}_{age_contrast}'  e.g. 'cortex_TG_vs_WT_6mo'
    'mukesh'   — Human NBB (outputs/reports/kinase_attribution_human/)

Memory: DuckDB-streamed, pool-filtered at scan.  No whole-file pandas reads
of substrate-set CSVs or stoichiometry matrices.

CLI:  python -m alz.cross_reference.substrate_motif_compare [args]
Task: pixi run substrate-compare
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

import duckdb
import numpy as np

# ─── biopython BLOSUM62 ─────────────────────────────────────────────────────
try:
    from Bio.Align import substitution_matrices as _sm
    _BLOSUM62 = _sm.load("BLOSUM62")
except ImportError:
    _BLOSUM62 = None

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─── Constants ───────────────────────────────────────────────────────────────
# Similarity floor separating 'conserved' from 'unique'.
# Motifs scoring below this against any profile-B candidate are A-only.
SIM_FLOOR: float = 0.50

MOTIF_LEN: int = 15          # canonical ±7 window
MOTIF_CENTER: int = 7        # 0-indexed center position

_FIVEXFAD_TISSUES = ("cortex", "hippocampus")

# ─── Data types ──────────────────────────────────────────────────────────────

class ProfileEntry(NamedTuple):
    direction: int          # +1 up / -1 down / 0 unknown
    support: int            # 1 for mouse; donor count for human per-donor
    gene: str
    site_position: str      # '' when not available (e.g. Song)
    track: str              # 'st' | 'py'

# Profile: dict[motif_upper → ProfileEntry]
Profile = dict


class SimilarityResult(NamedTuple):
    score: float            # normalized BLOSUM62 similarity in [0, 1]
    n_compared: int         # non-padded aligned positions compared
    match_class: str        # 'exact' | 'conserved' | 'unique'
    mismatch_positions: list  # 0-indexed positions where residues differ (non-pad)


@dataclasses.dataclass
class MatchedPair:
    motif_a: str            # uppercased
    motif_b: str            # uppercased
    gene_a: str
    gene_b: str
    site_a: str
    site_b: str
    track: str
    match_class: str
    similarity: float
    direction_a: int
    direction_b: int
    direction_agree: bool
    support_a: int
    support_b: int
    same_gene: bool         # gene_a.upper() == gene_b.upper()


@dataclasses.dataclass
class Decomposition:
    kinase_a: str
    kinase_b: str
    cohort_a: str
    cohort_b: str
    contrast_a: str
    contrast_b: str
    track: str
    pairs: list              # list[MatchedPair] — motifs shared (sim >= SIM_FLOOR)
    a_only: list             # list[str] motif_uppers with no B match
    b_only: list             # list[str] motif_uppers with no A match
    n_genes_unmatched: int   # matched pairs where gene_a != gene_b (convergent)

    @property
    def n_shared(self) -> int:
        return len(self.pairs)

    @property
    def n_a_only(self) -> int:
        return len(self.a_only)

    @property
    def n_b_only(self) -> int:
        return len(self.b_only)

    @property
    def jaccard(self) -> float:
        n = self.n_shared + self.n_a_only + self.n_b_only
        return self.n_shared / n if n > 0 else 0.0

    @property
    def direction_agree_frac(self) -> float:
        if not self.pairs:
            return float("nan")
        agree = sum(1 for p in self.pairs if p.direction_agree)
        return agree / len(self.pairs)

    @property
    def direction_corr(self) -> float:
        if not self.pairs:
            return float("nan")
        da = np.array([p.direction_a for p in self.pairs], dtype=float)
        db = np.array([p.direction_b for p in self.pairs], dtype=float)
        if np.std(da) == 0 or np.std(db) == 0:
            return float("nan")
        return float(np.corrcoef(da, db)[0, 1])


# ─── BLOSUM62 motif similarity ───────────────────────────────────────────────

def _blosum_get(a: str, b: str) -> float:
    """BLOSUM62 score for one position pair; returns 0 for unknown chars."""
    if _BLOSUM62 is None:
        return 1.0 if a == b else 0.0
    try:
        return float(_BLOSUM62[a, b])
    except (KeyError, IndexError):
        return 0.0


def _self_score(motif_upper: str) -> float:
    """Sum of BLOSUM62(c, c) for non-padded positions."""
    return sum(_blosum_get(c, c) for c in motif_upper if c != "_")


def motif_similarity(m_a: str, m_b: str, sim_floor: float = SIM_FLOOR) -> SimilarityResult:
    """Center-aligned BLOSUM62 similarity between two ±7-window motifs.

    Both motifs must be the same length (15-mer).  Positions where either
    character is '_' (padding) are skipped.  The central residue type must
    match (both S/T or both Y) for the result to be 'conserved' or 'exact';
    mismatching central types force 'unique' regardless of score.

    Score is normalized by the geometric mean of each motif's self-score so
    identical motifs land at 1.0.  Negative raw scores are clamped to 0.0.
    """
    ua, ub = m_a.upper(), m_b.upper()
    la, lb = len(ua), len(ub)

    # Align at center: if lengths differ, pad shorter on both sides
    if la != lb:
        # Extend the shorter to the longer with '_' on each side symmetrically
        maxl = max(la, lb)
        ua = ua.center(maxl, "_")
        ub = ub.center(maxl, "_")

    ctr = len(ua) // 2
    # Central residue type check
    ca, cb = ua[ctr], ub[ctr]
    _ST = {"S", "T"}
    same_central_type = (
        (ca in _ST and cb in _ST) or (ca == "Y" and cb == "Y")
    )

    raw = 0.0
    n_compared = 0
    mismatches = []
    for i, (a, b) in enumerate(zip(ua, ub)):
        if a == "_" or b == "_":
            continue
        raw += _blosum_get(a, b)
        n_compared += 1
        if a != b:
            mismatches.append(i)

    ss_a = _self_score(ua)
    ss_b = _self_score(ub)
    denom = (ss_a * ss_b) ** 0.5
    sim = max(0.0, raw / denom) if denom > 0 else 0.0

    if not same_central_type or sim < sim_floor:
        mc = "unique"
    elif sim >= 1.0 - 1e-9:
        mc = "exact"
    else:
        mc = "conserved"

    return SimilarityResult(
        score=round(sim, 6),
        n_compared=n_compared,
        match_class=mc,
        mismatch_positions=mismatches,
    )


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _repo_root() -> str:
    from alz.shared import config
    return config.REPO_ROOT


def _song_paths(track: str) -> dict:
    rr = _repo_root()
    sfx = "" if track == "st" else "_pY"
    base = os.path.join(rr, "outputs", "reports", "kinase_attribution")
    return {
        "mea":     os.path.join(base, f"mea_stoichiometry{sfx}.csv"),
        "matrix":  os.path.join(base, f"stoichiometry_matrix{sfx}.csv"),
        "ols":     os.path.join(base, f"site_level_ols{sfx}.csv"),
    }


def _fivexfad_paths(tissue: str, track: str) -> dict:
    rr = _repo_root()
    base = os.path.join(rr, "outputs", "reports", "kinase_attribution_5xfad")
    pfx = f"{tissue}_{track}"
    return {
        "mea":    os.path.join(base, f"{pfx}_mea_stoichiometry.csv"),
        "matrix": os.path.join(base, f"{pfx}_stoichiometry_matrix.csv"),
        "ols":    os.path.join(base, f"{pfx}_site_level_ols.csv"),
    }


def _human_paths(track: str) -> dict:
    rr = _repo_root()
    sfx = "" if track == "st" else "_pY"
    base = os.path.join(rr, "outputs", "reports", "kinase_attribution_human")
    pd_dir = os.path.join(base, "perdonor")
    return {
        "perdonor_mea":  os.path.join(pd_dir, f"mea_perdonor{sfx}.csv"),
        "matrix":        os.path.join(base, f"stoichiometry_matrix{sfx}.csv"),
        "sample_map":    os.path.join(rr, "outputs", "reports",
                                      "data_ingest_human", "sample_mapping.csv"),
    }


# ─── Utility ─────────────────────────────────────────────────────────────────

def _parse_leading(raw: str | None) -> list[str]:
    """Split 'Leading substrates' field into uppercase motif strings."""
    if not raw or not isinstance(raw, str):
        return []
    return [m.upper().strip() for m in raw.split(";") if m.strip()]


def _in_clause(strings: list[str]) -> str:
    """Build SQL IN clause values (single-quoted, escaped)."""
    return ", ".join(f"'{s.replace(chr(39), chr(39)*2)}'" for s in strings)


def _motif_join_key(motif_upper: str) -> str:
    """Strip leading/trailing '_' padding for matrix joins.

    Song's stoichiometry_matrix stores 13-char motifs (no '_' padding) while
    the MEA 'Leading substrates' stores 15-char motifs with leading/trailing '_'.
    Stripping gives the 13-char key that matches the matrix.
    5xFAD and Human matrices already store 15-char motifs — stripping is a no-op.
    """
    return motif_upper.strip("_")


def _sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


# ─── Profile builders ────────────────────────────────────────────────────────

def _build_song_profile(kinase: str, contrast: str, track: str) -> Profile:
    """Leading-edge substrates for Song mouse at one contrast.

    Direction = sign of stoich_lfc_{contrast} from site_level_ols.csv.
    site_position is not available in Song — field is ''.

    Motif join note: Song's stoichiometry_matrix stores 13-char motifs (no '_'
    padding) while the MEA 'Leading substrates' column stores 15-char motifs
    with leading/trailing '_'.  The matrix join uses TRIM('_' FROM ...) so both
    sides match.  Profile keys retain the 15-char padded form for BLOSUM62
    comparison (skips '_' positions automatically).
    """
    paths = _song_paths(track)
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Song/{track} artifact missing: {p}")

    conn = duckdb.connect()
    # 1. Get leading substrates for this kinase+contrast
    q = f"""
        SELECT "Leading substrates" as ls
        FROM read_csv_auto('{paths['mea']}')
        WHERE kinase = '{kinase}'
          AND contrast = '{contrast}'
        LIMIT 1
    """
    rows = conn.execute(q).fetchall()
    if not rows:
        return {}
    leading = _parse_leading(rows[0][0])  # 15-char uppercased, with '_' padding
    if not leading:
        return {}

    # Build join-key mapping: stripped_motif → original 15-char profile key
    # (Song matrix uses 13-char stripped motifs; Human/5xFAD use 15-char)
    stripped_to_original: dict[str, str] = {}
    for mu in leading:
        stripped_to_original.setdefault(_motif_join_key(mu), mu)

    stripped_keys = list(stripped_to_original.keys())
    in_clause = _in_clause(stripped_keys)

    # 2. Metadata: site_id, gene_symbol per motif
    # Use TRIM to match Song's 13-char matrix motifs against our stripped join keys
    meta_q = f"""
        SELECT site_id, gene_symbol, TRIM('_' FROM upper(motif)) AS motif_key
        FROM read_csv_auto('{paths['matrix']}')
        WHERE TRIM('_' FROM upper(motif)) IN ({in_clause})
    """
    meta_rows = conn.execute(meta_q).fetchall()
    if not meta_rows:
        return {}

    site_ids = [r[0] for r in meta_rows]
    site_in = _in_clause([str(s) for s in site_ids])

    # 3. LFC from site_level_ols (column stoich_lfc_{contrast})
    lfc_col = f"stoich_lfc_{contrast}"
    ols_q = f"""
        SELECT site_id, "{lfc_col}" as lfc
        FROM read_csv_auto('{paths['ols']}')
        WHERE CAST(site_id AS VARCHAR) IN ({site_in})
    """
    try:
        lfc_rows = conn.execute(ols_q).fetchall()
    except duckdb.CatalogException:
        lfc_rows = []
    lfc_map = {str(r[0]): float(r[1]) if r[1] is not None else 0.0
               for r in lfc_rows}

    # 4. Build profile: key = original 15-char padded motif; take first-seen per key
    profile: Profile = {}
    for site_id, gene, motif_key in meta_rows:
        # Recover the original 15-char profile key from the stripped join key
        mu = stripped_to_original.get(motif_key, motif_key)
        if mu not in profile:
            lfc = lfc_map.get(str(site_id), 0.0)
            profile[mu] = ProfileEntry(
                direction=_sign(lfc),
                support=1,
                gene=gene,
                site_position="",
                track=track,
            )
    return profile


def _fivexfad_parse_contrast(contrast: str) -> tuple[str, str]:
    """Split 'cortex_TG_vs_WT_6mo' → (tissue, 'TG_vs_WT_6mo').

    If no known tissue prefix is found, raises ValueError.
    """
    for tissue in _FIVEXFAD_TISSUES:
        pfx = tissue + "_"
        if contrast.startswith(pfx):
            return tissue, contrast[len(pfx):]
    raise ValueError(
        f"5xFAD contrast '{contrast}' must start with a known tissue prefix "
        f"({_FIVEXFAD_TISSUES}).  Format: '{{tissue}}_TG_vs_WT_{{age}}mo'."
    )


def _build_fivexfad_profile(kinase: str, contrast: str, track: str) -> Profile:
    """Leading-edge substrates for 5xFAD at one tissue×contrast.

    contrast encodes tissue: 'cortex_TG_vs_WT_6mo' → tissue=cortex.
    Direction = sign of stoich_lfc_{age_contrast} from site_level_ols.
    """
    tissue, age_contrast = _fivexfad_parse_contrast(contrast)
    paths = _fivexfad_paths(tissue, track)
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(f"5xFAD/{tissue}/{track} artifact missing: {p}")

    conn = duckdb.connect()
    q = f"""
        SELECT "Leading substrates" as ls
        FROM read_csv_auto('{paths['mea']}')
        WHERE kinase = '{kinase}'
          AND contrast = '{age_contrast}'
        LIMIT 1
    """
    rows = conn.execute(q).fetchall()
    if not rows:
        return {}
    leading = _parse_leading(rows[0][0])
    if not leading:
        return {}

    in_clause = _in_clause(leading)
    meta_q = f"""
        SELECT site_id, gene_symbol, motif, site_position
        FROM read_csv_auto('{paths['matrix']}')
        WHERE upper(motif) IN ({in_clause})
    """
    meta_rows = conn.execute(meta_q).fetchall()
    if not meta_rows:
        return {}

    site_ids = [r[0] for r in meta_rows]
    site_in = _in_clause([str(s) for s in site_ids])
    lfc_col = f"stoich_lfc_{age_contrast}"
    ols_q = f"""
        SELECT site_id, "{lfc_col}" as lfc
        FROM read_csv_auto('{paths['ols']}')
        WHERE CAST(site_id AS VARCHAR) IN ({site_in})
    """
    try:
        lfc_rows = conn.execute(ols_q).fetchall()
    except duckdb.CatalogException:
        lfc_rows = []
    lfc_map = {str(r[0]): float(r[1]) if r[1] is not None else 0.0
               for r in lfc_rows}

    profile: Profile = {}
    for site_id, gene, motif, site_pos in meta_rows:
        mu = motif.upper()
        if mu not in profile:
            lfc = lfc_map.get(str(site_id), 0.0)
            profile[mu] = ProfileEntry(
                direction=_sign(lfc),
                support=1,
                gene=gene,
                site_position=str(site_pos) if site_pos else "",
                track=track,
            )
    return profile


def _build_human_perdonor_profile(
    kinase: str, track: str, *, human_m: int = 1
) -> Profile:
    """Human per-donor leading-edge profile.

    Motif is included if it appears in the leading edge of >= human_m AD donors.
    direction = majority sign vote across AD donors that contributed the motif.
    support = number of AD donors in whose leading edge the motif appears.

    LFC per donor per site: donor_col - nanmean(ctrl_cols) from stoichiometry_matrix.
    """
    paths = _human_paths(track)
    for k, p in paths.items():
        if k != "sample_map" and not os.path.exists(p):
            raise FileNotFoundError(f"Human/{track} artifact missing: {p}")

    conn = duckdb.connect()

    # 1. Sample mapping → AD and CTRL donor lists
    if not os.path.exists(paths["sample_map"]):
        raise FileNotFoundError(f"Missing sample_mapping.csv: {paths['sample_map']}")
    smap = conn.execute(
        f'SELECT sample_id, "group" FROM read_csv_auto(\'{paths["sample_map"]}\')'
    ).fetchdf()
    ad_donors = sorted(smap.loc[smap["group"] == "AD", "sample_id"].tolist())
    ctrl_donors = sorted(smap.loc[smap["group"] == "CTRL", "sample_id"].tolist())
    if not ad_donors:
        raise RuntimeError("No AD donors found in sample_mapping.csv")

    # 2. Leading substrates per AD donor for this kinase
    ad_contrasts = [f"{d}_vs_CTRLmean" for d in ad_donors]
    ad_in = _in_clause(ad_contrasts)
    q = f"""
        SELECT contrast, "Leading substrates" as ls
        FROM read_csv_auto('{paths['perdonor_mea']}')
        WHERE kinase = '{kinase}'
          AND contrast IN ({ad_in})
    """
    rows = conn.execute(q).fetchall()
    if not rows:
        return {}

    # motif_upper → set of donors that have it in leading edge
    motif_donors: dict[str, set[str]] = defaultdict(set)
    for contrast, ls in rows:
        donor = contrast.replace("_vs_CTRLmean", "")
        for mu in _parse_leading(ls):
            motif_donors[mu].add(donor)

    # Filter to motifs with recurrence >= M
    motifs_kept = [mu for mu, donors in motif_donors.items()
                   if len(donors) >= human_m]
    if not motifs_kept:
        return {}

    # 3. Metadata from stoichiometry_matrix
    in_clause = _in_clause(motifs_kept)
    # Build column list: all AD and CTRL sample columns we need
    # First, discover which sample columns are actually in the matrix
    hdr_q = f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'stoichiometry_matrix'
    """
    # Use DuckDB's describe instead
    desc = conn.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{paths['matrix']}') LIMIT 0"
    ).fetchall()
    matrix_cols = [r[0] for r in desc]
    meta_cols = {"site_id", "protein_id", "gene_symbol", "site_position",
                 "motif", "matched_protein"}
    sample_cols_in_matrix = [c for c in matrix_cols if c not in meta_cols]
    ad_cols = [c for c in sample_cols_in_matrix if c in set(ad_donors)]
    ctrl_cols = [c for c in sample_cols_in_matrix if c in set(ctrl_donors)]

    if not ctrl_cols:
        raise RuntimeError("No CTRL columns found in stoichiometry_matrix")

    sp_expr = "site_position" if "site_position" in matrix_cols else "'' AS site_position"
    select_cols = ", ".join(
        ["site_id", "gene_symbol", "motif", sp_expr]
        + [f'"{c}"' for c in ad_cols]
        + [f'"{c}"' for c in ctrl_cols]
    )
    meta_q = f"""
        SELECT {select_cols}
        FROM read_csv_auto('{paths['matrix']}')
        WHERE upper(motif) IN ({in_clause})
    """
    meta_rows = conn.execute(meta_q).fetchdf()
    if meta_rows.empty:
        return {}

    # 4. Compute ctrl_mean and per-donor LFCs
    present_ctrl_cols = [c for c in ctrl_cols if c in meta_rows.columns]
    present_ad_cols = [c for c in ad_cols if c in meta_rows.columns]
    ctrl_block = meta_rows[present_ctrl_cols].astype(float)
    ctrl_mean_arr = np.nanmean(ctrl_block.values, axis=1)  # shape: (n_sites,)

    # Reset index for safe positional access
    meta_rows = meta_rows.reset_index(drop=True)
    profile: Profile = {}

    for row_idx in range(len(meta_rows)):
        mu = str(meta_rows.at[row_idx, "motif"]).upper()
        if mu not in motifs_kept:
            continue
        if mu in profile:
            continue  # keep first-seen site per motif

        donors_with_motif = motif_donors.get(mu, set())
        support = len(donors_with_motif)
        row_ctrl_mean = float(ctrl_mean_arr[row_idx])

        # Sign vote: per-donor LFC sign for this site's first occurrence
        n_up, n_down = 0, 0
        for donor in donors_with_motif:
            if donor not in present_ad_cols:
                continue
            val = meta_rows.at[row_idx, donor]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if np.isnan(fval):
                continue
            s = _sign(fval - row_ctrl_mean)
            if s > 0:
                n_up += 1
            elif s < 0:
                n_down += 1

        direction = 1 if n_up > n_down else (-1 if n_down > n_up else 0)
        sp = (str(meta_rows.at[row_idx, "site_position"])
              if "site_position" in meta_rows.columns else "")
        profile[mu] = ProfileEntry(
            direction=direction,
            support=support,
            gene=str(meta_rows.at[row_idx, "gene_symbol"]),
            site_position=sp,
            track=track,
        )
    return profile


def _build_human_pergroup_profile(kinase: str, contrast: str, track: str) -> Profile:
    """Human per-group leading-edge profile.

    Requires outputs/reports/kinase_attribution_human/human_group_mea_{track}.csv
    (emitted by _emit_human_group_mea on first use).  contrast should be one of
    'AD_vs_cleanCTRL' | 'suspect_vs_cleanCTRL' (matches concordance conventions).
    """
    rr = _repo_root()
    sfx = "" if track == "st" else "_pY"
    group_mea_path = os.path.join(
        rr, "outputs", "reports", "kinase_attribution_human",
        f"human_group_mea{sfx}.csv"
    )
    if not os.path.exists(group_mea_path):
        _emit_human_group_mea(track)
    if not os.path.exists(group_mea_path):
        raise FileNotFoundError(
            f"human_group_mea{sfx}.csv not found after emit; "
            "check stoichiometry_matrix and sample_mapping.csv"
        )

    conn = duckdb.connect()
    q = f"""
        SELECT "Leading substrates" as ls
        FROM read_csv_auto('{group_mea_path}')
        WHERE kinase = '{kinase}'
          AND contrast = '{contrast}'
        LIMIT 1
    """
    rows = conn.execute(q).fetchall()
    if not rows:
        return {}
    leading = _parse_leading(rows[0][0])
    if not leading:
        return {}

    paths = _human_paths(track)
    in_clause = _in_clause(leading)
    meta_q = f"""
        SELECT site_id, gene_symbol, motif, site_position
        FROM read_csv_auto('{paths['matrix']}')
        WHERE upper(motif) IN ({in_clause})
    """
    meta_rows = conn.execute(meta_q).fetchall()
    if not meta_rows:
        return {}

    # LFC from stoichiometry matrix group contrast
    # Determine AD vs clean CTRL columns from sample mapping
    if not os.path.exists(paths["sample_map"]):
        raise FileNotFoundError(paths["sample_map"])
    smap = conn.execute(
        f'SELECT sample_id, "group" FROM read_csv_auto(\'{paths["sample_map"]}\')'
    ).fetchdf()
    ad_donors = smap.loc[smap["group"] == "AD", "sample_id"].tolist()
    ctrl_donors = ["CTRL-01", "CTRL-02", "CTRL-03", "CTRL-04"]  # clean controls

    desc = conn.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{paths['matrix']}') LIMIT 0"
    ).fetchall()
    matrix_cols = [r[0] for r in desc]
    ad_cols = [c for c in matrix_cols if c in set(ad_donors)]
    ctrl_cols = [c for c in matrix_cols if c in set(ctrl_donors)]

    site_ids = [str(r[0]) for r in meta_rows]
    site_in = _in_clause(site_ids)
    select_cols = ", ".join(
        ["site_id"] + [f'"{c}"' for c in ad_cols] + [f'"{c}"' for c in ctrl_cols]
    )
    lfc_q = f"""
        SELECT {select_cols}
        FROM read_csv_auto('{paths['matrix']}')
        WHERE CAST(site_id AS VARCHAR) IN ({site_in})
    """
    lfc_df = conn.execute(lfc_q).fetchdf()

    if not lfc_df.empty and ctrl_cols:
        ctrl_block = lfc_df[[c for c in ctrl_cols if c in lfc_df.columns]].astype(float)
        ctrl_mean = np.nanmean(ctrl_block.values, axis=1)
        ad_block = lfc_df[[c for c in ad_cols if c in lfc_df.columns]].astype(float)
        group_lfc = np.nanmean(ad_block.values, axis=1) - ctrl_mean
        lfc_map = dict(zip(lfc_df["site_id"].astype(str), group_lfc))
    else:
        lfc_map = {}

    profile: Profile = {}
    for site_id, gene, motif, site_pos in meta_rows:
        mu = motif.upper()
        if mu not in profile:
            lfc = lfc_map.get(str(site_id), 0.0)
            profile[mu] = ProfileEntry(
                direction=_sign(lfc),
                support=len(ad_donors),
                gene=gene,
                site_position=str(site_pos) if site_pos else "",
                track=track,
            )
    return profile


def _emit_human_group_mea(track: str) -> None:
    """Emit human_group_mea_{track}.csv by lifting _group_contrast from concordance.

    Only runs once; subsequent calls are no-ops when the file exists.
    """
    from alz.ctrl_outlier_audit.concordance_overlap_AD_excl_01_03 import (
        _group_contrast, _sample_sets, AD_CONTRAST, SUSPECT_CONTRAST, CLEAN_CTRL, SUSPECT
    )
    from alz.cohorts.mukesh.mea import _load_track_matrix
    rr = _repo_root()
    sfx = "" if track == "st" else "_pY"
    out_path = os.path.join(
        rr, "outputs", "reports", "kinase_attribution_human",
        f"human_group_mea{sfx}.csv"
    )
    if os.path.exists(out_path):
        return
    matrix = _load_track_matrix(track, "stoich")
    if matrix is None:
        print(f"  [group-mea/{track}] stoichiometry matrix missing; cannot emit")
        return
    sets = _sample_sets(matrix.columns, SUSPECT)
    ctx = _group_contrast(track, sets["AD"], sets["CLEAN"], AD_CONTRAST)
    if ctx is None:
        print(f"  [group-mea/{track}] group contrast returned None; cannot emit")
        return
    ctx["mea"].to_csv(out_path, index=False)
    print(f"  [group-mea/{track}] wrote {out_path}")


def build_profile(
    kinase: str,
    cohort: str,
    contrast: str,
    track: str,
    *,
    human_mode: str = "perdonor",
    human_m: int = 1,
) -> Profile:
    """Build a substrate profile for one kinase at one cohort×contrast.

    Parameters
    ----------
    kinase     : kinase abbreviation (case-sensitive, as in MEA tables)
    cohort     : 'song' | 'fivexfad' | 'mukesh'
    contrast   : cohort-specific contrast string
                 Song:     '{genotype}_{age}mo'  e.g. 'App_2mo'
                 5xFAD:    '{tissue}_TG_vs_WT_{age}mo' e.g. 'cortex_TG_vs_WT_6mo'
                 Mukesh:   ignored for perdonor; 'AD_vs_cleanCTRL' for pergroup
    track      : 'st' | 'py'
    human_mode : 'perdonor' (default) | 'pergroup'
    human_m    : recurrence threshold for perdonor (motif in >= M donor LEs)

    Returns
    -------
    Profile — dict[motif_upper: ProfileEntry]
    Empty dict if kinase not found or no leading-edge substrates.
    """
    if cohort == "song":
        return _build_song_profile(kinase, contrast, track)
    elif cohort == "fivexfad":
        return _build_fivexfad_profile(kinase, contrast, track)
    elif cohort == "mukesh":
        if human_mode == "perdonor":
            return _build_human_perdonor_profile(kinase, track, human_m=human_m)
        else:
            return _build_human_pergroup_profile(kinase, contrast, track)
    else:
        raise ValueError(
            f"Unknown cohort '{cohort}'. Expected 'song' | 'fivexfad' | 'mukesh'."
        )


# ─── Comparison primitive ─────────────────────────────────────────────────────

def compare(
    profile_a: Profile,
    profile_b: Profile,
    *,
    kinase_a: str = "",
    kinase_b: str = "",
    cohort_a: str = "",
    cohort_b: str = "",
    contrast_a: str = "",
    contrast_b: str = "",
    track: str = "",
    sim_floor: float = SIM_FLOOR,
) -> Decomposition:
    """Symmetric overlap decomposition of two substrate profiles.

    For each motif in profile_a, find the highest-similarity motif in profile_b
    using center-aligned BLOSUM62.  If best sim >= sim_floor → matched pair
    (shared).  Motifs in A with no B counterpart → a_only.  Motifs in B not
    matched by any A motif → b_only.

    Direction agreement on shared pairs is per-pair (sign(direction_a) ==
    sign(direction_b)).  n_genes_unmatched counts matched pairs where the gene
    names differ (convergent/coincidental matching, not orthologous site).
    """
    pairs: list[MatchedPair] = []
    a_only: list[str] = []
    matched_b: set[str] = set()

    for mu_a, ea in profile_a.items():
        best_sim: Optional[SimilarityResult] = None
        best_mu_b: Optional[str] = None
        best_eb: Optional[ProfileEntry] = None

        for mu_b, eb in profile_b.items():
            sr = motif_similarity(mu_a, mu_b, sim_floor=sim_floor)
            if sr.match_class == "unique":
                continue
            if best_sim is None or sr.score > best_sim.score:
                best_sim = sr
                best_mu_b = mu_b
                best_eb = eb

        if best_sim is None or best_mu_b is None or best_eb is None:
            a_only.append(mu_a)
        else:
            matched_b.add(best_mu_b)
            same_gene = ea.gene.upper() == best_eb.gene.upper()
            pairs.append(MatchedPair(
                motif_a=mu_a,
                motif_b=best_mu_b,
                gene_a=ea.gene,
                gene_b=best_eb.gene,
                site_a=ea.site_position,
                site_b=best_eb.site_position,
                track=ea.track or track,
                match_class=best_sim.match_class,
                similarity=best_sim.score,
                direction_a=ea.direction,
                direction_b=best_eb.direction,
                direction_agree=(ea.direction == best_eb.direction),
                support_a=ea.support,
                support_b=best_eb.support,
                same_gene=same_gene,
            ))

    b_only = [mu_b for mu_b in profile_b if mu_b not in matched_b]
    n_genes_unmatched = sum(1 for p in pairs if not p.same_gene)

    return Decomposition(
        kinase_a=kinase_a,
        kinase_b=kinase_b,
        cohort_a=cohort_a,
        cohort_b=cohort_b,
        contrast_a=contrast_a,
        contrast_b=contrast_b,
        track=track,
        pairs=pairs,
        a_only=a_only,
        b_only=b_only,
        n_genes_unmatched=n_genes_unmatched,
    )


# ─── Output helpers ───────────────────────────────────────────────────────────

def _emit_pairs_csv(decompositions: list[Decomposition], out_path: Path) -> int:
    """Write substrate_pairs.csv.  Returns row count."""
    header = [
        "kinase_a", "kinase_b", "cohort_a", "contrast_a", "cohort_b", "contrast_b",
        "motif_a", "motif_b", "gene_a", "gene_b", "site_a", "site_b", "track",
        "match_class", "similarity", "direction_a", "direction_b", "direction_agree",
        "support_a", "support_b",
    ]
    n = 0
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for d in decompositions:
            for p in d.pairs:
                w.writerow([
                    d.kinase_a, d.kinase_b,
                    d.cohort_a, d.contrast_a, d.cohort_b, d.contrast_b,
                    p.motif_a, p.motif_b, p.gene_a, p.gene_b,
                    p.site_a, p.site_b, p.track,
                    p.match_class, f"{p.similarity:.6f}",
                    p.direction_a, p.direction_b, int(p.direction_agree),
                    p.support_a, p.support_b,
                ])
                n += 1
            # A-only rows
            for mu_a in d.a_only:
                ea = None  # might not have easy access here — handled differently
                w.writerow([
                    d.kinase_a, d.kinase_b,
                    d.cohort_a, d.contrast_a, d.cohort_b, d.contrast_b,
                    mu_a, "", "", "", "", "", d.track,
                    "a_only", "", "", "", "",
                    "", "",
                ])
                n += 1
            for mu_b in d.b_only:
                w.writerow([
                    d.kinase_a, d.kinase_b,
                    d.cohort_a, d.contrast_a, d.cohort_b, d.contrast_b,
                    "", mu_b, "", "", "", "", d.track,
                    "b_only", "", "", "", "",
                    "", "",
                ])
                n += 1
    return n


def _emit_pairs_csv_full(
    decompositions: list[Decomposition],
    profiles_a: dict[str, Profile],
    profiles_b: dict[str, Profile],
    out_path: Path,
) -> int:
    """Write substrate_pairs.csv with profile metadata for a-only and b-only rows."""
    header = [
        "kinase_a", "kinase_b", "cohort_a", "contrast_a", "cohort_b", "contrast_b",
        "motif_a", "motif_b", "gene_a", "gene_b", "site_a", "site_b", "track",
        "match_class", "similarity", "direction_a", "direction_b", "direction_agree",
        "support_a", "support_b",
    ]
    n = 0
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for d in decompositions:
            pa = profiles_a.get(d.kinase_a, {})
            pb = profiles_b.get(d.kinase_b, {})
            for p in d.pairs:
                w.writerow([
                    d.kinase_a, d.kinase_b,
                    d.cohort_a, d.contrast_a, d.cohort_b, d.contrast_b,
                    p.motif_a, p.motif_b, p.gene_a, p.gene_b,
                    p.site_a, p.site_b, p.track,
                    p.match_class, f"{p.similarity:.6f}",
                    p.direction_a, p.direction_b, int(p.direction_agree),
                    p.support_a, p.support_b,
                ])
                n += 1
            for mu_a in d.a_only:
                e = pa.get(mu_a)
                w.writerow([
                    d.kinase_a, d.kinase_b,
                    d.cohort_a, d.contrast_a, d.cohort_b, d.contrast_b,
                    mu_a, "", e.gene if e else "", "", e.site_position if e else "", "",
                    d.track, "a_only", "", e.direction if e else "", "", "",
                    e.support if e else "", "",
                ])
                n += 1
            for mu_b in d.b_only:
                e = pb.get(mu_b)
                w.writerow([
                    d.kinase_a, d.kinase_b,
                    d.cohort_a, d.contrast_a, d.cohort_b, d.contrast_b,
                    "", mu_b, "", e.gene if e else "", "", e.site_position if e else "",
                    d.track, "b_only", "", "", e.direction if e else "", "",
                    "", e.support if e else "",
                ])
                n += 1
    return n


def _emit_summary_csv(decompositions: list[Decomposition], out_path: Path) -> int:
    """Write kinase_summary.csv.  Returns row count."""
    header = [
        "kinase_a", "kinase_b", "cohort_a", "contrast_a", "cohort_b", "contrast_b",
        "track", "n_shared", "n_a_only", "n_b_only", "jaccard",
        "direction_agree_frac", "direction_corr",
        "n_exact", "n_conserved",
        "n_genes_unmatched",
    ]
    n = 0
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for d in decompositions:
            n_exact = sum(1 for p in d.pairs if p.match_class == "exact")
            n_conserved = sum(1 for p in d.pairs if p.match_class == "conserved")
            dc = d.direction_corr
            dc_str = f"{dc:.6f}" if dc == dc else ""  # NaN → ''
            daf = d.direction_agree_frac
            daf_str = f"{daf:.6f}" if daf == daf else ""
            w.writerow([
                d.kinase_a, d.kinase_b,
                d.cohort_a, d.contrast_a, d.cohort_b, d.contrast_b,
                d.track,
                d.n_shared, d.n_a_only, d.n_b_only,
                f"{d.jaccard:.6f}",
                daf_str, dc_str,
                n_exact, n_conserved,
                d.n_genes_unmatched,
            ])
            n += 1
    return n


def _make_figures(decompositions: list[Decomposition], out_dir: Path) -> None:
    """Per-kinase and aggregate figures: set overlap bars, similarity histogram,
    direction scatter.  Uses matplotlib (matplotlib_venn not required)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [figures] matplotlib not available; skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Per-kinase set overlap bar ────────────────────────────────────────
    for d in decompositions:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        ax1, ax2 = axes

        # Set overlap as horizontal bar
        counts = [d.n_a_only, d.n_shared, d.n_b_only]
        labels = ["A-only", "Shared", "B-only"]
        colors = ["#4E79A7", "#59A14F", "#E15759"]
        bars = ax1.barh(labels, counts, color=colors)
        ax1.set_xlabel("Substrate count")
        ax1.set_title(
            f"{d.kinase_a} vs {d.kinase_b}\n"
            f"Jaccard={d.jaccard:.2f}  dir_agree={d.direction_agree_frac:.2f}"
            if d.direction_agree_frac == d.direction_agree_frac
            else f"{d.kinase_a} vs {d.kinase_b}\nJaccard={d.jaccard:.2f}",
            fontsize=9,
        )
        for bar, cnt in zip(bars, counts):
            ax1.text(cnt + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(cnt), va="center", fontsize=8)

        # Similarity histogram for shared pairs
        sims = [p.similarity for p in d.pairs]
        if sims:
            ax2.hist(sims, bins=20, range=(0, 1), color="#76B7B2", edgecolor="white")
            ax2.set_xlabel("BLOSUM62 similarity")
            ax2.set_ylabel("Count")
            ax2.set_title(f"Shared motif similarity (n={len(sims)})", fontsize=9)
            ax2.axvline(SIM_FLOOR, color="red", linestyle="--", linewidth=0.8,
                        label=f"floor={SIM_FLOOR}")
            ax2.legend(fontsize=7)
        else:
            ax2.set_visible(False)

        plt.tight_layout()
        fname = f"{d.kinase_a}__vs__{d.kinase_b}.png"
        fig.savefig(fig_dir / fname, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Aggregate direction scatter ───────────────────────────────────────
    all_pairs = [p for d in decompositions for p in d.pairs]
    if all_pairs:
        fig, ax = plt.subplots(figsize=(5, 5))
        da = [p.direction_a for p in all_pairs]
        db = [p.direction_b for p in all_pairs]
        ax.scatter(da, db, alpha=0.3, s=20, color="#B07AA1")
        ax.set_xlabel("Direction A")
        ax.set_ylabel("Direction B")
        ax.set_title(f"Direction agreement — all pairs (n={len(all_pairs)})")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        fig.savefig(fig_dir / "direction_scatter_all.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


def _write_manifest(out_dir: Path, run_params: dict, n_pairs: int, n_kinases: int) -> None:
    lines = [
        "# Substrate motif comparison — MANIFEST",
        "",
        f"**Generated:** {_dt.date.today().isoformat()}  ",
        "**Generator:** `alz/cross_reference/substrate_motif_compare.py`  ",
        f"**Mode:** `{run_params.get('mode', '')}`  ",
        f"**Track:** `{run_params.get('track', '')}`  ",
        "",
        "## Cohort A",
        f"- Cohort: `{run_params.get('cohort_a', '')}`  ",
        f"- Contrast: `{run_params.get('contrast_a', '')}`  ",
        f"- Pool: {run_params.get('pool_a', [])}  ",
        "",
        "## Cohort B",
        f"- Cohort: `{run_params.get('cohort_b', '')}`  ",
        f"- Contrast: `{run_params.get('contrast_b', '')}`  ",
        f"- Pool: {run_params.get('pool_b', [])}  ",
        "",
        "## Parameters",
        f"- Similarity floor: `{run_params.get('sim_floor', SIM_FLOOR)}`  ",
        f"- Human recurrence M: `{run_params.get('human_m', 1)}`  ",
        f"- Human mode: `{run_params.get('human_mode', 'perdonor')}`  ",
        "",
        "## Results",
        f"- Kinase comparisons: {n_kinases}  ",
        f"- Substrate pair rows (shared + a-only + b-only): {n_pairs}  ",
        "",
        "## Outputs",
        "- `substrate_pairs.csv` — flat, viewer-consumable; one row per substrate  ",
        "- `kinase_summary.csv` — per-kinase aggregates  ",
        "- `figures/` — per-kinase set overlap + similarity histogram  ",
    ]
    (out_dir / "MANIFEST.md").write_text("\n".join(lines) + "\n")


# ─── Top-level runner ─────────────────────────────────────────────────────────

def run_comparison(
    pool_a: list[str],
    cohort_a: str,
    contrast_a: str,
    pool_b: list[str],
    cohort_b: str,
    contrast_b: str,
    track: str,
    *,
    mode: str = "matched",
    run_name: str = "run",
    human_mode: str = "perdonor",
    human_m: int = 1,
    sim_floor: float = SIM_FLOOR,
    out_root: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Run the full substrate comparison pipeline.

    Parameters
    ----------
    pool_a, pool_b  : explicit kinase name lists (caller's responsibility)
    cohort_a/b      : 'song' | 'fivexfad' | 'mukesh'
    contrast_a/b    : cohort-specific contrast strings
    track           : 'st' | 'py'
    mode            : 'matched' (pair by position) | 'all-pairs' (every pair in pool_a)
    run_name        : output subdirectory under out_root/
    human_mode      : 'perdonor' | 'pergroup'
    human_m         : recurrence threshold for human per-donor
    sim_floor       : BLOSUM62 similarity floor for 'shared' classification
    out_root        : root for outputs/reports/substrate_compare/ (default: repo root)
    verbose         : print progress

    Returns
    -------
    dict with keys: n_kinases, n_pairs, out_dir
    """
    if out_root is None:
        out_root = Path(_repo_root()) / "outputs" / "reports" / "substrate_compare"
    out_dir = Path(out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _vprint(*args):
        if verbose:
            print(*args)

    _vprint(f"[substrate-compare] mode={mode}  track={track}  run={run_name}")
    _vprint(f"  A: cohort={cohort_a}  contrast={contrast_a}  pool={pool_a}")
    _vprint(f"  B: cohort={cohort_b}  contrast={contrast_b}  pool={pool_b}")

    # ── Build profiles ─────────────────────────────────────────────────────
    def _bp(kinase, cohort, contrast):
        try:
            p = build_profile(
                kinase, cohort, contrast, track,
                human_mode=human_mode, human_m=human_m,
            )
            _vprint(f"    profile({kinase}/{cohort}/{contrast}): {len(p)} motifs")
            return p
        except Exception as exc:
            _vprint(f"    WARNING: profile({kinase}/{cohort}/{contrast}) failed: {exc}")
            return {}

    profiles_a: dict[str, Profile] = {k: _bp(k, cohort_a, contrast_a) for k in pool_a}

    if mode == "matched":
        # pair pool_a[i] with pool_b[i] by position
        if len(pool_a) != len(pool_b):
            raise ValueError(
                f"matched mode requires equal-length pools; "
                f"pool_a={len(pool_a)}, pool_b={len(pool_b)}"
            )
        profiles_b: dict[str, Profile] = {k: _bp(k, cohort_b, contrast_b) for k in pool_b}
        pairs_to_compare = list(zip(pool_a, pool_b))

    elif mode == "all-pairs":
        # within pool_a; cohort_b / contrast_b / pool_b ignored
        profiles_b = profiles_a
        n = len(pool_a)
        pairs_to_compare = [(pool_a[i], pool_a[j])
                            for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'matched' | 'all-pairs'.")

    # ── Compare ────────────────────────────────────────────────────────────
    decompositions: list[Decomposition] = []
    for ka, kb in pairs_to_compare:
        pa = profiles_a.get(ka, {})
        pb = profiles_b.get(kb, {})
        d = compare(
            pa, pb,
            kinase_a=ka, kinase_b=kb,
            cohort_a=cohort_a, cohort_b=cohort_b,
            contrast_a=contrast_a, contrast_b=contrast_b,
            track=track,
            sim_floor=sim_floor,
        )
        decompositions.append(d)
        _vprint(
            f"    {ka} vs {kb}: shared={d.n_shared} a_only={d.n_a_only} "
            f"b_only={d.n_b_only} jaccard={d.jaccard:.3f}"
        )

    # ── Outputs ────────────────────────────────────────────────────────────
    n_pairs = _emit_pairs_csv_full(
        decompositions, profiles_a, profiles_b,
        out_dir / "substrate_pairs.csv"
    )
    n_kinases = _emit_summary_csv(decompositions, out_dir / "kinase_summary.csv")
    _make_figures(decompositions, out_dir)

    run_params = dict(
        mode=mode, track=track,
        cohort_a=cohort_a, contrast_a=contrast_a, pool_a=pool_a,
        cohort_b=cohort_b, contrast_b=contrast_b, pool_b=pool_b,
        sim_floor=sim_floor, human_m=human_m, human_mode=human_mode,
    )
    _write_manifest(out_dir, run_params, n_pairs, n_kinases)

    _vprint(f"  wrote {out_dir}/  ({n_pairs} pair rows, {n_kinases} kinase summaries)")
    peak_mb = _peak_rss_mb()
    _vprint(f"  peak RSS: {peak_mb:.1f} MB")
    return {"n_kinases": n_kinases, "n_pairs": n_pairs, "out_dir": str(out_dir)}


def _peak_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-cohort substrate phosphosite comparator (D1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pool-a", required=True,
                        help="Comma-separated kinase names for cohort A")
    parser.add_argument("--cohort-a", required=True,
                        choices=["song", "fivexfad", "mukesh"],
                        help="Cohort for side A")
    parser.add_argument("--contrast-a", required=True,
                        help="Contrast for side A (cohort-specific format)")
    parser.add_argument("--pool-b",
                        help="Comma-separated kinase names for cohort B "
                             "(required for matched mode; ignored for all-pairs)")
    parser.add_argument("--cohort-b",
                        choices=["song", "fivexfad", "mukesh"],
                        help="Cohort for side B (required for matched mode)")
    parser.add_argument("--contrast-b",
                        help="Contrast for side B (required for matched mode)")
    parser.add_argument("--track", default="st", choices=["st", "py"],
                        help="Phospho track")
    parser.add_argument("--mode", default="matched",
                        choices=["matched", "all-pairs"],
                        help="Comparison mode: matched=pair by position, "
                             "all-pairs=every pair within pool-a")
    parser.add_argument("--run-name", default=None,
                        help="Output subdirectory name (default: auto-generated)")
    parser.add_argument("--human-mode", default="perdonor",
                        choices=["perdonor", "pergroup"],
                        help="Human profile builder mode")
    parser.add_argument("--human-m", type=int, default=1,
                        help="Recurrence threshold for human per-donor profiles")
    parser.add_argument("--sim-floor", type=float, default=SIM_FLOOR,
                        help="BLOSUM62 similarity floor for 'shared' classification")
    args = parser.parse_args(argv)

    pool_a = [k.strip() for k in args.pool_a.split(",") if k.strip()]

    if args.mode == "matched":
        if not args.pool_b or not args.cohort_b or not args.contrast_b:
            parser.error("matched mode requires --pool-b, --cohort-b, --contrast-b")
        pool_b = [k.strip() for k in args.pool_b.split(",") if k.strip()]
        cohort_b = args.cohort_b
        contrast_b = args.contrast_b
    else:
        pool_b = pool_a
        cohort_b = args.cohort_a
        contrast_b = args.contrast_a

    run_name = args.run_name or (
        f"{args.cohort_a}_{args.contrast_a}_vs_"
        f"{cohort_b}_{contrast_b}_{args.track}"
        f"_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    try:
        result = run_comparison(
            pool_a=pool_a,
            cohort_a=args.cohort_a,
            contrast_a=args.contrast_a,
            pool_b=pool_b,
            cohort_b=cohort_b,
            contrast_b=contrast_b,
            track=args.track,
            mode=args.mode,
            run_name=run_name,
            human_mode=args.human_mode,
            human_m=args.human_m,
            sim_floor=args.sim_floor,
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(f"substrate-compare complete: {result['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
