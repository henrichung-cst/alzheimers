"""Kinase → Incytr pathway integration bridge (B4).

For each MEA kinase/contrast/track row, maps floor-99 substrate genes to
pathway nodes (Ligand / Receptor / EM / Target) in the pair-mode
wide/*.parquet shards, annotates cell-type match, expression, and disease
context, and emits a flat backend artifact, a Receptor-EM fan
characterization, and a per-kinase ``#Backbones`` participation count.

Cohorts implemented:
  - song    : 9 contrasts, ST-only stoichiometry MEA, no expression layer
               (B5/snrna prereq); celltype from kinase_hypothesis_table
  - fivexfad: cortex + hippocampus, ST + pY MEA; celltype from
               fivexfad_celltype_mea.parquet; expression from
               fivexfad_expression_specificity.csv (age-pooled); disease
               context from fivexfad_snrna_attribution.csv (age-matched)
  - tcells  : one donor per run, using the donor's within-cohort state
               attribution emitted by tcell_within_cohort.py

Pathway set used:
  Song    : outputs/reports/incytr_pair_mode/wide/
  5xFAD   : outputs/reports/incytr_pair_mode_5xfad/{tissue}/wide/
  T-cells : outputs/reports/incytr_pair_mode_tcells/{donor}/wide/
  (Each cohort's gene_node_index.json.gz confirms its corresponding wide/ shards.)

B4 emits:
  - recep_em_fan.csv         : per Receptor-EM spine fan characterization
  - kinase_participation.csv : per-kinase participation over the gated wide/
      shards, at two grains — n_backbones (distinct Sender-Receiver-Receptor-EM
      spines, the kinase's own breadth) and n_paths (distinct full pathways the
      kinase sits along). Both count any-node (L/R/EM/T) participation, computed
      exactly via DuckDB (not estimated).

Usage:
  pixi run kinase-incytr-bridge            # Song + 5xFAD cohorts
  pixi run kinase-incytr-bridge -- --cohort song
  pixi run kinase-incytr-bridge -- --cohort fivexfad --tissue cortex
  pixi run kinase-incytr-bridge -- --cohort tcells --donor donor1
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import logging
import os
import re
import sys
import textwrap
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from alz.shared import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORTS = Path("outputs/reports")
SONG_MEA_DIR = REPORTS / "kinase_attribution"
FIVEXFAD_MEA_DIR = REPORTS / "kinase_attribution_5xfad"
SONG_WIDE_DIR = REPORTS / "incytr_pair_mode" / "wide"
FIVEXFAD_WIDE_DIR = REPORTS / "incytr_pair_mode_5xfad"
UNIFIED_VIEWER_DIR = REPORTS / "unified_viewer" / "edge_slices"
TCELL_VIEWER_DIR = REPORTS / "tcell_viewer" / "edge_slices"

SONG_INDEX = UNIFIED_VIEWER_DIR / "incytr_pathways" / "gene_node_index.json.gz"
FIVEXFAD_INDEX_TMPL = UNIFIED_VIEWER_DIR / "incytr_pathways_fivexfad_{tissue}" / "gene_node_index.json.gz"
TCELL_INDEX_TMPL = TCELL_VIEWER_DIR / "incytr_pathways" / "{donor}__gene_node_index.json.gz"

SONG_STOICH_MATRIX = SONG_MEA_DIR / "stoichiometry_matrix.csv"
SONG_STOICH_MATRIX_PY = SONG_MEA_DIR / "stoichiometry_matrix_pY.csv"
SONG_MEA_STOICH = SONG_MEA_DIR / "mea_stoichiometry.csv"
SONG_MEA_PY = SONG_MEA_DIR / "mea_stoichiometry_pY.csv"
SONG_SUBSTRATE_SETS = SONG_MEA_DIR / "mea_substrate_sets.csv"
SONG_SUBSTRATE_SETS_PY = SONG_MEA_DIR / "mea_substrate_sets_pY.csv"
SONG_KINASE_HYP = REPORTS / "attribution_recovery" / "kinase_hypothesis_table.csv"

FIVEXFAD_EXPR_SPEC = FIVEXFAD_MEA_DIR / "fivexfad_expression_specificity.csv"
FIVEXFAD_SNRNA_ATTR = FIVEXFAD_MEA_DIR / "fivexfad_snrna_attribution.csv"
FIVEXFAD_CELLTYPE_MEA = FIVEXFAD_MEA_DIR / "celltype_mea" / "fivexfad_celltype_mea.parquet"
TCELL_ATTRIBUTION_DIR = REPORTS / "kinase_attribution_tcells"

OUT_ROOT = REPORTS / "kinase_incytr_bridge"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Canonical pair-mode significance floor (identical to filter_significant_paths.py):
# a chain "passes" when either side's SigProb clears 0.1 AND |PDS| >= 0.2.
# #Backbones counts passing chains, so this floor is applied when enumerating
# wide/ full paths.
SIGPROB_CUTOFF = 0.1
ABS_PDS_CUTOFF = 0.2

POSITIONS = ("Ligand", "Receptor", "EM", "Target")


def _fivexfad_age_from_contrast(mea_contrast: str) -> int:
    """TG_vs_WT_3mo -> 3"""
    m = re.search(r"(\d+)mo$", mea_contrast)
    if m is None:
        raise ValueError(f"Cannot extract age from MEA contrast: {mea_contrast!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Step 1: load gene_node_index
# ---------------------------------------------------------------------------

def load_gene_node_index(path: Path) -> pd.DataFrame:
    """Decode the columnar gene_node_index.json.gz into a flat DataFrame.

    Returns columns: gene, role, sender, receiver, n_rows, best_abs_pds, best_pds
    """
    with gzip.open(path) as fh:
        idx = json.load(fh)
    genes = idx["genes"]
    roles = idx["roles"]
    senders = idx["senders"]
    receivers = idx["receivers"]
    return pd.DataFrame({
        "gene": [genes[i] for i in idx["gene_id"]],
        "role": [roles[i] for i in idx["role_id"]],
        "sender": [senders[i] for i in idx["sender_id"]],
        "receiver": [receivers[i] for i in idx["receiver_id"]],
        "n_rows": idx["n_rows"],
        "best_abs_pds": idx["best_abs_pds"],
    })


# ---------------------------------------------------------------------------
# Step 2: substrate bridge (floor-99 substrate sets → stoichiometry_matrix → gene)
# ---------------------------------------------------------------------------

SUBSTRATE_BRIDGE_COLS = [
    "kinase", "contrast", "channel", "NES", "FDR", "gene_symbol", "n_sites",
    "sites",
]


def load_floor_substrate_sets(
    path: Path,
    kl_floor: float = config.INCYTR_ATTRIBUTION_KL_PCT,
) -> pd.DataFrame:
    """Load only attribution-eligible rows from a substrate-set CSV.

    DuckDB performs the projection and KL-floor filter while scanning the CSV,
    which keeps the large 5xFAD cortex ST receipt out of pandas memory in full.
    The bridge applies the floor again at its public seam so callers cannot
    accidentally bypass the attribution rule with an unfiltered DataFrame.
    """
    columns = ["kinase", "contrast", "motif", "residue_type", "kl_percentile"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    safe_path = str(path).replace("'", "''")
    con = duckdb.connect()
    try:
        available = set(con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{safe_path}', header = true)"
        ).fetchdf()["column_name"])
        residue_expr = '"residue_type"' if "residue_type" in available else "NULL::VARCHAR"
        return con.execute(
            f"""
            SELECT kinase, contrast, motif, {residue_expr} AS residue_type,
                   TRY_CAST(kl_percentile AS DOUBLE) AS kl_percentile
            FROM read_csv_auto('{safe_path}', header = true)
            WHERE TRY_CAST(kl_percentile AS DOUBLE) >= $kl_floor
            """,
            {"kl_floor": kl_floor},
        ).df()
    finally:
        con.close()


def _motif_key(value: object) -> str:
    """Normalize substrate-set and stoichiometry motifs to a join key."""
    return str(value).strip().upper().strip("_")


def build_substrate_bridge(
    mea_stoich: pd.DataFrame,
    stoich_matrix: pd.DataFrame,
    substrate_sets: pd.DataFrame,
    kl_floor: float = config.INCYTR_ATTRIBUTION_KL_PCT,
) -> pd.DataFrame:
    """Map floor-99 substrate motifs to per-gene evidence rows.

    Substrate membership is the absolute kinase-library floor
    ``kl_percentile >= kl_floor`` from ``mea_substrate_sets.csv``; it is not
    gated by MEA FDR.  MEA ``NES`` and ``FDR`` are retained as annotations for
    the matching (kinase, contrast, track) row and are not used to select
    edges.

    ``n_sites`` is the number of distinct floor-qualified motifs for a
    (kinase, contrast) row that map to a gene.  ``sites`` is the compact JSON
    encoding of those same motifs, retaining the raw motif, residue type, and
    kinase-library percentile. Both are computed before the node-index join,
    so every role/sender/receiver fan-out retains the same raw evidence.

    Returns columns: kinase, contrast, channel, NES, FDR, gene_symbol, n_sites, sites
    """
    if substrate_sets.empty or mea_stoich.empty or stoich_matrix.empty:
        return pd.DataFrame(columns=SUBSTRATE_BRIDGE_COLS)

    substrate_columns = ["kinase", "contrast", "motif", "kl_percentile"]
    if "residue_type" in substrate_sets.columns:
        substrate_columns.insert(3, "residue_type")
    substrate = substrate_sets[substrate_columns].copy()
    if "residue_type" not in substrate.columns:
        substrate["residue_type"] = None
    substrate["kl_percentile"] = pd.to_numeric(
        substrate["kl_percentile"], errors="coerce"
    )
    substrate = substrate[substrate["kl_percentile"] >= kl_floor].copy()
    if substrate.empty:
        return pd.DataFrame(columns=SUBSTRATE_BRIDGE_COLS)
    substrate["motif_key"] = substrate["motif"].map(_motif_key)
    substrate = substrate[substrate["motif_key"].ne("")]
    substrate = substrate.sort_values(
        ["kinase", "contrast", "motif_key", "kl_percentile"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["kinase", "contrast", "motif_key"])

    motif_map = stoich_matrix[["motif", "gene_symbol"]].copy()
    motif_map["motif_key"] = motif_map["motif"].map(_motif_key)
    motif_map["gene_symbol"] = motif_map["gene_symbol"].astype("string")
    motif_map = motif_map.dropna(subset=["gene_symbol"])
    motif_map = motif_map[motif_map["motif_key"].ne("")].drop_duplicates(
        ["motif_key", "gene_symbol"]
    )

    mapped = substrate.merge(motif_map[["motif_key", "gene_symbol"]], on="motif_key", how="left")
    unmapped = mapped["gene_symbol"].isna()
    if unmapped.any():
        log.warning(
            "Substrate bridge: %s/%s floor-qualified kinase/contrast/motif rows "
            "did not map to a stoichiometry gene",
            int(unmapped.sum()),
            len(mapped),
        )
    log.info(
        "Substrate bridge: %s/%s floor-qualified rows mapped to a gene",
        int((~unmapped).sum()),
        len(mapped),
    )
    mapped = mapped.dropna(subset=["gene_symbol"])
    if mapped.empty:
        return pd.DataFrame(columns=SUBSTRATE_BRIDGE_COLS)
    mapped = mapped.drop_duplicates(["kinase", "contrast", "motif_key", "gene_symbol"])

    def encode_sites(group: pd.DataFrame) -> str:
        ordered = group.sort_values(
            ["kl_percentile", "motif_key"], ascending=[False, True]
        )
        records = []
        for row in ordered.itertuples(index=False):
            residue_type = row.residue_type
            records.append({
                "motif": str(row.motif),
                "residue_type": None if pd.isna(residue_type) else str(residue_type),
                "kl_percentile": float(row.kl_percentile),
            })
        return json.dumps(records, ensure_ascii=False, separators=(",", ":"))

    site_rows = []
    for keys, group in mapped.groupby(
        ["kinase", "contrast", "gene_symbol"], sort=False
    ):
        site_rows.append({
            "kinase": keys[0],
            "contrast": keys[1],
            "gene_symbol": keys[2],
            "sites": encode_sites(group),
        })
    site_lists = pd.DataFrame(site_rows)
    counts = (
        mapped.groupby(["kinase", "contrast", "gene_symbol"], as_index=False)
        .size()
        .rename(columns={"size": "n_sites"})
    )
    counts = counts.merge(site_lists, on=["kinase", "contrast", "gene_symbol"], how="left")
    for row in counts.itertuples(index=False):
        if len(json.loads(row.sites)) != int(row.n_sites):
            raise ValueError(
                "Substrate bridge site-list/count mismatch for "
                f"{row.kinase}/{row.contrast}/{row.gene_symbol}"
            )

    annotations = mea_stoich[["kinase", "contrast", "NES", "FDR"]].copy()
    annotations["channel"] = mea_stoich.get("track", "st")
    annotations = annotations.drop_duplicates(["kinase", "contrast", "channel"])
    result = annotations.merge(counts, on=["kinase", "contrast"], how="inner")
    result = result[SUBSTRATE_BRIDGE_COLS]
    return result


# ---------------------------------------------------------------------------
# Step 3: node join via gene_node_index (no parquet read for lookup)
# ---------------------------------------------------------------------------

def gene_node_hits(
    substrate_df: pd.DataFrame,
    node_index: pd.DataFrame,
) -> pd.DataFrame:
    """Join substrate genes to node_index to get all (gene, role, sender, receiver) hits.

    Position-aware ownership:
      Ligand -> sender cluster owns the node
      Receptor / EM / Target -> receiver cluster owns the node

    Returns flat hit table with columns:
      kinase, contrast, channel, NES, FDR, gene_symbol, n_sites, sites, role, sender, receiver,
      owning_cluster (the cluster that "owns" this node per position rule)
    """
    if substrate_df.empty:
        return pd.DataFrame()
    merged = substrate_df.merge(node_index, on="gene_symbol", how="inner")
    if merged.empty:
        return pd.DataFrame()
    # owning_cluster: Ligand -> sender; Receptor/EM/Target -> receiver
    merged["owning_cluster"] = merged["sender"].where(
        merged["role"].eq("Ligand"),
        merged["receiver"],
    )
    for c in ("contrast", "channel", "role", "sender", "receiver", "owning_cluster"):
        if c in merged.columns:
            merged[c] = merged[c].astype("category")
    return merged


# ---------------------------------------------------------------------------
# Step 4: cell-type match (position-aware)
# ---------------------------------------------------------------------------

def _apply_celltype_ranks(
    hits: pd.DataFrame,
    ranks: pd.DataFrame,
    join_cols: list[str],
) -> pd.DataFrame:
    """Annotate a long-form (join_cols..., celltype_match_rank) table onto hits.

    Adds celltype_match (bool) and celltype_match_rank (Int64, <NA> when no match).
    Join keys are string-cast on both sides to match the prior per-row str() compare.
    The rank table is tiny compared with the hit table, so use a keyed map instead
    of a pandas merge that duplicates every hit column in memory.
    """
    if hits.empty:
        hits["celltype_match"] = pd.Series(dtype=bool)
        hits["celltype_match_rank"] = pd.Series(dtype="Int64")
        return hits
    if ranks.empty:
        hits["celltype_match"] = False
        hits["celltype_match_rank"] = pd.Series(pd.NA, index=hits.index, dtype="Int64")
        return hits

    rank_map = _rank_map(ranks, join_cols, "celltype_match_rank")
    hit_key = _string_key(hits, join_cols)
    hit_rank = hit_key.map(rank_map)
    hits["celltype_match"] = hit_rank.notna().to_numpy()
    hits["celltype_match_rank"] = pd.array(hit_rank, dtype="Int64")
    return hits


def _string_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Build a stable string key without copying all payload columns."""
    key = df[cols[0]].astype("string")
    for col in cols[1:]:
        key = key.str.cat(df[col].astype("string"), sep="\x1f", na_rep="")
    return key


def _rank_map(df: pd.DataFrame, cols: list[str], value_col: str) -> pd.Series:
    key = _string_key(df, cols)
    return pd.Series(df[value_col].to_numpy(), index=key).groupby(level=0).min()


def _value_map(df: pd.DataFrame, cols: list[str], value_col: str) -> pd.Series:
    key = _string_key(df, cols)
    return pd.Series(df[value_col].to_numpy(), index=key).groupby(level=0).first()


def annotate_celltype_match_song(
    hits: pd.DataFrame,
    hyp: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate celltype_match from Song kinase_hypothesis_table.

    hyp has: kinase, top_celltype_1, top_celltype_2, top_celltype_3 (+ gene_symbol, residue_type)
    We use the first occurrence per kinase (has_high_conf_attribution kinases have full entries;
    others still carry top_celltype_1 as the best available guess).

    Returns hits with added: celltype_match (bool), celltype_match_rank (1/2/3/<NA>)
    """
    # First occurrence per kinase, melted to long form (kinase, owning_cluster, rank);
    # best (lowest) rank wins per (kinase, cluster). Vectorized join — the hit table
    # is millions of rows, so a per-row Python loop is not viable.
    hyp_first = hyp.drop_duplicates(subset=["kinase"], keep="first")
    rank_frames: list[pd.DataFrame] = []
    for col, rank in (("top_celltype_1", 1), ("top_celltype_2", 2), ("top_celltype_3", 3)):
        if col in hyp_first.columns:
            sub = hyp_first[["kinase", col]].dropna(subset=[col]).copy()
            sub = sub.rename(columns={col: "owning_cluster"})
            sub["celltype_match_rank"] = rank
            rank_frames.append(sub)
    if rank_frames:
        ranks = pd.concat(rank_frames, ignore_index=True)
        ranks["kinase"] = ranks["kinase"].astype(str)
        ranks["owning_cluster"] = ranks["owning_cluster"].astype(str)
        ranks = ranks.groupby(["kinase", "owning_cluster"], as_index=False)[
            "celltype_match_rank"
        ].min()
    else:
        ranks = pd.DataFrame(columns=["kinase", "owning_cluster", "celltype_match_rank"])

    return _apply_celltype_ranks(hits, ranks, ["kinase", "owning_cluster"])


def annotate_celltype_match_fivexfad(
    hits: pd.DataFrame,
    celltype_mea: pd.DataFrame,
    tissue: str,
) -> pd.DataFrame:
    """Annotate celltype_match for 5xFAD from celltype_mea.

    5xFAD kinase cell-type attribution: per (kinase, tissue, contrast), rank
    cell types by FDR then NES magnitude. Use the top-3 ranked cell types for
    the position-aware match.  The contrast is the MEA contrast (TG_vs_WT_Nmo).

    Returns hits with: celltype_match, celltype_match_rank
    """
    # Per (kinase, contrast): rank cell types by FDR asc then NES desc, take top 3,
    # melt to long form (kinase, contrast, owning_cluster, rank). Vectorized — best
    # (lowest) rank wins per (kinase, contrast, cluster).
    sub = celltype_mea[celltype_mea["tissue"] == tissue].copy()
    # Only named clusters (not cluster-*) can match node senders/receivers
    sub = sub[~sub["cell_type"].str.startswith("cluster-")]
    sub = sub.sort_values(["FDR", "NES"], ascending=[True, False])
    sub["celltype_match_rank"] = sub.groupby(["kinase", "contrast"]).cumcount() + 1
    sub = sub[sub["celltype_match_rank"] <= 3][
        ["kinase", "contrast", "cell_type", "celltype_match_rank"]
    ].rename(columns={"cell_type": "owning_cluster"})
    ranks = sub.groupby(
        ["kinase", "contrast", "owning_cluster"], as_index=False
    )["celltype_match_rank"].min()

    return _apply_celltype_ranks(hits, ranks, ["kinase", "contrast", "owning_cluster"])


def load_tcell_detected_attribution(donor: str) -> pd.DataFrame:
    """Load detected within-cohort T-cell states for one donor.

    ``tcell_within_cohort.py`` owns the state-detection calculation and its
    cross-cohort detection floor.  This bridge only consumes its emitted
    categorical call plus the raw fraction of cells expressing the kinase gene;
    it does not recreate an external-reference or cross-species attribution.

    Returns one row per (kinase, contrast, owning_cluster), with
    ``expression_fraction`` carrying the within-cohort fraction of cells
    expressing the kinase gene.  A state is eligible only when the upstream
    module has already called it detected.
    """
    path = TCELL_ATTRIBUTION_DIR / donor / "unified_attribution_tcells.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"T-cell {donor}: missing within-cohort attribution at {path}; "
            "cannot attribute motif hits without donor-specific state evidence."
        )

    cols = [
        "kinase", "contrast", "cell_type", "tcell_detected",
        "tcell_fraction_expressing",
    ]
    attribution = pd.read_csv(path, usecols=cols)
    detected = attribution.loc[
        attribution["tcell_detected"].astype("string").str.lower().eq("true"),
        cols,
    ].copy()
    if detected.empty:
        raise ValueError(f"T-cell {donor}: no detected state calls in {path}")

    # Import the gate from its owner, rather than copying a local threshold.
    from alz.cross_reference import tcell_within_cohort

    detected["tcell_fraction_expressing"] = pd.to_numeric(
        detected["tcell_fraction_expressing"], errors="raise"
    )
    below_floor = detected["tcell_fraction_expressing"] < (
        tcell_within_cohort.DETECTION_FRAC_MIN
    )
    if below_floor.any():
        raise ValueError(
            f"T-cell {donor}: detected state calls below the "
            "tcell_within_cohort detection floor"
        )

    detected = detected.rename(columns={
        "cell_type": "owning_cluster",
        "tcell_fraction_expressing": "expression_fraction",
    })
    key_cols = ["kinase", "contrast", "owning_cluster"]
    inconsistent = (
        detected.groupby(key_cols, dropna=False)["expression_fraction"]
        .nunique(dropna=False)
        .gt(1)
    )
    if inconsistent.any():
        raise ValueError(
            f"T-cell {donor}: inconsistent detected-state fractions in {path}"
        )
    return detected.drop_duplicates(subset=key_cols)[
        key_cols + ["expression_fraction"]
    ]


# ---------------------------------------------------------------------------
# Step 5: expression + disease context annotations (5xFAD only)
# ---------------------------------------------------------------------------

def annotate_fivexfad_expression(
    hits: pd.DataFrame,
    expr_spec: pd.DataFrame,
    tissue: str,
) -> pd.DataFrame:
    """Attach age-pooled expression fraction + concentration_tier for 5xFAD.

    expr_spec: fivexfad_expression_specificity.csv
    Records how expressed the KINASE gene is in each cell type × tissue (age-pooled).
    Join key: kinase × owning_cluster × tissue

    This answers: "is the kinase driving this substrate-node active in the cell type
    that owns the node?" — a proxy for kinase-in-context relevance.
    """
    sub = expr_spec[expr_spec["tissue"] == tissue][
        ["kinase", "cell_type", "fivexfad_fraction_cells_expressing",
         "fivexfad_concentration_tier"]
    ].drop_duplicates()
    sub = sub.rename(columns={
        "cell_type": "owning_cluster",
        "fivexfad_fraction_cells_expressing": "expression_fraction",
        "fivexfad_concentration_tier": "concentration_tier",
    })
    # Only named clusters
    sub = sub[~sub["owning_cluster"].str.startswith("cluster-")]
    join_cols = ["kinase", "owning_cluster"]
    hit_key = _string_key(hits, join_cols)
    for out_col in ("expression_fraction", "concentration_tier"):
        hits[out_col] = hit_key.map(_value_map(sub, join_cols, out_col)).to_numpy()
    return hits


def annotate_fivexfad_disease(
    hits: pd.DataFrame,
    snrna_attr: pd.DataFrame,
    tissue: str,
) -> pd.DataFrame:
    """Attach age-matched disease LFC from fivexfad_snrna_attribution.csv.

    Records the KINASE gene's own TG-vs-WT LFC in each cell type × tissue × age.
    The MEA contrast encodes age (TG_vs_WT_3mo → age_months=3). Join on
    kinase × owning_cluster × tissue × age_months.

    This answers: "how dysregulated is the kinase gene itself in the diseased
    owning cluster at the same age as the MEA contrast?"
    """
    sub = snrna_attr[snrna_attr["tissue"] == tissue][
        ["kinase", "cell_type", "age_months", "fivexfad_lfc"]
    ].drop_duplicates()
    sub = sub.rename(columns={"cell_type": "owning_cluster"})
    sub = sub[~sub["owning_cluster"].str.startswith("cluster-")]

    join_cols = ["kinase", "owning_cluster", "age_months"]
    disease = sub.rename(columns={"fivexfad_lfc": "disease_lfc"})
    age = hits["contrast"].astype("string").str.extract(r"(\d+)mo$", expand=False)
    key_df = pd.DataFrame({
        "kinase": hits["kinase"],
        "owning_cluster": hits["owning_cluster"],
        "age_months": age,
    }, index=hits.index)
    hits["disease_lfc"] = _string_key(key_df, join_cols).map(
        _value_map(disease, join_cols, "disease_lfc")
    ).to_numpy()
    return hits


# ---------------------------------------------------------------------------
# Step 6: Receptor-EM fan characterization
# ---------------------------------------------------------------------------

def build_recep_em_fan(hits_all: pd.DataFrame) -> pd.DataFrame:
    """Per Receptor-EM spine: count distinct Ligands (fan-in) and Targets (fan-out).

    Fan logic: for each unique (Receptor-gene, EM-gene) pair among matched substrate
    hits, find all (sender, receiver) pairs where BOTH genes appear at their respective
    roles. Then count distinct Ligand-genes and Target-genes that also appear in those
    same (sender, receiver) pairs. This characterizes the fan-in (upstream Ligands)
    and fan-out (downstream Targets) of each Receptor-EM spine.

    NOTE: (sender,receiver) is the linking key, so two substrate genes can appear in
    the same pair without necessarily being in the exact same 4-tuple path. This is a
    structural fan characterization, not a path enumeration (that is #Backbones).
    """
    empty_fan = pd.DataFrame(columns=[
        "Receptor", "EM", "n_ligands", "n_targets", "n_senders", "n_receivers",
        "n_sender_receiver_pairs", "example_sender", "example_receiver",
    ])
    if hits_all.empty:
        return empty_fan

    # Cell-type-matched hits only (default deliverable filters to matches)
    matched = hits_all[hits_all["celltype_match"] == True].copy() if "celltype_match" in hits_all.columns else hits_all.copy()
    if matched.empty:
        return empty_fan

    recep_df = matched[matched["role"] == "Receptor"][["gene_symbol", "sender", "receiver"]].drop_duplicates()
    em_df = matched[matched["role"] == "EM"][["gene_symbol", "sender", "receiver"]].drop_duplicates()
    lig_df = matched[matched["role"] == "Ligand"][["gene_symbol", "sender", "receiver"]].drop_duplicates()
    tgt_df = matched[matched["role"] == "Target"][["gene_symbol", "sender", "receiver"]].drop_duplicates()

    if recep_df.empty or em_df.empty:
        return empty_fan

    spine_df = recep_df.rename(columns={"gene_symbol": "Receptor"}).merge(
        em_df.rename(columns={"gene_symbol": "EM"}),
        on=["sender", "receiver"],
    )

    fan_rows: list[dict] = []
    for (receptor, em), grp in spine_df.groupby(["Receptor", "EM"]):
        pairs = set(zip(grp["sender"], grp["receiver"]))
        pair_idx = pd.MultiIndex.from_tuples(pairs)

        if not lig_df.empty:
            lig_pairs = lig_df.set_index(["sender", "receiver"])
            n_lig = lig_pairs.loc[lig_pairs.index.isin(pair_idx), "gene_symbol"].nunique()
        else:
            n_lig = 0

        if not tgt_df.empty:
            tgt_pairs = tgt_df.set_index(["sender", "receiver"])
            n_tgt = tgt_pairs.loc[tgt_pairs.index.isin(pair_idx), "gene_symbol"].nunique()
        else:
            n_tgt = 0

        ex_pair = next(iter(pairs))
        fan_rows.append({
            "Receptor": receptor,
            "EM": em,
            "n_ligands": int(n_lig),
            "n_targets": int(n_tgt),
            "n_senders": len({p[0] for p in pairs}),
            "n_receivers": len({p[1] for p in pairs}),
            "n_sender_receiver_pairs": len(pairs),
            "example_sender": ex_pair[0],
            "example_receiver": ex_pair[1],
        })

    if not fan_rows:
        return empty_fan
    return pd.DataFrame(fan_rows).sort_values(
        ["n_ligands", "n_targets"], ascending=False
    ).reset_index(drop=True)


def build_recep_em_fan_from_parquet(hits_parquet: Path, memory_limit: str = "4GB") -> pd.DataFrame:
    """DuckDB-backed fan characterization for streamed 5xFAD bridge output."""
    empty_fan = pd.DataFrame(columns=[
        "Receptor", "EM", "n_ligands", "n_targets", "n_senders", "n_receivers",
        "n_sender_receiver_pairs", "example_sender", "example_receiver",
    ])
    if not hits_parquet.exists():
        return empty_fan

    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    os.makedirs(spill, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{spill}'")
    safe = str(hits_parquet).replace("'", "''")
    sql = f"""
    WITH matched AS (
        SELECT gene_symbol, role, sender, receiver
        FROM read_parquet('{safe}')
        WHERE celltype_match = true
    ),
    recep AS (
        SELECT DISTINCT gene_symbol AS Receptor, sender, receiver
        FROM matched WHERE role = 'Receptor'
    ),
    em AS (
        SELECT DISTINCT gene_symbol AS EM, sender, receiver
        FROM matched WHERE role = 'EM'
    ),
    spine AS (
        SELECT DISTINCT r.Receptor, e.EM, r.sender, r.receiver
        FROM recep r
        JOIN em e USING (sender, receiver)
    ),
    base AS (
        SELECT Receptor, EM,
               COUNT(DISTINCT sender) AS n_senders,
               COUNT(DISTINCT receiver) AS n_receivers,
               COUNT(*) AS n_sender_receiver_pairs,
               MIN(sender) AS example_sender,
               MIN(receiver) AS example_receiver
        FROM spine
        GROUP BY Receptor, EM
    ),
    lig AS (
        SELECT s.Receptor, s.EM, COUNT(DISTINCT m.gene_symbol) AS n_ligands
        FROM spine s
        JOIN matched m
          ON s.sender = m.sender
         AND s.receiver = m.receiver
         AND m.role = 'Ligand'
        GROUP BY s.Receptor, s.EM
    ),
    tgt AS (
        SELECT s.Receptor, s.EM, COUNT(DISTINCT m.gene_symbol) AS n_targets
        FROM spine s
        JOIN matched m
          ON s.sender = m.sender
         AND s.receiver = m.receiver
         AND m.role = 'Target'
        GROUP BY s.Receptor, s.EM
    )
    SELECT b.Receptor, b.EM,
           COALESCE(l.n_ligands, 0) AS n_ligands,
           COALESCE(t.n_targets, 0) AS n_targets,
           b.n_senders, b.n_receivers, b.n_sender_receiver_pairs,
           b.example_sender, b.example_receiver
    FROM base b
    LEFT JOIN lig l USING (Receptor, EM)
    LEFT JOIN tgt t USING (Receptor, EM)
    ORDER BY n_ligands DESC, n_targets DESC
    """
    res = con.execute(sql).to_arrow_table().to_pandas()
    con.close()
    return res if not res.empty else empty_fan


# ---------------------------------------------------------------------------
# Step 7: per-kinase participation counts (n_backbones + n_paths)
# ---------------------------------------------------------------------------

def _q(col: str) -> str:
    """Double-quote a column name for DuckDB SQL."""
    return f'"{col}"'


def _detect_sigprob_cols(con: duckdb.DuckDBPyConnection, path: str) -> tuple[str, str]:
    """Return the two ``SigProb_*`` column names for one wide parquet."""
    names = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path}') LIMIT 0"
    ).fetchall()]
    sig = sorted(n for n in names if n.startswith("SigProb_"))
    if len(sig) != 2:
        raise SystemExit(f"{path}: expected 2 SigProb_* cols, found {sig!r}")
    if "PDS" not in names:
        raise SystemExit(f"{path}: no PDS column")
    return sig[0], sig[1]


def compute_participation_counts(
    hits_all: pd.DataFrame | str | Path,
    wide_glob: str,
    memory_limit: str = "8GB",
) -> pd.DataFrame:
    """Per-kinase pathway participation, at two grains, over the gated wide/ shards.

    A kinase *participates* in a gated path (canonical SigProb/PDS floor, pooled
    distinct across contrasts) when one of its floor-99 substrate genes appears at
    ANY node (L/R/EM/T) of that path in the matching sender×receiver pair.  Two
    counts capture different questions:

      n_backbones — distinct (Sender, Receiver, Receptor, EM) *spines* the kinase
                    acts on.  The kinase's own reach: one phosphorylation per
                    spine, the downstream Target fan-out collapsed.  This is the
                    breadth number the kinase tab shows.
      n_paths     — distinct full pathways (Sender, Receiver, Ligand, Receptor,
                    EM, Target) the kinase sits along.  Total end-to-end route
                    involvement; larger because each spine fans out across many
                    downstream targets.

    Entirely DuckDB-streamed: the wide shards (multi-GB decompressed) are never
    read whole into pandas; only the small per-kinase result is materialized.

    Returns columns: kinase, n_backbones, n_paths.
    """
    empty = pd.DataFrame(columns=["kinase", "n_backbones", "n_paths"])
    files = sorted(glob.glob(wide_glob))
    if not files:
        log.warning(f"participation: no wide parquets at {wide_glob}")
        return empty
    if isinstance(hits_all, (str, Path)):
        hits_parquet = Path(hits_all)
        if not hits_parquet.exists():
            log.warning(f"participation: hits parquet missing: {hits_parquet}")
            return empty
    elif hits_all.empty:
        return empty
    else:
        hits_parquet = None

    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    os.makedirs(spill, exist_ok=True)
    db_path = os.path.join(spill, f"kinase_bridge_participation_{os.getpid()}_{uuid.uuid4().hex}.duckdb")
    con = duckdb.connect(db_path)
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute("PRAGMA threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{spill}'")

    try:
        # Distinct kinase↔node attribution — ALL hits (matched or not): the
        # preamble counts a kinase's chains by substrate-phosphorylator
        # over-representation, not by cell-type match. Build it in DuckDB to
        # avoid a pandas drop_duplicates copy of the 5xFAD cortex hit table.
        if hits_parquet is None:
            con.register("hits_all", hits_all)
            con.execute("""
                CREATE TABLE attr AS
                SELECT DISTINCT kinase, gene_symbol, role, sender, receiver
                FROM hits_all
            """)
            con.unregister("hits_all")
        else:
            safe_hits = str(hits_parquet).replace("'", "''")
            con.execute(f"""
                CREATE TABLE attr AS
                SELECT DISTINCT kinase, gene_symbol, role, sender, receiver
                FROM read_parquet('{safe_hits}')
            """)
        con.execute("""
            CREATE TABLE kinase_dim AS
            SELECT ROW_NUMBER() OVER (ORDER BY kinase) AS kinase_id, kinase
            FROM (SELECT DISTINCT kinase FROM attr)
        """)
        for role in POSITIONS:
            con.execute(f"""
                CREATE TABLE attr_{role.lower()} AS
                SELECT DISTINCT k.kinase_id, a.sender, a.receiver, a.gene_symbol
                FROM attr a
                JOIN kinase_dim k USING (kinase)
                WHERE a.role = '{role}'
            """)

        con.execute("CREATE SEQUENCE path_id_seq START 1")
        con.execute("CREATE SEQUENCE spine_id_seq START 1")
        con.execute("CREATE TABLE path_dim (path_id BIGINT, path_key VARCHAR)")
        con.execute("CREATE TABLE spine_dim (spine_id BIGINT, spine_key VARCHAR)")
        con.execute("""
            CREATE TABLE path_touches (
                kinase_id BIGINT,
                path_id BIGINT
            )
        """)
        con.execute("""
            CREATE TABLE spine_touches (
                kinase_id BIGINT,
                spine_id BIGINT
            )
        """)
        for f in files:
            log.info(f"participation: processing {os.path.basename(f)}")
            sp1, sp2 = _detect_sigprob_cols(con, f)
            safe = f.replace("'", "''")
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE keyed_paths AS
                WITH paths AS (
                    SELECT DISTINCT
                           "Sender.group" AS sender,
                           "Receiver.group" AS receiver,
                           Ligand, Receptor, EM, Target
                    FROM read_parquet('{safe}')
                    WHERE ({_q(sp1)} > {SIGPROB_CUTOFF} OR {_q(sp2)} > {SIGPROB_CUTOFF})
                      AND ABS(PDS) >= {ABS_PDS_CUTOFF}
                )
                SELECT *,
                       CONCAT(
                           COALESCE(sender, ''), '\x1f',
                           COALESCE(receiver, ''), '\x1f',
                           COALESCE(Ligand, ''), '\x1f',
                           COALESCE(Receptor, ''), '\x1f',
                           COALESCE(EM, ''), '\x1f',
                           COALESCE(Target, '')
                       ) AS path_key,
                       CONCAT(
                           COALESCE(sender, ''), '\x1f',
                           COALESCE(receiver, ''), '\x1f',
                           COALESCE(Receptor, ''), '\x1f',
                           COALESCE(EM, '')
                       ) AS spine_key
                FROM paths
            """)
            con.execute("""
                INSERT INTO path_dim
                SELECT nextval('path_id_seq'), k.path_key
                FROM (SELECT DISTINCT path_key FROM keyed_paths) k
                LEFT JOIN path_dim d USING (path_key)
                WHERE d.path_key IS NULL
            """)
            con.execute("""
                INSERT INTO spine_dim
                SELECT nextval('spine_id_seq'), k.spine_key
                FROM (SELECT DISTINCT spine_key FROM keyed_paths) k
                LEFT JOIN spine_dim d USING (spine_key)
                WHERE d.spine_key IS NULL
            """)
            for role, col in (
                ("ligand", "Ligand"),
                ("receptor", "Receptor"),
                ("em", "EM"),
                ("target", "Target"),
            ):
                con.execute(f"""
                    INSERT INTO path_touches
                    SELECT DISTINCT a.kinase_id, d.path_id
                    FROM keyed_paths p
                    JOIN attr_{role} a
                      ON p.sender = a.sender
                     AND p.receiver = a.receiver
                     AND p.{_q(col)} = a.gene_symbol
                    JOIN path_dim d USING (path_key)
                    WHERE p.{_q(col)} IS NOT NULL
                """)
                con.execute(f"""
                    INSERT INTO spine_touches
                    SELECT DISTINCT a.kinase_id, d.spine_id
                    FROM keyed_paths p
                    JOIN attr_{role} a
                      ON p.sender = a.sender
                     AND p.receiver = a.receiver
                     AND p.{_q(col)} = a.gene_symbol
                    JOIN spine_dim d USING (spine_key)
                    WHERE p.{_q(col)} IS NOT NULL
                """)
            con.execute("""
                CREATE OR REPLACE TABLE path_touches AS
                SELECT DISTINCT kinase_id, path_id
                FROM path_touches
            """)
            con.execute("""
                CREATE OR REPLACE TABLE spine_touches AS
                SELECT DISTINCT kinase_id, spine_id
                FROM spine_touches
            """)
            con.execute("DROP TABLE keyed_paths")

        sql = """
        WITH nb AS (
            SELECT kinase_id, COUNT(*) AS n_backbones
            FROM spine_touches
            GROUP BY kinase_id
        ),
        np AS (
            SELECT kinase_id, COUNT(*) AS n_paths
            FROM path_touches
            GROUP BY kinase_id
        )
        SELECT k.kinase, nb.n_backbones, np.n_paths
        FROM np
        JOIN nb USING (kinase_id)
        JOIN kinase_dim k USING (kinase_id)
        ORDER BY n_backbones DESC, k.kinase
        """
        res = con.execute(sql).to_arrow_table().to_pandas()
    finally:
        con.close()
        for suffix in ("", ".wal"):
            try:
                os.remove(db_path + suffix)
            except FileNotFoundError:
                pass
    return res


# ---------------------------------------------------------------------------
# Song pipeline
# ---------------------------------------------------------------------------

def run_song(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build kinase_node_hits for Song cohort."""
    log.info("Song: loading MEA stoichiometry")
    mea_st = pd.read_csv(SONG_MEA_STOICH)
    # Song pY MEA if present
    mea_py = pd.read_csv(SONG_MEA_PY) if SONG_MEA_PY.exists() else pd.DataFrame()

    log.info("Song: loading stoichiometry matrix")
    stoich_st = pd.read_csv(SONG_STOICH_MATRIX, usecols=["site_id", "gene_symbol", "motif"])
    stoich_py = pd.read_csv(SONG_STOICH_MATRIX_PY, usecols=["site_id", "gene_symbol", "motif"])
    substrate_st = load_floor_substrate_sets(SONG_SUBSTRATE_SETS)
    substrate_py = load_floor_substrate_sets(SONG_SUBSTRATE_SETS_PY)

    log.info("Song: building substrate bridge (ST)")
    sub_st = build_substrate_bridge(mea_st, stoich_st, substrate_st)
    all_subs = [sub_st]
    if not mea_py.empty:
        log.info("Song: building substrate bridge (pY)")
        sub_py = build_substrate_bridge(mea_py, stoich_py, substrate_py)
        all_subs.append(sub_py)
    subs = pd.concat(all_subs, ignore_index=True).drop_duplicates()
    log.info(f"Song: {len(subs)} (kinase,contrast,gene) substrate rows")

    log.info("Song: loading gene_node_index")
    node_index = load_gene_node_index(SONG_INDEX)
    node_index = node_index.rename(columns={"gene": "gene_symbol"})

    log.info("Song: joining to node index")
    hits = gene_node_hits(subs, node_index)
    if hits.empty:
        log.warning("Song: no node hits found")
        return pd.DataFrame()

    log.info("Song: loading kinase_hypothesis_table for cell-type attribution")
    hyp = pd.read_csv(SONG_KINASE_HYP)
    hits = annotate_celltype_match_song(hits, hyp)

    hits["cohort"] = "song"
    hits["tissue"] = None
    # Song has no expression/disease columns
    hits["expression_fraction"] = None
    hits["concentration_tier"] = None
    hits["disease_lfc"] = None
    return hits


# ---------------------------------------------------------------------------
# 5xFAD pipeline
# ---------------------------------------------------------------------------

def run_fivexfad(tissue: str) -> pd.DataFrame:
    """Build kinase_node_hits for one 5xFAD tissue (cortex or hippocampus)."""
    log.info(f"5xFAD {tissue}: loading MEA stoichiometry")
    mea_st_path = FIVEXFAD_MEA_DIR / f"{tissue}_st_mea_stoichiometry.csv"
    mea_py_path = FIVEXFAD_MEA_DIR / f"{tissue}_py_mea_stoichiometry.csv"
    stoich_path = FIVEXFAD_MEA_DIR / f"{tissue}_st_stoichiometry_matrix.csv"
    stoich_py_path = FIVEXFAD_MEA_DIR / f"{tissue}_py_stoichiometry_matrix.csv"
    substrate_st_path = FIVEXFAD_MEA_DIR / f"{tissue}_st_mea_substrate_sets.csv"
    substrate_py_path = FIVEXFAD_MEA_DIR / f"{tissue}_py_mea_substrate_sets.csv"

    mea_st = pd.read_csv(mea_st_path)
    mea_py = pd.read_csv(mea_py_path) if mea_py_path.exists() else pd.DataFrame()
    stoich_st = pd.read_csv(stoich_path, usecols=["site_id", "gene_symbol", "motif"])
    stoich_py = (
        pd.read_csv(stoich_py_path, usecols=["site_id", "gene_symbol", "motif"])
        if stoich_py_path.exists()
        else pd.DataFrame()
    )
    substrate_st = load_floor_substrate_sets(substrate_st_path)
    substrate_py = load_floor_substrate_sets(substrate_py_path)

    log.info(f"5xFAD {tissue}: building substrate bridge")
    sub_st = build_substrate_bridge(mea_st, stoich_st, substrate_st)
    all_subs = [sub_st]
    if not mea_py.empty:
        sub_py = build_substrate_bridge(mea_py, stoich_py, substrate_py)
        all_subs.append(sub_py)
    subs = pd.concat(all_subs, ignore_index=True).drop_duplicates()
    log.info(f"5xFAD {tissue}: {len(subs)} substrate rows")

    log.info(f"5xFAD {tissue}: loading gene_node_index")
    idx_path = Path(str(FIVEXFAD_INDEX_TMPL).replace("{tissue}", tissue))
    node_index = load_gene_node_index(idx_path)
    node_index = node_index.rename(columns={"gene": "gene_symbol"})

    log.info(f"5xFAD {tissue}: joining to node index")
    hits = gene_node_hits(subs, node_index)
    if hits.empty:
        log.warning(f"5xFAD {tissue}: no node hits")
        return pd.DataFrame()

    log.info(f"5xFAD {tissue}: loading celltype_mea for attribution")
    celltype_mea = pd.read_parquet(FIVEXFAD_CELLTYPE_MEA)
    hits = annotate_celltype_match_fivexfad(hits, celltype_mea, tissue)

    log.info(f"5xFAD {tissue}: attaching expression specificity")
    expr_spec = pd.read_csv(FIVEXFAD_EXPR_SPEC)
    hits = annotate_fivexfad_expression(hits, expr_spec, tissue)

    log.info(f"5xFAD {tissue}: attaching disease LFC")
    snrna_attr = pd.read_csv(FIVEXFAD_SNRNA_ATTR)
    hits = annotate_fivexfad_disease(hits, snrna_attr, tissue)

    hits["cohort"] = "fivexfad"
    hits["tissue"] = tissue
    return hits


def write_fivexfad_streamed(tissue: str, out_dir: Path, wide_glob: str) -> bool:
    """Write 5xFAD bridge outputs without materializing the hit table in pandas."""
    log.info(f"5xFAD {tissue}: loading MEA stoichiometry")
    mea_st_path = FIVEXFAD_MEA_DIR / f"{tissue}_st_mea_stoichiometry.csv"
    mea_py_path = FIVEXFAD_MEA_DIR / f"{tissue}_py_mea_stoichiometry.csv"
    stoich_path = FIVEXFAD_MEA_DIR / f"{tissue}_st_stoichiometry_matrix.csv"
    stoich_py_path = FIVEXFAD_MEA_DIR / f"{tissue}_py_stoichiometry_matrix.csv"
    substrate_st_path = FIVEXFAD_MEA_DIR / f"{tissue}_st_mea_substrate_sets.csv"
    substrate_py_path = FIVEXFAD_MEA_DIR / f"{tissue}_py_mea_substrate_sets.csv"

    mea_st = pd.read_csv(mea_st_path)
    mea_py = pd.read_csv(mea_py_path) if mea_py_path.exists() else pd.DataFrame()
    stoich_st = pd.read_csv(stoich_path, usecols=["site_id", "gene_symbol", "motif"])
    stoich_py = (
        pd.read_csv(stoich_py_path, usecols=["site_id", "gene_symbol", "motif"])
        if stoich_py_path.exists()
        else pd.DataFrame()
    )
    substrate_st = load_floor_substrate_sets(substrate_st_path)
    substrate_py = load_floor_substrate_sets(substrate_py_path)

    log.info(f"5xFAD {tissue}: building substrate bridge")
    all_subs = [build_substrate_bridge(mea_st, stoich_st, substrate_st)]
    if not mea_py.empty:
        all_subs.append(build_substrate_bridge(mea_py, stoich_py, substrate_py))
    subs = pd.concat(all_subs, ignore_index=True).drop_duplicates()
    log.info(f"5xFAD {tissue}: {len(subs)} substrate rows")
    if subs.empty:
        return False

    log.info(f"5xFAD {tissue}: loading gene_node_index")
    idx_path = Path(str(FIVEXFAD_INDEX_TMPL).replace("{tissue}", tissue))
    node_index = load_gene_node_index(idx_path).rename(columns={"gene": "gene_symbol"})

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "kinase_node_hits.parquet"

    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    os.makedirs(spill, exist_ok=True)
    db_path = os.path.join(spill, f"kinase_bridge_fivexfad_{tissue}_{os.getpid()}_{uuid.uuid4().hex}.duckdb")
    con = duckdb.connect(db_path)
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute(f"SET temp_directory='{spill}'")
    try:
        con.register("subs_df", subs)
        con.register("node_index_df", node_index)
        con.execute("CREATE TABLE subs AS SELECT * FROM subs_df")
        con.execute("CREATE TABLE node_index AS SELECT * FROM node_index_df")
        con.unregister("subs_df")
        con.unregister("node_index_df")
        del subs, node_index

        safe_celltype = str(FIVEXFAD_CELLTYPE_MEA).replace("'", "''")
        safe_expr = str(FIVEXFAD_EXPR_SPEC).replace("'", "''")
        safe_snrna = str(FIVEXFAD_SNRNA_ATTR).replace("'", "''")
        safe_parquet = str(parquet_path).replace("'", "''")

        log.info(f"5xFAD {tissue}: writing kinase_node_hits")
        con.execute(f"""
            CREATE TABLE ranks AS
            WITH ranked AS (
                SELECT kinase, contrast, cell_type AS owning_cluster,
                       ROW_NUMBER() OVER (
                           PARTITION BY kinase, contrast
                           ORDER BY FDR ASC, NES DESC
                       ) AS celltype_match_rank
                FROM read_parquet('{safe_celltype}')
                WHERE tissue = $tissue
                  AND NOT starts_with(cell_type, 'cluster-')
            )
            SELECT kinase, contrast, owning_cluster, MIN(celltype_match_rank) AS celltype_match_rank
            FROM ranked
            WHERE celltype_match_rank <= 3
            GROUP BY kinase, contrast, owning_cluster
        """, {"tissue": tissue})
        con.execute(f"""
            CREATE TABLE expr AS
            SELECT DISTINCT kinase,
                   cell_type AS owning_cluster,
                   fivexfad_fraction_cells_expressing AS expression_fraction,
                   fivexfad_concentration_tier AS concentration_tier
            FROM read_csv_auto('{safe_expr}')
            WHERE tissue = $tissue
              AND NOT starts_with(cell_type, 'cluster-')
        """, {"tissue": tissue})
        con.execute(f"""
            CREATE TABLE disease AS
            SELECT DISTINCT kinase,
                   cell_type AS owning_cluster,
                   CAST(age_months AS INTEGER) AS age_months,
                   fivexfad_lfc AS disease_lfc
            FROM read_csv_auto('{safe_snrna}')
            WHERE tissue = $tissue
              AND NOT starts_with(cell_type, 'cluster-')
        """, {"tissue": tissue})

        final_sql = f"""
            WITH hits AS (
                SELECT s.kinase, s.contrast, s.channel, s.NES, s.FDR, s.n_sites, s.sites,
                       s.gene_symbol, n.role, n.sender, n.receiver,
                       CASE WHEN n.role = 'Ligand' THEN n.sender ELSE n.receiver END AS owning_cluster,
                       n.n_rows, n.best_abs_pds
                FROM subs s
                JOIN node_index n USING (gene_symbol)
            ),
            annotated AS (
                SELECT 'fivexfad' AS cohort,
                       $tissue AS tissue,
                       h.kinase, h.contrast, h.channel, h.NES, h.FDR, h.n_sites, h.sites,
                       h.gene_symbol, h.role, h.sender, h.receiver,
                       h.owning_cluster,
                       r.celltype_match_rank IS NOT NULL AS celltype_match,
                       r.celltype_match_rank,
                       h.n_rows, h.best_abs_pds,
                       e.expression_fraction, e.concentration_tier,
                       d.disease_lfc
                FROM hits h
                LEFT JOIN ranks r
                  ON h.kinase = r.kinase
                 AND h.contrast = r.contrast
                 AND h.owning_cluster = r.owning_cluster
                LEFT JOIN expr e
                  ON h.kinase = e.kinase
                 AND h.owning_cluster = e.owning_cluster
                LEFT JOIN disease d
                  ON h.kinase = d.kinase
                 AND h.owning_cluster = d.owning_cluster
                 AND CAST(regexp_extract(h.contrast, '(\\d+)mo$', 1) AS INTEGER) = d.age_months
            )
            SELECT {", ".join(_q(c) for c in FINAL_COLS)}
            FROM annotated
        """
        con.execute(f"COPY ({final_sql}) TO '{safe_parquet}' (FORMAT PARQUET)", {"tissue": tissue})
        stats = con.execute(f"""
            SELECT COUNT(*) AS n_total,
                   COUNT(DISTINCT kinase || '\x1f' || contrast) AS n_active,
                   SUM(CASE WHEN celltype_match THEN 1 ELSE 0 END) AS n_matched
            FROM read_parquet('{safe_parquet}')
        """).fetchone()
    finally:
        con.close()
        for suffix in ("", ".wal"):
            try:
                os.remove(db_path + suffix)
            except FileNotFoundError:
                pass

    n_total, n_active, n_matched = stats
    log.info(f"Wrote {n_total} rows to {parquet_path}")

    fan_df = build_recep_em_fan_from_parquet(parquet_path)
    fan_path = out_dir / "recep_em_fan.csv"
    fan_df.to_csv(fan_path, index=False)
    log.info(f"Wrote fan characterization ({len(fan_df)} Receptor-EM spines) to {fan_path}")

    part_df = compute_participation_counts(parquet_path, wide_glob)
    part_path = out_dir / "kinase_participation.csv"
    part_df.to_csv(part_path, index=False)
    log.info(f"Wrote participation (n_backbones + n_paths) for {len(part_df)} kinases to {part_path}")

    manifest = textwrap.dedent(f"""\
    # kinase_incytr_bridge — 5xFAD {tissue}

    Generated by alz/cross_reference/kinase_incytr_bridge.py (B4)

    ## Parameters
    - Attribution substrate floor: kl_percentile >= {config.INCYTR_ATTRIBUTION_KL_PCT}
      (no MEA-FDR gate; NES/FDR are retained as edge annotations)
    - Backbone floor: SigProb > {SIGPROB_CUTOFF} (either side) AND |PDS| >= {ABS_PDS_CUTOFF}

    ## Counts
    - Active (kinase,contrast) pairs: {n_active}
    - Total node hit rows: {n_total}
    - Cell-type-matched hits: {int(n_matched or 0)}

    ## Pathway set
    - 5xFAD: outputs/reports/incytr_pair_mode_5xfad/{tissue}/wide/

    ## Output files
    - kinase_node_hits.parquet : flat hit table
    - recep_em_fan.csv : per Receptor-EM spine fan characterization
    - kinase_participation.csv : per-kinase participation over gated paths
    - MANIFEST.md : this file
    """)
    manifest_path = out_dir / "MANIFEST.md"
    manifest_path.write_text(manifest)
    log.info(f"Wrote MANIFEST to {manifest_path}")
    return True


# ---------------------------------------------------------------------------
# Output + manifest
# ---------------------------------------------------------------------------

FINAL_COLS = [
    "cohort", "tissue", "kinase", "contrast", "channel", "NES", "FDR", "n_sites", "sites",
    "gene_symbol", "role", "sender", "receiver", "owning_cluster",
    "celltype_match", "celltype_match_rank",
    "n_rows", "best_abs_pds",
    "expression_fraction", "concentration_tier", "disease_lfc",
]


def write_outputs(hits_all: pd.DataFrame, out_dir: Path, cohort_label: str,
                  wide_glob: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reorder and fill missing columns in place. Avoid materializing a full copy of
    # the hit table; 5xFAD cortex can be millions of rows after KsG admission.
    for c in FINAL_COLS:
        if c not in hits_all.columns:
            hits_all[c] = None
    hit_cols = [c for c in FINAL_COLS if c in hits_all.columns]

    parquet_path = out_dir / "kinase_node_hits.parquet"
    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    os.makedirs(spill, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute(f"SET temp_directory='{spill}'")
    con.register("hits_all", hits_all)
    col_sql = ", ".join(_q(c) for c in hit_cols)
    safe_parquet = str(parquet_path).replace("'", "''")
    con.execute(f"COPY (SELECT {col_sql} FROM hits_all) TO '{safe_parquet}' (FORMAT PARQUET)")
    con.close()
    log.info(f"Wrote {len(hits_all)} rows to {parquet_path}")

    # Receptor-EM fan characterization
    fan_df = build_recep_em_fan(hits_all)
    fan_path = out_dir / "recep_em_fan.csv"
    fan_df.to_csv(fan_path, index=False)
    log.info(f"Wrote fan characterization ({len(fan_df)} Receptor-EM spines) to {fan_path}")

    # Per-kinase participation: n_backbones (spine breadth) + n_paths (full routes)
    part_df = compute_participation_counts(hits_all, wide_glob)
    part_path = out_dir / "kinase_participation.csv"
    part_df.to_csv(part_path, index=False)
    log.info(f"Wrote participation (n_backbones + n_paths) for {len(part_df)} "
             f"kinases to {part_path}")

    # MANIFEST
    n_active = hits_all[["kinase", "contrast"]].drop_duplicates().shape[0]
    n_matched = int(hits_all["celltype_match"].sum()) if "celltype_match" in hits_all.columns else "?"
    n_total = len(hits_all)
    manifest = textwrap.dedent(f"""\
    # kinase_incytr_bridge — {cohort_label}

    Generated by alz/cross_reference/kinase_incytr_bridge.py (B4)

    ## Parameters
    - Attribution substrate floor: kl_percentile >= {config.INCYTR_ATTRIBUTION_KL_PCT}
      (no MEA-FDR gate; NES/FDR are retained as edge annotations)
    - Backbone floor: SigProb > {SIGPROB_CUTOFF} (either side) AND |PDS| >= {ABS_PDS_CUTOFF}

    ## Counts
    - Active (kinase,contrast) pairs: {n_active}
    - Total node hit rows: {n_total}
    - Cell-type-matched hits: {n_matched}

    ## Pathway set
    - Song: outputs/reports/incytr_pair_mode/wide/
    - 5xFAD: outputs/reports/incytr_pair_mode_5xfad/{{tissue}}/wide/
    - T-cells: outputs/reports/incytr_pair_mode_tcells/{{donor}}/wide/
    (Each run uses the cohort-scoped gene_node_index.json.gz for its wide/ shards.)

    ## Exclusions
    - 15 cluster-* cell types in 5xFAD expression/attribution tables have no
      Sender.group/Receiver.group counterpart in pair-mode outputs and are excluded
      from celltype_match; they never appear in the owning_cluster column.
    - Song expression/disease-context columns (expression_fraction, concentration_tier,
      disease_lfc) are NULL — deferred to B5's snrna step.
    - T-cell runs are scoped to one donor and use that donor's existing
      within-cohort state-attribution output.

    ## Output files
    - kinase_node_hits.parquet : flat hit table (all rows; celltype_match=False rows
      retained for traceability; default filter = celltype_match==True)
    - recep_em_fan.csv : per Receptor-EM spine fan characterization
    - kinase_participation.csv : per-kinase participation over gated paths —
      n_backbones (distinct Sender-Receiver-Receptor-EM spines, the breadth number)
      and n_paths (distinct full Ligand-Receptor-EM-Target pathways, total routes),
      both counting any-node (L/R/EM/T) participation
    - MANIFEST.md : this file
    """)
    manifest_path = out_dir / "MANIFEST.md"
    manifest_path.write_text(manifest)
    log.info(f"Wrote MANIFEST to {manifest_path}")


def write_tcells_streamed(donor: str, out_dir: Path) -> bool:
    """Write one donor's T-cell bridge from its pre-built node index.

    The only cell-type attribution admitted here is the donor's detected
    within-cohort state call from ``tcell_within_cohort.py``.  Its raw fraction
    of state cells expressing the kinase is retained in ``expression_fraction``;
    the existing cross-cohort-only context fields remain null.
    """
    attribution_path = TCELL_ATTRIBUTION_DIR / donor / "unified_attribution_tcells.csv"
    if not attribution_path.exists():
        log.error(
            "T-cell %s: within-cohort attribution is unavailable at %s; "
            "no motif edges were emitted. Donor-specific attribution is required "
            "and donor1 evidence will not be reused.",
            donor,
            attribution_path,
        )
        return False

    mea_dir = TCELL_ATTRIBUTION_DIR / donor / "mea"
    mea_path = mea_dir / "mea_timecourse.csv"
    mea_py_path = mea_dir / "mea_timecourse_pY.csv"
    substrate_path = mea_dir / "mea_substrate_sets.csv"
    substrate_py_path = mea_dir / "mea_substrate_sets_pY.csv"
    stoich_path = TCELL_ATTRIBUTION_DIR / donor / "stoichiometry_matrix.csv"
    stoich_py_path = TCELL_ATTRIBUTION_DIR / donor / "stoichiometry_matrix_pY.csv"
    index_path = Path(str(TCELL_INDEX_TMPL).replace("{donor}", donor))
    required = (mea_path, substrate_path, stoich_path, index_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"T-cell {donor}: required bridge input missing: {', '.join(missing)}"
        )

    log.info("T-cell %s: loading MEA, floor-99 substrate sets, and stoichiometry matrix", donor)
    mea = pd.read_csv(mea_path)
    mea_py = pd.read_csv(mea_py_path) if mea_py_path.exists() else pd.DataFrame()
    stoich_st = pd.read_csv(stoich_path, usecols=["site_id", "gene_symbol", "motif"])
    stoich_py = (
        pd.read_csv(stoich_py_path, usecols=["site_id", "gene_symbol", "motif"])
        if stoich_py_path.exists()
        else pd.DataFrame()
    )
    substrate = load_floor_substrate_sets(substrate_path)
    substrate_py = load_floor_substrate_sets(substrate_py_path)
    all_subs = [build_substrate_bridge(mea, stoich_st, substrate)]
    if not mea_py.empty:
        all_subs.append(build_substrate_bridge(mea_py, stoich_py, substrate_py))
    subs = pd.concat(all_subs, ignore_index=True).drop_duplicates()
    log.info("T-cell %s: %s (kinase,contrast,gene) substrate rows", donor, len(subs))
    if subs.empty:
        log.warning("T-cell %s: no floor-99 substrate rows", donor)
        return False

    log.info("T-cell %s: loading gene_node_index", donor)
    node_index = load_gene_node_index(index_path).rename(columns={"gene": "gene_symbol"})
    attribution = load_tcell_detected_attribution(donor)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "kinase_node_hits.parquet"
    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    os.makedirs(spill, exist_ok=True)
    db_path = os.path.join(
        spill,
        f"kinase_bridge_tcells_{donor}_{os.getpid()}_{uuid.uuid4().hex}.duckdb",
    )
    con = duckdb.connect(db_path)
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{spill}'")
    try:
        con.register("subs_df", subs)
        con.register("node_index_df", node_index)
        con.register("attribution_df", attribution)
        con.execute("CREATE TABLE subs AS SELECT * FROM subs_df")
        con.execute("CREATE TABLE node_index AS SELECT * FROM node_index_df")
        con.execute("CREATE TABLE attribution AS SELECT * FROM attribution_df")
        con.unregister("subs_df")
        con.unregister("node_index_df")
        con.unregister("attribution_df")
        del subs, node_index, attribution

        safe_parquet = str(parquet_path).replace("'", "''")
        log.info("T-cell %s: writing kinase_node_hits", donor)
        final_sql = f"""
            WITH hits AS (
                SELECT s.kinase, s.contrast, s.channel, s.NES, s.FDR, s.n_sites, s.sites,
                       s.gene_symbol, n.role, n.sender, n.receiver,
                       CASE WHEN n.role = 'Ligand' THEN n.sender ELSE n.receiver END AS owning_cluster,
                       n.n_rows, n.best_abs_pds
                FROM subs s
                JOIN node_index n USING (gene_symbol)
            ),
            annotated AS (
                SELECT 'tcells' AS cohort,
                       $donor AS tissue,
                       h.kinase, h.contrast, h.channel, h.NES, h.FDR, h.n_sites, h.sites,
                       h.gene_symbol, h.role, h.sender, h.receiver,
                       h.owning_cluster,
                       a.owning_cluster IS NOT NULL AS celltype_match,
                       CAST(NULL AS INTEGER) AS celltype_match_rank,
                       h.n_rows, h.best_abs_pds,
                       a.expression_fraction,
                       CAST(NULL AS VARCHAR) AS concentration_tier,
                       CAST(NULL AS DOUBLE) AS disease_lfc
                FROM hits h
                LEFT JOIN attribution a
                  ON h.kinase = a.kinase
                 AND regexp_extract(h.contrast, '_(d[0-9]+)_vs_', 1) = a.contrast
                 AND h.owning_cluster = a.owning_cluster
            )
            SELECT {", ".join(_q(c) for c in FINAL_COLS)}
            FROM annotated
        """
        con.execute(
            f"COPY ({final_sql}) TO '{safe_parquet}' (FORMAT PARQUET)",
            {"donor": donor},
        )
        stats = con.execute(f"""
            SELECT COUNT(*) AS n_total,
                   COUNT(DISTINCT kinase || '\x1f' || contrast) AS n_active,
                   SUM(CASE WHEN celltype_match THEN 1 ELSE 0 END) AS n_matched
            FROM read_parquet('{safe_parquet}')
        """).fetchone()
    finally:
        con.close()
        for suffix in ("", ".wal"):
            try:
                os.remove(db_path + suffix)
            except FileNotFoundError:
                pass

    n_total, n_active, n_matched = stats
    log.info(
        "T-cell %s: %s node hits; %s active kinase-contrast pairs; %s state-attributed hits",
        donor,
        n_total,
        n_active,
        int(n_matched or 0),
    )

    manifest = textwrap.dedent(f"""\
    # kinase_incytr_bridge — T-cell {donor}

    Generated by alz/cross_reference/kinase_incytr_bridge.py (B4)

    ## Parameters
    - Attribution substrate floor: kl_percentile >= {config.INCYTR_ATTRIBUTION_KL_PCT}
      (no MEA-FDR gate; NES/FDR are retained as edge annotations)
    - Backbone floor: SigProb > {SIGPROB_CUTOFF} (either side) AND |PDS| >= {ABS_PDS_CUTOFF}

    ## Counts
    - Active (kinase,contrast) pairs: {n_active}
    - Total node hit rows: {n_total}
    - Cell-type-matched hits: {int(n_matched or 0)}

    ## Attribution
    - Donor-scoped within-cohort state calls from
      outputs/reports/kinase_attribution_tcells/{donor}/unified_attribution_tcells.csv.
    - celltype_match is true only for states already called detected by
      tcell_within_cohort.py; expression_fraction is the raw fraction of that
      state expressing the kinase gene.

    ## Pathway set
    - outputs/reports/incytr_pair_mode_tcells/{donor}/wide/

    ## Output files
    - kinase_node_hits.parquet : flat hit table
    - MANIFEST.md : this file
    """)
    manifest_path = out_dir / "MANIFEST.md"
    manifest_path.write_text(manifest)
    log.info("Wrote MANIFEST to %s", manifest_path)
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kinase → Incytr pathway bridge (B4)")
    parser.add_argument("--cohort", choices=["song", "fivexfad", "tcells", "all"], default="all")
    parser.add_argument("--tissue", choices=["cortex", "hippocampus", "both"], default="both",
                        help="5xFAD tissue (ignored for song cohort)")
    parser.add_argument("--donor", choices=["donor1", "donor2"],
                        help="T-cell donor (required with --cohort tcells)")
    args = parser.parse_args()

    if args.cohort == "tcells" and args.donor is None:
        parser.error("--donor is required with --cohort tcells")

    all_hits: list[pd.DataFrame] = []
    produced_any = False

    if args.cohort in ("song", "all"):
        con = duckdb.connect()
        song_hits = run_song(con)
        con.close()
        if not song_hits.empty:
            out_dir = OUT_ROOT / "song"
            wide_glob = str(SONG_WIDE_DIR / "*_incytr_output.parquet")
            write_outputs(song_hits, out_dir, "Song cohort", wide_glob)
            all_hits.append(song_hits)
            produced_any = True
        else:
            log.warning("Song: no hits produced")

    if args.cohort in ("fivexfad", "all"):
        tissues = ["cortex", "hippocampus"] if args.tissue == "both" else [args.tissue]
        for tissue in tissues:
            out_dir = OUT_ROOT / f"fivexfad_{tissue}"
            wide_glob = str(FIVEXFAD_WIDE_DIR / tissue / "wide" / "*_incytr_output.parquet")
            if write_fivexfad_streamed(tissue, out_dir, wide_glob):
                produced_any = True
            else:
                log.warning(f"5xFAD {tissue}: no hits produced")

    if args.cohort == "tcells":
        donor = args.donor
        assert donor is not None  # validated by argparse above
        out_dir = OUT_ROOT / f"tcells_{donor}"
        if write_tcells_streamed(donor, out_dir):
            produced_any = True
        else:
            log.warning("T-cell %s: no bridge output produced", donor)

    if all_hits:
        combined = pd.concat(all_hits, ignore_index=True)
        log.info(f"Combined: {len(combined)} total rows across all cohorts/tissues")
    elif not produced_any:
        log.warning("No hits produced for any cohort")


if __name__ == "__main__":
    main()
