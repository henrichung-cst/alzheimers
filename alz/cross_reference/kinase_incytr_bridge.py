"""Kinase → Incytr pathway integration bridge (B4).

For each active kinase (MEA FDR ≤ 0.25), maps its leading-edge substrate
genes to pathway nodes (Ligand / Receptor / EM / Target) in the pair-mode
wide/*.parquet shards, annotates cell-type match, expression, and disease
context, and emits a flat backend artifact plus a fan-structure
characterization for backbone-grain reconciliation with B5.

Cohorts implemented:
  - song    : 9 contrasts, ST-only stoichiometry MEA, no expression layer
               (B5/snrna prereq); celltype from kinase_hypothesis_table
  - fivexfad: cortex + hippocampus, ST + pY MEA; celltype from
               fivexfad_celltype_mea.parquet; expression from
               fivexfad_expression_specificity.csv (age-pooled); disease
               context from fivexfad_snrna_attribution.csv (age-matched)

Pathway set used:
  Song    : outputs/reports/incytr_pair_mode/wide/
  5xFAD   : outputs/reports/incytr_pair_mode_5xfad/{tissue}/wide/
  (gene_node_index.json.gz confirms the wide/ shards for both)

Backbone grain: NOT locked (deferred to B5). B4 emits:
  - recep_em_fan.csv        : per Receptor-EM spine fan characterization
  - kinase_backbone_counts.csv : n_backbones per kinase under parameterized key
  n_backbones is computed against BACKBONE_KEY (default: 'R-EM', provisional).
  Change BACKBONE_KEY to 'L-R-EM', 'R-EM-T', or 'full' to recount without
  code change — the key is the only parameter.

Usage:
  pixi run kinase-incytr-bridge            # all cohorts
  pixi run kinase-incytr-bridge -- --cohort song
  pixi run kinase-incytr-bridge -- --cohort fivexfad --tissue cortex
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

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

SONG_INDEX = UNIFIED_VIEWER_DIR / "incytr_pathways" / "gene_node_index.json.gz"
FIVEXFAD_INDEX_TMPL = UNIFIED_VIEWER_DIR / "incytr_pathways_fivexfad_{tissue}" / "gene_node_index.json.gz"

SONG_STOICH_MATRIX = SONG_MEA_DIR / "stoichiometry_matrix.csv"
SONG_MEA_STOICH = SONG_MEA_DIR / "mea_stoichiometry.csv"
SONG_MEA_PY = SONG_MEA_DIR / "mea_stoichiometry_pY.csv"
SONG_KINASE_HYP = REPORTS / "attribution_recovery" / "kinase_hypothesis_table.csv"

FIVEXFAD_EXPR_SPEC = FIVEXFAD_MEA_DIR / "fivexfad_expression_specificity.csv"
FIVEXFAD_SNRNA_ATTR = FIVEXFAD_MEA_DIR / "fivexfad_snrna_attribution.csv"
FIVEXFAD_CELLTYPE_MEA = FIVEXFAD_MEA_DIR / "celltype_mea" / "fivexfad_celltype_mea.parquet"

OUT_ROOT = REPORTS / "kinase_incytr_bridge"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEA_FDR_THRESH = 0.25

# Backbone key used for n_backbones rollup.  Candidates:
#   'R-EM'   : Receptor-EM spine (default, B4 provisional — deferred to B5)
#   'L-R-EM' : Ligand-Receptor-EM triple
#   'R-EM-T' : Receptor-EM-Target triple
#   'full'   : full 4-tuple (Ligand, Receptor, EM, Target)
BACKBONE_KEY: str = "R-EM"

POSITIONS = ("Ligand", "Receptor", "EM", "Target")

# Song MEA contrast → wide parquet filename mapping
_SONG_COND_MAP = {"App": "AppP", "Tau": "Ttau", "ApTt": "ApTt"}


def _song_contrast_to_parquet(contrast: str) -> str:
    """ApTt_2mo -> ma_2mo_ApTt_ma_2mo_WTyp_incytr_output.parquet"""
    condition, age = contrast.rsplit("_", 1)
    cond_file = _SONG_COND_MAP.get(condition, condition)
    return f"ma_{age}_{cond_file}_ma_{age}_WTyp_incytr_output.parquet"


def _fivexfad_age_from_contrast(mea_contrast: str) -> int:
    """TG_vs_WT_3mo -> 3"""
    m = re.search(r"(\d+)mo$", mea_contrast)
    if m is None:
        raise ValueError(f"Cannot extract age from MEA contrast: {mea_contrast!r}")
    return int(m.group(1))


def _fivexfad_parquet_name(age: int) -> str:
    """3 -> TG_3mo_WT_3mo_incytr_output.parquet"""
    return f"TG_{age}mo_WT_{age}mo_incytr_output.parquet"


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
# Step 2: substrate bridge (Leading substrates → stoichiometry_matrix → gene)
# ---------------------------------------------------------------------------

def build_substrate_bridge(mea_stoich: pd.DataFrame, stoich_matrix: pd.DataFrame) -> pd.DataFrame:
    """Parse 'Leading substrates' motif strings → unique (kinase, contrast, gene_symbol).

    'Leading substrates' motifs are 15-char _;-delimited strings like _QSTPSStPHASPK_.
    stoich_matrix.motif is 13-char uppercase (strip _ and uppercase to match).

    Returns columns: kinase, contrast, channel, NES, FDR, gene_symbol
    """
    rows: list[dict] = []
    motif_to_genes: dict[str, list[str]] = {}
    # Build motif -> gene_symbol map from stoich_matrix
    for _, r in stoich_matrix.iterrows():
        key = str(r["motif"]).upper()
        motif_to_genes.setdefault(key, []).append(str(r["gene_symbol"]))

    for _, row in mea_stoich.iterrows():
        if float(row["FDR"]) > MEA_FDR_THRESH:
            continue
        ls = str(row.get("Leading substrates", "") or "")
        if not ls:
            continue
        kinase = str(row["kinase"])
        contrast = str(row["contrast"])
        channel = str(row.get("track", "st"))
        nes = float(row["NES"]) if row["NES"] == row["NES"] else float("nan")
        fdr = float(row["FDR"])
        seen_genes: set[str] = set()
        for motif_raw in ls.split(";"):
            motif_raw = motif_raw.strip()
            if not motif_raw:
                continue
            key = motif_raw.upper().strip("_")
            for g in motif_to_genes.get(key, []):
                if g not in seen_genes:
                    seen_genes.add(g)
                    rows.append({
                        "kinase": kinase,
                        "contrast": contrast,
                        "channel": channel,
                        "NES": nes,
                        "FDR": fdr,
                        "gene_symbol": g,
                    })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["kinase", "contrast", "channel", "NES", "FDR", "gene_symbol"]
    )


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
      kinase, contrast, channel, NES, FDR, gene_symbol, role, sender, receiver,
      owning_cluster (the cluster that "owns" this node per position rule)
    """
    if substrate_df.empty:
        return pd.DataFrame()
    merged = substrate_df.merge(node_index, on="gene_symbol", how="inner")
    if merged.empty:
        return pd.DataFrame()
    # owning_cluster: Ligand -> sender; Receptor/EM/Target -> receiver
    merged["owning_cluster"] = merged.apply(
        lambda r: r["sender"] if r["role"] == "Ligand" else r["receiver"],
        axis=1,
    )
    return merged


# ---------------------------------------------------------------------------
# Step 4: cell-type match (position-aware)
# ---------------------------------------------------------------------------

def annotate_celltype_match_song(
    hits: pd.DataFrame,
    hyp: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate celltype_match from Song kinase_hypothesis_table.

    hyp has: kinase, top_celltype_1, top_celltype_2, top_celltype_3 (+ gene_symbol, residue_type)
    We use the first occurrence per kinase (has_high_conf_attribution kinases have full entries;
    others still carry top_celltype_1 as the best available guess).

    Returns hits with added: celltype_match (bool), celltype_match_rank (1/2/3/None)
    """
    # Build kinase -> [top_celltype_1, top_celltype_2, top_celltype_3]
    top_cts: dict[str, list[str]] = {}
    for _, r in hyp.iterrows():
        k = str(r["kinase"])
        if k not in top_cts:
            top_cts[k] = [
                str(r["top_celltype_1"]) if pd.notna(r.get("top_celltype_1")) else None,
                str(r["top_celltype_2"]) if pd.notna(r.get("top_celltype_2")) else None,
                str(r["top_celltype_3"]) if pd.notna(r.get("top_celltype_3")) else None,
            ]

    match_col: list[bool] = []
    rank_col: list[Optional[int]] = []
    for _, r in hits.iterrows():
        cts = top_cts.get(str(r["kinase"]), [None, None, None])
        owning = str(r["owning_cluster"])
        matched = False
        rank = None
        for i, ct in enumerate(cts):
            if ct and ct == owning:
                matched = True
                rank = i + 1
                break
        match_col.append(matched)
        rank_col.append(rank)
    hits = hits.copy()
    hits["celltype_match"] = match_col
    hits["celltype_match_rank"] = rank_col
    return hits


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
    # Build (kinase, contrast) -> [top_celltype_1, top_celltype_2, top_celltype_3]
    sub = celltype_mea[celltype_mea["tissue"] == tissue].copy()
    # Only named clusters (not cluster-*) can match node senders/receivers
    sub = sub[~sub["cell_type"].str.startswith("cluster-")]

    top_cts: dict[tuple[str, str], list[str]] = {}
    for (kinase, contrast), grp in sub.groupby(["kinase", "contrast"]):
        ranked = grp.sort_values(["FDR", "NES"], ascending=[True, False])
        ranked_cts = ranked["cell_type"].tolist()[:3]
        while len(ranked_cts) < 3:
            ranked_cts.append(None)
        top_cts[(str(kinase), str(contrast))] = ranked_cts

    match_col: list[bool] = []
    rank_col: list[Optional[int]] = []
    for _, r in hits.iterrows():
        key = (str(r["kinase"]), str(r["contrast"]))
        cts = top_cts.get(key, [None, None, None])
        owning = str(r["owning_cluster"])
        matched = False
        rank = None
        for i, ct in enumerate(cts):
            if ct and ct == owning:
                matched = True
                rank = i + 1
                break
        match_col.append(matched)
        rank_col.append(rank)
    hits = hits.copy()
    hits["celltype_match"] = match_col
    hits["celltype_match_rank"] = rank_col
    return hits


# ---------------------------------------------------------------------------
# Step 5: node-level fc from wide parquets (DuckDB-streamed)
# ---------------------------------------------------------------------------

def attach_node_fc(
    hits: pd.DataFrame,
    parquet_path: Path,
    contrast: str,
    channel: str,
) -> pd.DataFrame:
    """Attach per-node log2FC from a single wide parquet for one contrast.

    For each (gene_symbol, role, sender, receiver) in hits (for this contrast),
    pull the matching {role}_{channel}_log2FC + PDS from the parquet.
    Uses DuckDB so the parquet is never loaded whole into pandas.

    Returns: node_log2FC (channel-appropriate), node_PDS columns added.
    """
    if hits.empty:
        return hits

    fc_col = f"{{}}_{{channel}}_log2FC".replace("{channel}", channel)
    # Build a filter: gene at role AND sender AND receiver
    # We query each unique (role, sender, receiver, gene) combination
    con = duckdb.connect()
    # Register parquet as view
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet('{parquet_path}')")

    node_fc: dict[tuple[str, str, str, str], tuple[Optional[float], Optional[float]]] = {}
    for (role, sender, receiver, gene), subhits in hits.groupby(
        ["role", "sender", "receiver", "gene_symbol"]
    ):
        gene_col = role  # column name in parquet
        fc_colname = f"{role}_{channel}_log2FC"
        try:
            result = con.execute(f"""
                SELECT "{fc_colname}" AS fc, PDS
                FROM src
                WHERE "{gene_col}" = ?
                  AND "Sender.group" = ?
                  AND "Receiver.group" = ?
                LIMIT 1
            """, [gene, sender, receiver]).fetchone()
            if result:
                node_fc[(role, sender, receiver, gene)] = (result[0], result[1])
            else:
                node_fc[(role, sender, receiver, gene)] = (None, None)
        except Exception:
            node_fc[(role, sender, receiver, gene)] = (None, None)
    con.close()

    hits = hits.copy()
    hits["node_log2FC"] = hits.apply(
        lambda r: node_fc.get(
            (r["role"], r["sender"], r["receiver"], r["gene_symbol"]), (None, None)
        )[0],
        axis=1,
    )
    hits["node_PDS"] = hits.apply(
        lambda r: node_fc.get(
            (r["role"], r["sender"], r["receiver"], r["gene_symbol"]), (None, None)
        )[1],
        axis=1,
    )
    return hits


# ---------------------------------------------------------------------------
# Step 6: expression + disease context annotations (5xFAD only)
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
    return hits.merge(sub, on=["kinase", "owning_cluster"], how="left")


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

    # Extract age from contrast for each hit row
    hits = hits.copy()
    hits["_age"] = hits["contrast"].apply(
        lambda c: _fivexfad_age_from_contrast(c) if pd.notna(c) else None
    )
    hits = hits.merge(
        sub.rename(columns={"fivexfad_lfc": "disease_lfc"}),
        left_on=["kinase", "owning_cluster", "_age"],
        right_on=["kinase", "owning_cluster", "age_months"],
        how="left",
    )
    hits = hits.drop(columns=["_age", "age_months"], errors="ignore")
    return hits


# ---------------------------------------------------------------------------
# Step 7: fan characterization + backbone counts
# ---------------------------------------------------------------------------

def build_fan_and_backbone(
    hits_all: pd.DataFrame,
    backbone_key: str = "R-EM",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per Receptor-EM spine: count distinct Ligands (fan-in) and Targets (fan-out).

    Fan logic: for each unique (Receptor-gene, EM-gene) pair among matched substrate
    hits, find all (sender, receiver) pairs where BOTH genes appear at their respective
    roles. Then count distinct Ligand-genes and Target-genes that also appear in those
    same (sender, receiver) pairs. This characterizes the fan-in (upstream Ligands)
    and fan-out (downstream Targets) of each Receptor-EM spine — the Q3 deliverable
    for B5 backbone-grain reconciliation.

    NOTE: (sender,receiver) is used as the linking key, so two substrate genes can
    appear in the same pair without necessarily being in the exact same 4-tuple path.
    This is an upper-bound fan estimate; the exact bound requires loading the full
    pathway parquets, which is deferred heavy-compute (B5 gate).

    Returns:
        fan_df : recep_em_fan.csv
        backbone_df : kinase_backbone_counts.csv
    """
    if hits_all.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Use cell-type-matched hits only (plan: default deliverable filters to matches)
    matched = hits_all[hits_all["celltype_match"] == True].copy() if "celltype_match" in hits_all.columns else hits_all.copy()

    empty_fan = pd.DataFrame(columns=[
        "Receptor", "EM", "n_ligands", "n_targets", "n_senders", "n_receivers",
        "n_sender_receiver_pairs", "example_sender", "example_receiver",
    ])
    empty_bb = pd.DataFrame(columns=["kinase", "backbone_key", "n_backbones", "backbone_key_note"])
    if matched.empty:
        return empty_fan, empty_bb

    # Separate by role
    recep_df = matched[matched["role"] == "Receptor"][["gene_symbol", "sender", "receiver"]].drop_duplicates()
    em_df = matched[matched["role"] == "EM"][["gene_symbol", "sender", "receiver"]].drop_duplicates()
    lig_df = matched[matched["role"] == "Ligand"][["gene_symbol", "sender", "receiver"]].drop_duplicates()
    tgt_df = matched[matched["role"] == "Target"][["gene_symbol", "sender", "receiver"]].drop_duplicates()

    if recep_df.empty or em_df.empty:
        fan_df = empty_fan
    else:
        # Cross-join Receptor × EM on (sender, receiver)
        spine_df = recep_df.rename(columns={"gene_symbol": "Receptor"}).merge(
            em_df.rename(columns={"gene_symbol": "EM"}),
            on=["sender", "receiver"],
        )

        # Aggregate per (Receptor, EM) spine
        fan_rows: list[dict] = []
        for (receptor, em), grp in spine_df.groupby(["Receptor", "EM"]):
            pairs = set(zip(grp["sender"], grp["receiver"]))
            pair_idx = pd.MultiIndex.from_tuples(pairs)

            # Count Ligands in same (sender, receiver) pairs
            if not lig_df.empty:
                lig_pairs = lig_df.set_index(["sender", "receiver"])
                n_lig = lig_pairs.loc[lig_pairs.index.isin(pair_idx), "gene_symbol"].nunique()
            else:
                n_lig = 0

            # Count Targets in same (sender, receiver) pairs
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

        fan_df = pd.DataFrame(fan_rows).sort_values(
            ["n_ligands", "n_targets"], ascending=False
        ).reset_index(drop=True) if fan_rows else empty_fan

    # Backbone counts per kinase
    # Backbone key determines what constitutes a unique "backbone"
    def backbone_value(row: pd.Series) -> Optional[str]:
        gene = row["gene_symbol"]
        role = row["role"]
        sender = row["sender"]
        receiver = row["receiver"]
        if backbone_key == "R-EM":
            if role in ("Receptor", "EM"):
                return f"{sender}::{receiver}::{role}::{gene}"
            return None
        elif backbone_key == "L-R-EM":
            if role in ("Ligand", "Receptor", "EM"):
                return f"{sender}::{receiver}::{role}::{gene}"
            return None
        elif backbone_key == "R-EM-T":
            if role in ("Receptor", "EM", "Target"):
                return f"{sender}::{receiver}::{role}::{gene}"
            return None
        elif backbone_key == "full":
            return f"{sender}::{receiver}::{role}::{gene}"
        return None

    matched = matched.copy()
    matched["_backbone_value"] = matched.apply(backbone_value, axis=1)
    backbone_hits = matched[matched["_backbone_value"].notna()].copy()
    if backbone_hits.empty:
        backbone_df = pd.DataFrame(columns=["kinase", "backbone_key", "backbone_value", "n_backbones"])
    else:
        backbone_df = (
            backbone_hits.groupby("kinase")["_backbone_value"]
            .nunique()
            .reset_index()
            .rename(columns={"_backbone_value": "n_backbones"})
        )
        backbone_df["backbone_key"] = backbone_key
        backbone_df["backbone_key_note"] = "provisional — pending B5 grain reconciliation"
        backbone_df = backbone_df[["kinase", "backbone_key", "n_backbones", "backbone_key_note"]]

    return fan_df, backbone_df


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
    stoich = pd.read_csv(SONG_STOICH_MATRIX, usecols=["site_id", "gene_symbol", "motif"])

    log.info("Song: building substrate bridge (ST)")
    sub_st = build_substrate_bridge(mea_st, stoich)
    all_subs = [sub_st]
    if not mea_py.empty:
        log.info("Song: building substrate bridge (pY)")
        sub_py = build_substrate_bridge(mea_py, stoich)
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

    log.info("Song: attaching per-node log2FC from wide parquets")
    contrast_hits: list[pd.DataFrame] = []
    for contrast in hits["contrast"].unique():
        ch = hits[hits["contrast"] == contrast].copy()
        parquet_name = _song_contrast_to_parquet(contrast)
        parquet_path = SONG_WIDE_DIR / parquet_name
        if not parquet_path.exists():
            log.warning(f"Song: parquet not found for {contrast}: {parquet_path}")
            ch["node_log2FC"] = None
            ch["node_PDS"] = None
        else:
            # Determine channel from the 'channel' column
            channel = ch["channel"].iloc[0] if "channel" in ch.columns else "st"
            ch = attach_node_fc(ch, parquet_path, contrast, channel)
        contrast_hits.append(ch)
    hits = pd.concat(contrast_hits, ignore_index=True)

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

    mea_st = pd.read_csv(mea_st_path)
    mea_py = pd.read_csv(mea_py_path) if mea_py_path.exists() else pd.DataFrame()
    stoich = pd.read_csv(stoich_path, usecols=["site_id", "gene_symbol", "motif"])

    log.info(f"5xFAD {tissue}: building substrate bridge")
    sub_st = build_substrate_bridge(mea_st, stoich)
    all_subs = [sub_st]
    if not mea_py.empty:
        sub_py = build_substrate_bridge(mea_py, stoich)
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

    log.info(f"5xFAD {tissue}: attaching per-node log2FC from wide parquets")
    wide_dir = FIVEXFAD_WIDE_DIR / tissue / "wide"
    contrast_hits: list[pd.DataFrame] = []
    for contrast in hits["contrast"].unique():
        ch = hits[hits["contrast"] == contrast].copy()
        age = _fivexfad_age_from_contrast(contrast)
        parquet_name = _fivexfad_parquet_name(age)
        parquet_path = wide_dir / parquet_name
        if not parquet_path.exists():
            log.warning(f"5xFAD {tissue}: parquet not found for {contrast}: {parquet_path}")
            ch["node_log2FC"] = None
            ch["node_PDS"] = None
        else:
            channel = ch["channel"].iloc[0] if "channel" in ch.columns else "st"
            ch = attach_node_fc(ch, parquet_path, contrast, channel)
        contrast_hits.append(ch)
    hits = pd.concat(contrast_hits, ignore_index=True)

    log.info(f"5xFAD {tissue}: attaching expression specificity")
    expr_spec = pd.read_csv(FIVEXFAD_EXPR_SPEC)
    hits = annotate_fivexfad_expression(hits, expr_spec, tissue)

    log.info(f"5xFAD {tissue}: attaching disease LFC")
    snrna_attr = pd.read_csv(FIVEXFAD_SNRNA_ATTR)
    hits = annotate_fivexfad_disease(hits, snrna_attr, tissue)

    hits["cohort"] = "fivexfad"
    hits["tissue"] = tissue
    return hits


# ---------------------------------------------------------------------------
# Output + manifest
# ---------------------------------------------------------------------------

FINAL_COLS = [
    "cohort", "tissue", "kinase", "contrast", "channel", "NES", "FDR",
    "gene_symbol", "role", "sender", "receiver", "owning_cluster",
    "celltype_match", "celltype_match_rank",
    "node_log2FC", "node_PDS",
    "n_rows", "best_abs_pds",
    "expression_fraction", "concentration_tier", "disease_lfc",
]


def write_outputs(hits_all: pd.DataFrame, out_dir: Path, cohort_label: str,
                  backbone_key: str = BACKBONE_KEY) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reorder and fill missing columns
    for c in FINAL_COLS:
        if c not in hits_all.columns:
            hits_all[c] = None
    hits_out = hits_all[[c for c in FINAL_COLS if c in hits_all.columns]].copy()

    parquet_path = out_dir / "kinase_node_hits.parquet"
    hits_out.to_parquet(parquet_path, index=False, engine="pyarrow")
    log.info(f"Wrote {len(hits_out)} rows to {parquet_path}")

    # Fan characterization + backbone counts
    fan_df, backbone_df = build_fan_and_backbone(hits_all, backbone_key)
    fan_path = out_dir / "recep_em_fan.csv"
    fan_df.to_csv(fan_path, index=False)
    log.info(f"Wrote fan characterization ({len(fan_df)} Receptor-EM spines) to {fan_path}")

    bb_path = out_dir / "kinase_backbone_counts.csv"
    backbone_df.to_csv(bb_path, index=False)
    log.info(f"Wrote backbone counts ({len(backbone_df)} kinases) to {bb_path}")

    # MANIFEST
    n_active = hits_all[["kinase", "contrast"]].drop_duplicates().shape[0]
    n_matched = int(hits_all["celltype_match"].sum()) if "celltype_match" in hits_all.columns else "?"
    n_total = len(hits_all)
    manifest = textwrap.dedent(f"""\
    # kinase_incytr_bridge — {cohort_label}

    Generated by alz/cross_reference/kinase_incytr_bridge.py (B4)

    ## Parameters
    - MEA_FDR_THRESH: {MEA_FDR_THRESH}
    - BACKBONE_KEY: {backbone_key!r} (provisional — pending B5 backbone-grain reconciliation)

    ## Counts
    - Active (kinase,contrast) pairs: {n_active}
    - Total node hit rows: {n_total}
    - Cell-type-matched hits: {n_matched}

    ## Pathway set
    - Song: outputs/reports/incytr_pair_mode/wide/
    - 5xFAD: outputs/reports/incytr_pair_mode_5xfad/{{tissue}}/wide/
    (gene_node_index.json.gz confirms wide/ is the indexed pathway set for both cohorts)

    ## Exclusions
    - 15 cluster-* cell types in 5xFAD expression/attribution tables have no
      Sender.group/Receiver.group counterpart in pair-mode outputs and are excluded
      from celltype_match; they never appear in the owning_cluster column.
    - Song expression/disease-context columns (expression_fraction, concentration_tier,
      disease_lfc) are NULL — deferred to B5's snrna step.
    - T-cell cohort excluded (mea_timecourse.csv format, different scoping).

    ## Output files
    - kinase_node_hits.parquet : flat hit table (all rows; celltype_match=False rows
      retained for traceability; default filter = celltype_match==True)
    - recep_em_fan.csv : per Receptor-EM spine fan characterization (Q3 deliverable)
    - kinase_backbone_counts.csv : n_backbones per kinase at BACKBONE_KEY grain
      (provisional; recompute with --backbone-key to switch grain without code change)
    - MANIFEST.md : this file
    """)
    manifest_path = out_dir / "MANIFEST.md"
    manifest_path.write_text(manifest)
    log.info(f"Wrote MANIFEST to {manifest_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kinase → Incytr pathway bridge (B4)")
    parser.add_argument("--cohort", choices=["song", "fivexfad", "all"], default="all")
    parser.add_argument("--tissue", choices=["cortex", "hippocampus", "both"], default="both",
                        help="5xFAD tissue (ignored for song cohort)")
    parser.add_argument("--backbone-key", default=BACKBONE_KEY,
                        choices=["R-EM", "L-R-EM", "R-EM-T", "full"],
                        help="Backbone grain for n_backbones rollup (provisional, pending B5)")
    args = parser.parse_args()

    backbone_key_to_use = args.backbone_key

    all_hits: list[pd.DataFrame] = []

    if args.cohort in ("song", "all"):
        con = duckdb.connect()
        song_hits = run_song(con)
        con.close()
        if not song_hits.empty:
            out_dir = OUT_ROOT / "song"
            write_outputs(song_hits, out_dir, "Song cohort", backbone_key=backbone_key_to_use)
            all_hits.append(song_hits)
        else:
            log.warning("Song: no hits produced")

    if args.cohort in ("fivexfad", "all"):
        tissues = ["cortex", "hippocampus"] if args.tissue == "both" else [args.tissue]
        for tissue in tissues:
            fx_hits = run_fivexfad(tissue)
            if not fx_hits.empty:
                out_dir = OUT_ROOT / f"fivexfad_{tissue}"
                write_outputs(fx_hits, out_dir, f"5xFAD {tissue}", backbone_key=backbone_key_to_use)
                all_hits.append(fx_hits)
            else:
                log.warning(f"5xFAD {tissue}: no hits produced")

    if all_hits:
        combined = pd.concat(all_hits, ignore_index=True)
        log.info(f"Combined: {len(combined)} total rows across all cohorts/tissues")
    else:
        log.warning("No hits produced for any cohort")


if __name__ == "__main__":
    main()
