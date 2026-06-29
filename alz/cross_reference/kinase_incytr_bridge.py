"""Kinase → Incytr pathway integration bridge (B4).

For each active kinase (MEA FDR ≤ 0.25), maps its leading-edge substrate
genes to pathway nodes (Ligand / Receptor / EM / Target) in the pair-mode
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

Pathway set used:
  Song    : outputs/reports/incytr_pair_mode/wide/
  5xFAD   : outputs/reports/incytr_pair_mode_5xfad/{tissue}/wide/
  (gene_node_index.json.gz confirms the wide/ shards for both)

B4 emits:
  - recep_em_fan.csv         : per Receptor-EM spine fan characterization
  - kinase_participation.csv : per-kinase participation over the gated wide/
      shards, at two grains — n_backbones (distinct Sender-Receiver-Receptor-EM
      spines, the kinase's own breadth) and n_paths (distinct full pathways the
      kinase sits along). Both count any-node (L/R/EM/T) participation, computed
      exactly via DuckDB (not estimated).

Usage:
  pixi run kinase-incytr-bridge            # all cohorts
  pixi run kinase-incytr-bridge -- --cohort song
  pixi run kinase-incytr-bridge -- --cohort fivexfad --tissue cortex
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
from pathlib import Path

import duckdb
import pandas as pd

# Repo root on sys.path so the in-process R-EM-T backbone reduction (folded into
# this build step) imports cleanly whether run as a script or via `-m`.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from alz.incytr_pair import backbone_reduction  # noqa: E402

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

# Canonical pair-mode significance floor (identical to filter_significant_paths.py
# and backbone_reduction.py): a chain "passes" when either side's SigProb clears
# 0.1 AND |PDS| >= 0.2.  #Backbones counts passing chains, so this floor is
# applied when enumerating wide/ full paths.
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

def _apply_celltype_ranks(
    hits: pd.DataFrame,
    ranks: pd.DataFrame,
    join_cols: list[str],
) -> pd.DataFrame:
    """Left-join a long-form (join_cols..., celltype_match_rank) table onto hits.

    Adds celltype_match (bool) and celltype_match_rank (Int64, <NA> when no match).
    Join keys are string-cast on both sides to match the prior per-row str() compare.
    """
    hits = hits.copy()
    for c in join_cols:
        hits[c] = hits[c].astype(str)
    if not ranks.empty:
        ranks = ranks.copy()
        for c in join_cols:
            ranks[c] = ranks[c].astype(str)
    hits = hits.merge(ranks, on=join_cols, how="left")
    hits["celltype_match"] = hits["celltype_match_rank"].notna()
    hits["celltype_match_rank"] = hits["celltype_match_rank"].astype("Int64")
    return hits


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
    hits_all: pd.DataFrame,
    wide_glob: str,
    memory_limit: str = "8GB",
) -> pd.DataFrame:
    """Per-kinase pathway participation, at two grains, over the gated wide/ shards.

    A kinase *participates* in a gated path (canonical SigProb/PDS floor, pooled
    distinct across contrasts) when one of its leading-substrate genes appears at
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
    if hits_all.empty:
        return empty

    spill = os.environ.get(
        "DUCKDB_TEMP_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "duckdb"),
    )
    os.makedirs(spill, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{spill}'")

    # Distinct kinase↔node attribution — ALL hits (matched or not): the preamble
    # counts a kinase's chains by substrate-phosphorylator over-representation,
    # not by cell-type match.
    attr = hits_all[["kinase", "gene_symbol", "role", "sender", "receiver"]].drop_duplicates()
    con.register("attr", attr)

    parts: list[str] = []
    for f in files:
        sp1, sp2 = _detect_sigprob_cols(con, f)
        safe = f.replace("'", "''")
        parts.append(f"""
            SELECT "Sender.group","Receiver.group",Ligand,Receptor,EM,Target
            FROM read_parquet('{safe}')
            WHERE ({_q(sp1)} > {SIGPROB_CUTOFF} OR {_q(sp2)} > {SIGPROB_CUTOFF})
              AND ABS(PDS) >= {ABS_PDS_CUTOFF}""")
    gated_union = "\n            UNION ALL".join(parts)

    # Distinct full paths get a synthetic id, then are unpivoted into one row per
    # occupied node so the kinase↔node match is a single 4-key equi-join
    # (sender, receiver, role, gene) — a hash join, not the cross-product an
    # OR-of-positions join over the (sender,receiver) bucket would produce.
    sql = f"""
    WITH gated AS ({gated_union}
    ),
    paths AS (
        SELECT *, ROW_NUMBER() OVER () AS path_id
        FROM (
            SELECT DISTINCT "Sender.group","Receiver.group",Ligand,Receptor,EM,Target
            FROM gated
        )
    ),
    nodes AS (
        SELECT path_id, "Sender.group" AS sender, "Receiver.group" AS receiver,
               'Ligand' AS role, Ligand AS gene FROM paths WHERE Ligand IS NOT NULL
        UNION ALL
        SELECT path_id, "Sender.group", "Receiver.group",
               'Receptor', Receptor FROM paths WHERE Receptor IS NOT NULL
        UNION ALL
        SELECT path_id, "Sender.group", "Receiver.group",
               'EM', EM FROM paths WHERE EM IS NOT NULL
        UNION ALL
        SELECT path_id, "Sender.group", "Receiver.group",
               'Target', Target FROM paths WHERE Target IS NOT NULL
    ),
    touched AS (
        SELECT DISTINCT a.kinase, n.path_id
        FROM nodes n
        JOIN attr a
          ON n.sender = a.sender
         AND n.receiver = a.receiver
         AND n.role = a.role
         AND n.gene = a.gene_symbol
    ),
    spine_touched AS (
        SELECT DISTINCT t.kinase,
               p."Sender.group", p."Receiver.group", p.Receptor, p.EM
        FROM touched t
        JOIN paths p ON t.path_id = p.path_id
    ),
    nb AS (SELECT kinase, COUNT(*) AS n_backbones FROM spine_touched GROUP BY kinase),
    np AS (SELECT kinase, COUNT(*) AS n_paths FROM touched GROUP BY kinase)
    SELECT np.kinase, nb.n_backbones, np.n_paths
    FROM np JOIN nb USING (kinase)
    ORDER BY n_backbones DESC, np.kinase
    """
    res = con.execute(sql).to_arrow_table().to_pandas()
    con.close()
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
    "n_rows", "best_abs_pds",
    "expression_fraction", "concentration_tier", "disease_lfc",
]


def write_outputs(hits_all: pd.DataFrame, out_dir: Path, cohort_label: str,
                  wide_glob: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reorder and fill missing columns
    for c in FINAL_COLS:
        if c not in hits_all.columns:
            hits_all[c] = None
    hits_out = hits_all[[c for c in FINAL_COLS if c in hits_all.columns]].copy()

    parquet_path = out_dir / "kinase_node_hits.parquet"
    hits_out.to_parquet(parquet_path, index=False, engine="pyarrow")
    log.info(f"Wrote {len(hits_out)} rows to {parquet_path}")

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
    - MEA_FDR_THRESH: {MEA_FDR_THRESH}
    - Backbone floor: SigProb > {SIGPROB_CUTOFF} (either side) AND |PDS| >= {ABS_PDS_CUTOFF}

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Kinase → Incytr pathway bridge (B4)")
    parser.add_argument("--cohort", choices=["song", "fivexfad", "all"], default="all")
    parser.add_argument("--tissue", choices=["cortex", "hippocampus", "both"], default="both",
                        help="5xFAD tissue (ignored for song cohort)")
    args = parser.parse_args()

    all_hits: list[pd.DataFrame] = []

    if args.cohort in ("song", "all"):
        con = duckdb.connect()
        song_hits = run_song(con)
        con.close()
        if not song_hits.empty:
            out_dir = OUT_ROOT / "song"
            wide_glob = str(SONG_WIDE_DIR / "*_incytr_output.parquet")
            write_outputs(song_hits, out_dir, "Song cohort", wide_glob)
            all_hits.append(song_hits)

            # Fold: materialize the R-EM-T pathway backbone here (the build step),
            # not as a standalone pipeline step.  B2's sankey reads it lazily.
            rem_t_out = str(
                REPORTS / "incytr_pair_mode" / "backbone" / "backbone_rem_t.parquet"
            )
            log.info("Song: reducing R-EM-T backbones for B2 sankey")
            bb_summary = backbone_reduction.reduce(
                wide_dir=str(SONG_WIDE_DIR), out_path=rem_t_out, verbose=False
            )
            log.info(f"Song: wrote {bb_summary['n_backbone_paths']:,} R-EM-T "
                     f"backbones to {rem_t_out}")
        else:
            log.warning("Song: no hits produced")

    if args.cohort in ("fivexfad", "all"):
        tissues = ["cortex", "hippocampus"] if args.tissue == "both" else [args.tissue]
        for tissue in tissues:
            fx_hits = run_fivexfad(tissue)
            if not fx_hits.empty:
                out_dir = OUT_ROOT / f"fivexfad_{tissue}"
                wide_glob = str(FIVEXFAD_WIDE_DIR / tissue / "wide" / "*_incytr_output.parquet")
                write_outputs(fx_hits, out_dir, f"5xFAD {tissue}", wide_glob)
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
