"""Kinase→kinase interactome + per-node terminal-edge map (backend edge model).

Unifies two evidence sources into one provenance-tagged edge model that the viewer
walks to draw a single pathway's kinase sidechains:

  - literature (PhosphoSitePlus ``Kinase_Substrate_Dataset``): kinase→kinase edges
    where the substrate gene is itself a kinase (tags an edge ``psp`` / ``both``).
  - motif (the existing MEA/kldata bridge, ``kinase_incytr_bridge.py``): a kinase's
    floor-99 substrate sites mapped onto pathway nodes; terminal strength is the
    measured edge delta.

Two artifacts per cohort:

  interactome.csv    kinase→kinase edges (the internal signaling cascade the client
                     walks upstream from a terminal kinase). Provenance ``motif`` /
                     ``psp`` / ``both`` tags which evidence source(s) support the edge.
  terminal_edges.csv kinase→pathway-node edges (the last hop attaching a kinase to a
                     pathway). Motif-derived, PSP-corroborated (tag ``both`` when PSP
                     confirms the same kinase→node pair, else ``motif``).

Cohort-parameterized: ``is_mouse`` toggles human→mouse homology (song + 5xFAD are
mouse → homology-map PhosphoSitePlus and the motif kinase gene; t-cells are human, so
that cohort skips the map and reads PhosphoSitePlus in its native space). The
interactome/terminal builders take an already-native-space *motif-edge frame* as
input — no cohort is hardcoded in the builders.

Reused read-only:
  - alz/integration/build_yuyu_kldata.py  (map_human_to_mouse homologene)

The motif source is the bridge's kinase_node_hits.parquet under ``BRIDGE_ROOT``
(written by kinase_incytr_bridge.py) — a data dependency, not a code one. The wide
Incytr shards are never read here; the bridge already reduced them.

Terminal ranking retains |NES| for deterministic kinase-row selection, while the
viewer uses measured ``edge_delta`` for terminal width. Chain edges carry no fused
strength scalar; provenance is their only visual signal.

Usage:
  pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort song
  pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort fivexfad
  pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort tcells
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Read-only import (do NOT edit that module).
from alz.integration.build_yuyu_kldata import map_human_to_mouse
from alz.shared import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path("outputs/reports")

# PhosphoSitePlus dataset shipped inside the kinase_library package (read-only;
# do NOT vendor into the repo). 3-line date/license preamble precedes the header.
PSP_FILE = (
    REPO_ROOT
    / ".pixi/envs/default/lib/python3.11/site-packages/kinase_library"
    / "databases/substrates/Kinase_Substrate_Dataset_count_07_2021.txt"
)
PSP_SKIPROWS = 3  # "070821", license line, blank — header row starts "KINASE\t..."

KINASE_ABBREV_MAP = Path("data/derived/caches/kinase_to_gene_mapping.csv")
KINASE_ABBREV_OVERRIDES = Path("data/derived/caches/kinase_to_gene_overrides.csv")

BRIDGE_ROOT = REPORTS / "kinase_incytr_bridge"
OUT_ROOT = REPORTS / "kinase_kinase_edges"
SONG_CELLTYPE_EVIDENCE = REPORTS / "attribution_recovery" / "celltype_evidence_table.csv"

# cohort -> bridge output dirs (one per tissue; per donor for t-cells).
# t-cells are donor1-only: donor2 has no within-cohort attribution, so the bridge
# emits no donor2 motif source (see kinase_incytr_bridge.write_tcells_streamed).
COHORT_DIRS = {
    "song": ["song"],
    "fivexfad": ["fivexfad_cortex", "fivexfad_hippocampus"],
    "tcells": ["tcells_donor1"],
}
# song + 5xFAD are mouse cohorts → homology-map PhosphoSitePlus (human) to mouse.
# t-cells are human → PhosphoSitePlus is already in the cohort's native gene space.
MOUSE_COHORTS = {"song", "fivexfad"}


# ---------------------------------------------------------------------------
# kinase-abbreviation → gene reconciliation (dual-role node collapse)
# ---------------------------------------------------------------------------

def load_kinase_abbrev_map() -> dict[str, str]:
    """abbreviation → human gene symbol, overrides winning over the base mapping."""
    base = pd.read_csv(KINASE_ABBREV_MAP)
    mapping = dict(zip(base["kinase_abbreviation"], base["gene_symbol"]))
    if KINASE_ABBREV_OVERRIDES.exists():
        ov = pd.read_csv(KINASE_ABBREV_OVERRIDES)
        if not ov.empty:
            mapping.update(dict(zip(ov["kinase_abbreviation"], ov["gene_symbol"])))
    return mapping


# ---------------------------------------------------------------------------
# PhosphoSitePlus kinase→kinase edges (human; homology-mapped downstream)
# ---------------------------------------------------------------------------

def load_psp_kinase_edges() -> pd.DataFrame:
    """PhosphoSitePlus human→human kinase→kinase edges, references summed per pair.

    A kinase→kinase edge is a substrate row whose ``SUB_GENE`` is itself one of the
    dataset's kinase genes (``GENE``). Autophosphorylation self-loops (GENE==SUB_GENE)
    are dropped. IN_VIVO_REF_COUNT / IN_VITRO_REF_COUNT are summed across the sites of
    a pair (total literature support for the relationship).

    Returns: source_gene, target_gene, in_vivo_refs, in_vitro_refs  (all human).
    """
    psp = pd.read_csv(PSP_FILE, sep="\t", skiprows=PSP_SKIPROWS, encoding="latin-1")
    hum = psp[(psp["KIN_ORGANISM"] == "human") & (psp["SUB_ORGANISM"] == "human")]
    kinase_genes = set(hum["GENE"].dropna())
    kk = hum[hum["SUB_GENE"].isin(kinase_genes)].copy()
    kk = kk[kk["GENE"] != kk["SUB_GENE"]]  # drop autophosphorylation self-loops
    edges = (
        kk.groupby(["GENE", "SUB_GENE"], as_index=False)[
            ["IN_VIVO_REF_COUNT", "IN_VITRO_REF_COUNT"]
        ]
        .sum()
        .rename(
            columns={
                "GENE": "source_gene",
                "SUB_GENE": "target_gene",
                "IN_VIVO_REF_COUNT": "in_vivo_refs",
                "IN_VITRO_REF_COUNT": "in_vitro_refs",
            }
        )
    )
    return edges


# ---------------------------------------------------------------------------
# motif-edge frame (bridge kinase_node_hits → cohort-native kinase→node edges)
# ---------------------------------------------------------------------------

def load_motif_edges(bridge_dir: str, is_mouse: bool) -> pd.DataFrame:
    """Load the motif-edge frame for one bridge cohort dir, in cohort-native space.

    The bridge's kinase_node_hits.parquet is multi-GB decompressed (9-23M rows), so
    it is aggregated to distinct (kinase, node gene, role, contrast) via DuckDB under a
    memory cap — never read whole into pandas. The kinase abbreviation is reconciled to
    a human gene symbol, then homology-mapped to mouse when ``is_mouse``. The target
    (pathway-node) gene is already cohort-native (mouse for song/5xFAD).

    Contract schema (the T-cell bridge supplies direct-change fields; AD cohorts
    retain nulls for those fields until their cohort-specific change model lands):
      kinase, kinase_gene, target_gene, role, contrast, owning_cluster,
      best_abs_pds, best_abs_nes, signed_nes, best_fdr, n_sites, sites,
      n_significant_concordant, celltype_match
    """
    parquet = BRIDGE_ROOT / bridge_dir / "kinase_node_hits.parquet"
    if not parquet.exists():
        raise SystemExit(f"missing bridge motif source: {parquet}")

    spill = os.environ.get(
        "DUCKDB_TEMP_DIR", os.path.join(os.path.expanduser("~"), ".cache", "duckdb")
    )
    os.makedirs(spill, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute("PRAGMA threads=2")
    con.execute(f"SET temp_directory='{spill}'")
    safe = str(parquet).replace("'", "''")
    parquet_columns = set(con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{safe}')"
    ).fetchdf()["column_name"])
    sites_select = "sites" if "sites" in parquet_columns else "NULL::VARCHAR"
    has_direct_change_contract = "n_significant_concordant" in parquet_columns
    if bridge_dir.startswith("tcells_") and not has_direct_change_contract:
        raise ValueError(
            f"{bridge_dir}: direct-change bridge contract is required; "
            "motif-only terminal fallback is not supported"
        )
    n_sig_select = (
        "n_significant_concordant"
        if has_direct_change_contract
        else "NULL::INTEGER"
    )
    agg = con.execute(
        f"""
        WITH ranked_hits AS (
            -- Choose one deterministic max-|NES| row so signed_nes and n_sites
            -- remain aligned even when tracks tie on |NES|.
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY kinase, gene_symbol, role, contrast, owning_cluster
                       ORDER BY ABS(NES) DESC, FDR ASC, channel ASC
                   ) AS nes_rank
            FROM read_parquet('{safe}')
        )
        SELECT kinase,
               gene_symbol AS target_gene,
               role,
               contrast,
               owning_cluster,
               MAX(best_abs_pds) AS best_abs_pds,
               MAX(ABS(NES)) AS best_abs_nes,
               MAX(CASE WHEN nes_rank = 1 THEN NES END) AS signed_nes,
               MAX(CASE WHEN nes_rank = 1 THEN n_sites END) AS n_sites,
               MAX(CASE WHEN nes_rank = 1 THEN {sites_select} END) AS sites,
               MAX(CASE WHEN nes_rank = 1 THEN {n_sig_select} END) AS n_significant_concordant,
               MIN(FDR) AS best_fdr,
               BOOL_OR(celltype_match) AS celltype_match
        FROM ranked_hits
        GROUP BY kinase, gene_symbol, role, contrast, owning_cluster
        """
    ).to_arrow_table().to_pandas()
    con.close()

    for row in agg.itertuples(index=False):
        if pd.isna(row.sites):
            continue
        try:
            sites = json.loads(row.sites)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{bridge_dir}: invalid terminal site list for "
                f"{row.kinase}/{row.target_gene}/{row.contrast}"
            ) from exc
        if not isinstance(sites, list) or len(sites) != int(row.n_sites):
            raise ValueError(
                f"{bridge_dir}: site-list/count mismatch for "
                f"{row.kinase}/{row.target_gene}/{row.contrast}"
            )
        for site in sites:
            required_site_fields = {"motif", "residue_type", "kl_percentile"}
            if pd.notna(row.n_significant_concordant):
                required_site_fields.update({
                    "site_id", "site_position", "delta", "site_significance",
                    "concordant", "timecourse_consistency",
                })
            if not isinstance(site, dict) or not required_site_fields.issubset(site):
                raise ValueError(
                    f"{bridge_dir}: malformed terminal site evidence for "
                    f"{row.kinase}/{row.target_gene}/{row.contrast}"
                )
            if float(site["kl_percentile"]) < config.INCYTR_ATTRIBUTION_KL_PCT:
                raise ValueError(
                    f"{bridge_dir}: terminal site below floor for "
                    f"{row.kinase}/{row.target_gene}/{row.contrast}"
                )

    abbrev = load_kinase_abbrev_map()
    agg["kinase_gene"] = agg["kinase"].map(abbrev)
    unmapped = sorted(agg.loc[agg["kinase_gene"].isna(), "kinase"].unique())
    if unmapped:
        log.warning(f"{bridge_dir}: {len(unmapped)} kinase abbrevs unmapped, dropped: {unmapped}")
    agg = agg.dropna(subset=["kinase_gene"])

    if is_mouse:
        agg = _homology_map(agg, ["kinase_gene"])

    columns = [
        "kinase", "kinase_gene", "target_gene", "role", "contrast", "owning_cluster", "best_abs_pds",
        "best_abs_nes", "signed_nes", "best_fdr", "n_sites", "sites", "celltype_match",
    ]
    if has_direct_change_contract:
        columns.insert(-1, "n_significant_concordant")
    return agg[columns].reset_index(drop=True)


def assert_song_celltype_join(motif_edges: pd.DataFrame) -> None:
    """Fail the bridge build if Song Incytr cell types lack transcript evidence."""
    if not SONG_CELLTYPE_EVIDENCE.exists():
        raise FileNotFoundError(f"missing Song cell-type evidence: {SONG_CELLTYPE_EVIDENCE}")
    expected = set(
        pd.read_csv(SONG_CELLTYPE_EVIDENCE, usecols=["cell_type"])["cell_type"]
        .dropna()
        .astype(str)
    )
    observed = set(motif_edges["owning_cluster"].dropna().astype(str))
    unmatched = sorted(observed - expected)
    if unmatched:
        raise AssertionError(
            f"Song Incytr owning_cluster values lack transcript evidence: {unmatched}"
        )
    log.info("song: Incytr cell-type join unmatched=0 (%d observed labels)", len(observed))


def _homology_map(df: pd.DataFrame, gene_cols: list[str]) -> pd.DataFrame:
    """Map the given human gene columns to mouse in place; drop rows with any unmapped.

    Reuses map_human_to_mouse (homologene, cached). The mapping set is bounded to the
    kinase universe by callers, so this never triggers a mass mygene fetch.
    """
    genes: set[str] = set()
    for c in gene_cols:
        genes |= set(df[c].dropna())
    mapping = map_human_to_mouse(sorted(genes))
    out = df.copy()
    for c in gene_cols:
        out[c] = out[c].map(lambda g: mapping.get(g) or None)
    return out.dropna(subset=gene_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# edge fusion: motif/psp/both provenance tag
# ---------------------------------------------------------------------------

def build_interactome(motif_edges: pd.DataFrame, psp_edges: pd.DataFrame) -> pd.DataFrame:
    """Fuse the kinase→kinase evidence into one provenance-tagged edge list.

    Both frames must be in the same (cohort-native) gene space. The motif arm is
    restricted to edges whose target gene is itself a kinase (a source in either arm);
    the psp arm is the full literature kinase→kinase set. Motif contrasts are collapsed
    — the interactome is contrast-agnostic; per-contrast corroboration lives in
    terminal_edges.csv.

    Returns: source_gene, target_gene, provenance, in_vivo_refs, in_vitro_refs,
             n_motif_contrasts, motif_contrasts
    """
    kinase_set = (
        set(psp_edges["source_gene"])
        | set(psp_edges["target_gene"])
        | set(motif_edges["kinase_gene"])
    )
    mkk = motif_edges[motif_edges["target_gene"].isin(kinase_set)]
    motif_agg = (
        mkk.groupby(["kinase_gene", "target_gene"])
        .agg(
            n_motif_contrasts=("contrast", "nunique"),
            motif_contrasts=("contrast", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
        .rename(columns={"kinase_gene": "source_gene"})
    )

    merged = psp_edges.merge(
        motif_agg, on=["source_gene", "target_gene"], how="outer", indicator=True
    )
    merged["provenance"] = merged["_merge"].map(
        {"both": "both", "left_only": "psp", "right_only": "motif"}
    )
    merged["in_vivo_refs"] = merged["in_vivo_refs"].fillna(0).astype(int)
    merged["in_vitro_refs"] = merged["in_vitro_refs"].fillna(0).astype(int)
    merged["n_motif_contrasts"] = merged["n_motif_contrasts"].fillna(0).astype(int)
    merged["motif_contrasts"] = merged["motif_contrasts"].fillna("")

    cols = [
        "source_gene", "target_gene", "provenance",
        "in_vivo_refs", "in_vitro_refs", "n_motif_contrasts", "motif_contrasts",
    ]
    return merged[cols].sort_values(
        ["source_gene", "target_gene"]).reset_index(drop=True)


def build_terminal_map(motif_edges: pd.DataFrame, psp_edges: pd.DataFrame) -> pd.DataFrame:
    """Fuse motif kinase→pathway-node edges with PSP literature corroboration.

    A T-cell row is emitted only when its direct-change bridge count is at least
    one.  AD rows retain their existing motif-created behavior while the shared
    schema carries null direct-change fields. PSP corroborates a surviving row
    (tag ``both``) when it knows the same kinase→target pair; otherwise the row
    stays ``motif``. Pure-literature
    kinase→kinase edges with no motif support live in the interactome, not here.

    Returns: kinase, source_gene, target_gene, role, contrast, owning_cluster,
             celltype_match, provenance, best_abs_pds, best_abs_nes, signed_nes,
             best_fdr, n_sites, sites, n_significant_concordant, edge_delta
    """
    motif_edges = motif_edges.copy()
    if "sites" not in motif_edges.columns:
        motif_edges["sites"] = None
    has_direct_change_calls = (
        "n_significant_concordant" in motif_edges.columns
        and pd.to_numeric(
            motif_edges["n_significant_concordant"], errors="coerce"
        ).notna().any()
    )
    if has_direct_change_calls:
        motif_edges = motif_edges.loc[
            pd.to_numeric(motif_edges["n_significant_concordant"], errors="coerce").ge(1)
        ].copy()
    else:
        motif_edges["n_significant_concordant"] = None

    def edge_delta(sites: object) -> float:
        if sites is None or (isinstance(sites, float) and pd.isna(sites)):
            return float("nan")
        try:
            records = json.loads(sites) if isinstance(sites, str) else sites
        except (TypeError, json.JSONDecodeError):
            return float("nan")
        # The bridge already decided significance with its own alpha and stamped
        # each site's `changed` (BH-significant) flag — consume it, do not
        # re-threshold here (that would fork the significance cutoff).
        deltas = [
            float(site["delta"])
            for site in records or []
            if isinstance(site, dict)
            and site.get("changed") is True
            and site.get("concordant") is True
            and pd.notna(site.get("delta"))
        ]
        return float(np.mean(deltas)) if deltas else float("nan")

    psp_pairs = psp_edges[["source_gene", "target_gene"]].drop_duplicates().copy()
    psp_pairs["_has_lit"] = True

    df = motif_edges.rename(columns={"kinase_gene": "source_gene"}).merge(
        psp_pairs, on=["source_gene", "target_gene"], how="left"
    )
    df["provenance"] = np.where(df["_has_lit"].notna(), "both", "motif")
    df["edge_delta"] = df["sites"].map(edge_delta)

    cols = [
        "kinase", "source_gene", "target_gene", "role", "contrast", "owning_cluster",
        "celltype_match", "provenance", "best_abs_pds", "best_abs_nes", "signed_nes",
        "best_fdr", "n_sites", "sites", "n_significant_concordant", "edge_delta",
    ]
    return df[cols].sort_values("best_abs_nes", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# cycle-safe upstream walk (verification convenience; the JS client walks the
# edge list itself — this exists so the closure has a runnable cycle guard)
# ---------------------------------------------------------------------------

def walk_upstream(interactome: pd.DataFrame, terminal_gene: str, max_hops: int = 6) -> set[str]:
    """Kinases reachable upstream of ``terminal_gene`` via source→target edges.

    Breadth-first with a visited set, so it terminates on the cyclic kinase→kinase
    graph. Edges point source(kinase)→target(kinase); walking "upstream" from a node
    means following edges whose target is the current frontier back to their sources.
    """
    up: dict[str, list[str]] = {}
    for src, tgt in zip(interactome["source_gene"], interactome["target_gene"]):
        up.setdefault(tgt, []).append(src)
    visited: set[str] = set()
    frontier = [terminal_gene]
    hops = 0
    while frontier and hops < max_hops:
        nxt: list[str] = []
        for node in frontier:
            for src in up.get(node, ()):
                if src not in visited:
                    visited.add(src)
                    nxt.append(src)
        frontier = nxt
        hops += 1
    return visited


def _cycle_safety_check() -> None:
    """Assert the upstream walk terminates on a synthetic cyclic edge set."""
    cyclic = pd.DataFrame(
        {"source_gene": ["A", "B", "C", "A"], "target_gene": ["B", "C", "A", "C"]}
    )
    reachable = walk_upstream(cyclic, "C", max_hops=100)
    assert reachable == {"A", "B", "C"}, reachable  # full cycle reached, no hang
    assert walk_upstream(cyclic, "A", max_hops=100) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# per-cohort run + output
# ---------------------------------------------------------------------------

def run_cohort_dir(bridge_dir: str, is_mouse: bool) -> None:
    log.info(f"{bridge_dir}: loading PhosphoSitePlus kinase→kinase edges")
    psp = load_psp_kinase_edges()
    if is_mouse:
        # Bound the mapping set to the kinase universe (both endpoints are kinases).
        psp = _homology_map(psp, ["source_gene", "target_gene"])
        psp = (
            psp.groupby(["source_gene", "target_gene"], as_index=False)[
                ["in_vivo_refs", "in_vitro_refs"]
            ].sum()
        )
    log.info(f"{bridge_dir}: {len(psp)} PSP kinase→kinase edges")

    log.info(f"{bridge_dir}: loading motif-edge frame (bridge kinase_node_hits)")
    motif = load_motif_edges(bridge_dir, is_mouse)
    log.info(f"{bridge_dir}: {len(motif)} motif edges (distinct kinase×node×role×contrast)")
    if bridge_dir == "song":
        assert_song_celltype_join(motif)

    interactome = build_interactome(motif, psp)
    terminal = build_terminal_map(motif, psp)

    out_dir = OUT_ROOT / bridge_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    interactome.to_csv(out_dir / "interactome.csv", index=False)
    terminal.to_csv(out_dir / "terminal_edges.csv", index=False)

    prov = interactome["provenance"].value_counts().to_dict()
    n_nodes = len(set(interactome["source_gene"]) | set(interactome["target_gene"]))
    tprov = terminal["provenance"].value_counts().to_dict()
    manifest = (
        f"# kinase_kinase_edges — {bridge_dir}\n\n"
        f"Generated by alz/cross_reference/kinase_kinase_edges.py\n\n"
        f"## Interactome (kinase→kinase)\n"
        f"- edges: {len(interactome)}  nodes: {n_nodes}\n"
        f"- provenance: motif={prov.get('motif',0)} psp={prov.get('psp',0)} both={prov.get('both',0)}\n\n"
        f"## Terminal edges (kinase→pathway-node)\n"
        f"- edges: {len(terminal)}\n"
        f"- provenance: motif={tprov.get('motif',0)} both={tprov.get('both',0)}\n"
        f"- signed_nes: direction on terminal edges (+ enriched, − depleted); |NES| remains kinase evidence.\n"
        f"- n_sites: distinct physical floor-99 sites per terminal edge; edge_delta is mean Δ over significant-concordant sites.\n"
        f"- provenance (motif/psp/both) tags evidence source; rank on |NES|, never p_value.\n"
    )
    (out_dir / "MANIFEST.md").write_text(manifest)

    print(f"\n=== {bridge_dir} ===")
    print(f"interactome: {len(interactome)} edges, {n_nodes} nodes")
    print(f"  provenance: motif={prov.get('motif',0)} psp={prov.get('psp',0)} both={prov.get('both',0)}")
    print(f"terminal_edges: {len(terminal)} edges")
    print(f"  provenance: motif={tprov.get('motif',0)} both={tprov.get('both',0)}")
    print(f"  written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kinase→kinase interactome + terminal-edge map")
    parser.add_argument("--cohort", choices=sorted(COHORT_DIRS), required=True)
    args = parser.parse_args()

    _cycle_safety_check()  # runnable cycle-safety self-check, every invocation

    is_mouse = args.cohort in MOUSE_COHORTS
    for bridge_dir in COHORT_DIRS[args.cohort]:
        run_cohort_dir(bridge_dir, is_mouse)


if __name__ == "__main__":
    main()
