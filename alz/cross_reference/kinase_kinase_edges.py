"""Kinase→kinase interactome + per-node terminal-edge map (backend edge model).

Unifies two evidence sources into one weighted, provenance-tagged edge model that
the viewer walks to draw a single pathway's kinase sidechains:

  - literature (PhosphoSitePlus ``Kinase_Substrate_Dataset``): kinase→kinase edges
    where the substrate gene is itself a kinase; weight = log1p(IN_VIVO_REF_COUNT).
  - motif (the existing MEA/kldata bridge, ``kinase_incytr_bridge.py``): a kinase's
    leading-substrate genes mapped onto pathway nodes; corroboration strength = |PDS|.

Two artifacts per cohort:

  interactome.csv    kinase→kinase edges (the internal signaling cascade the client
                     walks upstream from a terminal kinase). Provenance ``motif`` /
                     ``psp`` / ``both``; weight = norm(log1p in_vivo) + norm(|PDS|).
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

Weight is normalized-additive and NOT clamped to [0,1]: each component is normalized
to [0,1] independently, then summed (range [0,2]). An edge corroborated by both
sources therefore scores above either single source — that sum *is* the corroboration
bonus. weight_lit / weight_motif are kept as separate columns for transparency.

Usage:
  pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort song
  pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort fivexfad
  pixi run python -m alz.cross_reference.kinase_kinase_edges --cohort tcells
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Read-only import (do NOT edit that module).
from alz.integration.build_yuyu_kldata import map_human_to_mouse

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

    Contract schema (also produced by subplan 02 for t-cells):
      kinase_gene, target_gene, role, contrast, best_abs_pds, celltype_match
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
    agg = con.execute(
        f"""
        SELECT kinase,
               gene_symbol AS target_gene,
               role,
               contrast,
               MAX(best_abs_pds) AS best_abs_pds,
               BOOL_OR(celltype_match) AS celltype_match
        FROM read_parquet('{safe}')
        GROUP BY kinase, gene_symbol, role, contrast
        """
    ).to_arrow_table().to_pandas()
    con.close()

    abbrev = load_kinase_abbrev_map()
    agg["kinase_gene"] = agg["kinase"].map(abbrev)
    unmapped = sorted(agg.loc[agg["kinase_gene"].isna(), "kinase"].unique())
    if unmapped:
        log.warning(f"{bridge_dir}: {len(unmapped)} kinase abbrevs unmapped, dropped: {unmapped}")
    agg = agg.dropna(subset=["kinase_gene"])

    if is_mouse:
        agg = _homology_map(agg, ["kinase_gene"])

    return agg[
        ["kinase_gene", "target_gene", "role", "contrast", "best_abs_pds", "celltype_match"]
    ].reset_index(drop=True)


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
# edge fusion: normalized-additive weight + motif/psp/both provenance tag
# ---------------------------------------------------------------------------

def _norm(series: pd.Series, ceiling: float) -> pd.Series:
    """Normalize to [0,1] by a fixed ceiling; 0 when the ceiling is degenerate."""
    if not np.isfinite(ceiling) or ceiling <= 0:
        return pd.Series(0.0, index=series.index)
    return (series / ceiling).clip(lower=0.0, upper=1.0)


def build_interactome(motif_edges: pd.DataFrame, psp_edges: pd.DataFrame) -> pd.DataFrame:
    """Fuse the kinase→kinase evidence into one provenance-tagged, weighted edge list.

    Both frames must be in the same (cohort-native) gene space. The motif arm is
    restricted to edges whose target gene is itself a kinase (a source in either arm);
    the psp arm is the full literature kinase→kinase set. Motif contrasts are collapsed
    (max |PDS|) — the interactome is contrast-agnostic; per-contrast corroboration lives
    in terminal_edges.csv.

    Returns: source_gene, target_gene, provenance, weight, weight_lit, weight_motif,
             in_vivo_refs, in_vitro_refs, n_motif_contrasts, motif_contrasts
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
            weight_motif_raw=("best_abs_pds", "max"),
            n_motif_contrasts=("contrast", "nunique"),
            motif_contrasts=("contrast", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
        .rename(columns={"kinase_gene": "source_gene"})
    )

    lit_ceiling = float(np.log1p(psp_edges["in_vivo_refs"]).max()) if not psp_edges.empty else 0.0
    motif_ceiling = float(motif_agg["weight_motif_raw"].max()) if not motif_agg.empty else 0.0

    psp = psp_edges.copy()
    psp["weight_lit_raw"] = np.log1p(psp["in_vivo_refs"])

    merged = psp.merge(motif_agg, on=["source_gene", "target_gene"], how="outer", indicator=True)
    merged["provenance"] = merged["_merge"].map(
        {"both": "both", "left_only": "psp", "right_only": "motif"}
    )
    merged["weight_lit"] = _norm(merged["weight_lit_raw"].fillna(0.0), lit_ceiling)
    merged["weight_motif"] = _norm(merged["weight_motif_raw"].fillna(0.0), motif_ceiling)
    merged["weight"] = merged["weight_lit"] + merged["weight_motif"]
    merged["in_vivo_refs"] = merged["in_vivo_refs"].fillna(0).astype(int)
    merged["in_vitro_refs"] = merged["in_vitro_refs"].fillna(0).astype(int)
    merged["n_motif_contrasts"] = merged["n_motif_contrasts"].fillna(0).astype(int)
    merged["motif_contrasts"] = merged["motif_contrasts"].fillna("")

    cols = [
        "source_gene", "target_gene", "provenance", "weight", "weight_lit", "weight_motif",
        "in_vivo_refs", "in_vitro_refs", "n_motif_contrasts", "motif_contrasts",
    ]
    return merged[cols].sort_values("weight", ascending=False).reset_index(drop=True)


def build_terminal_map(motif_edges: pd.DataFrame, psp_edges: pd.DataFrame) -> pd.DataFrame:
    """Fuse motif kinase→pathway-node edges with PSP literature corroboration.

    Motif-driven: every row is a motif terminal edge (kinase→node at a pathway role,
    per contrast). PSP corroborates a row (tag ``both``, adds weight_lit) when it knows
    the same kinase→target pair; otherwise the row stays ``motif``. Pure-literature
    kinase→kinase edges with no motif support live in the interactome, not here.

    Returns: source_gene, target_gene, role, contrast, celltype_match, provenance,
             weight, weight_lit, weight_motif, best_abs_pds
    """
    lit_ceiling = float(np.log1p(psp_edges["in_vivo_refs"]).max()) if not psp_edges.empty else 0.0
    motif_ceiling = float(motif_edges["best_abs_pds"].max()) if not motif_edges.empty else 0.0

    psp_lit = psp_edges[["source_gene", "target_gene", "in_vivo_refs"]].copy()
    psp_lit["weight_lit_raw"] = np.log1p(psp_lit["in_vivo_refs"])
    psp_lit = psp_lit[["source_gene", "target_gene", "weight_lit_raw"]]

    df = motif_edges.rename(columns={"kinase_gene": "source_gene"}).merge(
        psp_lit, on=["source_gene", "target_gene"], how="left"
    )
    df["provenance"] = np.where(df["weight_lit_raw"].notna(), "both", "motif")
    df["weight_lit"] = _norm(df["weight_lit_raw"].fillna(0.0), lit_ceiling)
    df["weight_motif"] = _norm(df["best_abs_pds"].fillna(0.0), motif_ceiling)
    df["weight"] = df["weight_lit"] + df["weight_motif"]

    cols = [
        "source_gene", "target_gene", "role", "contrast", "celltype_match", "provenance",
        "weight", "weight_lit", "weight_motif", "best_abs_pds",
    ]
    return df[cols].sort_values("weight", ascending=False).reset_index(drop=True)


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
        f"- provenance: motif={tprov.get('motif',0)} both={tprov.get('both',0)}\n\n"
        f"## Weight\n"
        f"- normalized-additive: norm(log1p(in_vivo_refs)) + norm(|PDS|), range [0,2].\n"
        f"- weight_lit / weight_motif kept separately. Rank on weight or |PDS|, never p_value.\n"
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
