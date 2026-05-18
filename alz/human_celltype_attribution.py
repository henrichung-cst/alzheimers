#!/usr/bin/env python3
"""Human Cell-Type Attribution: top-N specific cell types per kinase.

Derives a ranked "top-N specific cell types" list for each kinase from each
human reference (SEA-AD MTG and Allen HBCA).  Reads the specificity matrices
produced by ``human_reference_expression.py`` and emits a long-form CSV
suitable for the viewer payload.

Output:
  outputs/reports/kinase_attribution_human/celltype_specificity.csv
  Columns: kinase, reference, celltype, specificity_score, rank

Usage:
    python alz/human_celltype_attribution.py          # compute and save
    python alz/human_celltype_attribution.py --summary # print cached results
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.HUMAN_CELLTYPE_ATTRIBUTION_OUTPUT_DIR
OUT_FILE = config.CELLTYPE_SPECIFICITY_FILE
TOP_N = config.HUMAN_CELLTYPE_TOP_N

SEAAD_SPEC_FILE = config.SEAAD_KINASE_SPECIFICITY_FILE
HBCA_SPEC_FILE = config.HBCA_KINASE_SPECIFICITY_FILE

HBCA_CORTEX_HIPPOCAMPUS_SUPERCLUSTERS = {
    "Astrocyte",
    "CGE interneuron",
    "Committed oligodendrocyte precursor",
    "Deep-layer corticothalamic and 6b",
    "Deep-layer intratelencephalic",
    "Deep-layer near-projecting",
    "Ependymal",
    "Fibroblast",
    "Hippocampal CA1-3",
    "Hippocampal CA4",
    "Hippocampal dentate gyrus",
    "LAMP5-LHX6 and Chandelier",
    "MGE interneuron",
    "Microglia",
    "Oligodendrocyte",
    "Oligodendrocyte precursor",
    "Upper-layer intratelencephalic",
    "Vascular",
}

HBCA_SUPERCLUSTER_TO_LEVY_T5 = {
    "Astrocyte": [
        "Astrocytes",
        "Ptprz1-protoplasmic-astrocytes",
    ],
    "CGE interneuron": [
        "Erbb4-VIP-inhibitory-neurons",
        "Erbb4-inhibitory-neurons",
        "VIP-positive-interneuron",
        "Reln-neurons",
        "Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic",
        "GABAergic-inhibitory-interneurons-VIP-positive",
        "Inhibitory-Neurons",
    ],
    "Committed oligodendrocyte precursor": [
        "OPC",
    ],
    "Deep-layer corticothalamic and 6b": [
        "Foxp2-Excitatory-Neurons-layers-6-and-2-3",
        "Excitatory-neurons-Cajal-Retzius-cells-layer-I-Reelin",
    ],
    "Deep-layer intratelencephalic": [
        "Excitatory-Rorb",
        "Excitatory-Pyramidal",
    ],
    "Deep-layer near-projecting": [
        "Foxp2-Excitatory-Neurons-layers-6-and-2-3",
    ],
    "Ependymal": [
        "Ependymal-cell",
    ],
    "Fibroblast": [
        "Vascular-Leptomeningeal-Cells",
    ],
    "Hippocampal CA1-3": [
        "Excitatory-Pyramidal",
        "Excitatory-neurons",
    ],
    "Hippocampal CA4": [
        "Excitatory-neurons",
        "glutamatergic-excitatory-neurons",
    ],
    "Hippocampal dentate gyrus": [
        "Excitatory principal neurons in the hippocampal dentate gyrus",
    ],
    "LAMP5-LHX6 and Chandelier": [
        "Reln-neurons",
        "Ndnf-positive-neurogliaform-inhibitory-interneurons-GABAergic",
        "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4",
    ],
    "MGE interneuron": [
        "GABAergic-inhibitory-interneurons-Dlx6os1-Erbb4",
        "GABAergic inhibitory interneurons",
    ],
    "Microglia": [
        "Microglia",
    ],
    "Oligodendrocyte": [
        "Oligodendrocytes",
    ],
    "Oligodendrocyte precursor": [
        "OPC",
    ],
    "Upper-layer intratelencephalic": [
        "Excitatory-Pyramidal-Satb2-Cux2",
        "Glutamatergic-excitatory-neurons-Cortical-layer-2-4-pyramidal-neurons",
    ],
    "Vascular": [
        "Endothelial-cell",
        "Pericyte",
        "Vascular-Leptomeningeal-Cells",
        "Choroid-Plexus-Epithelial-Cells",
    ],
}


def _filter_hbca_cortex_hpc(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c in HBCA_CORTEX_HIPPOCAMPUS_SUPERCLUSTERS]
    if not keep:
        raise RuntimeError("HBCA cortex/hippocampus filter removed all columns")
    return df.loc[:, keep]


def _cluster_source_map_for_reference(reference: str) -> dict[str, list[tuple[str, float]]]:
    """Return Levy T5 cluster -> source cell type weights for a human reference."""
    spine = set(config.CLUSTER_SPINE)
    if reference == "seaad_mtg":
        return {
            cluster: [(ct, float(w)) for ct, w in entries]
            for cluster, entries in config.load_cluster_to_seaad_supertype_map().items()
            if cluster in spine and entries
        }
    if reference == "allen_hbca":
        out: dict[str, list[tuple[str, float]]] = {}
        for hbca_ct, clusters in HBCA_SUPERCLUSTER_TO_LEVY_T5.items():
            for cluster in clusters:
                if cluster in spine:
                    out.setdefault(cluster, []).append((hbca_ct, 1.0))
        return out
    raise ValueError(f"unknown human reference: {reference!r}")


def _rollup_matrix_to_levy_t5(
    matrix_df: pd.DataFrame,
    reference: str,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Aggregate source cell-type columns onto Levy T5 cluster columns.

    SEA-AD is many-supertypes -> one Levy T5 cluster via curated weights.
    HBCA is coarser than Levy T5 for several lineages, so a broad HBCA class
    can support multiple Levy T5 clusters, matching the lineage-level WMB
    inheritance used by the mouse attribution view.
    """
    cluster_to_sources = _cluster_source_map_for_reference(reference)
    rolled = pd.DataFrame(index=matrix_df.index)
    source_labels: dict[str, list[str]] = {}

    for cluster in config.CLUSTER_SPINE:
        entries = cluster_to_sources.get(cluster, [])
        present = [(src, weight) for src, weight in entries if src in matrix_df.columns]
        if not present:
            continue
        cols = [src for src, _ in present]
        weights = pd.Series([weight for _, weight in present], index=cols, dtype=float)
        denom = weights.sum()
        if denom <= 0:
            continue
        rolled[cluster] = matrix_df[cols].astype(float).mul(weights / denom, axis=1).sum(axis=1)
        source_labels[cluster] = cols

    if rolled.empty:
        raise RuntimeError(f"{reference} → Levy T5 rollup produced no mapped columns")
    return rolled, source_labels


def _rollup_seaad_lfc_to_levy_t5(
    seaad_lfc: dict[tuple[str, str], float] | None,
    kinases: list[str],
    source_labels: dict[str, list[str]],
) -> dict[tuple[str, str], float] | None:
    """Aggregate SEA-AD supertype LFC values onto Levy T5 clusters."""
    if seaad_lfc is None:
        return None
    cluster_to_sources = _cluster_source_map_for_reference("seaad_mtg")
    out: dict[tuple[str, str], float] = {}
    for kinase in kinases:
        for cluster, cols in source_labels.items():
            entries = [(src, weight) for src, weight in cluster_to_sources.get(cluster, [])
                       if src in cols and (kinase, src) in seaad_lfc]
            if not entries:
                continue
            weights = np.asarray([weight for _, weight in entries], dtype=float)
            vals = np.asarray([seaad_lfc[(kinase, src)] for src, _ in entries], dtype=float)
            finite = np.isfinite(vals) & np.isfinite(weights)
            if finite.any() and weights[finite].sum() > 0:
                out[(kinase, cluster)] = float(np.average(vals[finite], weights=weights[finite]))
    return out


# ---------------------------------------------------------------------------
# Core ranking
# ---------------------------------------------------------------------------


def _top_n_for_reference(
    spec_df: pd.DataFrame,
    reference: str,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Produce long-form top-N rows for a single reference.

    Parameters
    ----------
    spec_df : pd.DataFrame
        Shape kinase_id × celltype, values = log2(ct_mean / brain_mean).
    reference : str
        Reference label (e.g. "seaad_mtg" or "allen_hbca").
    top_n : int
        Number of top cell types per kinase to retain.

    Returns
    -------
    pd.DataFrame
        Long-form with columns: kinase, reference, celltype, specificity_score, rank.
    """
    rows = []
    for kinase in spec_df.index:
        scores = spec_df.loc[kinase]
        # Sort descending by score; rank is 1-based.
        ranked = scores.sort_values(ascending=False)
        for rank, (celltype, score) in enumerate(ranked.iloc[:top_n].items(), start=1):
            rows.append({
                "kinase": kinase,
                "reference": reference,
                "celltype": celltype,
                "specificity_score": round(float(score), 6) if np.isfinite(score) else 0.0,
                "rank": rank,
            })
    return pd.DataFrame(rows, columns=["kinase", "reference", "celltype",
                                       "specificity_score", "rank"])


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_human_celltype_attribution(force: bool = False) -> pd.DataFrame:
    """Compute top-N specific cell types per kinase from human references.

    Reads seaad_kinase_specificity.csv and hbca_kinase_specificity.csv,
    ranks cell types within each kinase for each reference, and concatenates
    into a single long-form output at CELLTYPE_SPECIFICITY_FILE.

    Missing reference files are skipped with a warning (allows partial runs
    when only one reference is available).

    Returns the combined DataFrame.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Human Cell-Type Attribution: top-N specific cell types")
    print("=" * 60)

    if not force and os.path.exists(OUT_FILE):
        print(f"  Cached: {OUT_FILE} (use --force to recompute)")
        return pd.read_csv(OUT_FILE)

    all_parts: list[pd.DataFrame] = []

    for ref_label, spec_path in [
        ("seaad_mtg", SEAAD_SPEC_FILE),
        ("allen_hbca", HBCA_SPEC_FILE),
    ]:
        if not os.path.exists(spec_path):
            print(f"  WARNING: {ref_label} specificity file not found at {spec_path} — skipping")
            print(f"  Run: python alz/human_reference_expression.py --ref "
                  f"{'seaad' if ref_label == 'seaad_mtg' else 'hbca'}")
            continue

        print(f"\n  Processing {ref_label} ...")
        spec_df = pd.read_csv(spec_path, index_col=0)
        if ref_label == "allen_hbca":
            before = spec_df.shape[1]
            spec_df = _filter_hbca_cortex_hpc(spec_df)
            print(f"    HBCA cortex/hippocampus filter: {before} → {spec_df.shape[1]} cell types")
        spec_df, source_labels = _rollup_matrix_to_levy_t5(spec_df, ref_label)
        print(f"    Levy T5 rollup: {len(source_labels)} clusters")
        print(f"    Shape: {spec_df.shape} (kinases × Levy T5 clusters)")

        part = _top_n_for_reference(spec_df, ref_label, TOP_N)
        all_parts.append(part)
        print(f"    Top-{TOP_N} rows: {len(part)}")

    if not all_parts:
        raise RuntimeError(
            "No specificity files found for either reference. "
            "Run atlas_reference.py and human_reference_expression.py first."
        )

    df = pd.concat(all_parts, ignore_index=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved {len(df)} rows to {OUT_FILE}")

    # Summary: top cell type per kinase per reference.
    for ref in df["reference"].unique():
        sub = df[df["reference"] == ref]
        top1 = sub[sub["rank"] == 1].head(5)
        print(f"\n  [{ref}] Top-1 cell type for first 5 kinases:")
        for _, row in top1.iterrows():
            print(f"    {row['kinase']}: {row['celltype']} (score={row['specificity_score']:.3f})")

    return df


# ---------------------------------------------------------------------------
# Payload builder helper
# ---------------------------------------------------------------------------


def build_celltype_specificity_payload() -> dict | None:
    """Build the PAYLOAD.human.celltype_specificity block for the viewer.

    Mirrors the mouse Attribution sub-tab — emits the full ranked cell-type
    list per kinase with per-row absolute expression and (SEA-AD only)
    AD-vs-control LFC from the full-CPS effect_sizes.h5ad.

    Returns None if neither specificity file exists (phase-2 data absent).

    Schema (per reference):
      {
        "celltypes": [...],
        "by_kinase": { kinase_id → [score per celltype] },
        "ranked_by_kinase": { kinase_id → [
            {celltype, rank, score, mean_log2_expression,
             sea_ad_lfc (SEA-AD only) },
        ...   # full ranked list, all Levy T5 clusters mapped from the reference
        ]},
      }
    """
    ref_map = {
        "seaad_mtg": (SEAAD_SPEC_FILE, config.SEAAD_KINASE_EXPRESSION_FILE),
        "allen_hbca": (HBCA_SPEC_FILE, config.HBCA_KINASE_EXPRESSION_FILE),
    }

    available = {ref: (sp, expr) for ref, (sp, expr) in ref_map.items()
                 if os.path.exists(sp)}
    if not available:
        return None

    # SEA-AD per-supertype LFC from the full-CPS effect_sizes.h5ad. Mirrors the
    # mouse Attribution "SEA-AD LFC" column after rolling supertype values onto
    # the same Levy T5 cluster rows used for specificity. None if anndata or
    # the h5ad is unavailable (degrades to no LFC column on the SEA-AD tab).
    seaad_lfc = _load_seaad_full_lfc()

    payload: dict = {"references": list(available.keys())}
    if seaad_lfc is not None:
        payload["seaad_lfc_stratum"] = "full"  # CPS range used (App/Tau/ApTt analog)

    for ref, (spec_path, expr_path) in available.items():
        spec_df = pd.read_csv(spec_path, index_col=0)
        if ref == "allen_hbca":
            spec_df = _filter_hbca_cortex_hpc(spec_df)
        expr_df = (
            pd.read_csv(expr_path, index_col=0)
            if os.path.exists(expr_path) else None
        )
        if expr_df is not None:
            if ref == "allen_hbca":
                expr_df = _filter_hbca_cortex_hpc(expr_df)
            spec_df, source_labels = _rollup_matrix_to_levy_t5(spec_df, ref)
            expr_df, _ = _rollup_matrix_to_levy_t5(expr_df, ref)
            expr_df = expr_df.reindex(columns=spec_df.columns)
        else:
            spec_df, source_labels = _rollup_matrix_to_levy_t5(spec_df, ref)
        celltypes = list(spec_df.columns)
        ref_lfc = (
            _rollup_seaad_lfc_to_levy_t5(seaad_lfc, list(spec_df.index), source_labels)
            if ref == "seaad_mtg" else None
        )

        by_kinase: dict[str, list] = {}
        for kinase in spec_df.index:
            scores = spec_df.loc[kinase].tolist()
            by_kinase[kinase] = [
                round(float(v), 4) if np.isfinite(v) else 0.0
                for v in scores
            ]

        ranked_by_kinase: dict[str, list[dict]] = {}
        for kinase in spec_df.index:
            scores = spec_df.loc[kinase]
            ranked = scores.sort_values(ascending=False)
            rows = []
            for i, (ct, sc) in enumerate(ranked.items()):
                row = {
                    "celltype": ct,
                    "rank": i + 1,
                    "score": round(float(sc), 4) if np.isfinite(sc) else 0.0,
                }
                if expr_df is not None and kinase in expr_df.index and ct in expr_df.columns:
                    val = expr_df.at[kinase, ct]
                    row["mean_log2_expression"] = (
                        round(float(val), 4) if np.isfinite(val) else None
                    )
                if ref == "seaad_mtg" and seaad_lfc is not None:
                    lfc = ref_lfc.get((kinase, ct)) if ref_lfc is not None else None
                    row["sea_ad_lfc"] = (
                        round(float(lfc), 4) if lfc is not None and np.isfinite(lfc) else None
                    )
                if ct in source_labels:
                    row["source_celltypes"] = source_labels[ct]
                rows.append(row)
            ranked_by_kinase[kinase] = rows

        payload[ref] = {
            "celltypes": celltypes,
            "by_kinase": by_kinase,
            "ranked_by_kinase": ranked_by_kinase,
        }

    return payload


def _load_seaad_full_lfc() -> dict[tuple[str, str], float] | None:
    """Load SEA-AD full-CPS effect sizes as a (gene_upper, supertype) → LFC dict.

    Returns None on ImportError or missing h5ad. Genes are upper-cased to match
    the kinase_id index used by the specificity matrix.
    """
    path = config.SEA_AD_EFFECT_SIZES.get("full")
    if not path or not os.path.exists(path):
        return None
    try:
        import anndata as ad
    except ImportError:
        return None
    adata = ad.read_h5ad(path)
    genes_upper = [g.upper() for g in adata.obs_names]
    supertypes = list(adata.var_names)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    out: dict[tuple[str, str], float] = {}
    for gi, g in enumerate(genes_upper):
        row = X[gi, :]
        for si, st in enumerate(supertypes):
            v = row[si]
            if np.isfinite(v):
                out[(g, st)] = float(v)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary() -> None:
    """Print cached attribution results."""
    if not os.path.exists(OUT_FILE):
        print(f"  No cached results at {OUT_FILE}")
        return
    df = pd.read_csv(OUT_FILE)
    print(f"  Celltype specificity: {len(df)} rows")
    for ref in df["reference"].unique():
        sub = df[df["reference"] == ref]
        n_kinases = sub["kinase"].nunique()
        n_ct = sub["celltype"].nunique()
        print(f"    [{ref}] {n_kinases} kinases, {n_ct} cell types")
        # Show top 3 rows for the first kinase.
        first_kinase = sub["kinase"].iloc[0]
        top3 = sub[sub["kinase"] == first_kinase].head(3)
        for _, row in top3.iterrows():
            print(f"      rank={row['rank']} {row['celltype']} "
                  f"(score={row['specificity_score']:.3f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Human Cell-Type Attribution: top-N specific cell types per kinase",
    )
    parser.add_argument("--summary", action="store_true",
                        help="Print cached results")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if cached results exist")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    else:
        compute_human_celltype_attribution(force=args.force)


if __name__ == "__main__":
    main()
