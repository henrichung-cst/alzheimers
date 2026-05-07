"""CTM-native proportional decomposition on the WMB-class spine.

Aggregates raw counts directly from the Song h5ad on Allen Cell Type Mapper
``class_name``, applies the proportional formula per group, and writes
per-track CSVs with ``{group}_{wmb_class}`` value columns.

Formula (per gene, per group, per WMB class)::

    deconv[gene, group, w] =
        bulk_median[gene, group]
      · (rna[gene, group, w] / Σ_w' rna[gene, group, w'])
      · size_factor[group, w]

with ``size_factor[group, w] = Σ_w' n_cells[group, w'] / n_cells[group, w]``.

Edge cases:
- Genes with zero raw count in any (group, w) cell: replace with
  ``min_nonzero / 10000`` before the share normalization, so shares default
  to uniform rather than NaN.
- Bulk values that are NaN propagate to NaN in the output.
- Genes absent from the snRNA pseudobulk are skipped (no cell-type prior).

Outputs go to ``outputs/reports/deconvolution/wmb_decomposition/``.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

# Repo root is two dirs above this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

WMB_MANIFEST_FILE = config.WMB_CLASS_MANIFEST_FILE
H5AD_FILE = config.SONG_H5AD_FILE

# Inputs were removed from disk on 2026-05-07; re-pull via `pixi run
# ingest-gdrive-shared` and copy into BULK_DIR before running this branch.
BULK_DIR = os.path.join(
    config.REPO_ROOT, "data", "incytr_collections", "song", "proteomics", "source",
)
SAMPLE_KEY_FILE = os.path.join(BULK_DIR, "yuyu_samplekey.csv")
PS_BULK_FILE = os.path.join(BULK_DIR, "imac_median.csv")
PY_BULK_FILE = os.path.join(BULK_DIR, "py_median.csv")
PR_BULK_FILE = os.path.join(BULK_DIR, "pr_median.csv")

OUTPUT_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "deconvolution", "wmb_decomposition",
)
PS_OUT_FILE = os.path.join(OUTPUT_DIR, "ps_wmb_decomposition.csv")
PY_OUT_FILE = os.path.join(OUTPUT_DIR, "py_wmb_decomposition.csv")
PR_OUT_FILE = os.path.join(OUTPUT_DIR, "pr_wmb_decomposition.csv")
SIZE_OUT_FILE = os.path.join(OUTPUT_DIR, "wmb_class_size.csv")
AUDIT_FILE = os.path.join(OUTPUT_DIR, "decomposition_audit.json")

PS_META_COLS = [
    "site_id", "protein_id", "gene_symbol", "prot_description",
    "site_position", "motif",
]
PY_META_COLS = [
    "protein_id", "gene_symbol", "prot_description", "site_position",
    "motif", "gene_id",
]
PR_META_COLS = [
    "protein_id", "gene_symbol", "geneID",
]


def _build_class_name_to_label() -> dict:
    m = pd.read_csv(WMB_MANIFEST_FILE)
    return dict(zip(m["class_name"], m["class_label"]))


def _read_bulk(path: str, gene_col: str, meta_cols: list[str]
               ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Return (meta_df, ms_id_columns, value_df) for a Yuyu bulk median CSV.

    The first column is an unnamed row index; drop it. Group columns are the
    24 MS_ID-style names like ``M_2mo_WT`` … ``F_6mo_T22/APP``.
    """
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed:")]

    if gene_col not in df.columns:
        raise KeyError(f"{path}: missing column {gene_col!r}; "
                       f"saw {list(df.columns)[:8]}")

    rename = {gene_col: "gene_symbol"} if gene_col != "gene_symbol" else {}
    df = df.rename(columns=rename)

    keep_meta = [c if rename.get(c, c) != "gene_symbol" else "gene_symbol"
                 for c in meta_cols]
    ms_cols = [c for c in df.columns if c not in keep_meta]
    if len(ms_cols) != 24:
        raise ValueError(
            f"{path}: expected 24 group columns, got {len(ms_cols)}: {ms_cols}"
        )

    meta = df[keep_meta].copy()
    values = df[ms_cols].apply(pd.to_numeric, errors="coerce")
    return meta, ms_cols, values


def _load_sample_key() -> dict:
    """Return MS_ID → SCRNA group (e.g. 'M_2mo_APP' → 'ma_2mo_AppP')."""
    sk = pd.read_csv(SAMPLE_KEY_FILE)
    return dict(zip(sk["MS_ID"], sk["Group"]))


def aggregate_h5ad_by_group_class(genes_needed: Iterable[str]
                                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Aggregate raw counts per (group, wmb_class, gene) from the Song h5ad.

    Returns:
        counts: DataFrame (group × wmb_class) MultiIndex rows, gene columns,
                raw count sums.
        n_cells: DataFrame (group × wmb_class) MultiIndex rows, single 'n_cells'
                 column.
        groups: ordered list of unique group labels (e.g. 'ma_2mo_WTyp').
    """
    import anndata as ad

    print(f"  Loading h5ad: {H5AD_FILE}")
    adata = ad.read_h5ad(H5AD_FILE)
    print(f"    {adata.shape[0]:,} nuclei × {adata.shape[1]:,} genes")

    cn_to_label = _build_class_name_to_label()
    mask_prob = adata.obs["class_prob"] >= config.SONG_MIN_SUBCLASS_PROB
    mask_mapped = adata.obs["class_name"].isin(cn_to_label.keys())
    mask = (mask_prob & mask_mapped).values
    obs = adata.obs[mask].copy()
    obs["wmb_class"] = obs["class_name"].map(cn_to_label)
    obs["group"] = obs["sample"].str.split("_", n=1).str[1]

    all_genes = adata.var_names.to_numpy()
    requested = pd.Index(list(genes_needed)).unique()
    gene_keep_mask = np.isin(all_genes, requested)
    if gene_keep_mask.sum() == 0:
        raise RuntimeError("No requested genes found in h5ad var_names.")
    keep_genes = all_genes[gene_keep_mask]
    print(f"    Keeping {len(keep_genes):,} of {len(all_genes):,} genes "
          f"that appear in bulk")

    X = adata.X[mask][:, gene_keep_mask].tocsr()
    del adata

    pairs = (
        obs.groupby(["group", "wmb_class"], observed=True)
        .indices  # dict {(group, class): np.array(row_idx)}
    )

    rows_meta = []
    rows_counts = []
    n_cells_rows = []
    for (group, wclass), idx in pairs.items():
        raw_sum = np.asarray(X[idx].sum(axis=0)).flatten()
        rows_meta.append((group, wclass))
        rows_counts.append(raw_sum)
        n_cells_rows.append({"group": group, "wmb_class": wclass,
                             "n_cells": int(len(idx))})

    counts = pd.DataFrame(np.vstack(rows_counts), columns=keep_genes)
    counts.index = pd.MultiIndex.from_tuples(rows_meta, names=["group", "wmb_class"])
    n_cells = pd.DataFrame(n_cells_rows).set_index(["group", "wmb_class"])

    groups = sorted(obs["group"].unique().tolist())
    print(f"    Aggregated: {len(counts)} (group, wmb_class) pairs, "
          f"{len(groups)} groups")
    return counts, n_cells, groups


def _decompose_track(meta: pd.DataFrame, ms_cols: list[str],
                     bulk_values: pd.DataFrame, ms_to_group: dict,
                     counts: pd.DataFrame, n_cells: pd.DataFrame,
                     groups: list[str]) -> pd.DataFrame:
    """Apply Yuyu's formula per (gene, group, class). Returns wide DataFrame
    with meta columns + ``{group}_{wmb_class}`` value columns.

    Rows whose ``gene_symbol`` is not in ``counts.columns`` are dropped — we
    cannot redistribute a parent without a cell-type prior. Genes with all-zero
    counts in a group fall back to a small floor (Yuyu's behavior).
    """
    snrna_genes = set(counts.columns)
    keep = meta["gene_symbol"].isin(snrna_genes)
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"    Dropping {n_dropped:,} of {len(meta):,} bulk rows "
              f"(gene not in snRNA pseudobulk)")
    meta = meta.loc[keep].reset_index(drop=True)
    bulk_values = bulk_values.loc[keep].reset_index(drop=True)

    wmb_classes = sorted({c for _, c in counts.index})
    n_sites = len(meta)
    n_groups = len(groups)
    n_classes = len(wmb_classes)

    # Pre-pivot snRNA counts into a (group × wmb_class × gene) tensor aligned to
    # the bulk rows' gene order; absent (group, wmb_class) cells get zeros.
    bulk_genes = meta["gene_symbol"].to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(counts.columns)}
    gene_idx = np.array([gene_to_idx[g] for g in bulk_genes])
    counts_aligned = counts.to_numpy(dtype=float)[:, gene_idx]  # (n_pairs, n_sites)
    pair_idx = {pair: i for i, pair in enumerate(counts.index)}

    # rna_tensor[g, w, s]
    rna_tensor = np.zeros((n_groups, n_classes, n_sites), dtype=float)
    n_cells_arr = np.zeros((n_groups, n_classes), dtype=float)
    for gi, group in enumerate(groups):
        for wi, wclass in enumerate(wmb_classes):
            i = pair_idx.get((group, wclass))
            if i is None:
                continue
            rna_tensor[gi, wi, :] = counts_aligned[i]
            n_cells_arr[gi, wi] = n_cells.loc[(group, wclass), "n_cells"]

    # Floor: per (group, gene), if the across-class total is zero, replace with
    # a tiny per-class floor so shares are uniform (matches Yuyu's behavior).
    totals = rna_tensor.sum(axis=1)  # (n_groups, n_sites)
    nonzero = rna_tensor[rna_tensor > 0]
    floor = (nonzero.min() / 10000.0) if nonzero.size else 1e-12
    zero_total = (totals == 0)  # (n_groups, n_sites)
    if zero_total.any():
        broadcast_mask = np.broadcast_to(
            zero_total[:, np.newaxis, :], rna_tensor.shape
        )
        rna_tensor = np.where(broadcast_mask, floor, rna_tensor)
        totals = rna_tensor.sum(axis=1)

    # Avoid div-by-zero per (group, site)
    safe_totals = np.where(totals > 0, totals, np.nan)
    shares = rna_tensor / safe_totals[:, np.newaxis, :]  # (g, w, s)

    # size_factor[group, w] = Σ_w' n_cells / n_cells[group, w]
    cells_total = n_cells_arr.sum(axis=1, keepdims=True)
    size_factor = np.where(
        n_cells_arr > 0,
        cells_total / np.where(n_cells_arr > 0, n_cells_arr, 1.0),
        0.0,
    )

    out = meta.copy()
    group_to_gi = {g: i for i, g in enumerate(groups)}
    for ms_id in ms_cols:
        group = ms_to_group.get(ms_id)
        if group is None:
            raise KeyError(f"MS_ID {ms_id!r} missing from yuyu_samplekey.csv")
        gi = group_to_gi.get(group)
        if gi is None:
            print(f"    WARN: group {group!r} (from {ms_id!r}) absent from "
                  f"snRNA aggregation; skipping that column")
            continue
        bulk_col = bulk_values[ms_id].to_numpy(dtype=float)  # (n_sites,)
        for wi, wclass in enumerate(wmb_classes):
            share_row = shares[gi, wi, :]
            sf = size_factor[gi, wi]
            out[f"{group}_{wclass}"] = bulk_col * share_row * sf

    return out


def write_size_table(n_cells: pd.DataFrame, groups: list[str],
                     wmb_classes: list[str]) -> None:
    """Replacement for `yuyu_clustersize.csv`: WMB classes × groups counts."""
    df = (
        n_cells.reset_index()
        .pivot(index="wmb_class", columns="group", values="n_cells")
        .reindex(index=wmb_classes, columns=groups)
        .fillna(0)
        .astype(int)
    )
    df.index.name = "wmb_class"
    df.to_csv(SIZE_OUT_FILE)
    print(f"  Wrote size table: {df.shape} → {SIZE_OUT_FILE}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover the gene universe needed by any bulk track
    print("Reading bulk medians ...")
    ps_meta, ps_cols, ps_vals = _read_bulk(PS_BULK_FILE, "gene_symbol", PS_META_COLS)
    py_meta, py_cols, py_vals = _read_bulk(PY_BULK_FILE, "gene_symbol", PY_META_COLS)
    pr_meta, pr_cols, pr_vals = _read_bulk(PR_BULK_FILE, "Gene Symbol", PR_META_COLS)
    # pr_median.csv has a few cust|... rows where Gene Symbol is "0.0";
    # drop those — they aren't real gene symbols and have no snRNA prior.
    pr_keep = pr_meta["gene_symbol"].astype(str).str.match(r"^[A-Za-z][\w\-\.]*$")
    pr_meta = pr_meta.loc[pr_keep].reset_index(drop=True)
    pr_vals = pr_vals.loc[pr_keep].reset_index(drop=True)

    if ps_cols != py_cols or ps_cols != pr_cols:
        raise ValueError("MS_ID columns disagree across bulk tracks")
    ms_cols = ps_cols

    ms_to_group = _load_sample_key()
    missing = [c for c in ms_cols if c not in ms_to_group]
    if missing:
        raise KeyError(f"yuyu_samplekey.csv missing MS_IDs: {missing}")

    all_genes = pd.Index(
        list(ps_meta["gene_symbol"]) + list(py_meta["gene_symbol"]) + list(pr_meta["gene_symbol"])
    ).dropna().unique()

    counts, n_cells, groups = aggregate_h5ad_by_group_class(all_genes)
    wmb_classes = sorted({c for _, c in counts.index})

    print(f"\nDecomposing ser/thr (imac_median) → {PS_OUT_FILE}")
    ps_out = _decompose_track(ps_meta, ms_cols, ps_vals, ms_to_group,
                              counts, n_cells, groups)
    ps_out.to_csv(PS_OUT_FILE, index=False)
    print(f"  {ps_out.shape}")

    print(f"\nDecomposing tyr (py_median) → {PY_OUT_FILE}")
    py_out = _decompose_track(py_meta, ms_cols, py_vals, ms_to_group,
                              counts, n_cells, groups)
    py_out.to_csv(PY_OUT_FILE, index=False)
    print(f"  {py_out.shape}")

    print(f"\nDecomposing total proteome (pr_median) → {PR_OUT_FILE}")
    pr_out = _decompose_track(pr_meta, ms_cols, pr_vals, ms_to_group,
                              counts, n_cells, groups)
    pr_out.to_csv(PR_OUT_FILE, index=False)
    print(f"  {pr_out.shape}")

    write_size_table(n_cells, groups, wmb_classes)

    # Mass-conservation invariant: after the size-factor step, Yuyu's formula
    # is *not* directly mass-conserving — mass conservation is restored in
    # `compute_site_fractions` (which re-normalizes by per-(sample, site) sum
    # of attributed = decomp * proportion[class, sample]). Verify the
    # equivalent invariant here: Σ_w (decomp[w] * (n_cells[w] / Σ n_cells)) ≈
    # bulk_median  (since size_factor[w] * proportion[w] = 1).
    print("\nMass-conservation audit on ST track (post-proportion):")
    val_cols = [c for c in ps_out.columns if c not in PS_META_COLS]
    sample_to_cols: dict[str, list[tuple[str, str]]] = {}
    for c in val_cols:
        parts = c.split("_", 3)
        if len(parts) < 4:
            continue
        sample = "_".join(parts[:3])
        sample_to_cols.setdefault(sample, []).append((c, parts[3]))
    n_cells_long = n_cells.reset_index()
    diffs = []
    bulk_finite_mask = ps_meta["gene_symbol"].isin(counts.columns).to_numpy()
    for ms_id, group in ms_to_group.items():
        pairs = sample_to_cols.get(group, [])
        if not pairs:
            continue
        nc_g = n_cells_long[n_cells_long["group"] == group].set_index("wmb_class")["n_cells"]
        nc_total = nc_g.sum()
        if nc_total == 0:
            continue
        attributed = np.zeros(len(ps_out))
        for col_name, wclass in pairs:
            prop = nc_g.get(wclass, 0) / nc_total
            attributed = attributed + ps_out[col_name].fillna(0).to_numpy() * prop
        bulk_arr = ps_vals[ms_id].to_numpy(dtype=float)[bulk_finite_mask]
        diffs.extend(np.abs(attributed - np.where(np.isnan(bulk_arr), 0, bulk_arr)))
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size:
        print(f"  Per-(group, site) |Σ_w decomp·prop - bulk|: "
              f"max={diffs.max():.3g}  mean={diffs.mean():.3g}  "
              f"frac<1e-3={float((diffs < 1e-3).mean()):.3f}")
    print(f"\n  WMB classes covered: {len(wmb_classes)} / 34")
    missing_classes = [c for c in config.WMB_CLASSES if c not in wmb_classes]
    if missing_classes:
        print(f"  Absent (biological sampling gap): {missing_classes}")


if __name__ == "__main__":
    main()
