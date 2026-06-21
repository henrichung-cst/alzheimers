"""Shared evidence loaders for cross-reference cell-type attribution.

Functions here build the SEA-AD / WMB / Song evidence tables that
`alz/bulk_mea/attribute.py` Stage 3 merges into the unified attribution.
They were extracted from `attribute.py` (task #16 of the repo organization
plan) so the same loaders can be reused by future cross-reference
consumers (viewer-side payload builders, sensitivity analyses) without
re-importing the entire Stage-3 module.

Direction conventions match the rest of the pipeline (+ = up in disease).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config


def compute_sea_ad_concordance(sig, cluster_to_seaad, sea_ad_paths):
    """Compute per-(kinase, contrast, cluster) SEA-AD LFCs + supertype audit.

    Direct cluster → SEA-AD supertype merge (no WMB hop). ``cluster_to_seaad``
    maps each spine cluster_name to a list of (supertype, weight) tuples;
    weights within a cluster sum to 1.0. Clusters mapped to ``n/a`` (empty
    list) get no SEA-AD evidence row.

    Many-to-many collapse: per (kinase, contrast, cluster), LFC is the
    weighted mean of supertype LFCs. Audit columns:
      - ``sea_ad_n_supertypes`` — number of finite supertype LFCs collapsed
      - ``sea_ad_direction_agreement`` — share of supertype LFCs with the same
        sign as the weighted mean (1.0 = all agree; 0.5 = perfectly mixed)

    Loads h5ads inside the function (anndata is an optional dep).
    Returns ``(sea_ad_df, supertype_df)``. On ImportError or missing h5ads,
    prints a warning and returns empty DataFrames so attribution can still
    degrade gracefully (Song/WMB-only tiers).
    """
    sea_ad_rows = []
    supertype_rows = []
    _supertype_emitted = set()

    try:
        import anndata as ad

        contrast_to_stratum = {}
        for contrast in sig["contrast"].unique():
            pathway = contrast.split("_")[0]
            if pathway not in config.SEA_AD_PATHWAY_MAP:
                raise ValueError(
                    f"Unknown pathway prefix '{pathway}' in contrast "
                    f"'{contrast}'. Expected one of "
                    f"{list(config.SEA_AD_PATHWAY_MAP)}")
            contrast_to_stratum[contrast] = config.SEA_AD_PATHWAY_MAP[pathway]

        needed_strata = set(contrast_to_stratum.values())
        adata_by_stratum = {}
        for stratum in needed_strata:
            path = sea_ad_paths[stratum]
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            adata_by_stratum[stratum] = ad.read_h5ad(path)
        strata_label = ", ".join(sorted(needed_strata))
        print(f"  Loading SEA-AD effect sizes ({strata_label})...")

        ref_adata = next(iter(adata_by_stratum.values()))
        sea_ad_genes_upper = {g.upper(): g for g in ref_adata.obs_names}
        supertypes = list(ref_adata.var_names)
        st_to_subclass = dict(zip(ref_adata.var_names, ref_adata.var["Subclass"]))
        st_to_idx = {st: i for i, st in enumerate(supertypes)}
        gene_to_idx = {g: i for i, g in enumerate(ref_adata.obs_names)}

        # Drop supertype refs that don't exist in the h5ad (defensive — bridge
        # is curated against SEA-AD var_names but log any drift).
        cluster_to_seaad_clean = {}
        missing_supertypes = set()
        for cluster, entries in cluster_to_seaad.items():
            kept = [(st, w) for st, w in entries if st in st_to_idx]
            for st, _w in entries:
                if st not in st_to_idx:
                    missing_supertypes.add(st)
            cluster_to_seaad_clean[cluster] = kept
        n_mapped = sum(1 for v in cluster_to_seaad_clean.values() if v)
        print(f"  SEA-AD bridge: {n_mapped}/{len(cluster_to_seaad_clean)} "
              f"clusters mapped (>=1 supertype)")
        if missing_supertypes:
            print(f"  WARNING: {len(missing_supertypes)} bridge supertypes "
                  f"not in SEA-AD h5ad: "
                  f"{sorted(missing_supertypes)[:5]}...")

        def _cluster_lfcs(adata, gene_idx):
            """Weighted-mean LFC per cluster + direction-agreement audit."""
            effects = adata.X[gene_idx, :]
            if hasattr(effects, "toarray"):
                effects = effects.toarray().flatten()
            else:
                effects = np.asarray(effects).flatten()
            out = {}
            for cluster, entries in cluster_to_seaad_clean.items():
                if not entries:
                    continue
                vals, weights = [], []
                for st, w in entries:
                    val = effects[st_to_idx[st]]
                    if np.isfinite(val):
                        vals.append(float(val))
                        weights.append(float(w))
                if not vals:
                    continue
                arr = np.asarray(vals)
                wts = np.asarray(weights)
                wts = wts / wts.sum() if wts.sum() > 0 else np.full_like(wts, 1.0 / len(wts))
                wmean = float((arr * wts).sum())
                if wmean == 0.0:
                    agree = 1.0
                else:
                    same_sign = (np.sign(arr) == np.sign(wmean))
                    agree = float(wts[same_sign].sum())
                out[cluster] = (wmean, len(vals), agree)
            return out

        _lfc_cache = {}

        for _, row in sig.iterrows():
            kinase = row["kinase"]
            contrast = row["contrast"]
            nes = row["NES"]
            fdr = row["FDR"]
            gene = row["gene_symbol"]
            gene_upper = gene.upper() if isinstance(gene, str) else ""

            if gene_upper not in sea_ad_genes_upper:
                continue

            gene_idx = gene_to_idx[sea_ad_genes_upper[gene_upper]]
            stratum = contrast_to_stratum[contrast]

            cache_key = (stratum, gene_idx)
            if cache_key not in _lfc_cache:
                _lfc_cache[cache_key] = _cluster_lfcs(
                    adata_by_stratum[stratum], gene_idx)
            cluster_lfcs = _lfc_cache[cache_key]

            if cache_key not in _supertype_emitted:
                _supertype_emitted.add(cache_key)
                effects = adata_by_stratum[stratum].X[gene_idx, :]
                if hasattr(effects, "toarray"):
                    effects = effects.toarray().flatten()
                else:
                    effects = np.asarray(effects).flatten()
                for i, st in enumerate(supertypes):
                    val = effects[i]
                    if not np.isfinite(val):
                        continue
                    supertype_rows.append({
                        "gene_symbol": gene,
                        "stratum": stratum,
                        "supertype": st,
                        "subclass": st_to_subclass[st],
                        "supertype_lfc": float(val),
                    })

            residue_type = row.get("residue_type", "ST")
            track_name = row.get("track", "st")
            for cluster, (wmean_lfc, n_st, agree) in cluster_lfcs.items():
                concordance = np.sign(nes) * wmean_lfc
                sea_ad_rows.append({
                    "kinase": kinase,
                    "gene_symbol": gene,
                    "contrast": contrast,
                    "NES": nes,
                    "FDR": fdr,
                    "residue_type": residue_type,
                    "track": track_name,
                    "cell_type": cluster,
                    "sea_ad_lfc": wmean_lfc,
                    "sea_ad_n_supertypes": n_st,
                    "sea_ad_direction_agreement": agree,
                    "sea_ad_stratum": stratum,
                    "concordance_score": concordance,
                })

        print(f"  SEA-AD concordance: {len(sea_ad_rows)} "
              f"(kinase, contrast, cluster) rows")

    except (ImportError, FileNotFoundError) as e:
        print("  " + "!" * 70)
        print(f"  WARNING: SEA-AD not available ({type(e).__name__}: {e})")
        print("  Skipping human-concordance evidence; confidence tiers will")
        print("  be downgraded for kinases lacking Song/WMB support.")
        print("  " + "!" * 70)

    sea_ad_df = pd.DataFrame(sea_ad_rows)
    supertype_df = pd.DataFrame(supertype_rows)
    return sea_ad_df, supertype_df


def prepare_wmb_specificity(wmb_df):
    """Reduce WMB expression export to top-(gene, wmb_class) rows for merging.

    The WMB CSV is keyed on ``cell_type`` = WMB class (retained subset, ~9
    classes). Spine clusters look up their parent class via the
    ``cluster_to_wmb_class`` crosswalk in ``_assemble_unified``.
    """
    if wmb_df is None or len(wmb_df) == 0:
        return None
    wmb = wmb_df.copy()
    wmb["_gene_upper"] = wmb["gene_symbol"].str.upper()
    # Standard attribution metric (detection-gated): per-(gene, class) detection
    # + detected-set concentration / tier, replacing the share `specificity_score`.
    wmb_top = (wmb.sort_values("fraction_cells_expressing")
                  .drop_duplicates(["_gene_upper", "cell_type"], keep="last")
                  [["_gene_upper", "cell_type", "detected", "concentration",
                    "concentration_tier", "mean_log2_expression",
                    "fraction_cells_expressing", "binary_expressed"]]
                  .rename(columns={
                      "cell_type": "wmb_class",
                      "detected": "wmb_detected",
                      "concentration": "wmb_concentration",
                      "concentration_tier": "wmb_concentration_tier",
                      "mean_log2_expression": "wmb_mean_log2_expression",
                      "fraction_cells_expressing": "wmb_fraction_cells_expressing",
                      "binary_expressed": "wmb_binary_expressed",
                  }))
    print(f"  WMB detection: {len(wmb_top)} (gene, wmb_class) pairs loaded")
    return wmb_top


def prepare_song_specificity(song_sp_df):
    if song_sp_df is None or len(song_sp_df) == 0:
        return None
    song_sp = song_sp_df.copy()
    song_sp["_gene_upper"] = song_sp["gene_symbol"].str.upper()
    # Standard attribution metric (detection-gated): per-(gene, cluster)
    # detection + detected-set concentration / tier, plus the per-gene breadth
    # summary (effective number of cell types, top cell type). Replaces the
    # share `specificity_score` and the rejected `tau`.
    cols = ["_gene_upper", "cell_type", "detected", "concentration",
            "concentration_tier", "fraction_cells_expressing"]
    rename = {"detected": "song_detected",
              "concentration": "song_concentration",
              "concentration_tier": "song_concentration_tier",
              "fraction_cells_expressing": "song_fraction_cells_expressing"}
    for src, dst in (("effective_n_celltypes", "song_effective_n"),
                     ("top_celltype", "song_top_celltype"),
                     ("top_concentration", "song_top_concentration")):
        if src in song_sp.columns:
            cols.append(src)
            rename[src] = dst
    song_spec_top = (song_sp.sort_values("fraction_cells_expressing")
                            .drop_duplicates(["_gene_upper", "cell_type"], keep="last")
                            [cols]
                            .rename(columns=rename))
    print(f"  Song detection: {len(song_spec_top)} (gene, cell_type) pairs loaded")
    return song_spec_top


def prepare_song_concordance(song_cd_df):
    """Returns (song_cd_top, key_is_contrast) or (None, False) if absent."""
    if song_cd_df is None or len(song_cd_df) == 0:
        return None, False
    contrast_col = "contrast" if "contrast" in song_cd_df.columns else "pathway"
    key_is_contrast = (contrast_col == "contrast")
    keep_cols = ["gene_symbol", "cell_type", contrast_col, "song_lfc"]
    for opt in ("song_pval", "song_fdr"):
        if opt in song_cd_df.columns:
            keep_cols.append(opt)
    song_cd_top = song_cd_df[keep_cols].copy()
    song_cd_top["_gene_upper"] = song_cd_top["gene_symbol"].str.upper()
    song_cd_top = song_cd_top.rename(columns={contrast_col: "_song_contrast"})
    song_cd_top = song_cd_top.drop(columns=["gene_symbol"])
    print(f"  Song concordance: {len(song_cd_top)} (gene, cell_type, "
          f"{contrast_col}) entries loaded")
    return song_cd_top, key_is_contrast
