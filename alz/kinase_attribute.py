"""Stage 3 of kinase attribution: unified cell-type attribution.

Analysis spine is the Levy-t5 31-cluster spine (`config.CLUSTER_SPINE`),
not WMB-34. Evidence sources merge via single-hop crosswalks:
  - SEA-AD per-supertype effect sizes (pathway-matched: App→early CPS,
    Tau→late CPS, ApTt→full CPS), joined direct cluster → SEA-AD supertype(s)
    via `cluster_to_seaad_supertype.csv` (many-to-many; weighted mean LFC).
  - WMB expression specificity, joined cluster → parent WMB class via
    `cluster_to_wmb_class.csv` (1:1, lineage-level; clusters sharing a parent
    inherit identical WMB scores).
  - Song within-cohort transcriptomic concordance + specificity, keyed
    directly on cluster_name (identity join).

Inputs:
  outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv
  data/external/sea_ad/effect_sizes{,_early,_late}.h5ad
  outputs/reports/wmb_expression/wmb_kinase_expression.csv
  outputs/reports/snrna_integration/song_{specificity,concordance}.csv (optional)
  data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv
  config.CLUSTER_TO_WMB_CLASS_FILE
  config.CLUSTER_TO_SEAAD_SUPERTYPE_FILE

Outputs (under outputs/reports/kinase_attribution/):
  unified_attribution.csv         — confidence != "none" rows (sorted)
  unified_attribution_full.csv    — full sig × cluster grid (n_kinases × 9 × 31)
  sea_ad_supertype_lfc.csv        — per-(gene, stratum, supertype) audit
  attribution_summary.json
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_confidence_and_basis_vectorized(unified):
    """Vectorized confidence + evidence-basis assignment for the unified table.

    Non-MEA-significant rows short-circuit to ('none', 'weak') in the same pass.
    """
    eligible = unified["mea_significant"].astype(bool) & (unified["effective_concordance"] > 0)
    wmb = unified["wmb_specificity"]
    sea_lfc = unified["sea_ad_lfc"].fillna(0.0)
    song_lfc_col = (unified["song_lfc"].fillna(0.0)
                    if "song_lfc" in unified.columns
                    else pd.Series(0.0, index=unified.index))
    song_contributed = unified["concordance_source"].isin(("song", "both"))

    has_wmb = wmb >= config.SPECIFICITY_LOW
    has_wmb_high = wmb >= config.SPECIFICITY_HIGH
    has_sea_ad = sea_lfc.abs() > config.SEA_AD_LFC_MIN
    has_song = song_lfc_col.abs() > config.SONG_LFC_MIN

    # Confidence tiers, priority high → moderate → low (np.select picks first match)
    high_mask = eligible & song_contributed & has_wmb_high & (has_song | has_sea_ad)
    mod_song_mask = eligible & song_contributed & (has_wmb | has_sea_ad | has_song)
    mod_sea_ad_mask = eligible & ~song_contributed & (has_wmb | has_sea_ad)
    low_mask = eligible & ~song_contributed & ~(has_wmb | has_sea_ad)
    conf = pd.Series(
        np.select(
            [high_mask, mod_song_mask, mod_sea_ad_mask, low_mask],
            ["high", "moderate", "moderate", "low"],
            default="none",
        ),
        index=unified.index,
    )

    basis = pd.Series(
        np.select(
            [
                eligible & has_wmb & has_sea_ad & has_song,
                eligible & has_wmb & has_song & ~has_sea_ad,
                eligible & has_wmb & has_sea_ad & ~has_song,
                eligible & has_wmb & ~has_sea_ad & ~has_song,
                eligible & has_song & ~has_wmb & ~has_sea_ad,
                eligible & has_sea_ad & ~has_wmb & ~has_song,
            ],
            [
                "three_way",
                "within_cohort",
                "cross_species",
                "mouse_expression_only",
                "song_only",
                "human_concordance_only",
            ],
            default="weak",
        ),
        index=unified.index,
    )
    return conf, basis


def _combine_mea_tracks(mea_by_track):
    """Concatenate per-track MEA results, injecting residue_type/track if absent.

    `mea_by_track` is an iterable of (track_cfg, df) pairs. Returns the combined
    `sig` DataFrame (no FDR filter — attribution gates on FDR later via the
    `mea_significant` column on the cross-joined grid).
    """
    mea_frames = []
    for track_cfg, df in mea_by_track:
        if df is None:
            continue
        df = df.copy()
        if "residue_type" not in df.columns:
            df["residue_type"] = track_cfg["residue"]
        if "track" not in df.columns:
            df["track"] = track_cfg["name"]
        mea_frames.append(df)
        print(f"  Track {track_cfg['name']}: loaded {len(df)} rows")
    if not mea_frames:
        raise FileNotFoundError(
            "No MEA outputs available; run the enrich pipeline first."
        )
    mea = pd.concat(mea_frames, ignore_index=True, sort=False)
    n_sig = (mea["FDR"] < config.MEA_FDR_THRESH).sum()
    print(f"  MEA results (combined): {len(mea)} total, {n_sig} significant "
          f"(FDR<{config.MEA_FDR_THRESH}) — building attribution from full grid")
    if "residue_type" in mea.columns:
        residue_breakdown = mea["residue_type"].value_counts().to_dict()
        print(f"  By residue_type: {residue_breakdown}")
    return mea


def _map_kinases_to_genes(sig, k2g_df):
    """Inject gene_symbol on `sig` from the kinase→gene mapping cache."""
    sig = sig.copy()
    kinase_to_gene = dict(zip(k2g_df["kinase_abbreviation"], k2g_df["gene_symbol"]))
    sig["gene_symbol"] = sig["kinase"].map(lambda k: kinase_to_gene.get(k, k))
    return sig


def _compute_sea_ad_concordance(sig, cluster_to_seaad, sea_ad_paths):
    """Compute per-(kinase, contrast, cluster) SEA-AD LFCs + supertype audit.

    Direct cluster → SEA-AD supertype merge (no WMB hop). `cluster_to_seaad`
    maps each spine cluster_name to a list of (supertype, weight) tuples;
    weights within a cluster sum to 1.0. Clusters mapped to `n/a` (empty
    list) get no SEA-AD evidence row.

    Many-to-many collapse: per (kinase, contrast, cluster), LFC is the
    weighted mean of supertype LFCs. Audit columns:
      - `sea_ad_n_supertypes` — number of finite supertype LFCs collapsed
      - `sea_ad_direction_agreement` — share of supertype LFCs with the same
        sign as the weighted mean (1.0 = all agree; 0.5 = perfectly mixed)

    Loads h5ads inside the function (anndata is not a built-in Kedro dataset).
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


def _prepare_wmb_specificity(wmb_df):
    """Reduce WMB expression export to top-(gene, wmb_class) rows for merging.

    The WMB CSV is keyed on `cell_type` = WMB class (retained subset, ~9
    classes). Spine clusters look up their parent class via the
    `cluster_to_wmb_class` crosswalk in `_assemble_unified`.
    """
    if wmb_df is None or len(wmb_df) == 0:
        return None
    wmb = wmb_df.copy()
    wmb["_gene_upper"] = wmb["gene_symbol"].str.upper()
    wmb_top = (wmb.sort_values("specificity_score")
                  .drop_duplicates(["_gene_upper", "cell_type"], keep="last")
                  [["_gene_upper", "cell_type", "specificity_score",
                    "mean_log2_expression", "fraction_cells_expressing",
                    "binary_expressed"]]
                  .rename(columns={
                      "cell_type": "wmb_class",
                      "specificity_score": "wmb_specificity",
                      "mean_log2_expression": "wmb_mean_log2_expression",
                      "fraction_cells_expressing": "wmb_fraction_cells_expressing",
                      "binary_expressed": "wmb_binary_expressed",
                  }))
    print(f"  WMB specificity: {len(wmb_top)} (gene, wmb_class) pairs loaded")
    return wmb_top


def _prepare_song_specificity(song_sp_df):
    if song_sp_df is None or len(song_sp_df) == 0:
        return None
    song_sp = song_sp_df.copy()
    song_sp["_gene_upper"] = song_sp["gene_symbol"].str.upper()
    song_spec_top = (song_sp.sort_values("specificity_score")
                            .drop_duplicates(["_gene_upper", "cell_type"], keep="last")
                            [["_gene_upper", "cell_type", "specificity_score"]]
                            .rename(columns={"specificity_score": "song_specificity"}))
    print(f"  Song specificity: {len(song_spec_top)} (gene, cell_type) pairs loaded")
    return song_spec_top


def _prepare_song_concordance(song_cd_df):
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


def _assemble_unified(sig, sea_ad_df, wmb_top, song_spec_top,
                      song_cd_top, song_key_is_contrast):
    """Cross-join sig × WMB classes, layer in evidence merges, score, and label.

    Returns ``(unified_full, unified_attributed, summary_dict)``.
    `unified_full` keeps every row (including confidence='none'); `unified_attributed`
    is the sorted, filtered slice that downstream consumers use.
    """
    print(f"  Building unified base: {len(sig)} sig rows × "
          f"{len(config.CLUSTER_SPINE)} spine clusters")
    sig_base = sig[["kinase", "gene_symbol", "contrast", "NES", "FDR"]].copy()
    sig_base["_gene_upper"] = sig_base["gene_symbol"].fillna("").astype(str).str.upper()
    sig_base["mea_significant"] = (np.isfinite(sig_base["FDR"])
                                    & (sig_base["FDR"] < config.MEA_FDR_THRESH))
    sig_base["residue_type"] = sig.get("residue_type", "ST")
    sig_base["track"] = sig.get("track", "st")

    cell_type_df = pd.DataFrame({"cell_type": list(config.CLUSTER_SPINE)})
    unified = sig_base.merge(cell_type_df, how="cross")

    # Attach the parent WMB class for each cluster via 1:1 crosswalk; used to
    # join WMB expression specificity at lineage level (clusters sharing a
    # parent inherit identical WMB scores).
    cluster_to_wmb = config.load_cluster_to_wmb_class_map()
    unified["wmb_class"] = unified["cell_type"].map(cluster_to_wmb)
    missing_wmb_parent = sorted(unified.loc[unified["wmb_class"].isna(), "cell_type"].unique())
    if missing_wmb_parent:
        print(f"  WARNING: {len(missing_wmb_parent)} spine clusters have no "
              f"WMB parent in cluster_to_wmb_class.csv: {missing_wmb_parent}")

    if sea_ad_df is not None and len(sea_ad_df) > 0:
        sea_ad_cols = [
            "kinase", "contrast", "cell_type",
            "sea_ad_lfc", "sea_ad_n_supertypes", "sea_ad_stratum",
            "concordance_score",
        ]
        if "sea_ad_direction_agreement" in sea_ad_df.columns:
            sea_ad_cols.append("sea_ad_direction_agreement")
        unified = unified.merge(
            sea_ad_df[sea_ad_cols], on=["kinase", "contrast", "cell_type"], how="left"
        )
    else:
        unified["sea_ad_lfc"] = np.nan
        unified["sea_ad_n_supertypes"] = np.nan
        unified["sea_ad_direction_agreement"] = np.nan
        unified["sea_ad_stratum"] = np.nan
        unified["concordance_score"] = np.nan

    if wmb_top is not None:
        # Lineage-level WMB join: (_gene_upper, wmb_class) → specificity.
        unified = unified.merge(wmb_top, on=["_gene_upper", "wmb_class"], how="left")
    else:
        unified["wmb_specificity"] = np.nan
        unified["wmb_mean_log2_expression"] = np.nan
        unified["wmb_fraction_cells_expressing"] = np.nan
        unified["wmb_binary_expressed"] = np.nan
    unified["wmb_specificity"] = unified["wmb_specificity"].fillna(0.0)
    unified["wmb_binary_expressed"] = unified["wmb_binary_expressed"].fillna(False).astype(bool)

    if song_spec_top is not None:
        unified = unified.merge(song_spec_top, on=["_gene_upper", "cell_type"], how="left")
    else:
        unified["song_specificity"] = np.nan

    if song_cd_top is not None:
        unified["_song_contrast"] = (unified["contrast"] if song_key_is_contrast
                                     else unified["contrast"].str.split("_").str[0])
        unified = unified.merge(song_cd_top,
                                 on=["_gene_upper", "cell_type", "_song_contrast"],
                                 how="left")
        unified = unified.drop(columns=["_song_contrast"])
    for _col in ("song_lfc", "song_pval", "song_fdr"):
        if _col not in unified.columns:
            unified[_col] = np.nan
    unified["song_concordance_score"] = np.where(
        np.isfinite(unified["song_lfc"]),
        np.sign(unified["NES"]) * unified["song_lfc"],
        np.nan,
    )

    w_song = config.SONG_CONCORDANCE_WEIGHT
    w_sea_ad = config.SEA_AD_CONCORDANCE_WEIGHT
    song_cs = unified["song_concordance_score"]
    sea_ad_cs = unified["concordance_score"]
    has_song_v = np.isfinite(song_cs)
    has_sea_ad_v = np.isfinite(sea_ad_cs) & (sea_ad_cs != 0)
    has_both = has_song_v & has_sea_ad_v
    has_song_only = has_song_v & ~has_sea_ad_v
    has_sea_ad_only = ~has_song_v & has_sea_ad_v
    unified["effective_concordance"] = np.select(
        [has_both, has_song_only, has_sea_ad_only],
        [(w_song * song_cs + w_sea_ad * sea_ad_cs) / (w_song + w_sea_ad),
         song_cs, sea_ad_cs],
        default=0.0,
    )
    unified["concordance_source"] = np.select(
        [has_both, has_song_only, has_sea_ad_only],
        ["both", "song", "sea_ad"],
        default="none",
    )

    unified["combined_score"] = (
        unified["effective_concordance"]
        * (config.COMBINED_SCORE_SPECIFICITY_BASE + unified["wmb_specificity"]))
    conf, basis = _assign_confidence_and_basis_vectorized(unified)
    unified["combined_confidence"] = conf
    unified["evidence_basis"] = basis

    unified = unified.drop(columns=["_gene_upper"], errors="ignore")

    expected = len(sig) * len(config.CLUSTER_SPINE)
    if len(unified) != expected:
        raise AssertionError(
            f"unified row count {len(unified)} != expected "
            f"{expected} (n_sig {len(sig)} × n_clusters "
            f"{len(config.CLUSTER_SPINE)}) — silent drop in merge")

    attributed = unified[unified["combined_confidence"] != "none"].copy()
    attributed = attributed.sort_values("combined_score", ascending=False)

    if len(attributed) > 0:
        print(f"\n  Attribution summary:")
        n_kinase_contrast = attributed.groupby(["kinase", "contrast"]).ngroups
        n_unique_kinases = attributed["kinase"].nunique()
        print(f"    {n_kinase_contrast} kinase-contrast pairs attributed "
              f"({n_unique_kinases} unique kinases)")
        print(f"\n  By confidence:")
        for conf, cnt in attributed[
                "combined_confidence"].value_counts().items():
            print(f"    {conf}: {cnt}")
        print(f"\n  By cell type (spine cluster):")
        for ct, cnt in attributed[
                "cell_type"].value_counts().items():
            print(f"    {ct}: {cnt}")

    summary = {
        "n_mea_significant": int(len(sig)),
        "n_total_rows": int(len(unified)),
        "n_attributed": int(len(attributed)),
        "by_confidence": (attributed["combined_confidence"].value_counts()
                          .to_dict() if len(attributed) > 0 else {}),
        "by_cell_type": (attributed["cell_type"].value_counts()
                         .to_dict() if len(attributed) > 0 else {}),
        "by_contrast": (attributed["contrast"].value_counts()
                        .to_dict() if len(attributed) > 0 else {}),
    }
    return unified, attributed, summary


# ===========================================================================
# CLI
# ===========================================================================

def main():
    """CLI shim: delegates to `kedro run --pipeline=attribute`."""
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    bootstrap_project(Path(__file__).resolve().parent.parent)
    with KedroSession.create() as session:
        session.run(pipeline_name="attribute")


if __name__ == "__main__":
    main()
