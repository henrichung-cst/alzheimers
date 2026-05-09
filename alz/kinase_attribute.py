"""Stage 3 of kinase attribution: unified cell-type attribution.

Combines MEA results (S/T + pY tracks) with three evidence sources:
  - SEA-AD per-supertype effect sizes (pathway-matched: App→early CPS,
    Tau→late CPS, ApTt→full CPS), aggregated to WMB classes
  - WMB expression specificity (Allen WMB 10Xv3, 34 classes)
  - Song within-cohort transcriptomic concordance + specificity

Inputs:
  outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv
  data/external/sea_ad/effect_sizes{,_early,_late}.h5ad
  outputs/reports/wmb_expression/wmb_kinase_expression.csv
  outputs/reports/snrna_integration/song_{specificity,concordance}.csv (optional)
  data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv
  config.SEAAD_TO_WMB_CLASS_FILE

Outputs (under outputs/reports/kinase_attribution/):
  unified_attribution.csv         — confidence != "none" rows (sorted)
  unified_attribution_full.csv    — full sig × WMB-class grid
  sea_ad_supertype_lfc.csv        — per-(gene, stratum, supertype) audit
  attribution_summary.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import config
from kinase_enrich import _resolve_track, _track_output

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = config.KINASE_ATTRIBUTION_OUTPUT_DIR
WMB_EXPRESSION_FILE = config.WMB_EXPRESSION_FILE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def _compute_sea_ad_concordance(sig, seaad_to_class_df, sea_ad_paths):
    """Compute per-(kinase, contrast, cell_type) SEA-AD LFCs + supertype audit.

    Loads h5ads inside the function (anndata is not a built-in Kedro dataset and
    the files are optional in degraded environments). Returns
    ``(sea_ad_df, supertype_df)``. On ImportError or missing h5ads, prints a
    warning and returns empty DataFrames so attribution can still degrade
    gracefully (Song/WMB-only tiers).
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
        gene_to_idx = {g: i for i, g in enumerate(ref_adata.obs_names)}

        seaad_to_class = dict(
            zip(seaad_to_class_df["seaad_subclass"],
                seaad_to_class_df["wmb_class_label"])
        )
        st_to_wmb_class = {
            st: seaad_to_class.get(sc) for st, sc in st_to_subclass.items()
        }
        n_st_mapped = sum(1 for v in st_to_wmb_class.values() if v is not None)
        print(f"  SEA-AD supertypes mapped to WMB class: "
              f"{n_st_mapped}/{len(supertypes)}")

        def _class_lfcs(adata, gene_idx):
            effects = adata.X[gene_idx, :]
            if hasattr(effects, "toarray"):
                effects = effects.toarray().flatten()
            else:
                effects = np.asarray(effects).flatten()
            cls_vals, cls_counts = {}, {}
            for i, st in enumerate(supertypes):
                wmb_class = st_to_wmb_class.get(st)
                if wmb_class is None:
                    continue
                val = effects[i]
                if np.isfinite(val):
                    cls_vals.setdefault(wmb_class, []).append(val)
                    cls_counts[wmb_class] = cls_counts.get(wmb_class, 0) + 1
            return (
                {c: float(np.median(v)) for c, v in cls_vals.items()},
                cls_counts,
            )

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
                _lfc_cache[cache_key] = _class_lfcs(
                    adata_by_stratum[stratum], gene_idx)
            cls_lfcs, cls_counts = _lfc_cache[cache_key]

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
            for wmb_class, median_lfc in cls_lfcs.items():
                concordance = np.sign(nes) * median_lfc
                sea_ad_rows.append({
                    "kinase": kinase,
                    "gene_symbol": gene,
                    "contrast": contrast,
                    "NES": nes,
                    "FDR": fdr,
                    "residue_type": residue_type,
                    "track": track_name,
                    "cell_type": wmb_class,
                    "sea_ad_lfc": median_lfc,
                    "sea_ad_n_supertypes": cls_counts[wmb_class],
                    "sea_ad_stratum": stratum,
                    "concordance_score": concordance,
                })

        print(f"  SEA-AD concordance: {len(sea_ad_rows)} "
              f"(kinase, contrast, cell_type) rows")

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
    """Reduce WMB expression export to top-(gene, cell_type) rows for merging."""
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
                      "specificity_score": "wmb_specificity",
                      "mean_log2_expression": "wmb_mean_log2_expression",
                      "fraction_cells_expressing": "wmb_fraction_cells_expressing",
                      "binary_expressed": "wmb_binary_expressed",
                  }))
    print(f"  WMB specificity: {len(wmb_top)} (gene, cell_type) pairs loaded")
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
          f"{len(config.WMB_CLASSES)} WMB classes")
    sig_base = sig[["kinase", "gene_symbol", "contrast", "NES", "FDR"]].copy()
    sig_base["_gene_upper"] = sig_base["gene_symbol"].fillna("").astype(str).str.upper()
    sig_base["mea_significant"] = (np.isfinite(sig_base["FDR"])
                                    & (sig_base["FDR"] < config.MEA_FDR_THRESH))
    sig_base["residue_type"] = sig.get("residue_type", "ST")
    sig_base["track"] = sig.get("track", "st")

    cell_type_df = pd.DataFrame({"cell_type": list(config.WMB_CLASSES)})
    unified = sig_base.merge(cell_type_df, how="cross")

    if sea_ad_df is not None and len(sea_ad_df) > 0:
        sea_ad_join = sea_ad_df[[
            "kinase", "contrast", "cell_type",
            "sea_ad_lfc", "sea_ad_n_supertypes", "sea_ad_stratum",
            "concordance_score",
        ]]
        unified = unified.merge(
            sea_ad_join, on=["kinase", "contrast", "cell_type"], how="left"
        )
    else:
        unified["sea_ad_lfc"] = np.nan
        unified["sea_ad_n_supertypes"] = np.nan
        unified["sea_ad_stratum"] = np.nan
        unified["concordance_score"] = np.nan

    if wmb_top is not None:
        unified = unified.merge(wmb_top, on=["_gene_upper", "cell_type"], how="left")
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
        print(f"\n  By cell type (WMB class):")
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
# Stage 3: Unified cell-type attribution
# ===========================================================================

def step_attribute():
    """Stage 3: Unified cell-type attribution (SEA-AD concordance + WMB expression)."""
    _ensure_output_dir()
    print("\n=== Stage 3: Unified Cell-Type Attribution ===\n")

    # 3a. Load MEA results from every track that produced an output and concat.
    mea_by_track = []
    for track_name, track_cfg in config.PHOSPHO_TRACKS.items():
        mea_path = _track_output("mea_stoichiometry.csv", track_cfg)
        if not os.path.exists(mea_path):
            print(f"  Track {track_name}: {mea_path} not present, skipping")
            continue
        mea_by_track.append((track_cfg, pd.read_csv(mea_path)))
    sig = _combine_mea_tracks(mea_by_track)
    if len(sig) == 0:
        print("  No MEA rows found. Stage 3 complete.")
        return

    # 3b. Map kinases to genes.
    k2g = pd.read_csv(config.MAPPING_CACHE_FILE)
    sig = _map_kinases_to_genes(sig, k2g)

    # 3c. SEA-AD concordance.
    if not os.path.exists(config.SEAAD_TO_WMB_CLASS_FILE):
        raise FileNotFoundError(
            f"SEA-AD → WMB class mapping not found at "
            f"{config.SEAAD_TO_WMB_CLASS_FILE}. Generate via Phase 1c."
        )
    seaad_to_class_df = pd.read_csv(config.SEAAD_TO_WMB_CLASS_FILE)
    sea_ad_df, supertype_df = _compute_sea_ad_concordance(
        sig, seaad_to_class_df, config.SEA_AD_EFFECT_SIZES
    )

    if len(supertype_df) > 0:
        supertype_path = os.path.join(OUTPUT_DIR, "sea_ad_supertype_lfc.csv")
        supertype_df.to_csv(supertype_path, index=False)
        print(f"  SEA-AD supertype LFCs: {len(supertype_df)} rows → "
              f"{supertype_path}")

    # 3d. WMB expression specificity.
    wmb_df = pd.read_csv(WMB_EXPRESSION_FILE) if os.path.exists(WMB_EXPRESSION_FILE) else None
    if wmb_df is None:
        print(f"  WMB expression file not found at {WMB_EXPRESSION_FILE}")
    wmb_top = _prepare_wmb_specificity(wmb_df)

    # 3d′. Song within-cohort evidence (specificity + concordance).
    song_sp_df = (pd.read_csv(config.SONG_EXPRESSION_FILE)
                  if os.path.exists(config.SONG_EXPRESSION_FILE) else None)
    song_spec_top = _prepare_song_specificity(song_sp_df)

    song_cd_df = (pd.read_csv(config.SONG_CONCORDANCE_FILE)
                  if os.path.exists(config.SONG_CONCORDANCE_FILE) else None)
    song_cd_top, song_key_is_contrast = _prepare_song_concordance(song_cd_df)

    # 3e. Combine into unified attribution table.
    unified, attributed, summary = _assemble_unified(
        sig, sea_ad_df, wmb_top, song_spec_top, song_cd_top, song_key_is_contrast
    )

    out_path = os.path.join(OUTPUT_DIR, "unified_attribution.csv")
    attributed.to_csv(out_path, index=False)
    print(f"\n  Saved {out_path} ({len(attributed)} attributed rows)")

    full_path = os.path.join(OUTPUT_DIR, "unified_attribution_full.csv")
    unified.to_csv(full_path, index=False)

    summary_path = os.path.join(OUTPUT_DIR, "attribution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Stage 3 complete.")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 of kinase attribution: unified cell-type "
                    "attribution (SEA-AD + WMB + Song).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args()
    step_attribute()


if __name__ == "__main__":
    main()
