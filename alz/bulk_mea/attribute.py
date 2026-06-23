"""Stage 3 of kinase attribution: unified cell-type attribution.

Analysis spine is the Levy-t5 31-cluster spine (`config.CLUSTER_SPINE`),
not WMB-34. Evidence sources merge via single-hop crosswalks:
  - SEA-AD per-supertype effect sizes (pathway-matched: App→early CPS,
    Tau→late CPS, ApTt→full CPS), joined direct cluster → SEA-AD supertype(s)
    via `cluster_to_seaad_supertype.csv` (many-to-many; weighted mean LFC).
  - WMB expression specificity, joined cluster → parent WMB class via
    `cluster_to_wmb_class.csv` (1:1, lineage-level; clusters sharing a parent
    inherit identical WMB specificity values).
  - Song within-cohort transcriptomic concordance + specificity, keyed
    directly on cluster_name (identity join).

Inputs:
  outputs/reports/kinase_attribution/mea_stoichiometry{,_pY}.csv
  data/external/sea_ad/effect_sizes{,_early,_late}.h5ad
  outputs/reports/wmb_expression/wmb_kinase_expression.csv
  outputs/reports/snrna_integration/song_{specificity,concordance}.csv (optional)
  data/derived/caches/kinase_to_gene_mapping.csv
  config.CLUSTER_TO_WMB_CLASS_FILE
  config.CLUSTER_TO_SEAAD_SUPERTYPE_FILE

Outputs (under outputs/reports/kinase_attribution/):
  unified_attribution.csv         — confidence != "none" rows (sorted)
  unified_attribution_full.csv    — full sig × cluster grid (n_kinases × 9 × 31)
  sea_ad_supertype_lfc.csv        — per-(gene, stratum, supertype) audit
  attribution_summary.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.shared import config
from alz.bulk_mea.confidence import (
    CONFIDENCE_RANK,
    assign_confidence,
    load_decomposition_crosscheck,
    prepare_human_location_specificity,
)
from alz.bulk_mea.specificity_class import assign_specificity_class
from alz.cross_reference.evidence import (
    compute_sea_ad_concordance,
    prepare_song_concordance,
    prepare_song_specificity,
    prepare_wmb_specificity,
)


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


def _assemble_unified(sig, sea_ad_df, wmb_top, song_spec_top,
                      song_cd_top, song_key_is_contrast,
                      human_location_top=None, decomp_top=None):
    """Cross-join sig × WMB classes, layer in evidence merges, and label.

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
        unified = unified.rename(columns={"concordance_score": "_sea_ad_directional_lfc"})
    else:
        unified["sea_ad_lfc"] = np.nan
        unified["sea_ad_n_supertypes"] = np.nan
        unified["sea_ad_direction_agreement"] = np.nan
        unified["sea_ad_stratum"] = np.nan
        unified["_sea_ad_directional_lfc"] = np.nan

    if wmb_top is not None:
        # Lineage-level WMB join: (_gene_upper, wmb_class) → detection metric.
        unified = unified.merge(wmb_top, on=["_gene_upper", "wmb_class"], how="left")
    else:
        for _c in ("wmb_detected", "wmb_concentration", "wmb_concentration_tier",
                   "wmb_mean_log2_expression", "wmb_fraction_cells_expressing",
                   "wmb_binary_expressed", "wmb_top_celltype"):
            unified[_c] = np.nan
    if "wmb_top_celltype" not in unified.columns:
        unified["wmb_top_celltype"] = ""
    unified["wmb_detected"] = unified["wmb_detected"].fillna(False).astype(bool)
    unified["wmb_concentration"] = unified["wmb_concentration"].fillna(0.0)
    unified["wmb_concentration_tier"] = unified["wmb_concentration_tier"].fillna(0).astype(int)
    unified["wmb_binary_expressed"] = unified["wmb_binary_expressed"].fillna(False).astype(bool)
    unified["wmb_top_celltype"] = unified["wmb_top_celltype"].fillna("").astype(str)

    if song_spec_top is not None:
        unified = unified.merge(song_spec_top, on=["_gene_upper", "cell_type"], how="left")
    for _col in ("song_detected", "song_concentration", "song_concentration_of_total",
                 "song_concentration_tier", "song_fraction_cells_expressing",
                 "song_effective_n", "song_top_celltype", "song_top_concentration"):
        if _col not in unified.columns:
            unified[_col] = np.nan
    unified["song_detected"] = unified["song_detected"].fillna(False).astype(bool)
    unified["song_concentration"] = unified["song_concentration"].fillna(0.0)
    unified["song_concentration_of_total"] = unified["song_concentration_of_total"].fillna(0.0)
    unified["song_concentration_tier"] = unified["song_concentration_tier"].fillna(0).astype(int)

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
    unified["_song_directional_lfc"] = np.where(
        np.isfinite(unified["song_lfc"]),
        np.sign(unified["NES"]) * unified["song_lfc"],
        np.nan,
    )

    if human_location_top is not None:
        unified = unified.merge(
            human_location_top,
            on=["_gene_upper", "cell_type"],
            how="left",
        )
    for _col in ("seaad_location_score", "hbca_location_score", "human_location_score"):
        if _col not in unified.columns:
            unified[_col] = np.nan

    if decomp_top is not None:
        unified = unified.merge(
            decomp_top,
            on=["kinase", "contrast", "cell_type"],
            how="left",
        )
    for _col in ("decomp_nes", "decomp_fdr"):
        if _col not in unified.columns:
            unified[_col] = np.nan

    w_song = config.SONG_CONCORDANCE_WEIGHT
    w_sea_ad = config.SEA_AD_CONCORDANCE_WEIGHT
    song_cs = unified["_song_directional_lfc"]
    sea_ad_cs = unified["_sea_ad_directional_lfc"]
    has_song_v = np.isfinite(song_cs)
    has_sea_ad_v = np.isfinite(sea_ad_cs) & (sea_ad_cs != 0)
    has_both = has_song_v & has_sea_ad_v
    has_song_only = has_song_v & ~has_sea_ad_v
    has_sea_ad_only = ~has_song_v & has_sea_ad_v
    unified["_effective_concordance"] = np.select(
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

    unified = assign_confidence(unified)
    unified = assign_specificity_class(unified)

    unified = unified.drop(columns=["_gene_upper"], errors="ignore")

    expected = len(sig) * len(config.CLUSTER_SPINE)
    if len(unified) != expected:
        raise AssertionError(
            f"unified row count {len(unified)} != expected "
            f"{expected} (n_sig {len(sig)} × n_clusters "
            f"{len(config.CLUSTER_SPINE)}) — silent drop in merge")

    attributed = unified[unified["confidence_tier"] != "none"].copy()
    attributed["_confidence_rank"] = (
        attributed["confidence_tier"].map(CONFIDENCE_RANK).fillna(0).astype(int)
    )
    attributed["_song_lfc_abs"] = attributed["song_lfc"].abs()
    attributed["_sea_ad_lfc_abs"] = attributed["sea_ad_lfc"].abs()
    attributed = attributed.sort_values(
        [
            "_confidence_rank",
            "song_concentration",
            "decomp_agrees_bulk",
            "wmb_concentration",
            "human_location_score",
            "_song_lfc_abs",
            "_sea_ad_lfc_abs",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).drop(columns=["_confidence_rank", "_song_lfc_abs", "_sea_ad_lfc_abs"])

    internal_cols = [
        "_effective_concordance",
        "_song_directional_lfc",
        "_sea_ad_directional_lfc",
    ]
    unified = unified.drop(columns=internal_cols, errors="ignore")
    attributed = attributed.drop(columns=internal_cols, errors="ignore")

    if len(attributed) > 0:
        print(f"\n  Attribution summary:")
        n_kinase_contrast = attributed.groupby(["kinase", "contrast"]).ngroups
        n_unique_kinases = attributed["kinase"].nunique()
        print(f"    {n_kinase_contrast} kinase-contrast pairs attributed "
              f"({n_unique_kinases} unique kinases)")
        print(f"\n  By confidence:")
        for conf, cnt in attributed[
                "confidence_tier"].value_counts().items():
            print(f"    {conf}: {cnt}")
        print(f"\n  By cell type (spine cluster):")
        for ct, cnt in attributed[
                "cell_type"].value_counts().items():
            print(f"    {ct}: {cnt}")

    summary = {
        "n_mea_significant": int(len(sig)),
        "n_total_rows": int(len(unified)),
        "n_attributed": int(len(attributed)),
        "by_confidence": (attributed["confidence_tier"].value_counts()
                          .to_dict() if len(attributed) > 0 else {}),
        "by_cell_type": (attributed["cell_type"].value_counts()
                         .to_dict() if len(attributed) > 0 else {}),
        "by_contrast": (attributed["contrast"].value_counts()
                        .to_dict() if len(attributed) > 0 else {}),
        **config.provenance_stamp(),
    }
    return unified, attributed, summary


# ===========================================================================
# CLI
# ===========================================================================

def main():
    """Run unified cell-type attribution directly (no Kedro)."""
    print("\n=== Stage 3: Unified Cell-Type Attribution ===\n")

    params = config.load_params()
    sea_ad_paths = params["sea_ad_paths"]
    wmb_expression_path = params["wmb_expression_path"]
    song_specificity_path = params["song_specificity_path"]
    song_concordance_path = params["song_concordance_path"]

    kinase_attr_dir = config.KINASE_ATTRIBUTION_OUTPUT_DIR
    os.makedirs(kinase_attr_dir, exist_ok=True)

    # Load per-track MEA stoichiometry
    mea_by_track = []
    for track_name, track_cfg in config.PHOSPHO_TRACKS.items():
        suffix = track_cfg["output_suffix"]
        fname = f"mea_stoichiometry{suffix}.csv"
        path = os.path.join(kinase_attr_dir, fname)
        if not os.path.exists(path):
            print(f"  [{track_name}] {path} not found; skipping.")
            mea_by_track.append((track_cfg, None))
        else:
            mea_by_track.append((track_cfg, pd.read_csv(path)))

    # Load kinase-to-gene mapping
    k2g_path = config.MAPPING_CACHE_FILE
    if not os.path.exists(k2g_path):
        raise FileNotFoundError(
            f"kinase_to_gene_mapping.csv not found at {k2g_path}")
    k2g_df = pd.read_csv(k2g_path)

    # Combine MEA tracks and inject gene symbols
    sig = _combine_mea_tracks(mea_by_track)
    sig = _map_kinases_to_genes(sig, k2g_df)

    # SEA-AD concordance
    cluster_to_seaad = config.load_cluster_to_seaad_supertype_map()
    sea_ad_df, supertype_df = compute_sea_ad_concordance(
        sig, cluster_to_seaad, sea_ad_paths)

    # Assemble unified attribution
    wmb_df = (pd.read_csv(wmb_expression_path)
              if wmb_expression_path and os.path.exists(wmb_expression_path)
              else None)
    if wmb_df is None and wmb_expression_path:
        print(f"  WMB expression file not found at {wmb_expression_path}")
    wmb_top = prepare_wmb_specificity(wmb_df)

    song_sp_df = (pd.read_csv(song_specificity_path)
                  if song_specificity_path and os.path.exists(song_specificity_path)
                  else None)
    song_spec_top = prepare_song_specificity(song_sp_df)

    song_cd_df = (pd.read_csv(song_concordance_path)
                  if song_concordance_path and os.path.exists(song_concordance_path)
                  else None)
    song_cd_top, song_key_is_contrast = prepare_song_concordance(song_cd_df)

    human_location_top = prepare_human_location_specificity()
    decomp_top = load_decomposition_crosscheck()
    if decomp_top is None:
        print("  Decomposition cross-check: not available; no very_high promotion")
    else:
        print(f"  Decomposition cross-check: {len(decomp_top)} rows loaded")

    unified, attributed, summary = _assemble_unified(
        sig, sea_ad_df, wmb_top, song_spec_top, song_cd_top, song_key_is_contrast,
        human_location_top=human_location_top, decomp_top=decomp_top)

    # Save outputs
    attributed.to_csv(
        os.path.join(kinase_attr_dir, "unified_attribution.csv"), index=False)
    unified.to_csv(
        os.path.join(kinase_attr_dir, "unified_attribution_full.csv"), index=False)
    print(f"\n  Saved unified_attribution.csv ({len(attributed)} attributed rows)")
    print(f"  Saved unified_attribution_full.csv ({len(unified)} rows)")

    if supertype_df is not None and len(supertype_df) > 0:
        supertype_df.to_csv(
            os.path.join(kinase_attr_dir, "sea_ad_supertype_lfc.csv"), index=False)
        print(f"  Saved sea_ad_supertype_lfc.csv ({len(supertype_df)} rows)")

    summary_path = os.path.join(kinase_attr_dir, "attribution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved attribution_summary.json")
    print("\n  Stage 3 complete.")


if __name__ == "__main__":
    main()
