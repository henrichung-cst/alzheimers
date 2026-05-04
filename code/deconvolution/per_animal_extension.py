"""Per-animal expansion of the group-level decomposition.

The 24 group-level samples are too few for the 10-parameter factorial OLS
(dof = 2 for males-only). This module re-estimates contrasts at per-animal
grain (~33 male animals, dof = 23) by:

    attributed[g, c, s] = proportion[g, c] * signal[g, c, s]
    frac[g, c, s]       = attributed[g, c, s] / sum_c attributed[g, c, s]
    deconv[a, c, s]     = frac[group(a), c, s] * bulk[a, s]

frac is site-specific within each group, so the cell axis carries real
information that survives median-centering before MEA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deconvolution import paths
from deconvolution.load_deconvoluted import (
    DeconvoluatedTrack, load_track, safe_log2,
)
from deconvolution.factorial_ols import (
    PARAM_NAMES, _ols_batch, _contrast_lfc_se, _contrast_vector,
)

GENOTYPE_PER_ANIMAL_CODING = {
    "WT":      {"App": 0, "Tau": 0, "Int": 0},
    "APP":     {"App": 1, "Tau": 0, "Int": 0},
    "T22":     {"App": 0, "Tau": 1, "Int": 0},
    "T22/APP": {"App": 1, "Tau": 1, "Int": 1},
}


def compute_site_fractions(track: DeconvoluatedTrack,
                           cluster_size_df: pd.DataFrame) -> pd.DataFrame:
    """Compute frac[group, cell_type, site] from a deconvolved track.

    Returns a DataFrame with the same shape as ``track.values``
    (n_sites x MultiIndex(sample, cell_type)) where each entry is the
    fraction of bulk attributable to that cell type at that site in that
    group. Entries sum to 1 across cell types at each (sample, site).

    Index of cluster_size_df must match the level-1 label of the track's
    column MultiIndex (`wmb_class` for the WMB-projected track).

    Sites where the per-(sample, site) total attribution is 0 (e.g. NaN
    bulk) get 0s — those sites are dropped downstream by safe_log2.
    """
    # Normalize cluster sizes per column → proportion[cluster, group]
    totals = cluster_size_df.sum(axis=0)
    if (totals <= 0).any():
        bad = totals[totals <= 0].index.tolist()
        raise ValueError(f"cluster_size has empty columns: {bad}")
    prop = cluster_size_df.div(totals, axis=1)  # cluster x group

    sig = track.values  # n_sites x MultiIndex(sample, cluster)
    cols = sig.columns

    # Per-column weight = proportion[cluster, sample]
    weights = np.array([
        prop.loc[c, s] if (c in prop.index and s in prop.columns) else np.nan
        for s, c in cols
    ])
    if not np.all(np.isfinite(weights)):
        missing_idx = np.where(~np.isfinite(weights))[0]
        sample_examples = [(cols[i]) for i in missing_idx[:5]]
        raise ValueError(
            f"Missing proportion entries (cluster, sample); "
            f"first few: {sample_examples}"
        )

    attributed = sig.values.astype(float) * weights[np.newaxis, :]
    # NaN signal means Yuyu's NNLS didn't fit at that (group, cluster, site).
    # bulk_yuyu (sum across clusters at (sample, site)) is computed with nansum
    # so a missing cluster doesn't drag the entire group×site into NaN.
    sample_codes = pd.Index([s for s, _ in cols])
    unique_samples = sample_codes.unique()
    sample_to_idx = {s: i for i, s in enumerate(unique_samples)}
    sample_idx_per_col = np.array([sample_to_idx[s] for s, _ in cols])
    bulk_yuyu = np.zeros((sig.shape[0], len(unique_samples)))
    for s_name, s_idx in sample_to_idx.items():
        col_mask = sample_idx_per_col == s_idx
        bulk_yuyu[:, s_idx] = np.nansum(attributed[:, col_mask], axis=1)

    denom = bulk_yuyu[:, sample_idx_per_col]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_arr = attributed / denom
    # Where bulk_yuyu is 0 (all clusters missing at this group×site), frac is undefined.
    # Where attributed is NaN, frac is already NaN via division (NaN preserved).
    frac_arr = np.where(denom > 0, frac_arr, np.nan)
    frac = pd.DataFrame(frac_arr, index=sig.index, columns=cols)
    return frac


def _build_design_matrix_per_animal(mapping_subset: pd.DataFrame) -> np.ndarray:
    """10-parameter factorial design from per-animal metadata.

    mapping_subset must have columns: genotype, timepoint. Rows are in the
    desired animal order (matching the Y matrix column order downstream).
    """
    n = len(mapping_subset)
    X = np.zeros((n, len(PARAM_NAMES)))
    X[:, 0] = 1.0  # const
    geno = mapping_subset["genotype"].values
    for i, g in enumerate(geno):
        if g not in GENOTYPE_PER_ANIMAL_CODING:
            raise KeyError(f"Unknown genotype: {g!r}")
        c = GENOTYPE_PER_ANIMAL_CODING[g]
        X[i, 1] = c["App"]
        X[i, 2] = c["Tau"]
        X[i, 3] = c["Int"]
    times = mapping_subset["timepoint"].values
    X[:, 4] = (times == "4mo").astype(float)
    X[:, 5] = (times == "6mo").astype(float)
    X[:, 6] = X[:, 1] * X[:, 4]  # App_x_time4
    X[:, 7] = X[:, 1] * X[:, 5]  # App_x_time6
    X[:, 8] = X[:, 2] * X[:, 4]  # Tau_x_time4
    X[:, 9] = X[:, 2] * X[:, 5]  # Tau_x_time6
    return X


def select_male_animals(sample_mapping: pd.DataFrame,
                        exclusions_path: str | None = None) -> pd.DataFrame:
    """Return the male animals with outlier-flagged samples removed.

    Mirrors the live pipeline's ANALYSIS_MODE=males_only behaviour.
    """
    males = sample_mapping[sample_mapping["sex"] == "M"].copy()
    if exclusions_path is not None:
        excl = pd.read_csv(exclusions_path)
        bad_ids = set(
            excl.loc[excl["excluded"].astype(bool), "animal_id"].astype(str)
        )
        males = males[~males["animal_id"].astype(str).isin(bad_ids)]
    return males.reset_index(drop=True)


def run_per_animal_track(track: DeconvoluatedTrack,
                         frac: pd.DataFrame,
                         raw_phospho: pd.DataFrame,
                         male_mapping: pd.DataFrame,
                         cell_types: list[str] | None = None) -> pd.DataFrame:
    """Run per-animal factorial OLS for each cell type on the deconvolved track.

    Parameters
    ----------
    track : DeconvoluatedTrack on the WMB-class axis
    frac : output of compute_site_fractions; n_sites x (sample, wmb_class)
    raw_phospho : per-animal bulk; must have site_id column + per-animal columns
    male_mapping : output of select_male_animals; rows define OLS sample order
    cell_types : subset of track.clusters (= WMB classes); default = all
    """
    if cell_types is None:
        cell_types = track.clusters

    # Align raw_phospho rows to track sites (via site_id)
    yuyu_site_ids = track.site_id().astype(str).values
    rp = raw_phospho.copy()
    rp["site_id"] = rp["site_id"].astype(str)
    rp = rp.set_index("site_id").reindex(yuyu_site_ids)

    bulk_cols = male_mapping["column_name"].tolist()
    missing = [c for c in bulk_cols if c not in rp.columns]
    if missing:
        raise KeyError(f"raw_phospho missing animal columns: {missing[:5]}")
    bulk_per_animal = rp[bulk_cols].to_numpy(dtype=float)  # n_sites x n_animals

    groups_per_animal = male_mapping["phospho_group_id"].tolist()

    X = _build_design_matrix_per_animal(male_mapping)
    XtX_inv = np.linalg.inv(X.T @ X)
    contrast_vecs = {c: _contrast_vector(c) for c in paths.CONTRASTS}

    site_id = track.site_id().values
    motif = track.meta["motif"].astype(str).values
    gene_symbol = track.meta["gene_symbol"].astype(str).values
    n_sites = len(site_id)
    n_contr = len(contrast_vecs)

    out_frames = []
    for ci, ct in enumerate(cell_types, 1):
        # Pull frac[group, site] for this cell type across all groups per animal
        frac_ct_cols = [(g, ct) for g in groups_per_animal]
        missing_fc = [c for c in frac_ct_cols if c not in frac.columns]
        if missing_fc:
            print(f"    [{track.track}] wmb_class {ct!r}: "
                  f"missing frac cols {missing_fc[:3]}; skipping")
            continue
        frac_for_animals = frac[frac_ct_cols].to_numpy(dtype=float)
        # n_sites x n_animals; element (s, a) = frac[group(a), wmb_class, s]

        deconv = frac_for_animals * bulk_per_animal
        Y = safe_log2(deconv)

        betas, sigma2, dof = _ols_batch(Y, X, XtX_inv)

        lfc_arr = np.empty(n_sites * n_contr)
        se_arr = np.empty(n_sites * n_contr)
        p_arr = np.empty(n_sites * n_contr)
        contrast_arr = np.empty(n_sites * n_contr, dtype=object)
        for j, (contrast_name, cvec) in enumerate(contrast_vecs.items()):
            lfc, se_c, pval = _contrast_lfc_se(betas, sigma2, XtX_inv, cvec, dof)
            sl = slice(j * n_sites, (j + 1) * n_sites)
            lfc_arr[sl] = lfc
            se_arr[sl] = se_c
            p_arr[sl] = pval
            contrast_arr[sl] = contrast_name

        out_frames.append(pd.DataFrame({
            "site_id": np.tile(site_id, n_contr),
            "gene_symbol": np.tile(gene_symbol, n_contr),
            "motif": np.tile(motif, n_contr),
            "wmb_class": ct,
            "contrast": contrast_arr,
            "lfc": lfc_arr,
            "se": se_arr,
            "pval": p_arr,
            "track": track.track,
        }))
        if ci % 5 == 0 or ci == len(cell_types):
            print(f"    [{track.track}] per-animal OLS done for "
                  f"{ci}/{len(cell_types)} wmb_classes (dof={dof})")

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)
