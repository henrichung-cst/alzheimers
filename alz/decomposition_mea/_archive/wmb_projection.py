"""Project Yuyu's 46-cluster deconvolution onto the WMB-class spine.

Mapping source: `yuyu_46_to_wmb_class_soft.csv` — row-stochastic 46×27
matrix derived per nucleus from Allen Cell Type Mapper `class_name`
(see `remap_clusters_via_ctm.py`).

Projection:
    track.values_wmb[s, (sample, w)] = sum_c track.values[s, (sample, c)] * S[c, w]
    cluster_size_wmb[w, group]       = sum_c cluster_size[c, group]       * S[c, w]
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from deconvolution import paths
from deconvolution.load_deconvoluted import DeconvoluatedTrack


def load_soft_mass() -> pd.DataFrame:
    """Return the 46 × 27 row-stochastic WMB-projection matrix S[c, w].

    Index: 46 Yuyu cluster names. Columns: WMB class labels (e.g. "30 Astro-Epen").
    The auxiliary `n_nuclei` column is dropped after an audit log line.
    """
    df = pd.read_csv(paths.SOFT_MASS_FILE)
    if "cluster_name" not in df.columns:
        raise ValueError(
            f"Expected 'cluster_name' column in {paths.SOFT_MASS_FILE}; "
            f"found {list(df.columns)[:5]}"
        )
    df = df.set_index("cluster_name")
    n_nuclei = df.pop("n_nuclei") if "n_nuclei" in df.columns else None

    # Sanity: each row sums to ~1 across WMB-class columns.
    row_sums = df.sum(axis=1)
    bad = row_sums[(row_sums - 1.0).abs() > 1e-3]
    if len(bad):
        raise ValueError(
            f"Soft-mass rows not row-stochastic for clusters: "
            f"{bad.head().to_dict()}"
        )

    print(f"  Loaded soft-mass crosswalk: {df.shape[0]} clusters → "
          f"{df.shape[1]} WMB classes")
    if n_nuclei is not None:
        print(f"    Total nuclei represented: {int(n_nuclei.sum()):,}")
    return df


def project_track_to_wmb(track: DeconvoluatedTrack,
                         S: pd.DataFrame) -> DeconvoluatedTrack:
    """Return a track whose value-column inner level is WMB class, not cluster.

    The MultiIndex name of the inner level becomes "wmb_class" so downstream
    code that introspects column names sees the new axis explicitly.

    Mass-conserving up to the 27-class restriction: any cluster mass that fell
    in CTM classes outside the 27 reachable on this tissue is already absent
    from S (those columns don't exist), so projected sums may be slightly
    below the original cluster-level sums. That gap is the same gap surfaced
    by the calibration audit and is documented as the biological-sampling
    floor on this branch.
    """
    sig = track.values  # n_sites × MultiIndex(sample, cluster)
    samples = pd.Index([s for s, _ in sig.columns]).unique()
    wmb_classes = list(S.columns)

    # Build a (n_clusters_in_track × n_wmb) matrix matching track.clusters
    # row order, so a per-sample slice of `sig` can be matmul'd directly.
    tc = list(track.clusters)
    missing = [c for c in tc if c not in S.index]
    if missing:
        raise KeyError(
            f"Soft-mass crosswalk is missing rows for {len(missing)} clusters: "
            f"{missing[:5]}"
        )
    S_aligned = S.loc[tc].to_numpy(dtype=float)  # n_clusters × n_wmb

    out_cols = []
    out_blocks = []
    for sample in samples:
        # Pick the cluster columns for this sample, in track.clusters order.
        cols = [(sample, c) for c in tc]
        block = sig.loc[:, cols].to_numpy(dtype=float)  # n_sites × n_clusters
        # NaN policy: a NaN in a single (cluster, sample, site) drags the
        # corresponding (wmb, sample, site) to NaN under naive matmul. Use
        # nan-safe contraction so a missing cluster doesn't blank out an
        # entire WMB class. Coverage of NaNs in Yuyu's NNLS output is sparse;
        # this matches `compute_site_fractions`'s nansum semantics.
        nan_mask = np.isnan(block)
        block_filled = np.where(nan_mask, 0.0, block)
        # Effective weight: zero out the column slice of S that maps from a
        # NaN cluster, on a per-site basis.
        # projected[i, w] = sum_c block_filled[i, c] * S_aligned[c, w]
        projected = block_filled @ S_aligned  # n_sites × n_wmb
        out_blocks.append(projected)
        out_cols.extend([(sample, w) for w in wmb_classes])

    full = np.hstack(out_blocks)
    new_columns = pd.MultiIndex.from_tuples(out_cols,
                                            names=["sample", "wmb_class"])
    new_values = pd.DataFrame(full, index=sig.index, columns=new_columns)

    return DeconvoluatedTrack(
        track=track.track,
        meta=track.meta,
        values=new_values,
        samples=list(samples),
        clusters=wmb_classes,
    )


def project_cluster_size(cluster_size_df: pd.DataFrame,
                         S: pd.DataFrame) -> pd.DataFrame:
    """Return a (n_wmb × n_groups) WMB-class nucleus-count matrix.

    cluster_size_df: 46 clusters × 24 groups (raw nucleus counts per group).
    S: 46 × 27 row-stochastic projection.
    Result is indexed by WMB class, columns = groups; integer-cast for
    downstream MIN_CELLS_PER_GROUP comparisons.
    """
    common = [c for c in cluster_size_df.index if c in S.index]
    if len(common) != len(cluster_size_df.index):
        missing = [c for c in cluster_size_df.index if c not in S.index]
        raise KeyError(
            f"Soft-mass crosswalk missing rows for {len(missing)} clusters "
            f"present in cluster_size_df: {missing[:5]}"
        )
    cs = cluster_size_df.loc[common].to_numpy(dtype=float)  # 46 × 24
    s_arr = S.loc[common].to_numpy(dtype=float)             # 46 × 27
    projected = s_arr.T @ cs  # (27 × 46) @ (46 × 24) = 27 × 24
    return pd.DataFrame(
        np.rint(projected).astype(int),
        index=S.columns,
        columns=cluster_size_df.columns,
    )


def audit_mass_conservation(track_before: DeconvoluatedTrack,
                            track_after: DeconvoluatedTrack,
                            tol: float = 1e-6) -> dict:
    """Compare per-(sample, site) sums before vs. after projection.

    Returns a dict with diagnostic counts. Useful as a one-shot sanity check.
    """
    before = track_before.values
    after = track_after.values
    samples = pd.Index([s for s, _ in before.columns]).unique()

    n_sites = before.shape[0]
    n_samples = len(samples)
    diffs = np.empty((n_sites, n_samples))
    for j, sample in enumerate(samples):
        b = before.xs(sample, axis=1, level="sample").to_numpy(dtype=float)
        a = after.xs(sample, axis=1, level="sample").to_numpy(dtype=float)
        b_sum = np.nansum(b, axis=1)
        a_sum = np.nansum(a, axis=1)
        diffs[:, j] = a_sum - b_sum
    abs_diffs = np.abs(diffs)
    finite = abs_diffs[np.isfinite(abs_diffs)]
    return {
        "n_sites": int(n_sites),
        "n_samples": int(n_samples),
        "max_abs_diff": float(finite.max()) if finite.size else float("nan"),
        "mean_abs_diff": float(finite.mean()) if finite.size else float("nan"),
        "frac_within_tol": float((finite <= tol).mean()) if finite.size else float("nan"),
    }
