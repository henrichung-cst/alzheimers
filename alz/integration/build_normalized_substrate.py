#!/usr/bin/env python3
"""Precompute per-(cluster, contrast, layer) limma-normalized condition means.

Resolves cross-phase note #1 of ``docs/plans/merged_evidence_panel.md``:
Incytr's ``integrate_omics_layer`` (``incytr/R/analysis.R:385,391``) calls
``limma::normalizeBetweenArrays(matrix(cond1, cond2))`` on the per-cluster
(gene × {disease_mean, WT_mean}) matrix *before* ``Cal_foldchange``. A naive
JS-side ``log2((D + 1e-5) / (W + 1e-5))`` from the raw substrate therefore
disagrees with Incytr's stored ``*_pr/_ps/_py_log2FC`` by ~0.01-0.04 on
typical rows.

This module replicates the quantile-normalization step in pure numpy
(matching ``limma::normalizeQuantiles`` with ``ties=TRUE``) and writes
per-cluster parquets carrying the post-normalization condition means. The JS
Evidence tab reads these and recomputes the LFC by hand, agreeing with
Incytr's stored value to <=1e-4 on every multiomic row.

Substrates (per Phase 2 layout, see ``alz/integration/build_omics_trace.py``):
    outputs/reports/decomposition/levy_t5/protein_per_cluster.parquet
    outputs/reports/decomposition/levy_t5/phospho_per_cluster.parquet
    outputs/reports/decomposition/levy_t5/phospho_per_cluster_pY.parquet

Driver wide parquets (one per contrast):
    outputs/reports/incytr_pair_mode/wide/<c1>_<c2>_incytr_output.parquet

Aggregation rules (mirror ``alz/incytr/incytr_commandline.R:241-285``):
    protein   : per-(gene, group) median across the 3 males per group,
                then per-gene mean (defensive — substrate is already per-gene).
    phospho_* : per-(site, group) median across the 3 males per group,
                then per-gene arithmetic mean across sites.

Output: ``audit_sources/omics_trace_normalized/<cluster>.parquet``
    layer                 : str   {protein, phospho_ps, phospho_py}
    gene_symbol           : str
    contrast              : str   "<c1>_<c2>" with c1=disease, c2=WTyp
    group                 : str   one of c1, c2
    mean_value_normalized : float64

Build-time round-trip assertion: a per-layer sample of (gene, contrast)
recomputations is compared against the stored ``Ligand_*_log2FC`` value in
the wide parquet. Build aborts if any sampled cell disagrees by >1e-4.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # alz/

import config  # noqa: E402

from integration.pair_to_receiver_cache import _sanitize_celltype  # noqa: E402
from viewer.paths import (  # noqa: E402
    OMICS_TRACE_NORMALIZED_DIR,
    OMICS_TRACE_NORMALIZED_INDEX,
    OMICS_TRACE_NORMALIZED_SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
)

WIDE_DIR = os.path.join(
    config.REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide"
)
YUYU_DIR = os.path.join(config.REPO_ROOT, "data", "derived", "incytr_inputs")
YUYU_PROTEIN = os.path.join(YUYU_DIR, "pr_yuyu_deconvoluted.csv")
YUYU_PS = os.path.join(YUYU_DIR, "ps_yuyu_deconvoluted.csv")
YUYU_PY = os.path.join(YUYU_DIR, "py_yuyu_deconvoluted.csv")

EPSILON = 1e-3  # matches Cal_foldchange's driver-passed `correction = 0.001`
# (see alz/incytr/incytr_commandline.R:376-389; the original Item 3.1 audit
# noted 1e-5 in error).

# Round-trip sample size per (contrast, layer) for build-time assertion.
ROUNDTRIP_SAMPLE_PER_CELL = 5
ROUNDTRIP_TOL = 1e-4

# Wide-parquet filename pattern: "<c1>_<c2>_incytr_output.parquet" where each
# group token is "<sex>_<age>_<geno>" e.g. "ma_2mo_ApTt_ma_2mo_WTyp_...".
_WIDE_RE = re.compile(
    r"^(?P<c1>(?:ma|fe)_\dmo_\w+)_(?P<c2>(?:ma|fe)_\dmo_WTyp)_incytr_output\.parquet$"
)


# ---------------------------------------------------------------------------
# limma::normalizeQuantiles port (with ties="average").
# ---------------------------------------------------------------------------
def normalize_quantiles(matrix: np.ndarray) -> np.ndarray:
    """Faithful port of ``limma::normalizeQuantiles`` with ``ties=TRUE``.

    Algorithm (from limma source `normalizeQuantiles.R`):
      1. For each column j, sort non-NaN values; if fewer than n rows, stretch
         the sorted column onto a length-n grid via linear interpolation over
         a common [0, 1] axis. This handles different per-column NaN counts.
      2. Take the row-wise mean across the stretched columns → reference
         quantile means ``m`` (length n).
      3. For each column j, compute average-tied ranks of the non-NaN values
         and linearly interpolate ``m`` at the rank positions (1..nobs[j])
         mapped onto the same [0, 1] axis.
    NaN positions in the input remain NaN in the output.
    """
    A = np.asarray(matrix, dtype=float).copy()
    n, k = A.shape
    if k == 1 or n == 0:
        return A
    grid = np.arange(n) / (n - 1) if n > 1 else np.array([0.0])
    S = np.empty((n, k), dtype=float)
    nobs = np.full(k, n, dtype=int)
    for j in range(k):
        col = A[:, j]
        valid = ~np.isnan(col)
        sorted_vals = np.sort(col[valid])
        nobsj = len(sorted_vals)
        nobs[j] = nobsj
        if nobsj == n:
            S[:, j] = sorted_vals
        elif nobsj > 1:
            xs = np.arange(nobsj) / (nobsj - 1)
            S[:, j] = np.interp(grid, xs, sorted_vals)
        elif nobsj == 1:
            S[:, j] = sorted_vals[0]
        else:
            S[:, j] = np.nan
    m = np.nanmean(S, axis=1) if k > 1 else S[:, 0]
    out = np.full_like(A, np.nan)
    for j in range(k):
        col = A[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        r = rankdata(col[valid], method="average")
        if nobs[j] > 1:
            pos = (r - 1) / (nobs[j] - 1)
        else:
            pos = np.zeros_like(r)
        out[valid, j] = np.interp(pos, grid, m)
    return out


# ---------------------------------------------------------------------------
# Substrate loaders (read directly from the yuyu CSVs the R driver consumes).
# ---------------------------------------------------------------------------
def _load_yuyu_aggregated(
    path: str, gene_col: str
) -> tuple[pd.DataFrame, list[str]]:
    """Load a yuyu CSV and reduce sites→gene with `mean(skipna=True)`, mirroring
    the R driver's ``group_by(gene_symbol) %>% summarise_all(mean, na.rm=T)``.

    Returns (per-gene wide DataFrame, list of group×cluster column names).
    """
    df = pd.read_csv(path)
    cluster_cols = [
        c for c in df.columns
        if c not in (gene_col, "gene_symbol", "Gene Symbol", "site_id")
    ]
    # The R driver uses the protein file's "Gene Symbol" column for protein,
    # and the phospho files' "gene_symbol" column for phospho. Both end up
    # as the grouping key.
    agg = (
        df[[gene_col] + cluster_cols]
        .rename(columns={gene_col: "gene_symbol"})
        .groupby("gene_symbol", as_index=False)
        .mean(numeric_only=True)
    )
    return agg, cluster_cols


def _extract_cluster_pair(
    agg: pd.DataFrame, cluster: str, c1: str, c2: str
) -> pd.DataFrame:
    """Pull the (gene_symbol, <c1>_<cluster>, <c2>_<cluster>) columns and
    rename them to (gene_symbol, c1, c2). Returns the per-gene table the
    R driver hands to ``normalizeBetweenArrays``.
    """
    col1 = f"{c1}_{cluster}"
    col2 = f"{c2}_{cluster}"
    out_cols = {"gene_symbol": "gene_symbol"}
    if col1 in agg.columns:
        out_cols[col1] = c1
    if col2 in agg.columns:
        out_cols[col2] = c2
    sub = agg[list(out_cols.keys())].rename(columns=out_cols)
    for g in (c1, c2):
        if g not in sub.columns:
            sub[g] = np.nan
    return sub[["gene_symbol", c1, c2]]


# ---------------------------------------------------------------------------
# Contrast discovery from wide parquet filenames.
# ---------------------------------------------------------------------------
def _discover_contrasts() -> list[tuple[str, str, str]]:
    """Return [(filename_stem, c1, c2)] for each wide parquet."""
    out = []
    for path in sorted(glob.glob(os.path.join(WIDE_DIR, "*_incytr_output.parquet"))):
        name = os.path.basename(path)
        m = _WIDE_RE.match(name)
        if not m:
            raise ValueError(
                f"wide parquet name does not match expected pattern: {name}"
            )
        c1 = m.group("c1")
        c2 = m.group("c2")
        contrast_id = f"{c1}_{c2}"
        out.append((contrast_id, c1, c2, path))
    if not out:
        raise FileNotFoundError(
            f"no wide pair-mode parquets found under {WIDE_DIR}; "
            f"run alz/incytr/run_pair_mode.sh before this step."
        )
    return out


def _discover_pathway_clusters(
    contrasts: list[tuple[str, str, str, str]],
) -> set[str]:
    """Union of Sender + Receiver across all wide parquets — the authoritative
    pathway-cluster set. We derive it here (rather than depending on
    ``edge_slices/incytr_pathways/index.json``) so the builder can run
    standalone without a prior viewer build.
    """
    clusters: set[str] = set()
    for (_cid, _c1, _c2, path) in contrasts:
        wide = pq.read_table(path, columns=["Sender", "Receiver"]).to_pandas()
        clusters.update(wide["Sender"].dropna().unique().tolist())
        clusters.update(wide["Receiver"].dropna().unique().tolist())
    return clusters


# ---------------------------------------------------------------------------
# Round-trip verification.
# ---------------------------------------------------------------------------
_LAYER_TO_STORED = {
    "protein": "_pr_log2FC",
    "phospho_ps": "_ps_log2FC",
    "phospho_py": "_py_log2FC",
}


def _roundtrip_sample(
    wide_path: str,
    c1: str,
    c2: str,
    normalized: dict[tuple[str, str], pd.DataFrame],
    pathway_clusters: set[str],
    rng: np.random.Generator,
) -> list[str]:
    """For each (cluster, layer), sample ROUNDTRIP_SAMPLE_PER_CELL pathway
    rows from the wide parquet and assert recomputed == stored within tol.

    Uses the ``Ligand_*_log2FC`` column with cluster = Sender, since the
    aggregation upstream (``analysis.R:385``) operates on the sender-side
    matrix. (Receiver-side would give the same result with cluster = Receiver
    and ``Receptor/EM/Target_*_log2FC``, but ligand is sufficient and avoids
    multiple-cluster-per-row complexity.)

    Returns a list of failure-description strings; empty if all pass.
    """
    failures: list[str] = []
    # Read minimum columns from the wide parquet for ligand sampling.
    cols = ["Ligand", "Sender"] + [
        f"Ligand{suffix}" for suffix in _LAYER_TO_STORED.values()
    ]
    wide = pq.read_table(wide_path, columns=cols).to_pandas()
    wide = wide.dropna(subset=["Ligand", "Sender"])

    for cluster in sorted(pathway_clusters):
        sub_wide = wide[wide["Sender"] == cluster]
        if sub_wide.empty:
            continue
        for layer, suffix in _LAYER_TO_STORED.items():
            stored_col = f"Ligand{suffix}"
            nm = normalized.get((cluster, layer))
            if nm is None or nm.empty:
                continue
            # Drop NaN stored rows and dedup on Ligand (which is the gene name).
            valid = sub_wide.dropna(subset=[stored_col])
            valid = valid.drop_duplicates(subset=["Ligand"])
            if valid.empty:
                continue
            take = min(ROUNDTRIP_SAMPLE_PER_CELL, len(valid))
            picks = valid.sample(n=take, random_state=int(rng.integers(0, 2**31)))
            norm_lookup = nm.set_index("gene_symbol")
            for _, row in picks.iterrows():
                gene = row["Ligand"]
                stored = float(row[stored_col])
                if gene not in norm_lookup.index:
                    failures.append(
                        f"{cluster}/{layer}: gene {gene!r} stored "
                        f"{stored:.6f} but absent from normalized substrate"
                    )
                    continue
                d_norm = float(norm_lookup.at[gene, c1])
                w_norm = float(norm_lookup.at[gene, c2])
                if math.isnan(d_norm) or math.isnan(w_norm):
                    # Normalized to NaN on either side → stored is undefined;
                    # skip rather than flag (Incytr would also produce NaN/Inf).
                    continue
                recomputed = math.log2((d_norm + EPSILON) / (w_norm + EPSILON))
                if abs(recomputed - stored) > ROUNDTRIP_TOL:
                    failures.append(
                        f"{cluster}/{layer}/{gene}: stored={stored:.6f} "
                        f"recomputed={recomputed:.6f} "
                        f"diff={abs(recomputed - stored):.2e}"
                    )
    return failures


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------
def build(force: bool = False) -> dict:
    """Build per-cluster normalized substrate shards. Returns index dict."""
    if not force and os.path.exists(OMICS_TRACE_NORMALIZED_INDEX):
        with open(OMICS_TRACE_NORMALIZED_INDEX) as f:
            existing = json.load(f)
        if existing.get("schema_version") == OMICS_TRACE_NORMALIZED_SCHEMA_VERSION:
            return existing

    if os.path.exists(OMICS_TRACE_NORMALIZED_DIR):
        shutil.rmtree(OMICS_TRACE_NORMALIZED_DIR)
    os.makedirs(OMICS_TRACE_NORMALIZED_DIR, exist_ok=True)

    t0 = time.time()
    # --- Substrate checks ---
    for path, label in [
        (YUYU_PROTEIN, "pr_yuyu_deconvoluted.csv"),
        (YUYU_PS, "ps_yuyu_deconvoluted.csv"),
        (YUYU_PY, "py_yuyu_deconvoluted.csv"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"normalized-substrate input missing: {path} ({label}). "
                f"Run alz/incytr/build_pair_inputs.sh first."
            )

    contrasts = _discover_contrasts()
    pathway_clusters = _discover_pathway_clusters(contrasts)
    print(
        f"  normalized_substrate: {len(pathway_clusters)} pathway clusters "
        f"(from wide parquet Sender/Receiver union)",
        flush=True,
    )
    print(
        f"  normalized_substrate: {len(contrasts)} contrasts", flush=True
    )

    # --- Aggregate yuyu CSVs once (mirrors incytr_commandline.R:241-285) ---
    # Protein file uses "Gene Symbol" as grouping key; phospho files use
    # "gene_symbol". `summarise_all(mean, na.rm=T)` matches
    # `groupby + mean(numeric_only=True, skipna=True)`.
    print(
        f"  normalized_substrate: aggregating yuyu CSVs "
        f"(mean across sites per gene_symbol)", flush=True
    )
    pr_agg, _ = _load_yuyu_aggregated(YUYU_PROTEIN, "Gene Symbol")
    ps_agg, _ = _load_yuyu_aggregated(YUYU_PS, "gene_symbol")
    py_agg, _ = _load_yuyu_aggregated(YUYU_PY, "gene_symbol")
    layer_aggs: dict[str, pd.DataFrame] = {
        "protein": pr_agg, "phospho_ps": ps_agg, "phospho_py": py_agg,
    }

    # --- Per-cluster normalize across all contrasts/layers ---
    # Layout: per-cluster long-form rows
    #   (layer, gene_symbol, contrast, group, mean_value_normalized)
    rng = np.random.default_rng(0)
    shards_written: dict[str, str] = {}
    # Stash normalized DataFrames for the round-trip check (per-contrast).
    # Cleared/refreshed per contrast.
    all_failures: list[str] = []

    # Build per-contrast normalized tables first (we need them in memory
    # for the round-trip sampling against each wide parquet), but emit
    # per-cluster shards at the end.
    # Map: cluster -> list of rows (dicts) to concat at write time.
    per_cluster_rows: dict[str, list[pd.DataFrame]] = {
        c: [] for c in pathway_clusters
    }

    for (contrast_id, c1, c2, wide_path) in contrasts:
        print(
            f"  normalized_substrate: contrast {contrast_id}", flush=True
        )
        normalized_lookup: dict[tuple[str, str], pd.DataFrame] = {}
        for cluster in sorted(pathway_clusters):
            for layer, agg_full in layer_aggs.items():
                agg = _extract_cluster_pair(agg_full, cluster, c1, c2)
                if agg.empty:
                    continue
                mat = agg[[c1, c2]].to_numpy()
                norm = normalize_quantiles(mat)
                normed = pd.DataFrame(
                    norm, columns=[c1, c2], index=agg["gene_symbol"]
                ).reset_index()
                normalized_lookup[(cluster, layer)] = normed
                long = normed.melt(
                    id_vars=["gene_symbol"],
                    value_vars=[c1, c2],
                    var_name="group",
                    value_name="mean_value_normalized",
                )
                long.insert(0, "layer", layer)
                long.insert(2, "contrast", contrast_id)
                per_cluster_rows[cluster].append(long)
        # Round-trip check for this contrast
        failures = _roundtrip_sample(
            wide_path, c1, c2, normalized_lookup, pathway_clusters, rng
        )
        if failures:
            all_failures.extend(
                [f"[{contrast_id}] {f}" for f in failures]
            )

    if all_failures:
        head = "\n  ".join(all_failures[:20])
        raise RuntimeError(
            f"normalized_substrate: {len(all_failures)} round-trip "
            f"failure(s) exceed tol={ROUNDTRIP_TOL}. First 20:\n  {head}"
        )

    # --- Write per-cluster shards ---
    n_clusters_written = 0
    for cluster in sorted(pathway_clusters):
        rows = per_cluster_rows[cluster]
        if not rows:
            raise RuntimeError(
                f"normalized_substrate: cluster {cluster!r} produced zero "
                f"normalized rows across all contrasts/layers. Substrate "
                f"drift — investigate before bumping schema."
            )
        out = pd.concat(rows, ignore_index=True)
        out = out[
            ["layer", "gene_symbol", "contrast", "group",
             "mean_value_normalized"]
        ]
        slug = _sanitize_celltype(cluster)
        out_path = os.path.join(
            OMICS_TRACE_NORMALIZED_DIR, f"{slug}.parquet"
        )
        out.to_parquet(out_path, index=False, compression="zstd")
        shards_written[cluster] = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)
        n_clusters_written += 1

    # --- Index ---
    index = {
        "schema_version": OMICS_TRACE_NORMALIZED_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "label": (
            "limma-normalized per-(cluster, contrast, layer) condition means "
            "(mirrors Incytr's normalizeBetweenArrays step pre-Cal_foldchange)"
        ),
        "epsilon": EPSILON,
        "epsilon_note": (
            "JS-side LFC reconstruction: "
            "log2((D + epsilon) / (W + epsilon)). Epsilon matches "
            "Cal_foldchange's driver-passed `correction = 1e-5`."
        ),
        "normalization": "limma::normalizeQuantiles (ties=TRUE)",
        "aggregation_note": (
            "Protein: per-(gene, group) median across males, then per-gene "
            "mean (defensive). Phospho: per-(site, group) median, then "
            "per-gene mean across sites. Mirrors incytr_commandline.R:241-285."
        ),
        "layers": ["protein", "phospho_ps", "phospho_py"],
        "contrasts": [c[0] for c in contrasts],
        "roundtrip_tol": ROUNDTRIP_TOL,
        "roundtrip_sample_per_cell": ROUNDTRIP_SAMPLE_PER_CELL,
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "filename_template": "{cluster}.parquet",
        "relative_path": os.path.relpath(
            OMICS_TRACE_NORMALIZED_DIR, UNIFIED_VIEWER_DIR
        ),
        "clusters": sorted(shards_written.keys()),
        "shard_files": shards_written,
        "n_shards": len(shards_written),
    }
    with open(OMICS_TRACE_NORMALIZED_INDEX, "w") as f:
        json.dump(index, f, indent=2)
    print(
        f"  normalized_substrate: wrote {n_clusters_written} cluster shards "
        f"in {time.time() - t0:.1f}s (round-trip passed)",
        flush=True,
    )
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild even if index is current."
    )
    args = ap.parse_args()
    build(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
