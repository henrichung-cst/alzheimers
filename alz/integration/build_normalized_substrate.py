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

Aggregation rules (mirror ``alz/incytr_pair/incytr_commandline.R:241-285``):
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

This module only *produces* the substrate. Correctness of the JS-side LFC
reconstruction against Incytr's stored ``*_log2FC`` is asserted by the dedicated
sampled harness ``alz/integration/verify_pathway_round_trip.py`` (run inside the
viewer build) — the recompute lives there, not duplicated here.
"""

from __future__ import annotations

import argparse
import glob
import json
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
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root

from alz.shared import config  # noqa: E402
from alz.shared.incytr_constants import EPSILON_OMICS  # noqa: E402

from incytr_pair.pair_to_receiver_cache import _sanitize_celltype  # noqa: E402
from viewer.paths import (  # noqa: E402
    OMICS_TRACE_NORMALIZED_DIR,
    OMICS_TRACE_NORMALIZED_INDEX,
    OMICS_TRACE_NORMALIZED_SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
)

# Pair-mode wide parquets — the per-group protein/phospho values this substrate
# normalizes must come from the SAME run that produced the shards the viewer
# round-trips against. Resolved from INCYTR_PAIR_MODE_INPUT_DIR (the env var the
# viewer build reads at build_unified_viewer.py:1993), defaulting to the live
# filtered nboot=0 set. The old nboot=100 `wide/` is superseded — not a fallback.
WIDE_DIR = os.environ.get(
    "INCYTR_PAIR_MODE_INPUT_DIR",
    os.path.join(config.REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "wide_nboot0"),
)
YUYU_DIR = os.path.join(config.REPO_ROOT, "data", "derived", "incytr_inputs")
YUYU_PROTEIN = os.path.join(YUYU_DIR, "pr_yuyu_deconvoluted.csv")
YUYU_PS = os.path.join(YUYU_DIR, "ps_yuyu_deconvoluted.csv")
YUYU_PY = os.path.join(YUYU_DIR, "py_yuyu_deconvoluted.csv")

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
            f"run alz/incytr_pair/run_pair_mode.sh before this step."
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
                f"Run alz/incytr_pair/build_pair_inputs.sh first."
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
    # sce4 parity override #2: floor deconvoluted PROTEIN values <1 to 1 BEFORE
    # quantile-normalization, mirroring incytr_commandline.R `floor_pr`
    # (pr_1/pr_2 only — ps/py are NOT floored). Without this the ~1e-5
    # deconvolution residuals survive into the normalized substrate and produce
    # ~15-log2 one-sided outliers that the scored pr_log2FC (floored) never sees.
    _pr_num = [c for c in pr_agg.columns if c != "gene_symbol"]
    pr_agg[_pr_num] = pr_agg[_pr_num].clip(lower=1.0)
    layer_aggs: dict[str, pd.DataFrame] = {
        "protein": pr_agg, "phospho_ps": ps_agg, "phospho_py": py_agg,
    }

    # --- Per-cluster normalize across all contrasts/layers ---
    # Layout: per-cluster long-form rows
    #   (layer, gene_symbol, contrast, group, mean_value_normalized)
    # Correctness of the JS reconstruction against Incytr's stored *_log2FC is
    # asserted downstream by verify_pathway_round_trip.py (sampled, in the
    # viewer build) — not recomputed here.
    shards_written: dict[str, str] = {}
    # Map: cluster -> list of rows (dicts) to concat at write time.
    per_cluster_rows: dict[str, list[pd.DataFrame]] = {
        c: [] for c in pathway_clusters
    }

    for (contrast_id, c1, c2, _wide_path) in contrasts:
        print(
            f"  normalized_substrate: contrast {contrast_id}", flush=True
        )
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
                long = normed.melt(
                    id_vars=["gene_symbol"],
                    value_vars=[c1, c2],
                    var_name="group",
                    value_name="mean_value_normalized",
                )
                long.insert(0, "layer", layer)
                long.insert(2, "contrast", contrast_id)
                per_cluster_rows[cluster].append(long)

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
        "epsilon": EPSILON_OMICS,
        "epsilon_note": (
            "JS-side LFC reconstruction: "
            "log2((D + epsilon) / (W + epsilon)). Epsilon = 1e-3 matches "
            "incytr_commandline.R's explicit `correction = 0.001` passed to "
            "Cal_foldchange for all three omics layers (pr/ps/py). "
            "Transcript uses a separate epsilon = 0.01 "
            "(`Cal_scFC(correction = 0.01)` at incytr_commandline.R:435, "
            "sce4-parity override of the analysis.R:248 default 1e-5) "
            "applied directly in the JS without this substrate."
        ),
        "normalization": "limma::normalizeQuantiles (ties=TRUE)",
        "aggregation_note": (
            "Protein: per-(gene, group) median across males, then per-gene "
            "mean (defensive). Phospho: per-(site, group) median, then "
            "per-gene mean across sites. Mirrors incytr_commandline.R:241-285."
        ),
        "layers": ["protein", "phospho_ps", "phospho_py"],
        "contrasts": [c[0] for c in contrasts],
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
        f"in {time.time() - t0:.1f}s",
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
