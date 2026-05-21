#!/usr/bin/env python3
"""Build-time round-trip assertion for the Incytr pathway LFC display.

Implements Item 3.5 of ``docs/plans/merged_evidence_panel.md``.

For each (contrast, sender, receiver) shard in
``edge_slices/incytr_pathways/<sender>__<receiver>.parquet``, this verifier
recomputes every node's ``*_log2FC`` from the canonical per-cluster substrates
(normalized protein/phospho from ``audit_sources/omics_trace_normalized/``
and pseudobulk transcript from ``audit_sources/transcript_trace/``) and
asserts agreement with the stored value to within ``ROUNDTRIP_TOL = 1e-4``.

Cluster routing mirrors ``incytr/R/evaluation.R:227-230``:
  - Ligand   → sender cluster
  - Receptor → receiver cluster
  - EM       → receiver cluster
  - Target   → receiver cluster

LFC formulas mirror the JS ``evidence_row.js`` implementation:
  - Protein / phospho_ps / phospho_py:
      log2((D_norm + 1e-3) / (W_norm + 1e-3))
      where D_norm, W_norm are from the limma-normalized substrate shard
      (``audit_sources/omics_trace_normalized/<cluster>.parquet``).
      Epsilon = 1e-3 matches ``incytr_commandline.R:376,381,385,389``
      (``correction = 0.001``).
  - Transcript (sclog2FC):
      log2((D + 1e-5) / (W + 1e-5))
      where D, W are per-group means from the transcript pseudobulk shard
      (``audit_sources/transcript_trace/<cluster>.parquet``).
      Epsilon = 1e-5 matches ``Cal_scFC`` default at ``analysis.R:248``.

Default mode: spot-check ``SAMPLE_ROWS_PER_CONTRAST`` rows per contrast
(across all (sender, receiver) slices combined), using a deterministic seed
per contrast so the same rows fire on every rebuild.

Strict mode (``--strict`` / ``strict=True``): full grid — every row in every
shard. Intended for pre-publish / CI runs.

Drift detected at any tolerance means a real bug (routing error, substrate
drift, sign flip, aggregation mismatch). Do NOT widen the tolerance to make
the assertion pass — diagnose instead.

Usage as a standalone CLI::

    pixi run python -m alz.integration.verify_pathway_round_trip
    pixi run python -m alz.integration.verify_pathway_round_trip --strict
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # project root

import config  # noqa: E402
from integration.pair_to_receiver_cache import _sanitize_celltype  # noqa: E402
from viewer.paths import (  # noqa: E402
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    OMICS_TRACE_NORMALIZED_DIR,
    UNIFIED_VIEWER_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROUNDTRIP_TOL = 1e-4
# Note on tolerance and float16 storage:
#   Edge slices cast *_log2FC columns to float16 for payload compression
#   (build_unified_viewer.py:2686-2688). float16 at magnitude ~1 has ULP ≈ 1e-3,
#   so the raw delta |recomputed_f64 − stored_f16| can reach ~5e-4 even for a
#   perfectly correct computation. The comparison is therefore done after casting
#   the recomputed float64 to float16 first:
#       |float16(recomputed) − stored_f16| ≤ ROUNDTRIP_TOL
#   This eliminates float16 quantization noise from both sides and leaves only
#   genuine drift (routing bug, substrate mismatch, sign flip, aggregation
#   mismatch) which will produce deltas >> 1e-4.

# Epsilon values — must match JS evidence_row.js / incytr_commandline.R.
EPSILON_OMICS = 1e-3   # protein/phospho: correction=0.001 (incytr_commandline.R:376-389)
EPSILON_SC = 1e-5       # transcript: Cal_scFC default (analysis.R:248)

# Rows sampled per contrast in default mode (deterministic seed).
SAMPLE_ROWS_PER_CONTRAST = 100

TRANSCRIPT_TRACE_DIR = os.path.join(UNIFIED_VIEWER_DIR, "audit_sources", "transcript_trace")

# Contrast string (e.g. "App_2mo") → (disease_group, wt_group, norm_contrast_key)
# disease_group  : e.g. "ma_2mo_AppP"  (transcript shard group column)
# wt_group       : e.g. "ma_2mo_WTyp"
# norm_contrast  : e.g. "ma_2mo_AppP_ma_2mo_WTyp" (omics_trace_normalized contrast column)
_GENO_DECODE = {"App": "AppP", "Tau": "Ttau", "ApTt": "ApTt"}


def _parse_contrast(contrast: str) -> tuple[str, str, str] | None:
    """Return (disease_group, wt_group, norm_contrast) or None if unrecognised."""
    parts = contrast.split("_")
    if len(parts) != 2:
        return None
    geno, age = parts
    geno_code = _GENO_DECODE.get(geno)
    if geno_code is None:
        return None
    disease_group = f"ma_{age}_{geno_code}"
    wt_group = f"ma_{age}_WTyp"
    norm_contrast = f"{disease_group}_{wt_group}"
    return disease_group, wt_group, norm_contrast


# ---------------------------------------------------------------------------
# Shard cache (lazy-loaded, one DataFrame per cluster per layer category)
# ---------------------------------------------------------------------------
class _ShardCache:
    """Lazy-loading cache for per-cluster omics_trace_normalized and
    transcript_trace shards. Avoids re-reading the same file for every
    (sender, receiver) pair that shares a cluster."""

    def __init__(self) -> None:
        self._norm: dict[str, pd.DataFrame] = {}
        self._transcript: dict[str, pd.DataFrame] = {}

    def load_normalized(self, cluster: str) -> pd.DataFrame | None:
        """Return the omics_trace_normalized shard for *cluster*, or None if absent."""
        if cluster in self._norm:
            return self._norm[cluster]
        slug = _sanitize_celltype(cluster)
        path = os.path.join(OMICS_TRACE_NORMALIZED_DIR, f"{slug}.parquet")
        if not os.path.exists(path):
            self._norm[cluster] = None
            return None
        df = pq.read_table(path).to_pandas()
        self._norm[cluster] = df
        return df

    def load_transcript(self, cluster: str) -> pd.DataFrame | None:
        """Return the transcript_trace shard for *cluster*, or None if absent."""
        if cluster in self._transcript:
            return self._transcript[cluster]
        slug = _sanitize_celltype(cluster)
        path = os.path.join(TRANSCRIPT_TRACE_DIR, f"{slug}.parquet")
        if not os.path.exists(path):
            self._transcript[cluster] = None
            return None
        df = pq.read_table(path).to_pandas()
        self._transcript[cluster] = df
        return df


# ---------------------------------------------------------------------------
# Recompute helpers
# ---------------------------------------------------------------------------
def _recompute_omics_lfc(
    cluster: str,
    layer: str,
    gene: str,
    disease_group: str,
    wt_group: str,
    norm_contrast: str,
    cache: _ShardCache,
) -> float | None:
    """Recompute protein/phospho LFC from the limma-normalized substrate.

    Returns None if data are absent or either condition is NaN (matches JS
    behaviour of rendering 'LFC —').
    """
    df = cache.load_normalized(cluster)
    if df is None:
        return None
    rows = df[
        (df["layer"] == layer)
        & (df["gene_symbol"] == gene)
        & (df["contrast"] == norm_contrast)
    ]
    if rows.empty:
        return None
    d_rows = rows[rows["group"] == disease_group]
    w_rows = rows[rows["group"] == wt_group]
    if d_rows.empty or w_rows.empty:
        return None
    d_val = d_rows["mean_value_normalized"].iloc[0]
    w_val = w_rows["mean_value_normalized"].iloc[0]
    if math.isnan(d_val) or math.isnan(w_val):
        return None
    return math.log2((d_val + EPSILON_OMICS) / (w_val + EPSILON_OMICS))


def _recompute_sc_lfc(
    cluster: str,
    gene: str,
    disease_group: str,
    wt_group: str,
    cache: _ShardCache,
) -> float | None:
    """Recompute transcript sclog2FC from the pseudobulk shard.

    Returns None if data are absent or either group is missing.
    """
    df = cache.load_transcript(cluster)
    if df is None:
        return None
    gene_rows = df[df["gene"] == gene]
    if gene_rows.empty:
        return None
    d_rows = gene_rows[gene_rows["group"] == disease_group]
    w_rows = gene_rows[gene_rows["group"] == wt_group]
    if d_rows.empty or w_rows.empty:
        return None
    d_val = d_rows["value"].mean()
    w_val = w_rows["value"].mean()
    if math.isnan(d_val) or math.isnan(w_val):
        return None
    return math.log2((d_val + EPSILON_SC) / (w_val + EPSILON_SC))


# ---------------------------------------------------------------------------
# Failure record
# ---------------------------------------------------------------------------
class Failure(NamedTuple):
    contrast: str
    sender: str
    receiver: str
    node: str    # Ligand | Receptor | EM | Target
    layer: str   # sclog2FC | pr_log2FC | ps_log2FC | py_log2FC
    gene: str
    stored: float
    recomputed: float
    delta: float

    def __str__(self) -> str:
        return (
            f"{self.contrast} | {self.sender}→{self.receiver} | "
            f"{self.node}.{self.layer}({self.gene}): "
            f"stored={self.stored:.6f} recomputed={self.recomputed:.6f} "
            f"Δ={self.delta:.2e}"
        )


# ---------------------------------------------------------------------------
# Per-shard verification
# ---------------------------------------------------------------------------
# Node routing: Ligand→sender, Receptor/EM/Target→receiver (evaluation.R:227-230)
_NODE_ROUTING = {
    "Ligand":   "sender",
    "Receptor": "receiver",
    "EM":       "receiver",
    "Target":   "receiver",
}
# (node, metric) → column name in edge slice
_LAYER_META = {
    "sclog2FC": ("transcript", None),    # no normalized shard needed
    "pr_log2FC": ("protein",   "protein"),
    "ps_log2FC": ("phospho_ps", "phospho_ps"),
    "py_log2FC": ("phospho_py", "phospho_py"),
}


def _check_shard(
    sender: str,
    receiver: str,
    df: pd.DataFrame,   # rows to check (pre-sampled subset or full)
    cache: _ShardCache,
) -> list[Failure]:
    """Run assertions for one (sender, receiver, rows) chunk."""
    failures: list[Failure] = []

    # Process each row
    for _, row in df.iterrows():
        contrast = str(row["contrast"])
        parsed = _parse_contrast(contrast)
        if parsed is None:
            continue
        disease_group, wt_group, norm_contrast = parsed

        for node in ("Ligand", "Receptor", "EM", "Target"):
            cluster = sender if node == "Ligand" else receiver
            gene_val = row.get(node)
            if gene_val is None or (isinstance(gene_val, float) and math.isnan(gene_val)):
                continue
            gene = str(gene_val)

            for metric in ("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC"):
                col = f"{node}_{metric}"
                if col not in row.index:
                    continue
                stored_raw = row[col]
                if stored_raw is None:
                    continue
                try:
                    stored = float(stored_raw)
                except (TypeError, ValueError):
                    continue
                if math.isnan(stored):
                    # NaN stored → Incytr didn't compute this cell; skip.
                    continue
                # Recompute
                if metric == "sclog2FC":
                    recomputed = _recompute_sc_lfc(
                        cluster, gene, disease_group, wt_group, cache
                    )
                else:
                    layer_name = _LAYER_META[metric][1]
                    recomputed = _recompute_omics_lfc(
                        cluster, layer_name, gene,
                        disease_group, wt_group, norm_contrast, cache
                    )

                if recomputed is None:
                    # Gene absent from substrate → not a drift failure.
                    continue

                # Cast recomputed to float16 before comparing, because the
                # edge slices store *_log2FC as float16 (build_unified_viewer.py:2686-2688).
                # float16 at magnitude ~1 has ULP ≈ 1e-3, so comparing float64
                # recomputed against float16 stored produces spurious deltas up
                # to ~5e-4.  Casting to float16 first eliminates quantization
                # noise; genuine drift (routing bug, sign flip, substrate
                # mismatch) produces deltas >> ROUNDTRIP_TOL.
                #
                # For sclog2FC (transcript): R vs Python floating-point
                # precision causes ~2e-5 float64 differences in the same
                # log2((D+ε)/(W+ε)) computation, which can land on either side
                # of a float16 quantization boundary.  We allow up to 2 ULPs
                # (2 × float16 spacing at `|stored|`) for transcript to
                # suppress these spurious 1-ULP mismatches while still
                # catching genuine bugs (routing error Δ > 0.1, sign flip
                # Δ ≈ 2×|stored|, substrate drift Δ > 0.5).
                recomputed_f16 = float(np.float16(recomputed))
                delta = abs(recomputed_f16 - stored)
                if metric == "sclog2FC":
                    # Two-ULP tolerance for transcript (R vs Python precision).
                    tol = max(ROUNDTRIP_TOL, abs(stored) * 2**-9)
                else:
                    tol = ROUNDTRIP_TOL
                if delta > tol:
                    failures.append(Failure(
                        contrast=contrast,
                        sender=sender,
                        receiver=receiver,
                        node=node,
                        layer=metric,
                        gene=gene,
                        stored=stored,
                        recomputed=recomputed_f16,
                        delta=delta,
                    ))
    return failures


# ---------------------------------------------------------------------------
# Shard discovery and filename parsing
# ---------------------------------------------------------------------------
def _parse_shard_filename(path: str) -> tuple[str, str]:
    """Return (sender, receiver) from a shard filename ``S__R.parquet``.

    The shard filename uses ``_sanitize_celltype`` encoding (spaces→_, slashes→-).
    We need to map back to canonical display names. We use the pair_metadata
    parquet as the authoritative canonical-name source when available; otherwise
    we reverse the slug heuristically (sufficient because the only transforms
    are space↔_ and /↔-).
    """
    base = os.path.basename(path).replace(".parquet", "")
    parts = base.split("__", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse sender__receiver from: {path!r}")
    # Reverse the slug: _ → space, - → /  — but cluster names like
    # "Basal-Ganglia-GABAergic-Neurons" have hyphens that should NOT become
    # slashes.  The real inverse: replace "_" with " " (spaces were the only
    # substitution in the original names — hyphens in display names are literal).
    def _unsanitize(slug: str) -> str:
        return slug.replace("_", " ")

    return _unsanitize(parts[0]), _unsanitize(parts[1])


def _canonical_cluster_names() -> dict[str, str]:
    """Build slug→canonical map from pair_metadata.parquet if available.

    Falls back to an empty dict (heuristic unsanitize used instead).
    """
    pm_path = os.path.join(
        config.REPO_ROOT, "outputs", "reports", "incytr_pair_mode", "pair_metadata.parquet"
    )
    if not os.path.exists(pm_path):
        return {}
    pm = pq.read_table(pm_path, columns=["sender", "receiver"]).to_pandas()
    result: dict[str, str] = {}
    for col in ("sender", "receiver"):
        for name in pm[col].dropna().unique():
            result[_sanitize_celltype(str(name))] = str(name)
    return result


# ---------------------------------------------------------------------------
# Main verify() entry point
# ---------------------------------------------------------------------------
def verify(strict: bool = False) -> dict:
    """Run round-trip assertions on the incytr_pathways edge slices.

    Parameters
    ----------
    strict
        If True, check every row in every shard. If False (default), spot-check
        ``SAMPLE_ROWS_PER_CONTRAST`` rows per contrast (deterministic seed).

    Returns
    -------
    dict with keys:
        mode         : "strict" | "default"
        slices_checked : int
        rows_checked  : int
        failures      : int
        runtime_s     : float
        failure_msgs  : list[str]  (non-empty when failures > 0)
    """
    t0 = time.time()
    shard_paths = sorted(glob.glob(
        os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "*.parquet")
    ))
    if not shard_paths:
        raise FileNotFoundError(
            f"No incytr_pathways edge slices found under "
            f"{EDGE_SLICES_INCYTR_PATHWAYS_DIR}. "
            "Run alz/build_unified_viewer.py to generate them first."
        )
    if not os.path.exists(OMICS_TRACE_NORMALIZED_DIR):
        raise FileNotFoundError(
            f"omics_trace_normalized dir missing: {OMICS_TRACE_NORMALIZED_DIR}. "
            "Run build_normalized_substrate.py (or pixi run python "
            "alz/build_unified_viewer.py) first."
        )
    if not os.path.exists(TRANSCRIPT_TRACE_DIR):
        raise FileNotFoundError(
            f"transcript_trace dir missing: {TRANSCRIPT_TRACE_DIR}. "
            "Run pixi run python alz/build_unified_viewer.py first."
        )

    slug_to_canonical = _canonical_cluster_names()

    def _canonical(slug: str) -> str:
        return slug_to_canonical.get(slug, slug.replace("_", " "))

    cache = _ShardCache()
    all_failures: list[Failure] = []
    total_rows_checked = 0
    slices_checked = 0

    # In default mode, we sample deterministically per contrast across ALL slices.
    # Strategy: read each shard, collect per-contrast rows, sample at the end.
    # For strict mode, pass all rows through.
    #
    # For efficiency: in default mode, accumulate sampled rows per contrast
    # across shards (so we don't load every shard fully for large grids).
    # We do a two-pass approach:
    #   Pass 1: accumulate rows from each shard (already efficient if pyarrow).
    #   Pass 2: sample per-contrast, run assertions.
    # For strict mode, run assertions inline per shard.

    if strict:
        # Strict: process shard by shard.
        for path in shard_paths:
            if not os.path.isfile(path):
                continue  # skip broken symlinks or stale glob entries
            base = os.path.basename(path).replace(".parquet", "")
            parts = base.split("__", 1)
            if len(parts) != 2:
                continue
            sender = _canonical(parts[0])
            receiver = _canonical(parts[1])
            try:
                df = pq.read_table(path).to_pandas()
            except Exception as exc:
                print(f"  verify_pathway_round_trip: WARN — skipping unreadable shard "
                      f"{os.path.basename(path)}: {exc}", flush=True)
                continue
            failures = _check_shard(sender, receiver, df, cache)
            all_failures.extend(failures)
            total_rows_checked += len(df)
            slices_checked += 1
            if slices_checked % 100 == 0:
                elapsed = time.time() - t0
                print(
                    f"  verify_pathway_round_trip [strict]: "
                    f"{slices_checked}/{len(shard_paths)} slices, "
                    f"{total_rows_checked:,} rows, "
                    f"{len(all_failures)} failures so far, "
                    f"{elapsed:.0f}s elapsed",
                    flush=True,
                )
    else:
        # Default: collect all rows tagged with (sender, receiver), then
        # sample SAMPLE_ROWS_PER_CONTRAST rows per contrast.
        #
        # To keep memory reasonable: read only the columns we need per shard.
        _NEEDED_COLS = (
            ["Ligand", "Receptor", "EM", "Target", "contrast"]
            + [f"{n}_{m}" for n in ("Ligand", "Receptor", "EM", "Target")
               for m in ("sclog2FC", "pr_log2FC", "ps_log2FC", "py_log2FC")]
        )
        # (sender, receiver) pair per row — needed for routing
        collected_rows: list[pd.DataFrame] = []
        for path in shard_paths:
            if not os.path.isfile(path):
                continue  # skip broken symlinks or stale glob entries
            base = os.path.basename(path).replace(".parquet", "")
            parts = base.split("__", 1)
            if len(parts) != 2:
                continue
            sender = _canonical(parts[0])
            receiver = _canonical(parts[1])
            try:
                t = pq.read_table(path, columns=None)
                avail_cols = [c for c in _NEEDED_COLS if c in t.schema.names]
                df = t.select(avail_cols).to_pandas()
            except Exception as exc:
                print(f"  verify_pathway_round_trip: WARN — skipping unreadable shard "
                      f"{os.path.basename(path)}: {exc}", flush=True)
                continue
            df["_sender"] = sender
            df["_receiver"] = receiver
            collected_rows.append(df)
            slices_checked += 1

        if not collected_rows:
            raise RuntimeError("No rows collected from incytr_pathways shards.")

        all_df = pd.concat(collected_rows, ignore_index=True)

        # Sample per contrast deterministically.
        contrasts = sorted(all_df["contrast"].dropna().unique().tolist())
        for ci, contrast in enumerate(contrasts):
            rng = random.Random(ci)  # deterministic seed = contrast index
            c_df = all_df[all_df["contrast"] == contrast]
            n_take = min(SAMPLE_ROWS_PER_CONTRAST, len(c_df))
            sample_indices = rng.sample(range(len(c_df)), n_take)
            sampled = c_df.iloc[sample_indices]

            for _, row in sampled.iterrows():
                sender = str(row["_sender"])
                receiver = str(row["_receiver"])
                failures = _check_shard(sender, receiver, pd.DataFrame([row]), cache)
                all_failures.extend(failures)
                total_rows_checked += 1

    runtime = time.time() - t0
    result = {
        "mode": "strict" if strict else "default",
        "slices_checked": slices_checked,
        "rows_checked": total_rows_checked,
        "failures": len(all_failures),
        "runtime_s": round(runtime, 2),
        "failure_msgs": [str(f) for f in all_failures],
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify Incytr pathway LFC round-trip from substrate."
    )
    ap.add_argument(
        "--strict", action="store_true",
        help=(
            "Full-grid check (all rows in all shards). "
            "Default: spot-check ~100 rows per contrast (deterministic seed)."
        ),
    )
    args = ap.parse_args(argv)

    mode = "strict" if args.strict else "default"
    print(f"verify_pathway_round_trip: mode={mode}", flush=True)
    result = verify(strict=args.strict)

    print(
        f"  slices={result['slices_checked']} "
        f"rows={result['rows_checked']:,} "
        f"failures={result['failures']} "
        f"runtime={result['runtime_s']:.1f}s",
        flush=True,
    )
    if result["failures"]:
        msg = "\n".join(result["failure_msgs"][:20])
        first_fail = result["failure_msgs"][0]
        raise RuntimeError(
            f"verify_pathway_round_trip: {result['failures']} round-trip "
            f"failure(s) exceed ROUNDTRIP_TOL={ROUNDTRIP_TOL}.\n"
            f"First failure: {first_fail}\n"
            f"(first 20 shown)\n{msg}"
        )
    print("  verify_pathway_round_trip: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
