#!/usr/bin/env python3
"""Build per-cluster, per-SAMPLE protein + phospho deconvolution shards for the
5xFAD contexts of the Incytr Pathways "Evidence" panel.

5xFAD analog of ``build_omics_trace.py``. Where the Song builder ships
per-animal decomposed abundances, this builder forward-projects each 5xFAD
*sample* (not just the condition mean) into per-cell-type abundance so the
Evidence panel can draw per-replicate dot-bars exactly as Song does.

Provenance — identical multiplicative deconvolution to ``fivexfad_decompose``:

    P_c(key, sample) = (N_total[cond] / N_per[(cl, cond)])
                       × 2**log2(key, sample)
                       × share[cond][cl][gene(key)]

where ``cond = group_map[sample] = "<geno>_<age>mo"`` and the per-condition
``share`` matrix + cell counts are the *same* objects
``fivexfad_decompose`` consumed to write the condition-level
``{pr,ps,py}_deconvoluted.csv`` files behind the Incytr LFC chips. Because
``_linear_group_bulk`` builds the condition bulk as ``mean_sample(2**log2)``
and the share/size-factor are constant across a condition's samples, the mean
over a condition's per-sample ``P_c`` is identically the condition-level
deconvoluted value. The build asserts this reconciliation per (key, cond, cl)
to a relative 1e-6 — guaranteeing the dot-bars and the LFC chip describe the
same data. This is the 5xFAD analog of ``verify_pathway_round_trip.py``.

Inputs (already on disk; NOTHING upstream is recomputed):
  - per-sample log2 matrices in ``outputs/reports/kinase_attribution_5xfad/``:
      {tissue}_total_proteome_normalized.csv     (protein,     gene-keyed)
      {tissue}_st_raw_phospho_normalized.csv     (phospho_ps,  site-keyed)
      {tissue}_py_raw_phospho_normalized.csv     (phospho_py,  site-keyed)
  - the scRNA share + cell counts via ``fivexfad_decompose._load_aggexp`` /
    ``_shares_by_condition`` / ``_load_counts`` (reused, not duplicated)
  - the per-sample → condition map via ``ingest._sample_group_map`` (reused)
  - routed evidence genes from the pyarrow-readable pair-mode wide parquets at
    ``outputs/reports/incytr_pair_mode_5xfad/{tissue}/wide/`` (the browser-only
    edge_slices shards are not pyarrow-readable, so we read their substrate)
  - the JS-facing cluster vocabulary from
    ``edge_slices/incytr_pathways_fivexfad_<tissue>/index.json``

Output: per-cluster parquet shards under
``audit_sources/omics_trace_fivexfad_<tissue>/<slug>.parquet`` in the Song
omics_trace schema so ``OmicsTraceStore`` reads them unchanged. ``animal_id``
carries the ``biological_sample_id``; ``genotype`` ∈ {TG, WT}; ``timepoint`` ∈
{3mo, 6mo, 9mo, 12mo}. ``sex`` is omitted (5xFAD is not sex-split here).

Schema version: 1 — bump OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION in
alz/viewer/paths.py on any schema change.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root

from alz.shared import config  # noqa: E402
from alz.cohorts.fivexfad.ingest import (  # noqa: E402
    INCYTR_INPUT_DIR,
    OUTPUT_DIR,
    _sample_group_map,
)
from alz.ingest.fivexfad_decompose import (  # noqa: E402
    KEY_COLS,
    OUT_FILE,
    _load_aggexp,
    _load_counts,
    _shares_by_condition,
)
from alz.incytr_pair.pair_to_receiver_cache import _sanitize_celltype  # noqa: E402
from alz.viewer.paths import (  # noqa: E402
    OMICS_TRACE_FIVEXFAD_CORTEX_DIR,
    OMICS_TRACE_FIVEXFAD_CORTEX_INDEX,
    OMICS_TRACE_FIVEXFAD_HIPPO_DIR,
    OMICS_TRACE_FIVEXFAD_HIPPO_INDEX,
    OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
)

# Per-tissue output dir + index + the JS-facing pathway index.
_TISSUE_OUT = {
    "cortex": (OMICS_TRACE_FIVEXFAD_CORTEX_DIR, OMICS_TRACE_FIVEXFAD_CORTEX_INDEX,
               "incytr_pathways_fivexfad_cortex"),
    "hippocampus": (OMICS_TRACE_FIVEXFAD_HIPPO_DIR, OMICS_TRACE_FIVEXFAD_HIPPO_INDEX,
                    "incytr_pathways_fivexfad_hippocampus"),
}

# layer → (deconvolution channel, per-sample matrix filename, key columns).
_LAYER_SPEC = {
    "protein":    ("pr", "{tissue}_total_proteome_normalized.csv"),
    "phospho_ps": ("ps", "{tissue}_st_raw_phospho_normalized.csv"),
    "phospho_py": ("py", "{tissue}_py_raw_phospho_normalized.csv"),
}
_LAYERS = list(_LAYER_SPEC)

_SHARD_COLS = [
    "layer", "gene_symbol", "site_id", "motif", "animal_id",
    "genotype", "timepoint", "value", "log2_value",
]

_RECON_RTOL = 1e-6


def _pathway_index_path(tissue: str) -> str:
    return os.path.join(UNIFIED_VIEWER_DIR, "edge_slices",
                        _TISSUE_OUT[tissue][2], "index.json")


@functools.lru_cache(maxsize=None)
def _load_pathway_clusters(index_path: str) -> set[str]:
    """Union of sender/receiver cluster names from the JS-facing pathway index.

    Cached: the omics and transcript 5xFAD builders both call this for the same
    tissue in one ``build_unified_viewer`` run; the index is immutable per run.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"5xFAD incytr_pathways index missing at {index_path}; build the "
            f"5xFAD pathway shards (build_unified_viewer) before omics_trace_fivexfad."
        )
    with open(index_path) as f:
        idx = json.load(f)
    clusters: set[str] = set()
    for pair in idx.get("present") or []:
        if len(pair) >= 2:
            clusters.add(str(pair[0]))
            clusters.add(str(pair[1]))
    return clusters


@functools.lru_cache(maxsize=None)
def _load_evidence_genes(tissue: str) -> dict[str, set[str]]:
    """Routed evidence genes per cluster from the pair-mode wide parquets.

    Ligand → sender cluster; Receptor/EM/Target → receiver cluster
    (incytr/R/evaluation.R:227-230). Read across all contrasts (the index
    unions them), so the result covers every pathway cluster.

    Cached: the omics and transcript 5xFAD builders both call this for the same
    tissue in one ``build_unified_viewer`` run, and reading every wide parquet
    is the dominant I/O — callers only read (never mutate) the returned dict.
    """
    wide_dir = os.path.join(config.REPO_ROOT, "outputs", "reports",
                            "incytr_pair_mode_5xfad", tissue, "wide")
    files = sorted(glob.glob(os.path.join(wide_dir, "*_incytr_output.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no 5xFAD pair-mode wide parquets in {wide_dir}; cannot route "
            f"evidence genes for omics_trace_fivexfad."
        )
    out: dict[str, set[str]] = {}

    def add(cluster: object, gene: object) -> None:
        if cluster is None or gene is None:
            return
        if pd.isna(cluster) or pd.isna(gene):
            return
        c, g = str(cluster), str(gene)
        if c and g:
            out.setdefault(c, set()).add(g)

    cols = ["Sender", "Receiver", "Ligand", "Receptor", "EM", "Target"]
    for fpath in files:
        df = pq.read_table(fpath, columns=cols).to_pandas()
        for snd, lig in df[["Sender", "Ligand"]].itertuples(index=False):
            add(snd, lig)
        for rcv, rec, em, tgt in df[["Receiver", "Receptor", "EM", "Target"]].itertuples(index=False):
            add(rcv, rec)
            add(rcv, em)
            add(rcv, tgt)
    return out


def _read_decon(tissue: str, channel: str, conds: list[str],
                clusters: set[str]) -> tuple[pd.DataFrame, str]:
    """Condition-level deconvoluted CSV restricted to the value columns we gate,
    indexed by its row key (gene_symbol for pr; site_id for ps/py)."""
    path = os.path.join(INCYTR_INPUT_DIR, tissue, OUT_FILE[channel])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"condition-level deconvoluted CSV missing: {path}. Run "
            f"alz/ingest/fivexfad_decompose.py before omics_trace_fivexfad."
        )
    key_cols = KEY_COLS[channel]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    want_vals = [f"{cond}_{cl}" for cl in clusters for cond in conds]
    usecols = key_cols + [c for c in want_vals if c in header]
    decon = pd.read_csv(path, usecols=usecols)
    key = "site_id" if "site_id" in key_cols else "gene_symbol"
    decon[key] = decon[key].astype(str)
    return decon.set_index(key), key


def build_tissue(tissue: str, force: bool = False) -> dict:
    """Build (or reuse) the per-sample deconvolution shards for one tissue."""
    out_dir, index_path, _ = _TISSUE_OUT[tissue]

    if not force and os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
        if existing.get("omics_schema_version") == OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION:
            return existing

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    print(f"\n=== omics_trace_fivexfad {tissue} ===", flush=True)

    # --- Shared deconvolution objects (same as fivexfad_decompose) ---
    manifest = pd.read_csv(os.path.join(OUTPUT_DIR, "sample_manifest.csv"))
    group_map = _sample_group_map(manifest, tissue)
    agg, parsed, floor = _load_aggexp(tissue)
    shares = _shares_by_condition(agg, parsed, floor)
    n_per, n_total = _load_counts(tissue)
    conds = sorted(shares)

    pathway_clusters = _load_pathway_clusters(_pathway_index_path(tissue))
    evidence_genes = _load_evidence_genes(tissue)
    evid_all = set().union(*evidence_genes.values()) if evidence_genes else set()
    print(f"  {len(pathway_clusters)} pathway clusters, "
          f"{sum(len(v) for v in evidence_genes.values()):,} routed cluster-gene pairs, "
          f"{len(conds)} conditions", flush=True)

    # --- Load per-sample matrices once; filter to evidence genes; linearize ---
    loaded: dict[str, dict] = {}
    for layer, (channel, matrix_tmpl) in _LAYER_SPEC.items():
        mpath = os.path.join(OUTPUT_DIR, matrix_tmpl.format(tissue=tissue))
        if not os.path.exists(mpath):
            raise FileNotFoundError(
                f"per-sample matrix missing: {mpath} ({layer}). Run "
                f"alz/cohorts/fivexfad/ingest.py (run_ingest) before omics_trace_fivexfad."
            )
        m = pd.read_csv(mpath)
        key_cols = KEY_COLS[channel]
        sample_cols = [c for c in m.columns if c in group_map]
        m = m[m["gene_symbol"].astype(str).isin(evid_all)].reset_index(drop=True)
        # Linearize to numpy once here — the per-cluster loop only row-masks it,
        # so converting per cluster (one full-matrix to_numpy each) is wasted.
        lin = np.power(
            2.0, m[sample_cols].apply(pd.to_numeric, errors="coerce")
        ).to_numpy(dtype=float)
        decon, decon_key = _read_decon(tissue, channel, conds, pathway_clusters)
        loaded[layer] = {
            "key_cols": key_cols, "sample_cols": sample_cols,
            "m": m, "lin": lin, "decon": decon, "decon_key": decon_key,
            "gene": m["gene_symbol"].astype(str),
        }
        print(f"  {layer}: {len(m):,} evidence-gene rows × {len(sample_cols)} samples",
              flush=True)

    # --- Per-cluster shard build (one shard per cluster, all 3 layers) ---
    shards_written: dict[str, str] = {}
    max_rel_err = 0.0
    n_gate = 0
    for cl in sorted(pathway_clusters):
        genes_cl = evidence_genes.get(cl, set())
        parts: list[pd.DataFrame] = []
        for layer in _LAYERS:
            L = loaded[layer]
            sub_mask = L["gene"].isin(genes_cl).to_numpy()
            if not sub_mask.any():
                continue
            sub = L["m"][sub_mask]
            sub_lin = L["lin"][sub_mask]
            sub_gene = L["gene"][sub_mask]
            decon, decon_key = L["decon"], L["decon_key"]
            sub_keys = sub[decon_key].astype(str).to_numpy()
            for cond in conds:
                sh = shares[cond]
                if cl not in sh.columns:
                    continue
                nc = n_per.get((cl, cond), 0)
                if nc <= 0:
                    continue
                cols_c = [j for j, c in enumerate(L["sample_cols"])
                          if group_map[c] == cond]
                if not cols_c:
                    continue
                sf = n_total[cond] / nc
                share_vec = sh[cl].reindex(sub_gene).to_numpy()
                block = sf * sub_lin[:, cols_c] * share_vec[:, None]

                # --- Reconciliation gate vs the condition-level deconvoluted CSV ---
                colname = f"{cond}_{cl}"
                if colname in decon.columns:
                    with np.errstate(invalid="ignore"), warnings.catch_warnings():
                        # All-NaN rows (site undetected across a condition's
                        # samples) yield NaN means — expected; not an error.
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        row_mean = np.nanmean(
                            np.where(np.isfinite(block), block, np.nan), axis=1)
                    expected = decon[colname].reindex(sub_keys).to_numpy()
                    cmp = np.isfinite(row_mean) & np.isfinite(expected) & (expected != 0)
                    if cmp.any():
                        rel = np.abs(row_mean[cmp] - expected[cmp]) / np.abs(expected[cmp])
                        max_rel_err = max(max_rel_err, float(rel.max()))
                        n_gate += int(cmp.sum())

                geno, tp = cond.split("_", 1)
                sample_names = [L["sample_cols"][j] for j in cols_c]
                bdf = pd.DataFrame(block, columns=sample_names)
                for kc in L["key_cols"]:
                    bdf[kc] = sub[kc].to_numpy()
                long = bdf.melt(id_vars=L["key_cols"], var_name="animal_id",
                                value_name="value")
                long["layer"] = layer
                long["genotype"] = geno
                long["timepoint"] = tp
                if "site_id" not in L["key_cols"]:
                    long["site_id"] = None
                    long["motif"] = None
                parts.append(long)

        if not parts:
            print(f"  (warn) cluster {cl!r}: no deconvolved rows — no shard", flush=True)
            continue
        shard = pd.concat(parts, ignore_index=True)
        shard = shard[np.isfinite(shard["value"])].reset_index(drop=True)
        if shard.empty:
            print(f"  (warn) cluster {cl!r}: all rows zero/NaN — no shard", flush=True)
            continue
        shard["log2_value"] = np.where(
            shard["value"].to_numpy() > 0, np.log2(shard["value"].to_numpy()), np.nan)
        shard["site_id"] = shard["site_id"].astype("string")
        shard["motif"] = shard["motif"].astype("string")
        shard = shard[_SHARD_COLS]
        slug = _sanitize_celltype(cl)
        out_path = os.path.join(out_dir, f"{slug}.parquet")
        shard.to_parquet(out_path, index=False, compression="zstd")
        shards_written[cl] = os.path.relpath(out_path, UNIFIED_VIEWER_DIR)

    if not shards_written:
        raise RuntimeError(
            f"omics_trace_fivexfad {tissue}: no shards written — logic error.")
    if max_rel_err >= _RECON_RTOL:
        raise AssertionError(
            f"omics_trace_fivexfad {tissue}: per-sample mean does NOT reconcile to "
            f"the condition-level deconvoluted value (max rel err {max_rel_err:.3g} "
            f">= {_RECON_RTOL:g} over {n_gate:,} compared cells). The dot-bars and "
            f"the Incytr LFC chip would describe different data."
        )

    missing = pathway_clusters - set(shards_written)
    if missing:
        print(f"  (warn) {len(missing)} pathway cluster(s) without a shard "
              f"(no routed evidence rows): {sorted(missing)}", flush=True)

    index = {
        "omics_schema_version": OMICS_TRACE_FIVEXFAD_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "tissue": tissue,
        "label": f"5xFAD {tissue} per-sample deconvoluted abundance "
                 f"(protein + phospho pS + pY)",
        "layers": list(_LAYERS),
        "gene_scope": "routed_incytr_pathway_evidence_genes",
        "reconciliation_max_rel_err": max_rel_err,
        "reconciliation_cells_compared": n_gate,
        "reconciliation_note": (
            "Per (key, condition, cluster): mean over a condition's per-sample P_c "
            "== the condition-level *_deconvoluted.csv value behind the Incytr LFC "
            f"chip, to relative {_RECON_RTOL:g}."
        ),
        "sanitize_rule": "replace('/', '-'); replace(' ', '_')",
        "filename_template": "{cluster}.parquet",
        "relative_path": os.path.relpath(out_dir, UNIFIED_VIEWER_DIR),
        "clusters": sorted(shards_written.keys()),
        "shard_files": shards_written,
        "n_shards": len(shards_written),
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  wrote {len(shards_written)} cluster shards; reconciliation max "
          f"rel err {max_rel_err:.3g} over {n_gate:,} cells in {time.time() - t0:.1f}s",
          flush=True)
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tissue", choices=["cortex", "hippocampus", "all"],
                    default="all")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if index is current.")
    args = ap.parse_args()
    tissues = ["cortex", "hippocampus"] if args.tissue == "all" else [args.tissue]
    for t in tissues:
        build_tissue(t, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
