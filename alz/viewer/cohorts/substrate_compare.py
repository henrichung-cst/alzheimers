"""Substrate Conservation slice — D1 cross-cohort substrate overlap.

Backs the viewer's "Substrate Conservation" tab from the latest C5 run
(``outputs/reports/substrate_compare/c5_mukesh_5xfad_<ts>/``). Two parts:

* **inline** ``PAYLOAD.substrate_compare`` — per-(kinase × context) summary:
  gene-identity overlap counts (shared / human_only / mouse_only), coverage splits
  (engaged / unmeasured), site-level refinement counts, overlap_frac_gene,
  blosum_similarity, direction_agree_frac, and a precomputed BLOSUM-similarity
  histogram (so the detail histogram needs no shard fetch).

* **lazy shards** — one parquet per kinase under
  ``edge_slices/substrate_pairs/<sanitized_name>.parquet`` holding that kinase's
  substrate rows across all 8 contexts with partition and coverage columns.
  Fetched on kinase selection for the detail table.

Cross-cohort: this slice reads only the C5 artifact. Kinase NES direction glyphs
(human vs mouse) are joined by kinase NAME against the existing ``human`` and
``supporting_5xfad`` payload sections in the JS tab — this builder does not
re-query MEA. The slice is emitted only when a C5 run exists; otherwise ``None``
(the tab is gated off).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from alz.viewer.paths import (
    EDGE_SLICES_SUBSTRATE_PAIRS_DIR,
    SUBSTRATE_COMPARE_GLOB,
    SUBSTRATE_TIER_FILE,
)
from alz.viewer.shared.cohort_slice import CohortViewerSlice, EdgeSliceContribution

# Canonical context order for the 2×4 (tissue × age) enrichment mini-heatmap.
_TISSUES = ("cortex", "hippocampus")
_AGES = ("3mo", "6mo", "9mo", "12mo")
_CONTEXTS = [f"{t}_{a}" for t in _TISSUES for a in _AGES]

_HIST_BINS = 10
_HIST_RANGE = (0.0, 1.0)

_SHARD_TEMPLATE = "{name}.parquet"


def sanitize_kinase(name: str) -> str:
    """Filesystem/URL-safe shard basename. The JS applies the identical rule."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name))


def _latest_c5_dir() -> str | None:
    dirs = sorted(glob.glob(SUBSTRATE_COMPARE_GLOB))
    return dirs[-1] if dirs else None


# Curated effector-tier labels. Section headers in kinases.txt map to the
# reader-facing label shown as a pill in the kinase table.
_TIER_LABELS = {"GREEN": "Credible Effector", "YELLOW": "Plausible"}


def _load_tiers() -> dict[str, str]:
    """Parse kinases.txt (#GREEN / #YELLOW sections) → {kinase: tier_key}.

    tier_key is 'green' | 'yellow'; the JS maps it to the pill label/color.
    Missing file → empty map (tab still builds, no annotations).
    """
    tiers: dict[str, str] = {}
    if not os.path.exists(SUBSTRATE_TIER_FILE):
        return tiers
    current: str | None = None
    with open(SUBSTRATE_TIER_FILE) as fh:
        for line in fh:
            tok = line.strip()
            if not tok:
                continue
            if tok.startswith("#"):
                current = tok[1:].strip().upper()
                continue
            if current in _TIER_LABELS:
                tiers[tok] = current.lower()
    return tiers


def _parse_hist(raw: str) -> list[int]:
    if not raw:
        return [0] * _HIST_BINS
    return [int(x) for x in raw.split(";")]


def _num(v):
    """CSV cell → float, or None for the empty-string sentinel."""
    if v is None or v == "":
        return None
    return float(v)


def _build_summary_block(c5_dir: str) -> dict:
    """Read kinase_summary.csv into the inline columnar block."""
    summary_csv = os.path.join(c5_dir, "kinase_summary.csv")
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT kinase, context, "
        f"n_shared_gene, n_human_only_gene, n_mouse_only_gene, "
        f"n_human_only_engaged, n_human_only_unmeasured, "
        f"n_mouse_only_engaged, n_mouse_only_unmeasured, "
        f"n_shared_site, n_diffsite, "
        f"overlap_frac_gene, blosum_similarity, "
        f"direction_agree_frac, sim_hist "
        f"FROM read_csv_auto('{summary_csv}', header=true) "
        f"ORDER BY kinase, context"
    ).fetchall()

    kinase, context = [], []
    n_shared_gene, n_human_only_gene, n_mouse_only_gene = [], [], []
    n_human_only_engaged, n_human_only_unmeasured = [], []
    n_mouse_only_engaged, n_mouse_only_unmeasured = [], []
    n_shared_site, n_diffsite = [], []
    overlap_frac_gene, blosum_similarity, direction_agree_frac, sim_hist = [], [], [], []
    for r in rows:
        (k, ctx,
         nsg, nhog, nmog,
         nhoe, nhou, nmoe, nmou,
         nss, nds,
         ofg, bsim,
         daf, hist) = r
        kinase.append(k)
        context.append(ctx)
        n_shared_gene.append(int(nsg) if nsg is not None else 0)
        n_human_only_gene.append(int(nhog) if nhog is not None else 0)
        n_mouse_only_gene.append(int(nmog) if nmog is not None else 0)
        n_human_only_engaged.append(int(nhoe) if nhoe is not None else 0)
        n_human_only_unmeasured.append(int(nhou) if nhou is not None else 0)
        n_mouse_only_engaged.append(int(nmoe) if nmoe is not None else 0)
        n_mouse_only_unmeasured.append(int(nmou) if nmou is not None else 0)
        n_shared_site.append(int(nss) if nss is not None else 0)
        n_diffsite.append(int(nds) if nds is not None else 0)
        overlap_frac_gene.append(_num(ofg))
        blosum_similarity.append(_num(bsim))
        direction_agree_frac.append(_num(daf))
        sim_hist.append(_parse_hist(hist))

    generated_at = None
    manifest_path = os.path.join(c5_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                generated_at = json.load(fh).get("generated_at")
        except (ValueError, OSError):
            generated_at = None

    return {
        "schema_version": 2,
        "source": os.path.basename(c5_dir),
        "generated_at": generated_at,
        "contexts": _CONTEXTS,
        "tissues": list(_TISSUES),
        "ages": list(_AGES),
        "hist_bins": _HIST_BINS,
        "hist_range": list(_HIST_RANGE),
        "tier_labels": {"green": _TIER_LABELS["GREEN"], "yellow": _TIER_LABELS["YELLOW"]},
        "tiers": _load_tiers(),
        "kinase": kinase,
        "context": context,
        "n_shared_gene": n_shared_gene,
        "n_human_only_gene": n_human_only_gene,
        "n_mouse_only_gene": n_mouse_only_gene,
        "n_human_only_engaged": n_human_only_engaged,
        "n_human_only_unmeasured": n_human_only_unmeasured,
        "n_mouse_only_engaged": n_mouse_only_engaged,
        "n_mouse_only_unmeasured": n_mouse_only_unmeasured,
        "n_shared_site": n_shared_site,
        "n_diffsite": n_diffsite,
        "overlap_frac_gene": overlap_frac_gene,
        "blosum_similarity": blosum_similarity,
        "direction_agree_frac": direction_agree_frac,
        "sim_hist": sim_hist,
    }


def _write_pair_shards(c5_dir: str) -> list[str]:
    """One parquet per kinase (all match classes, all contexts). Returns names.

    DuckDB-streamed: reads each per-context pairs CSV — matched (exact/conserved)
    plus cohort-unique (a_only/b_only) rows — so the detail table shows shared and
    unique substrates alike. The full set is ~210K rows across all kinases ×
    contexts; grouped per kinase it stays small on disk.
    """
    shutil.rmtree(EDGE_SLICES_SUBSTRATE_PAIRS_DIR, ignore_errors=True)
    os.makedirs(EDGE_SLICES_SUBSTRATE_PAIRS_DIR, exist_ok=True)

    con = duckdb.connect()
    parts = []
    for ctx in _CONTEXTS:
        f = os.path.join(c5_dir, f"kinase_pairs_{ctx}.csv")
        if not os.path.exists(f):
            continue
        t = con.execute(
            f"""
            SELECT kinase, '{ctx}' AS context,
                   gene_a, site_a, motif_a, gene_b, site_b, motif_b,
                   TRY_CAST(similarity AS DOUBLE)       AS similarity,
                   partition,
                   coverage,
                   TRY_CAST(direction_a AS INTEGER)     AS direction_a,
                   TRY_CAST(direction_b AS INTEGER)     AS direction_b,
                   TRY_CAST(direction_agree AS INTEGER)  AS direction_agree,
                   TRY_CAST(support_a AS INTEGER)       AS support_a,
                   TRY_CAST(support_b AS INTEGER)       AS support_b
            FROM read_csv_auto('{f}', header=true, all_varchar=true)
            """
        ).fetch_arrow_table()
        parts.append(t)

    present: list[str] = []
    if not parts:
        return present

    df = pa.concat_tables(parts).to_pandas()
    for name, g in df.groupby("kinase", sort=True):
        out = g.drop(columns=["kinase"]).reset_index(drop=True)
        fname = _SHARD_TEMPLATE.format(name=sanitize_kinase(name))
        pq.write_table(
            pa.Table.from_pandas(out, preserve_index=False),
            os.path.join(EDGE_SLICES_SUBSTRATE_PAIRS_DIR, fname),
            compression="zstd",
        )
        present.append(str(name))

    index = {
        "schema_version": 2,
        "slice_count": len(present),
        "present_kinases": present,
        "filename_template": _SHARD_TEMPLATE,
        "sanitize": "non-alphanumeric runs → '_'",
    }
    with open(os.path.join(EDGE_SLICES_SUBSTRATE_PAIRS_DIR, "index.json"), "w") as fh:
        json.dump(index, fh)
    return present


def build_substrate_compare_slice() -> CohortViewerSlice | None:
    """Assemble the cross-reference slice, or None when no C5 run exists."""
    c5_dir = _latest_c5_dir()
    if c5_dir is None:
        print("  (info) no C5 substrate-compare run found; "
              "substrate_compare tab omitted", flush=True)
        return None

    print(f"  substrate_compare: source {os.path.basename(c5_dir)}", flush=True)
    block = _build_summary_block(c5_dir)
    present = _write_pair_shards(c5_dir)
    print(f"  substrate_compare: {len(set(block['kinase']))} kinases × "
          f"{len(block['contexts'])} contexts; {len(present)} pair shards",
          flush=True)

    edge = EdgeSliceContribution(
        family="substrate_pairs",
        entries={
            "substrate_pairs_url": "edge_slices/substrate_pairs/",
            "substrate_pairs_index": "edge_slices/substrate_pairs/index.json",
            "present_substrate_pairs_kinases": present,
        },
    )
    return CohortViewerSlice(
        cohort_id="cross_reference",
        context_ids=(),
        owned_sections={"substrate_compare": block},
        capabilities={"substrate_compare": True},
        edge_slice_ref=(edge,),
        provenance={"source": os.path.basename(c5_dir), "n_pair_shards": len(present)},
    )
