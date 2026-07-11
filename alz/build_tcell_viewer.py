#!/usr/bin/env python3
"""T-cell viewer builder: single-file HTML deliverable for the T-cell cohort.

Reads the T-cell bulk MEA (donor1 only; donor2 has no IMAC) and the per-donor
pair-mode Incytr wide outputs from the evidence-backed per-cell-state run.
Emits `outputs/reports/tcell_viewer/index.html` with the columnar payload
inlined as `<script type="application/json" id="payload-data">` plus per-pair
parquet shards under `edge_slices/incytr_pathways/` fetched on demand.

The mouse-cohort and human-cohort builders live at `alz/build_unified_viewer.py`.
This is a fully independent builder — no shared code paths flag-gated on cohort.

Usage:
    python alz/build_tcell_viewer.py              # payload + html (default)
    python alz/build_tcell_viewer.py --summary    # input row counts
    python alz/build_tcell_viewer.py --payload    # JSON only
    python alz/build_tcell_viewer.py --html       # write HTML (needs payload)
    python alz/build_tcell_viewer.py --validate   # write report md
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import resource
import sys

import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, HERE)

from alz.shared import config  # noqa: E402
from alz.viewer.shared.payload_helpers import _sanitize, _build_kinase_motifs  # noqa: E402

from tcell_viewer.paths import (  # noqa: E402
    EDGE_SLICES_INCYTR_PATHWAYS_DIR,
    INCYTR_PAIR_MODE_TCELLS_DIR,
    KINASE_ATTRIBUTION_TCELLS_DIR,
    PAYLOAD_JSON,
    PAYLOAD_JSON_GZ,
    REPORT_MD,
    SCHEMA_VERSION,
    UNIFIED_VIEWER_DIR,
    UNIFIED_VIEWER_HTML,
    resolve_incytr_pair_mode_tcells_dir,
)

# Slice modules — cohort slice logic lives here, not inline.
from alz.tcell_viewer.common import (  # noqa: E402
    DONORS,
    DONOR_WITH_MEA,
    TCELL_ATTRIBUTION_CAVEAT,
    TIMEPOINT_COLOR_MAP,
)
from alz.tcell_viewer.slices_incytr import (  # noqa: E402
    _write_tcell_pair_pathways,
    _timepoint_label,
)
from alz.tcell_viewer.slices_kinase import (  # noqa: E402
    _build_donor_kinases_slice,
    _build_celltypes_slice,
    _load_projected_state_mea_payload,
    _build_decomposition_index_from_state_mea,
)
from alz.tcell_viewer.slices_traces import (  # noqa: E402
    _write_tcell_transcript_trace,
    _write_tcell_omics_trace,
)
from alz.tcell_viewer.slices_audit import (  # noqa: E402
    build_tcell_audit_manifest,
    AUDIT_TABLE_SPECS,
)
from alz.tcell_viewer.state_contract import load_donor_states  # noqa: E402


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

from jinja2 import Environment, FileSystemLoader  # noqa: E402

_TEMPLATE_DIR = os.path.join(HERE, "tcell_viewer", "template")
_SHARED_TEMPLATE_DIR = os.path.join(HERE, "viewer_shared", "template")
_VIEWER_SPECIFIC_TAB_INCLUDES = [
    "js/tabs/attribution_manifest_tcell.js",
]


def _render_template() -> str:
    def _raw(path: str) -> str:
        local_path = os.path.join(_TEMPLATE_DIR, path)
        shared_path = os.path.join(_SHARED_TEMPLATE_DIR, path)
        source = local_path if os.path.exists(local_path) else shared_path
        with open(source) as f:
            return f.read()

    env = Environment(
        loader=FileSystemLoader([_TEMPLATE_DIR, _SHARED_TEMPLATE_DIR]),
        keep_trailing_newline=True,
    )
    env.globals["raw"] = _raw
    return env.get_template("index.html.j2").render(
        viewer_specific_tab_includes=_VIEWER_SPECIFIC_TAB_INCLUDES
    )


def write_html(payload: dict, json_str: str | None = None, *,
               inline_payload: bool = False) -> dict:
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    # Audit P8: default hosted mode keeps index.html small and loads
    # tcell_viewer.payload.json.gz via client-side DecompressionStream (mirrors
    # the unified viewer). --inline-payload preserves the air-gapped single-file
    # mode by baking the JSON into the payload-data script tag.
    if inline_payload and json_str is None:
        if payload is None:
            raise ValueError("payload or json_str is required for inline HTML")
        json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_text = json_str if inline_payload else "null"
    html = _render_template()
    palette = payload.get("meta", {}).get("timepoint_color_map", {})
    d13 = palette.get("d13", TIMEPOINT_COLOR_MAP["d13"])
    d17 = palette.get("d17", TIMEPOINT_COLOR_MAP["d17"])
    d20 = palette.get("d20", TIMEPOINT_COLOR_MAP["d20"])
    # styles.css lifts verbatim from unified_viewer, which carries mouse-
    # genotype color sentinels (__APP_COLOR__ etc). Map them to day colors:
    # mid (App→d17) / early (Tau→d13) / late (ApTt→d20).
    for sentinel, value in (
        ("__D13_COLOR__", d13),
        ("__D17_COLOR__", d17),
        ("__D20_COLOR__", d20),
        ("__APP_COLOR__", d17),
        ("__TAU_COLOR__", d13),
        ("__APTT_COLOR__", d20),
        ("__PAYLOAD_SENTINEL__", payload_text),
    ):
        html = html.replace(sentinel, value)
    raw = html.encode("utf-8")
    with open(UNIFIED_VIEWER_HTML, "wb") as f:
        f.write(raw)
    return {"html_bytes": len(raw), "output": UNIFIED_VIEWER_HTML}


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def build_tcell_payload() -> dict:
    """Assemble the T-cell payload — donor-scoped data nested under by_context."""
    print("[build_tcell_payload] kinase slices per donor:", flush=True)
    kinases_by_context: dict[str, dict] = {}
    contrast_union: list[str] = []
    fdr_thresh = 0.25
    attribution_index: dict | None = None
    for donor in DONORS:
        block = _build_donor_kinases_slice(donor)
        if block is None:
            print(f"  {donor}: no MEA (expected for donor2)", flush=True)
            continue
        kinases_by_context[donor] = block["kinases_slice"]
        for c in block["contrasts"]:
            if c not in contrast_union:
                contrast_union.append(c)
        fdr_thresh = block["fdr_threshold"]
        # Attribution is MEA-donor scoped (donor1 only). A single top-level
        # attribution_index suffices — the explorer renders only for the MEA
        # donor, which getScopedAttribution reads globally.
        if block.get("attribution_index") and attribution_index is None:
            attribution_index = block["attribution_index"]
            n_attr = len(attribution_index["kinase_id"])
            print(f"  {donor}: within-cohort attribution rows: {n_attr}",
                  flush=True)
        print(f"  {donor}: {len(block['kinase_names'])} kinases × "
              f"{len(block['contrasts'])} contrasts", flush=True)

    # Empty donor slice: same column names, zero rows. Keeps the JS contract
    # stable (donor swap toggles between two slice objects, never null).
    if kinases_by_context:
        template_cols = next(iter(kinases_by_context.values()))
        empty_slice = {k: [] for k in template_cols}
        for donor in DONORS:
            kinases_by_context.setdefault(donor, empty_slice)

    kinases_slice = {
        "by_context": kinases_by_context,
    }

    print("[build_tcell_payload] celltypes slice:", flush=True)
    donor_states = {d: load_donor_states(d) for d in DONORS}
    celltypes_slice = _build_celltypes_slice(donor_states)
    celltype_id_by_name = {
        name: idx for idx, name in zip(celltypes_slice["id"], celltypes_slice["name"])
    }
    celltypes_by_context: dict[str, dict] = {}
    for donor, clusters in donor_states.items():
        ordered = [c for c in celltypes_slice["name"] if c in set(clusters)]
        celltypes_by_context[donor] = {
            "id": [celltype_id_by_name[c] for c in ordered],
            "name": ordered,
            "tissue_category": ["T-cell"] * len(ordered),
            "available_donors": [[donor] for _ in ordered],
        }
    celltypes_slice["by_context"] = celltypes_by_context
    print(f"  {len(celltypes_slice['id'])} cluster(s) across {len(DONORS)} donors",
          flush=True)

    print("[build_tcell_payload] pair-mode shards:", flush=True)
    incytr_pathways_block = _write_tcell_pair_pathways()

    print("[build_tcell_payload] transcript_trace shards:", flush=True)
    transcript_trace_meta = _write_tcell_transcript_trace()
    total = sum(len(b["clusters"])
                for b in transcript_trace_meta.get("by_context", {}).values())
    print(f"  {total} cluster shard(s) total across {len(DONORS)} donors",
          flush=True)

    print("[build_tcell_payload] omics_trace shards:", flush=True)
    omics_trace_meta = _write_tcell_omics_trace()
    total_omics = sum(len(b["clusters"])
                      for b in omics_trace_meta.get("by_context", {}).values())
    print(f"  {total_omics} omics shard(s) total across {len(DONORS)} donors",
          flush=True)

    projected_state_mea = _load_projected_state_mea_payload()
    projected_state_contexts = set(
        (projected_state_mea or {}).get("by_context", {}).keys()
    )
    if projected_state_mea is not None:
        n_rows = sum(
            len(block.get("rows", []))
            for block in projected_state_mea.get("by_context", {}).values()
        )
        print(f"[build_tcell_payload] projected state MEA rows: {n_rows}",
              flush=True)

    family_map: dict[str, str] = {}
    union_kinases: list[str] = []
    seen: set[str] = set()
    for donor in DONORS:
        for k in (kinases_by_context.get(donor, {}).get("name") or []):
            if k not in seen:
                seen.add(k)
                union_kinases.append(k)
    if union_kinases:
        try:
            from kinase_library.modules import data as kl_data
            family_map = {
                str(k): str(v) for k, v in
                kl_data.get_kinase_family(union_kinases).to_dict().items()
                if v is not None and str(v) != "nan"
            }
        except Exception as e:
            print(f"  (warn) family resolve failed: {e}; using empty map",
                  flush=True)

    kinase_motifs = _build_kinase_motifs(union_kinases)
    audit_tables = build_tcell_audit_manifest()

    # Timepoints actually seen across both donors → palette subset.
    timepoint_set: set[str] = set()
    for block in incytr_pathways_block.get("by_context", {}).values():
        for c in block.get("contrasts", []):
            timepoint_set.add(_timepoint_label(c))
            timepoint_set.add(c.split("_", 1)[1] if "_" in c else c)
    timepoint_set.update(contrast_union)
    palette = {tp: TIMEPOINT_COLOR_MAP.get(tp, "#808080")
               for tp in sorted(timepoint_set)}

    contexts: list[dict] = []
    for donor in DONORS:
        ip_block = incytr_pathways_block.get("by_context", {}).get(donor, {})
        donor_contrasts = ip_block.get("contrasts") or []
        capabilities = {
            "kinases": donor in DONOR_WITH_MEA and len(
                kinases_by_context.get(donor, {}).get("id", [])
            ) > 0,
            "celltypes": len(celltypes_by_context.get(donor, {}).get("id", [])) > 0,
            "incytr": bool(ip_block.get("slice_index", {}).get("present")),
            "decomp_ols": False,
            "song_concordance": False,
            "human_reference": True,
            "subclass_breakdown": False,
            # Within-cohort cell-type attribution (specificity + concordance vs
            # bulk NES). Donor1 only — requires the IMAC kinase MEA.
            "within_cohort_attribution": (
                donor in DONOR_WITH_MEA and attribution_index is not None),
            "audit_tables": True,
            "transcript_trace": donor in transcript_trace_meta.get("by_context", {}),
            "omics_trace": len(
                omics_trace_meta.get("by_context", {}).get(donor, {}).get("clusters", [])
            ) > 0,
            "projected_state_mea": donor in projected_state_contexts,
        }
        notes = []
        if not capabilities["kinases"]:
            notes.append("No IMAC kinase MEA is available for this donor.")
        elif capabilities["within_cohort_attribution"]:
            notes.append(
                "Cell-type attribution is within-cohort: the bulk kinase signal "
                "is localized to evidence-backed per-cell states using this cohort's own scRNA "
                "(transcript specificity + concordance vs the bulk NES). Single "
                "donor, single library per day — no per-state significance test "
                "(direction + timecourse consistency only).")
            notes.append(TCELL_ATTRIBUTION_CAVEAT)
        contexts.append({
            "id": donor,
            "label": donor.replace("donor", "Donor "),
            "cohort": "tcell",
            "axis_kind": "donor",
            "contrasts": donor_contrasts,
            "contrast_axis": {
                "primary": "day",
                "baseline": "d2",
                "groups": ip_block.get("diseases", []),
                "timepoints": ip_block.get("timepoints", []),
            },
            "celltypes": celltypes_by_context.get(donor, {}).get("name", []),
            "capabilities": capabilities,
            "notes": notes,
        })

    meta = {
        "schema_version": SCHEMA_VERSION,
        "viewer_payload_schema_version": 2,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "cohort": "tcell",
        "default_context": "donor1",
        "contexts": contexts,
        "capabilities": {
            "contexts": True,
            "kinases": any(c["capabilities"]["kinases"] for c in contexts),
            "celltypes": any(c["capabilities"]["celltypes"] for c in contexts),
            "incytr": any(c["capabilities"]["incytr"] for c in contexts),
            "decomp_ols": False,
            "song_concordance": False,
            "human_reference": True,
            "subclass_breakdown": False,
            "within_cohort_attribution": any(
                c["capabilities"]["within_cohort_attribution"] for c in contexts),
            "audit_tables": True,
            "transcript_trace": any(c["capabilities"]["transcript_trace"] for c in contexts),
            "omics_trace": any(c["capabilities"]["omics_trace"] for c in contexts),
            "projected_state_mea": any(
                c["capabilities"].get("projected_state_mea") for c in contexts),
        },
        "donors": list(DONORS),
        "donors_with_mea": list(DONOR_WITH_MEA),
        "contrasts": contrast_union or sorted(timepoint_set),
        "timepoints": sorted(timepoint_set),
        "timepoint_color_map": palette,
        "familyMap": family_map,
        "fdr_threshold": fdr_thresh,
        "mea_kinase_donor": "donor1",
        "tcell_attribution_caveat": TCELL_ATTRIBUTION_CAVEAT,
        "transcript_trace": transcript_trace_meta,
        "omics_trace": omics_trace_meta,
        "notes": {
            "donor2_mea": "Donor 2 has no IMAC; kinase MEA unavailable for donor 2.",
        },
    }

    payload = {
        "kinases": kinases_slice,
        "kinase_motifs": kinase_motifs,
        "celltypes": celltypes_slice,
        "audit_tables": audit_tables,
        "edge_slice_ref": {
            "schema_version": SCHEMA_VERSION,
            "incytr_pathways_url": "edge_slices/incytr_pathways/",
            "incytr_pathways_index": "edge_slices/incytr_pathways/index.json",
        },
        "incytr_pathways": incytr_pathways_block,
        "meta": meta,
    }
    if attribution_index is not None:
        payload["attribution_index"] = attribution_index
    if projected_state_mea is not None:
        payload["projected_state_mea"] = projected_state_mea
        decomposition_index = _build_decomposition_index_from_state_mea(
            projected_state_mea, kinases_slice, meta)
        if decomposition_index is not None:
            payload["decomposition_index"] = decomposition_index
    return _sanitize(payload)


def write_payload(payload: dict) -> dict:
    os.makedirs(UNIFIED_VIEWER_DIR, exist_ok=True)
    json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = json_str.encode("utf-8")
    with open(PAYLOAD_JSON, "wb") as f:
        f.write(raw)
    gz = gzip.compress(raw, compresslevel=6)
    with open(PAYLOAD_JSON_GZ, "wb") as f:
        f.write(gz)
    return {"raw_bytes": len(raw), "gzip_bytes": len(gz), "json_str": json_str}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(payload: dict | None = None) -> str:
    errors: list[str] = []
    warnings: list[str] = []

    if not os.path.exists(PAYLOAD_JSON):
        errors.append(f"payload JSON missing: {PAYLOAD_JSON}")
        raw_bytes = gzip_bytes = 0
    else:
        raw_bytes = os.path.getsize(PAYLOAD_JSON)
        gzip_bytes = os.path.getsize(PAYLOAD_JSON_GZ) if os.path.exists(PAYLOAD_JSON_GZ) else 0
        if payload is None:
            with open(PAYLOAD_JSON) as f:
                payload = json.load(f)

    if raw_bytes >= 100 * 1024 * 1024:
        errors.append(f"payload raw {raw_bytes/1e6:.1f} MB exceeds 100 MB cap")
    if gzip_bytes >= 20 * 1024 * 1024:
        errors.append(f"payload gzip {gzip_bytes/1e6:.1f} MB exceeds 20 MB cap")

    if payload is not None:
        meta = payload.get("meta", {})
        if meta.get("viewer_payload_schema_version") != 2:
            errors.append("meta.viewer_payload_schema_version != 2")
        context_ids = [c.get("id") for c in meta.get("contexts", [])]
        if meta.get("default_context") not in context_ids:
            errors.append("meta.default_context is not present in meta.contexts")
        for key in ("kinases", "celltypes", "incytr_pathways"):
            if "by_context" not in (payload.get(key) or {}):
                errors.append(f"{key}.by_context missing")

        # Donor1 must have MEA.
        kinases_by_context = payload.get("kinases", {}).get("by_context", {})
        d1_rows = len(kinases_by_context.get("donor1", {}).get("id", []))
        if d1_rows == 0:
            errors.append("donor1 kinases slice is empty — expected MEA outputs")

        # Both donors must have Incytr pair-mode pathways.
        ip_donors = set(payload.get("incytr_pathways", {}).get("donors", []))
        for d in DONORS:
            if d not in ip_donors:
                errors.append(f"{d} missing from incytr_pathways block")

        ip_idx_path = os.path.join(EDGE_SLICES_INCYTR_PATHWAYS_DIR, "index.json")
        if not os.path.exists(ip_idx_path):
            errors.append(f"missing edge_slices/incytr_pathways/index.json")

        celltypes_by_context = payload.get("celltypes", {}).get("by_context", {})
        incytr_by_context = payload.get("incytr_pathways", {}).get("by_context", {})
        transcript_by_context = (
            payload.get("meta", {}).get("transcript_trace", {})
            .get("by_context", {})
        )
        for donor in DONORS:
            roster = set(load_donor_states(donor))
            payload_states = set(
                celltypes_by_context.get(donor, {}).get("name", [])
            )
            if payload_states != roster:
                errors.append(
                    f"{donor} celltype roster mismatch: payload="
                    f"{sorted(payload_states)} canonical={sorted(roster)}"
                )
            block = incytr_by_context.get(donor, {})
            endpoints = set(block.get("senders", [])) | set(
                block.get("receivers", [])
            )
            if not endpoints.issubset(roster):
                errors.append(
                    f"{donor} Incytr endpoints outside canonical roster: "
                    f"{sorted(endpoints - roster)}"
                )
            forbidden = {
                "low_signal_celltypes",
                "pathway_counts_low_signal_excluded",
            }
            present_forbidden = sorted(forbidden & set(block))
            qc = block.get("celltype_qc", {})
            raw_count_fields = {
                "median_n", "mean_n", "min_n", "total_n", "n_timepoints",
                "by_day",
            }
            qc_days = set(qc.get("days", []))
            qc_states = set(qc.get("by_celltype", {}))
            if qc_states != roster:
                errors.append(
                    f"{donor} raw cell-count roster mismatch: "
                    f"{sorted(qc_states)}"
                )
            for state, record in qc.get("by_celltype", {}).items():
                missing_raw = raw_count_fields - set(record)
                if missing_raw:
                    errors.append(
                        f"{donor}/{state} cell-count evidence missing "
                        f"{sorted(missing_raw)}"
                    )
                    continue
                by_day = record["by_day"]
                if set(by_day) != qc_days:
                    errors.append(
                        f"{donor}/{state} day-count evidence mismatch: "
                        f"{sorted(by_day)} != {sorted(qc_days)}"
                    )
                if record["n_timepoints"] != len(qc_days):
                    errors.append(
                        f"{donor}/{state} n_timepoints does not cover scoped days"
                    )

            counts_path = os.path.join(config.REPO_ROOT, qc.get("source", ""))
            if os.path.exists(counts_path):
                source_counts = pd.read_csv(counts_path)
                source_counts["day"] = pd.to_numeric(
                    source_counts["day"], errors="coerce"
                )
                source_counts["n_cells"] = pd.to_numeric(
                    source_counts["n_cells"], errors="coerce"
                )
                source_counts = source_counts.dropna(
                    subset=["state", "day", "n_cells"]
                )
                expected_state_day = (
                    source_counts.groupby(["state", "day"])["n_cells"].sum()
                )
                for state, record in qc.get("by_celltype", {}).items():
                    for day_label, observed_n in record.get("by_day", {}).items():
                        day = int(day_label.removeprefix("d"))
                        expected_n = int(expected_state_day.get((state, day), 0))
                        if observed_n != expected_n:
                            errors.append(
                                f"{donor}/{state}/{day_label} raw cell count "
                                f"{observed_n} != source {expected_n}"
                            )
            present_forbidden.extend(sorted(
                {
                    "low_signal_rule",
                    "low_signal_median_n_threshold",
                    "low_signal_celltypes",
                } & set(qc)
            ))
            if present_forbidden:
                errors.append(
                    f"{donor} exports legacy low-signal keys: "
                    f"{sorted(set(present_forbidden))}"
                )
            if any(
                "low_signal_endpoint" in row
                for row in block.get("top_instances", {}).get("rows", [])
            ):
                errors.append(f"{donor} top pathway rows export low_signal_endpoint")
            if any(
                "low_signal_median_le_3" in row
                for row in block.get("celltype_pathway_qc", {}).get("rows", [])
            ):
                errors.append(f"{donor} cell-count evidence exports a gate call")

            wide_dir = os.path.join(INCYTR_PAIR_MODE_TCELLS_DIR, donor, "wide")
            wide_files = sorted(glob.glob(
                os.path.join(wide_dir, "*_incytr_output.parquet")
            ))
            expected_rows = sum(
                pq.ParquetFile(path).metadata.num_rows for path in wide_files
            )
            if int(block.get("n_total_rows", -1)) != expected_rows:
                errors.append(
                    f"{donor} pathway total {block.get('n_total_rows')} != "
                    f"source parquet total {expected_rows}"
                )
            expected_contrasts = {
                os.path.basename(path).removesuffix("_incytr_output.parquet")
                for path in wide_files
            }
            if set(block.get("contrasts", [])) != expected_contrasts:
                errors.append(
                    f"{donor} contrast roster does not match source parquets"
                )

            trace = transcript_by_context.get(donor, {})
            declared_trace_states = set(trace.get("clusters", []))
            if declared_trace_states != roster:
                errors.append(
                    f"{donor} transcript-trace roster mismatch: "
                    f"{sorted(declared_trace_states)}"
                )
            trace_dir = os.path.join(
                UNIFIED_VIEWER_DIR,
                trace.get("relative_path", f"audit_sources/transcript_trace/{donor}"),
            )
            trace_files = {
                os.path.splitext(os.path.basename(path))[0]
                for path in glob.glob(os.path.join(trace_dir, "*.parquet"))
            }
            if trace_files != roster:
                errors.append(
                    f"{donor} transcript-trace file roster mismatch: "
                    f"files={sorted(trace_files)} canonical={sorted(roster)}"
                )

        expected_source = (
            "pair_mode_tcells ("
            f"{os.path.relpath(INCYTR_PAIR_MODE_TCELLS_DIR, config.REPO_ROOT)})"
        )
        if payload.get("incytr_pathways", {}).get("source") != expected_source:
            errors.append("incytr_pathways source does not match selected input root")

        production_default = os.path.join(
            config.REPO_ROOT, "outputs", "reports",
            "incytr_pair_mode_tcells_percell_posneg",
        )
        if resolve_incytr_pair_mode_tcells_dir({}) != production_default:
            errors.append("T-cell pair-mode resolver does not select production default")
        diagnostic_override = os.path.join(config.REPO_ROOT, "diagnostic-pair-mode")
        if resolve_incytr_pair_mode_tcells_dir({
            "TCELL_INCYTR_PAIR_MODE_DIR": diagnostic_override,
        }) != diagnostic_override:
            errors.append("T-cell pair-mode resolver ignores explicit override")

        legacy_trace_files = glob.glob(os.path.join(
            UNIFIED_VIEWER_DIR, "audit_sources", "transcript_trace", "*.parquet"
        ))
        if legacy_trace_files:
            errors.append("transcript-trace root contains legacy unscoped shards")

        attribution_states = set(
            payload.get("attribution_index", {}).get("cell_type", [])
        )
        unknown_attribution_states = (
            attribution_states - set(load_donor_states("donor1"))
        )
        if unknown_attribution_states:
            errors.append(
                "donor1 attribution contains states outside the per-cell audit: "
                f"{sorted(unknown_attribution_states)}"
            )

        for donor, projected in (
            payload.get("projected_state_mea", {}).get("by_context", {}).items()
        ):
            roster = set(load_donor_states(donor))
            projected_states = {
                str(row.get("state"))
                for row in projected.get("rows", [])
                if row.get("state") is not None
            }
            if not projected_states.issubset(roster):
                errors.append(
                    f"{donor} projected-state MEA contains historical states: "
                    f"{sorted(projected_states - roster)}"
                )

    peak_mb = _peak_rss_mb()
    lines = [
        "# T-cell Viewer Payload Report",
        "",
        f"_Generated {pd.Timestamp.utcnow().isoformat()}_",
        "",
        "## Sizes",
        "",
        f"- Payload JSON (raw): {raw_bytes/1e6:.2f} MB (cap 100)",
        f"- Payload JSON (gzip): {gzip_bytes/1e6:.2f} MB (cap 20)",
        "",
        f"- Peak RSS: {peak_mb:.0f} MB",
        "",
        "## Invariants",
        "",
    ]
    if errors:
        lines.append("### FAIL")
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("All structural invariants pass.")
    if warnings:
        lines.append("")
        lines.append("### Warnings")
        for w in warnings:
            lines.append(f"- {w}")
    report = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    if errors:
        raise SystemExit(f"validation failed: {len(errors)} error(s)")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Print input row counts")
    ap.add_argument("--payload", action="store_true", help="Write JSON payload")
    ap.add_argument("--html", action="store_true", help="Write tcell_viewer HTML (requires payload)")
    ap.add_argument("--validate", action="store_true", help="Write payload validation report")
    ap.add_argument("--inline-payload", action="store_true",
                    help="Bake the payload into index.html (air-gapped single-file mode); "
                         "default emits a small index.html + .json.gz sidecar")
    args = ap.parse_args(argv)

    if not any([args.summary, args.payload, args.html, args.validate]):
        args.payload = True
        args.html = True

    if args.summary:
        for donor in DONORS:
            mea_dir = os.path.join(KINASE_ATTRIBUTION_TCELLS_DIR, donor, "mea")
            manifest_path = os.path.join(mea_dir, "mea_manifest.json")
            mea_state = "n/a"
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    mea_state = f"{len(json.load(f).get('mea_ran', []))} tracks"
            wide_dir = os.path.join(INCYTR_PAIR_MODE_TCELLS_DIR, donor, "wide")
            wide_files = (sorted(glob.glob(os.path.join(wide_dir, "*_incytr_output.parquet")))
                          if os.path.isdir(wide_dir) else [])
            print(f"  {donor}: MEA={mea_state}, "
                  f"wide_parquets={len(wide_files)} "
                  f"({[os.path.basename(f) for f in wide_files]})")

    payload = None
    json_str = None
    if args.payload:
        payload = build_tcell_payload()
        sizes = write_payload(payload)
        json_str = sizes.pop("json_str")
        print(f"  payload raw={sizes['raw_bytes']/1e6:.2f} MB "
              f"gzip={sizes['gzip_bytes']/1e6:.2f} MB")

    if args.html:
        if payload is None:
            if not os.path.exists(PAYLOAD_JSON):
                raise SystemExit(f"payload missing at {PAYLOAD_JSON}; run --payload first")
            with open(PAYLOAD_JSON) as f:
                json_str = f.read()
            payload = json.loads(json_str)
        info = write_html(payload, json_str=json_str, inline_payload=args.inline_payload)
        print(f"  html {info['html_bytes']/1e6:.2f} MB -> {info['output']}")

    if args.validate:
        validate(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
