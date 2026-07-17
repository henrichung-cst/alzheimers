"""Validate viewer payload schema v2.

This is a fast structural check for static viewer payloads. It does not inspect
large parquet shards; it verifies that the JSON payload routes all shared blocks
through contexts and that context capabilities agree with the available blocks.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from alz.viewer.shared.payload_helpers import (
    _INCYTR_SIDECHAIN_INTERACTOME_COLUMNS,
    _INCYTR_SIDECHAIN_TERMINAL_COLUMNS,
)


REQUIRED_CONTEXT_FIELDS = {
    "id",
    "label",
    "cohort",
    "axis_kind",
    "capabilities",
}
CONTEXT_BLOCKS = ("kinases", "celltypes", "incytr_pathways")


def _load_payload(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("payload root is not a JSON object")
    return payload


def _n_rows(block: Any) -> int:
    if not isinstance(block, dict):
        return 0
    ids = block.get("id")
    if isinstance(ids, list):
        return len(ids)
    names = block.get("name")
    if isinstance(names, list):
        return len(names)
    return 0


def _check_contexts(payload: dict[str, Any], errors: list[str]) -> list[str]:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta missing or not an object")
        return []

    if meta.get("viewer_payload_schema_version") != 2:
        errors.append("meta.viewer_payload_schema_version must be 2")

    contexts = meta.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        errors.append("meta.contexts must be a non-empty list")
        return []

    context_ids: list[str] = []
    seen: set[str] = set()
    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, dict):
            errors.append(f"meta.contexts[{i}] is not an object")
            continue
        missing = REQUIRED_CONTEXT_FIELDS - set(ctx)
        if missing:
            errors.append(f"context {i} missing fields: {sorted(missing)}")
        ctx_id = ctx.get("id")
        if not isinstance(ctx_id, str) or not ctx_id:
            errors.append(f"context {i} has invalid id")
            continue
        if ctx_id in seen:
            errors.append(f"duplicate context id: {ctx_id}")
        seen.add(ctx_id)
        context_ids.append(ctx_id)
        if not isinstance(ctx.get("capabilities"), dict):
            errors.append(f"context {ctx_id} capabilities must be an object")

    default_context = meta.get("default_context")
    if default_context not in seen:
        errors.append("meta.default_context is not present in meta.contexts")

    return context_ids


def _check_context_blocks(
    payload: dict[str, Any],
    context_ids: list[str],
    errors: list[str],
) -> None:
    for block_name in CONTEXT_BLOCKS:
        block = payload.get(block_name)
        if not isinstance(block, dict):
            errors.append(f"{block_name} missing or not an object")
            continue
        if "by_donor" in block:
            errors.append(f"{block_name}.by_donor is deprecated; use by_context")
        by_context = block.get("by_context")
        if not isinstance(by_context, dict):
            errors.append(f"{block_name}.by_context missing or not an object")
            continue
        missing = sorted(set(context_ids) - set(by_context))
        if missing:
            errors.append(f"{block_name}.by_context missing contexts: {missing}")

    tt = payload.get("meta", {}).get("transcript_trace")
    if isinstance(tt, dict) and "by_donor" in tt:
        errors.append("meta.transcript_trace.by_donor is deprecated; use by_context")


def _check_incytr(payload: dict[str, Any], context_ids: list[str], errors: list[str]) -> None:
    by_context = payload.get("incytr_pathways", {}).get("by_context", {})
    if not isinstance(by_context, dict):
        return
    for ctx_id in context_ids:
        block = by_context.get(ctx_id)
        if not isinstance(block, dict):
            continue
        idx = block.get("slice_index")
        if not isinstance(idx, dict):
            errors.append(f"incytr_pathways.by_context.{ctx_id}.slice_index missing")
            continue
        if not idx.get("filename_template"):
            errors.append(f"incytr slice_index for {ctx_id} missing filename_template")
        present = idx.get("present")
        if not isinstance(present, list):
            errors.append(f"incytr slice_index for {ctx_id} present must be a list")
        elif block.get("n_total_rows", 0) and not present:
            errors.append(f"incytr slice_index for {ctx_id} has rows but no present pairs")


def _check_incytr_sidechains(
    payload: dict[str, Any],
    context_ids: list[str],
    payload_path: Path,
    errors: list[str],
) -> None:
    """Validate the context-indexed kinase-sidechain lazy shards when present."""
    ref = payload.get("edge_slice_ref", {})
    if not isinstance(ref, dict):
        errors.append("edge_slice_ref missing or not an object")
        return
    url = ref.get("incytr_sidechains_url")
    index_rel = ref.get("incytr_sidechains_index")
    if url is None and index_rel is None:
        return
    if not isinstance(url, str) or not isinstance(index_rel, str):
        errors.append("Incytr sidechain URL and index must both be strings")
        return
    index_path = payload_path.parent / index_rel
    if not index_path.is_file():
        errors.append(f"Incytr sidechain index missing: {index_rel}")
        return
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    by_context = index.get("by_context")
    if (
        index.get("schema_version") != 1
        or index.get("slice_type") != "incytr_kinase_sidechains"
        or not isinstance(by_context, dict)
        or not by_context
    ):
        errors.append("Incytr sidechain index has an invalid schema")
        return
    unknown_contexts = set(by_context) - set(context_ids)
    if unknown_contexts:
        errors.append(
            f"Incytr sidechain index has unknown contexts: {sorted(unknown_contexts)}"
        )
    expected_contexts = (
        {"donor1"}
        if payload.get("meta", {}).get("cohort") == "tcell"
        else set(context_ids)
    )
    if set(by_context) != expected_contexts:
        errors.append(
            "Incytr sidechain index context coverage mismatch: "
            f"expected {sorted(expected_contexts)}, found {sorted(by_context)}"
        )
    for context_id, entry in by_context.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            errors.append(f"Incytr sidechain index entry {context_id!r} is invalid")
            continue
        if not entry["url"].startswith(url):
            errors.append(
                f"Incytr sidechain shard {context_id!r} is outside its declared URL"
            )
            continue
        shard_path = payload_path.parent / entry["url"]
        if not shard_path.is_file():
            errors.append(f"Incytr sidechain shard missing: {entry['url']}")
            continue
        with gzip.open(shard_path, "rt", encoding="utf-8") as fh:
            shard = json.load(fh)
        if (
            shard.get("schema_version") != 1
            or shard.get("slice_type") != "incytr_kinase_sidechains"
            or shard.get("context_id") != context_id
        ):
            errors.append(f"Incytr sidechain shard {context_id!r} has an invalid schema")
            continue
        observed_counts = (
            shard.get("interactome_edge_count"),
            shard.get("interactome_node_count"),
            shard.get("terminal_edge_count"),
        )
        index_counts = (
            entry.get("interactome_edge_count"),
            entry.get("interactome_node_count"),
            entry.get("terminal_edge_count"),
        )
        if index_counts != observed_counts:
            errors.append(
                f"Incytr sidechain index counts for {context_id!r} do not match shard"
            )
        for table, count_key, expected_columns in (
            ("interactome", "interactome_edge_count", _INCYTR_SIDECHAIN_INTERACTOME_COLUMNS),
            ("terminal_edges", "terminal_edge_count", _INCYTR_SIDECHAIN_TERMINAL_COLUMNS),
        ):
            columns = shard.get(table)
            count = shard.get(count_key)
            if not isinstance(columns, dict) or not isinstance(count, int):
                errors.append(f"Incytr sidechain shard {context_id!r} lacks {table}")
                continue
            if tuple(columns) != expected_columns:
                errors.append(
                    f"Incytr sidechain shard {context_id!r} has an invalid {table} schema"
                )
                continue
            if any(not isinstance(values, list) for values in columns.values()):
                errors.append(
                    f"Incytr sidechain shard {context_id!r} {table} is not columnar"
                )
                continue
            lengths = {len(values) for values in columns.values()}
            if not lengths or len(lengths) != 1 or count not in lengths:
                errors.append(
                    f"Incytr sidechain shard {context_id!r} {table} count mismatch"
                )
        interactome = shard.get("interactome", {})
        if isinstance(interactome, dict):
            nodes = set(interactome.get("source_gene", []))
            nodes.update(interactome.get("target_gene", []))
            if shard.get("interactome_node_count") != len(nodes):
                errors.append(
                    f"Incytr sidechain shard {context_id!r} node count mismatch"
                )


def _check_capabilities(
    payload: dict[str, Any],
    context_ids: list[str],
    errors: list[str],
) -> None:
    contexts = {
        c.get("id"): c
        for c in payload.get("meta", {}).get("contexts", [])
        if isinstance(c, dict)
    }
    kinases = payload.get("kinases", {}).get("by_context", {})
    celltypes = payload.get("celltypes", {}).get("by_context", {})
    incytr = payload.get("incytr_pathways", {}).get("by_context", {})
    for ctx_id in context_ids:
        caps = contexts.get(ctx_id, {}).get("capabilities", {})
        if not isinstance(caps, dict):
            continue
        has_kinases = _n_rows(kinases.get(ctx_id)) > 0
        if bool(caps.get("kinases")) != has_kinases:
            errors.append(
                f"context {ctx_id} capability kinases={caps.get('kinases')} "
                f"but kinase rows={_n_rows(kinases.get(ctx_id))}"
            )
        has_celltypes = _n_rows(celltypes.get(ctx_id)) > 0
        if bool(caps.get("celltypes")) != has_celltypes:
            errors.append(
                f"context {ctx_id} capability celltypes={caps.get('celltypes')} "
                f"but celltype rows={_n_rows(celltypes.get(ctx_id))}"
            )
        idx = incytr.get(ctx_id, {}).get("slice_index", {})
        has_incytr = bool(isinstance(idx, dict) and idx.get("present"))
        if bool(caps.get("incytr")) != has_incytr:
            errors.append(
                f"context {ctx_id} capability incytr={caps.get('incytr')} "
                f"but present pairs={len(idx.get('present', [])) if isinstance(idx, dict) else 0}"
            )


def _check_tcell_contract(payload: dict[str, Any], errors: list[str]) -> None:
    """Hard checks for the evidence-backed T-cell state payload."""
    if payload.get("meta", {}).get("cohort") != "tcell":
        return
    celltypes = payload.get("celltypes", {}).get("by_context", {})
    incytr = payload.get("incytr_pathways", {}).get("by_context", {})
    all_states: set[str] = set()
    for donor, state_block in celltypes.items():
        roster = {str(value) for value in state_block.get("name", [])}
        all_states.update(roster)
        block = incytr.get(donor, {})
        endpoints = {
            str(value)
            for value in [*block.get("senders", []), *block.get("receivers", [])]
        }
        unknown = sorted(endpoints - roster)
        if unknown:
            errors.append(f"T-cell context {donor} has foreign endpoints: {unknown}")
        forbidden = {
            "low_signal_celltypes",
            "pathway_counts_low_signal_excluded",
        } & set(block)
        qc_forbidden = {
            "low_signal_rule",
            "low_signal_median_n_threshold",
            "low_signal_celltypes",
        } & set(block.get("celltype_qc", {}))
        qc = block.get("celltype_qc", {})
        if set(qc.get("by_celltype", {})) != roster:
            errors.append(
                f"T-cell context {donor} raw cell-count roster mismatch"
            )
        qc_days = set(qc.get("days", []))
        for state, evidence in qc.get("by_celltype", {}).items():
            by_day = evidence.get("by_day")
            if not isinstance(by_day, dict) or set(by_day) != qc_days:
                errors.append(
                    f"T-cell context {donor}/{state} lacks complete day-count evidence"
                )
            if evidence.get("n_timepoints") != len(qc_days):
                errors.append(
                    f"T-cell context {donor}/{state} timepoint count is incomplete"
                )
        if forbidden or qc_forbidden:
            errors.append(
                f"T-cell context {donor} exports legacy low-signal keys: "
                f"{sorted(forbidden | qc_forbidden)}"
            )
        if any(
            "low_signal_endpoint" in row
            for row in block.get("top_instances", {}).get("rows", [])
        ):
            errors.append(
                f"T-cell context {donor} top rows export low_signal_endpoint"
            )
        if any(
            "low_signal_median_le_3" in row
            for row in block.get("celltype_pathway_qc", {}).get("rows", [])
        ):
            errors.append(
                f"T-cell context {donor} cell-count rows export a gate call"
            )
    attr_states = {
        str(value) for value in payload.get("attribution_index", {}).get(
            "cell_type", []
        )
    }
    if not attr_states.issubset(all_states):
        errors.append(
            "T-cell attribution contains states absent from the context rosters: "
            f"{sorted(attr_states - all_states)}"
        )


def validate(path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    payload = _load_payload(path)
    errors: list[str] = []
    context_ids = _check_contexts(payload, errors)
    if context_ids:
        _check_context_blocks(payload, context_ids, errors)
        _check_incytr(payload, context_ids, errors)
        _check_incytr_sidechains(payload, context_ids, path, errors)
        _check_capabilities(payload, context_ids, errors)
        _check_tcell_contract(payload, errors)
    summary = {
        "path": str(path),
        "contexts": context_ids,
        "default_context": payload.get("meta", {}).get("default_context"),
        "schema_version": payload.get("meta", {}).get("viewer_payload_schema_version"),
    }
    return not errors, errors, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("payload", nargs="+", type=Path)
    args = ap.parse_args()

    ok_all = True
    for path in args.payload:
        ok, errors, summary = validate(path)
        print(
            f"{path}: schema={summary['schema_version']} "
            f"default={summary['default_context']} "
            f"contexts={','.join(summary['contexts']) or '<none>'} "
            f"pass={ok}"
        )
        for err in errors:
            print(f"  ERROR: {err}")
        ok_all = ok_all and ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
