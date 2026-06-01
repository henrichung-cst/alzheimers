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


def validate(path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    payload = _load_payload(path)
    errors: list[str] = []
    context_ids = _check_contexts(payload, errors)
    if context_ids:
        _check_context_blocks(payload, context_ids, errors)
        _check_incytr(payload, context_ids, errors)
        _check_capabilities(payload, context_ids, errors)
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
