"""Shared build-cache helpers for unified-viewer shard writers.

The cache directory lives at outputs/reports/unified_viewer/.build_cache/.
Each shard family gets one gzip-compressed JSON file keyed by family name.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os

from alz.shared import config
from alz.viewer.paths import UNIFIED_VIEWER_DIR

_BUILD_CACHE_SCHEMA_VERSION = 1
_VIEWER_BUILD_CACHE_DIR = os.path.join(UNIFIED_VIEWER_DIR, ".build_cache")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_fingerprint(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "path": os.path.relpath(path, config.REPO_ROOT),
            "missing": True,
        }
    st = os.stat(path)
    return {
        "path": os.path.relpath(path, config.REPO_ROOT),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _input_signature(family: str, paths: list[str], params: dict) -> dict:
    signature = {
        "cache_schema_version": _BUILD_CACHE_SCHEMA_VERSION,
        "family": family,
        "params": params,
        "files": [_file_fingerprint(p) for p in sorted(set(paths))],
    }
    return json.loads(json.dumps(signature, sort_keys=True, separators=(",", ":")))


def _build_cache_path(family: str) -> str:
    return os.path.join(_VIEWER_BUILD_CACHE_DIR, f"{family}.json.gz")


def _load_build_cache(
    family: str,
    signature: dict,
    output_dir: str,
) -> dict | None:
    path = _build_cache_path(family)
    if not os.path.exists(path):
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  {family}: ignoring unreadable build cache ({e})", flush=True)
        return None

    # Fast path: compare mtime_ns + size only (skip SHA-256 on hit).
    # SHA-256 is still written on cache-write; the fast path is read-only.
    cached_files = cached.get("input_signature", {}).get("files", [])
    current_files = signature.get("files", [])
    if len(cached_files) != len(current_files):
        return None
    # Also verify non-file parts of the signature match (params, family, version).
    cached_sig_nonfiles = {
        k: v for k, v in cached.get("input_signature", {}).items() if k != "files"
    }
    current_sig_nonfiles = {k: v for k, v in signature.items() if k != "files"}
    if cached_sig_nonfiles != current_sig_nonfiles:
        return None
    # Per-file fast check.
    for cf, sf in zip(cached_files, current_files):
        cf_missing = cf.get("missing", False)
        sf_missing = sf.get("missing", False)
        if cf_missing != sf_missing:
            return None
        if cf_missing:
            continue
        if cf["size"] != sf["size"] or cf["mtime_ns"] != sf["mtime_ns"]:
            # Fall through to full-signature comparison below.
            if cached.get("input_signature") != signature:
                return None
            break

    missing = [
        rel for rel in cached.get("output_files", [])
        if not os.path.exists(os.path.join(output_dir, rel))
    ]
    if missing:
        print(
            f"  {family}: build cache stale; {len(missing)} output file(s) missing",
            flush=True,
        )
        return None
    payload = cached.get("payload")
    if payload is None:
        return None
    print(f"  {family}: cache hit; reusing existing shards", flush=True)
    return payload


def _write_build_cache(
    family: str,
    signature: dict,
    output_dir: str,
    output_files: list[str],
    payload: dict,
) -> None:
    os.makedirs(_VIEWER_BUILD_CACHE_DIR, exist_ok=True)
    existing = [
        rel for rel in output_files
        if os.path.exists(os.path.join(output_dir, rel))
    ]
    cache = {
        "cache_schema_version": _BUILD_CACHE_SCHEMA_VERSION,
        "family": family,
        "input_signature": signature,
        "output_files": sorted(existing),
        "payload": payload,
    }
    with gzip.open(_build_cache_path(family), "wt", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, separators=(",", ":"))
