"""Verify the chunked unified-viewer Jinja template.

Usage:
    pixi run python alz/viewer/verify_template.py

The legacy monolithic ``HTML_TEMPLATE`` string was removed when the viewer was
split into ``alz/viewer/template`` and ``alz/viewer_shared/template`` chunks.
This verifier now checks the live contract:

- every ``{{ raw("...") }}`` include in ``index.html.j2`` resolves from the
  local template directory or the shared template fallback;
- this script's independent render is byte-equivalent to
  ``build_unified_viewer._render_template()``;
- sentinels substituted by ``write_html()`` are still present before rendering
  the final artifact.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
TEMPLATE_DIR = Path(HERE) / "template"
SHARED_TEMPLATE_DIR = Path(HERE).parent / "viewer_shared" / "template"
INDEX_TEMPLATE = SHARED_TEMPLATE_DIR / "index.html.j2"
VIEWER_SPECIFIC_TAB_INCLUDES = [
    "js/tabs/kinase_human.js",
    "js/tabs/kinase_fivexfad.js",
    "js/tabs/kinase_crosstable.js",
]
REQUIRED_SENTINELS = (
    "__APP_COLOR__",
    "__TAU_COLOR__",
    "__APTT_COLOR__",
    "__PAYLOAD_SENTINEL__",
)


def _raw_refs() -> list[str]:
    text = INDEX_TEMPLATE.read_text()
    refs = re.findall(r"""raw\(\s*['"]([^'"]+)['"]\s*\)""", text)
    refs.extend(VIEWER_SPECIFIC_TAB_INCLUDES)
    return refs


def _resolve_raw(path: str) -> Path:
    local_path = TEMPLATE_DIR / path
    shared_path = SHARED_TEMPLATE_DIR / path
    if local_path.exists():
        return local_path
    if shared_path.exists():
        return shared_path
    raise FileNotFoundError(
        f"raw include not found: {path} "
        f"(checked {local_path} and {shared_path})"
    )


def verify_raw_refs() -> list[tuple[str, str]]:
    refs = _raw_refs()
    if not refs:
        raise SystemExit(f"No raw() includes found in {INDEX_TEMPLATE}")
    resolved: list[tuple[str, str]] = []
    for ref in refs:
        source = _resolve_raw(ref)
        origin = "local" if source.is_relative_to(TEMPLATE_DIR) else "shared"
        resolved.append((ref, origin))
    return resolved


def render_jinja() -> str:
    def raw(path: str) -> str:
        return _resolve_raw(path).read_text()

    env = Environment(
        loader=FileSystemLoader([TEMPLATE_DIR, SHARED_TEMPLATE_DIR]),
        keep_trailing_newline=True,
    )
    env.globals["raw"] = raw
    return env.get_template("index.html.j2").render(
        viewer_specific_tab_includes=VIEWER_SPECIFIC_TAB_INCLUDES
    )


def render_builder() -> str:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from alz import build_unified_viewer

    return build_unified_viewer._render_template()


def verify_sentinels(rendered: str) -> None:
    missing = [s for s in REQUIRED_SENTINELS if s not in rendered]
    if missing:
        raise SystemExit(f"Missing write_html sentinel(s): {', '.join(missing)}")


def main() -> int:
    resolved = verify_raw_refs()
    rendered = render_jinja()
    builder = render_builder()
    verify_sentinels(rendered)

    if builder == rendered:
        local = sum(1 for _, origin in resolved if origin == "local")
        shared = sum(1 for _, origin in resolved if origin == "shared")
        print(
            f"TEMPLATE OK: {len(rendered):,} bytes · "
            f"{len(resolved)} raw includes ({local} local, {shared} shared)"
        )
        return 0

    print(f"DIVERGENT: builder={len(builder):,} independent={len(rendered):,}")
    # Find first divergence
    for i, (a, b) in enumerate(zip(builder, rendered)):
        if a != b:
            ctx_start = max(0, i - 80)
            ctx_end = i + 80
            print(f"\nFirst divergence at byte {i}:")
            print(f"  builder    : {builder[ctx_start:ctx_end]!r}")
            print(f"  independent: {rendered[ctx_start:ctx_end]!r}")
            break
    else:
        if len(builder) < len(rendered):
            print(
                "\nIndependent render has extra trailing bytes: "
                f"{rendered[len(builder):len(builder)+200]!r}"
            )
        else:
            print(
                "\nBuilder render has extra trailing bytes: "
                f"{builder[len(rendered):len(rendered)+200]!r}"
            )
    return 1


if __name__ == "__main__":
    sys.exit(main())
