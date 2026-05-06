"""Verify Jinja-rendered template == legacy HTML_TEMPLATE bytes.

Usage: pixi run python code/viewer/verify_template.py

Loads the legacy HTML_TEMPLATE raw string from build_unified_viewer.py and
compares against the rendered template/index.html.j2. Sentinel substitutions
(__APP_COLOR__, etc.) are NOT performed — both sides keep them as-is.
"""

from __future__ import annotations

import os
import re
import sys
from jinja2 import Environment, FileSystemLoader

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "template")
SRC = os.path.join(os.path.dirname(HERE), "build_unified_viewer.py")


def load_legacy_template() -> str:
    text = open(SRC).read()
    m = re.search(r'HTML_TEMPLATE = r"""(.*?)"""', text, re.DOTALL)
    if not m:
        raise SystemExit("HTML_TEMPLATE not found in build_unified_viewer.py")
    return m.group(1)


def render_jinja() -> str:
    def raw(path: str) -> str:
        return open(os.path.join(TEMPLATE_DIR, path)).read()

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), keep_trailing_newline=True)
    env.globals["raw"] = raw
    return env.get_template("index.html.j2").render()


def main() -> int:
    legacy = load_legacy_template()
    rendered = render_jinja()

    if legacy == rendered:
        print(f"BYTE-EQUIVALENT: {len(legacy):,} bytes")
        return 0

    print(f"DIVERGENT: legacy={len(legacy):,} rendered={len(rendered):,}")
    # Find first divergence
    for i, (a, b) in enumerate(zip(legacy, rendered)):
        if a != b:
            ctx_start = max(0, i - 80)
            ctx_end = i + 80
            print(f"\nFirst divergence at byte {i}:")
            print(f"  legacy  : {legacy[ctx_start:ctx_end]!r}")
            print(f"  rendered: {rendered[ctx_start:ctx_end]!r}")
            break
    else:
        if len(legacy) < len(rendered):
            print(f"\nRendered has extra trailing bytes: {rendered[len(legacy):len(legacy)+200]!r}")
        else:
            print(f"\nLegacy has extra trailing bytes: {legacy[len(rendered):len(rendered)+200]!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
