import os
import sys

# Phase 2 bridge: live scripts use flat imports (import config) which work via
# sys.path[0] when invoked as `python alz/foo.py`. Kedro imports modules as
# `alz.foo`, where the package's directory is NOT on sys.path. Add it here so
# the flat imports keep resolving until Phase 4 rewrites them as package-relative.
_ALZ_DIR = os.path.dirname(os.path.abspath(__file__))
if _ALZ_DIR not in sys.path:
    sys.path.insert(0, _ALZ_DIR)

__version__ = "0.1.0"
