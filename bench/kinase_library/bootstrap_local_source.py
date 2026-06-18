"""Copy the installed kinase-library package into the local MEA bench area.

This creates an ignored source tree at ``bench/kinase_library/local_src``. The
copy is for profiling and experimental edits only; production code continues to
import the package from the active Python environment unless a benchmark command
explicitly prepends this local source path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import shutil
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET = BENCH_DIR / "local_src"


def _ignore_generated(dir_name: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".pytest_cache"}
    ignored.update(name for name in names if name.endswith((".pyc", ".pyo")))
    return ignored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Directory that will contain the local kinase_library source tree.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing local source tree.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    target = args.target.resolve()
    package_target = target / "kinase_library"

    import kinase_library

    package_source = Path(kinase_library.__file__).resolve().parent
    version = importlib.metadata.version("kinase-library")

    if package_target.exists():
        if not args.force:
            raise SystemExit(
                f"{package_target} already exists; pass --force to replace it."
            )
        shutil.rmtree(package_target)

    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package_source, package_target, ignore=_ignore_generated)

    metadata = {
        "package": "kinase-library",
        "version": version,
        "source": str(package_source),
        "target": str(package_target),
        "copied_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    metadata_path = target / "SOURCE_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
