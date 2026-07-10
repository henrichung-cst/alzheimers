#!/usr/bin/env python3
"""Extract every displayed table from Matt's historical HTML into its fixture."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.section = "document"
        self.sections: list[str] = []
        self._heading_tag: str | None = None
        self._heading_parts: list[str] | None = None
        self._table_rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self.tables: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3"}:
            self._heading_parts = []
            self._heading_tag = tag
        elif tag == "table":
            self._table_rows = []
        elif tag == "tr" and self._table_rows is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        elif self._heading_parts is not None:
            self._heading_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3"} and self._heading_parts is not None:
            heading = _normalize(" ".join(self._heading_parts))
            if heading and heading != "Table of contents":
                self.section = heading
                if self._heading_tag == "h2":
                    self.sections.append(heading)
            self._heading_parts = None
            self._heading_tag = None
        elif tag in {"th", "td"} and self._cell_parts is not None:
            assert self._row is not None
            self._row.append(_normalize(" ".join(self._cell_parts)))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                assert self._table_rows is not None
                self._table_rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_rows is not None:
            self.tables.append(
                {
                    "table_index": len(self.tables),
                    "section": self.section,
                    "row_count_including_header": len(self._table_rows),
                    "rows": self._table_rows,
                }
            )
            self._table_rows = None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_displayed_tables(report: Path) -> list[dict[str, object]]:
    parser = _TableParser()
    parser.feed(report.read_text(encoding="utf-8"))
    return parser.tables


def extract_top_level_sections(report: Path) -> list[str]:
    parser = _TableParser()
    parser.feed(report.read_text(encoding="utf-8"))
    return parser.sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    tables = extract_displayed_tables(Path(fixture["source_report"]))
    if not tables:
        raise ValueError("no displayed tables found in Matt's report")
    fixture["displayed_table_count"] = len(tables)
    fixture["displayed_tables"] = tables
    sections = extract_top_level_sections(Path(fixture["source_report"]))
    if not sections:
        raise ValueError("no top-level sections found in Matt's report")
    fixture["historical_sections"] = sections
    state_counts: dict[str, dict[str, int]] = {"donor1": {}, "donor2": {}}
    for table in tables:
        rows = table["rows"]
        if not rows:
            continue
        header = rows[0]
        if "state" not in header or "n d1" not in header or "n d2" not in header:
            continue
        state_index = header.index("state")
        for donor, count_column in (("donor1", "n d1"), ("donor2", "n d2")):
            count_index = header.index(count_column)
            for row in rows[1:]:
                if len(row) <= max(state_index, count_index):
                    continue
                state, count = row[state_index], row[count_index]
                if state and count and count != "NaN":
                    state_counts[donor][state] = int(float(count))
    fixture["displayed_state_target_counts"] = state_counts
    args.fixture.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(tables)} displayed tables to {args.fixture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
