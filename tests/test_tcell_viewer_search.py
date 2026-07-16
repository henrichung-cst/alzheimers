from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_INDEX = (
    ROOT / "alz" / "viewer_shared" / "template" / "js" / "tabs" /
    "incytr_global_index.js"
)
TCELL_STATE = ROOT / "alz" / "tcell_viewer" / "template" / "js" / "01_state.js"
UNIFIED_STATE = ROOT / "alz" / "viewer" / "template" / "js" / "01_state.js"
INCYTR_PATHWAYS = (
    ROOT / "alz" / "viewer_shared" / "template" / "js" / "tabs" /
    "incytr_pathways.js"
)


class TCellViewerSearchTests(unittest.TestCase):
    def _run_node(self, script: str) -> None:
        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gene_search_requires_an_exact_symbol(self) -> None:
        script = textwrap.dedent(
            f"""
            const fs = require("fs");
            const vm = require("vm");
            let src = fs.readFileSync({str(GLOBAL_INDEX)!r}, "utf8");
            src = src.replace(
              "return {{ available, loaded, manifest, ensureLoaded, filterRank, materialize,",
              "return {{ _member, available, loaded, manifest, ensureLoaded, filterRank, materialize,"
            );
            const ctx = {{window: {{}}}};
            vm.runInNewContext(src, ctx);
            const member = ctx.window.IncytrGlobalIndex._member;
            const vocab = ["pdcd1", "pdcd10", "pdcd11"];
            const exact = Array.from(member(vocab, "pdcd1"));
            const absent = Array.from(member(vocab, "pdcd"));
            const descriptive = Array.from(member(vocab, "pdcd", true));
            if (exact.join(",") !== "1,0,0") {{
              throw new Error(`exact PDCD1 mask was ${{exact}}`);
            }}
            if (absent.join(",") !== "0,0,0") {{
              throw new Error(`non-exact PDCD gene mask was ${{absent}}`);
            }}
            if (descriptive.join(",") !== "1,1,1") {{
              throw new Error(`descriptive PDCD mask was ${{descriptive}}`);
            }}
            """
        )
        self._run_node(script)

    def test_unified_state_shares_exact_token_helpers(self) -> None:
        # The two 01_state.js copies are duplicated, not shared, so the
        # exact-token search helpers must stay byte-identical or the unified
        # viewer silently reverts to substring matching.
        def _block(path: Path) -> str:
            src = path.read_text()
            start = src.index("function _searchValueHasExactToken")
            end = src.index("// Canonical metric glossary", start)
            return src[start:end].strip()

        self.assertEqual(_block(UNIFIED_STATE), _block(TCELL_STATE))
        self.assertIn(
            "_searchValuesMatch(Object.values(r), q)",
            UNIFIED_STATE.read_text(),
        )

    def test_table_and_pair_search_use_gene_token_boundaries(self) -> None:
        script = f"""
            const fs = require("fs");
            const vm = require("vm");

            const stateSrc = fs.readFileSync({str(TCELL_STATE)!r}, "utf8");
            const searchStart = stateSrc.indexOf("function _searchValueHasExactToken");
            const searchEnd = stateSrc.indexOf("// Canonical metric glossary", searchStart);
            const searchCtx = {{}};
            vm.runInNewContext(stateSrc.slice(searchStart, searchEnd), searchCtx);
            if (searchCtx._searchValuesMatch(["PDCD10", "PDCD11"], "pdcd1")) {{
              throw new Error("table search matched PDCD10/11 for PDCD1");
            }}
            if (!searchCtx._searchValuesMatch(["GZMB*PDCD1*TOX"], "pdcd1")) {{
              throw new Error("table search missed an exact PDCD1 path token");
            }}

            const pathSrc = fs.readFileSync({str(INCYTR_PATHWAYS)!r}, "utf8");
            const pairStart = pathSrc.indexOf("function _ipRowHasExactGeneSearchValue");
            const pairEnd = pathSrc.indexOf("function _ipBuildPathIndexes", pairStart);
            const pairCtx = {{}};
            vm.runInNewContext(pathSrc.slice(pairStart, pairEnd), pairCtx);
            const row = {{
              _sender: "CD8Exhausted", _receiver: "CD4RestingMemory",
              Ligand: "PDCD10", Receptor: "PDCD11", EM: "TOX", Target: "GZMB",
              contrast: "d20_d2",
            }};
            if (pairCtx._ipRowHasExactGeneSearchValue(row, "pdcd1")) {{
              throw new Error("pair search matched PDCD10/11 for PDCD1");
            }}
            if (!pairCtx._ipRowHasDescriptiveSearchValue(row, "exhaust")) {{
              throw new Error("descriptive state substring search regressed");
            }}

            // The reported bug: "TOX" must match the EM gene node, not the
            // "cytoTOXic" substring inside a sender/receiver state name.
            const rowMatches = (r, tok) =>
              pairCtx._ipRowHasExactGeneSearchValue(r, tok)
              || pairCtx._ipRowHasDescriptiveSearchValue(r, tok);
            const cytoRow = {{
              _sender: "CD8Cytotoxic", _receiver: "CD4Cytotoxic",
              Ligand: "GZMB", Receptor: "PDCD1", EM: "IFNG", Target: "IL2",
              contrast: "d20_d2",
            }};
            if (rowMatches(cytoRow, "tox")) {{
              throw new Error("TOX swallowed by a Cytotoxic sender/receiver");
            }}
            const toxNodeRow = {{
              _sender: "CD4RestingMemory", _receiver: "CD8Naive",
              Ligand: "GZMB", Receptor: "PDCD1", EM: "TOX", Target: "IL2",
              contrast: "d20_d2",
            }};
            if (!rowMatches(toxNodeRow, "tox")) {{
              throw new Error("TOX missed where it is an actual path node");
            }}
        """
        self._run_node(script)


if __name__ == "__main__":
    unittest.main()
