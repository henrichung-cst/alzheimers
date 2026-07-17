from __future__ import annotations

import subprocess
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from alz.cross_reference import kinase_kinase_edges as edges


ROOT = Path(__file__).resolve().parents[1]
SIDECHAINS_JS = (
    ROOT / "alz" / "viewer_shared" / "template" / "js" / "tabs" /
    "incytr_sidechains.js"
)


class KinaseSidechainBackendTests(unittest.TestCase):
    def test_terminal_and_interactome_motif_weights_use_abs_nes(self) -> None:
        motif = pd.DataFrame(
            {
                "kinase_gene": ["LOW", "HIGH", "LOW", "HIGH"],
                "target_gene": ["NODE", "NODE", "HIGH", "HIGH"],
                "role": ["Receptor", "Receptor", "Receptor", "Receptor"],
                "contrast": ["d13_d2"] * 4,
                "best_abs_pds": [100.0, 1.0, 100.0, 1.0],
                "best_abs_nes": [1.0, 4.0, 1.0, 4.0],
                "best_fdr": [0.01, 0.01, 0.01, 0.01],
                "celltype_match": [True] * 4,
            }
        )
        psp = pd.DataFrame(
            columns=["source_gene", "target_gene", "in_vivo_refs", "in_vitro_refs"]
        )

        terminal = edges.build_terminal_map(motif.iloc[:2], psp)
        by_kinase = terminal.set_index("source_gene")
        self.assertEqual(
            list(terminal.columns),
            [
                "source_gene", "target_gene", "role", "contrast", "celltype_match",
                "provenance", "weight", "weight_lit", "weight_motif", "best_abs_pds",
                "best_abs_nes", "best_fdr",
            ],
        )
        self.assertEqual(by_kinase.loc["LOW", "best_abs_pds"], 100.0)
        self.assertEqual(by_kinase.loc["HIGH", "best_abs_pds"], 1.0)
        self.assertEqual(by_kinase.loc["LOW", "weight_motif"], 0.25)
        self.assertEqual(by_kinase.loc["HIGH", "weight_motif"], 1.0)

        interactome = edges.build_interactome(motif.iloc[2:], psp)
        self.assertEqual(interactome.iloc[0]["source_gene"], "HIGH")
        self.assertEqual(interactome.iloc[0]["weight_motif"], 1.0)

    def test_load_motif_edges_carries_nes_and_fdr(self) -> None:
        bridge_root = self._tmp_path()
        source = bridge_root / "cohort" / "kinase_node_hits.parquet"
        source.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "kinase": ["K1", "K1"],
                "gene_symbol": ["GENE", "GENE"],
                "role": ["Receptor", "Receptor"],
                "contrast": ["d13_d2", "d13_d2"],
                "best_abs_pds": [2.0, 3.0],
                "NES": [-2.5, 1.5],
                "FDR": [0.02, 0.01],
                "celltype_match": [False, True],
            }
        ).to_parquet(source, index=False)

        original_root = edges.BRIDGE_ROOT
        original_map = edges.load_kinase_abbrev_map
        try:
            edges.BRIDGE_ROOT = bridge_root
            edges.load_kinase_abbrev_map = lambda: {"K1": "KIN1"}
            result = edges.load_motif_edges("cohort", is_mouse=False)
        finally:
            edges.BRIDGE_ROOT = original_root
            edges.load_kinase_abbrev_map = original_map

        self.assertEqual(
            list(result.columns),
            [
                "kinase_gene", "target_gene", "role", "contrast", "best_abs_pds",
                "best_abs_nes", "best_fdr", "celltype_match",
            ],
        )
        row = result.iloc[0]
        self.assertEqual(row["best_abs_pds"], 3.0)
        self.assertEqual(row["best_abs_nes"], 2.5)
        self.assertEqual(row["best_fdr"], 0.01)
        self.assertTrue(row["celltype_match"])

    @staticmethod
    def _tmp_path() -> Path:
        import tempfile

        return Path(tempfile.mkdtemp(prefix="kinase-sidechain-test-"))


class KinaseSidechainViewerTests(unittest.TestCase):
    def test_viewer_selects_fdr_gated_top_n_and_induced_chain(self) -> None:
        script = textwrap.dedent(
            f"""
            const fs = require("fs");
            const vm = require("vm");
            const src = fs.readFileSync({str(SIDECHAINS_JS)!r}, "utf8");
            const ctx = {{window: {{}}}};
            vm.runInNewContext(src, ctx);

            const terminalFields = [
              "source_gene", "target_gene", "role", "contrast", "provenance", "weight",
              "best_abs_nes", "best_fdr",
            ];
            const terminalRows = [];
            for (let i = 0; i < 12; i++) {{
              terminalRows.push({{
                source_gene: `RK${{i}}`, target_gene: "R", role: "Receptor",
                contrast: "d13_d2", provenance: "motif", weight: i,
                best_abs_nes: i, best_fdr: 0.01,
              }});
            }}
            terminalRows.push({{
              source_gene: "FDR_FAIL", target_gene: "R", role: "Receptor",
              contrast: "d13_d2", provenance: "motif", weight: 100,
              best_abs_nes: 100, best_fdr: 0.05,
            }});
            const terminal = {{}};
            for (const field of terminalFields) terminal[field] = terminalRows.map(row => row[field]);

            const chainRows = [
              {{source_gene: "RK2", target_gene: "RK3", provenance: "motif", weight: 0.5}},
              {{source_gene: "UPSTREAM", target_gene: "RK2", provenance: "motif", weight: 100}},
              {{source_gene: "RK0", target_gene: "RK12", provenance: "motif", weight: 100}},
              {{source_gene: "OUTSIDE", target_gene: "OUTSIDE2", provenance: "motif", weight: 100}},
            ];
            const interactome = {{}};
            for (const field of ["source_gene", "target_gene", "provenance", "weight"])
              interactome[field] = chainRows.map(row => row[field]);

            const graph = ctx._isGraphForRow(
              {{interactome, terminal_edges: terminal}},
              {{Ligand: "L", Receptor: "R", EM: "E", Target: "T", contrast: "d13_d2"}},
            );
            if (graph.terminalEdges.length !== 10) throw new Error("top-N selection failed");
            if (graph.terminalEdges.some(edge => edge.source_gene === "FDR_FAIL"))
              throw new Error("FDR boundary was not gated");
            if (graph.hiddenTerminalCounts.Receptor !== 2)
              throw new Error(`hidden count was ${{graph.hiddenTerminalCounts.Receptor}}`);
            if (graph.chainEdges.length !== 1 || graph.chainEdges[0].source_gene !== "RK2")
              throw new Error("chain edges were not induced on drawn kinases");
            if (graph.observedMax !== 11) throw new Error(`observed max was ${{graph.observedMax}}`);
            for (const edge of graph.chainEdges) {{
              if (!graph.kinaseGenes.has(edge.source_gene) || !graph.kinaseGenes.has(edge.target_gene))
                throw new Error("chain edge escaped selected kinase set");
            }}
            """
        )
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
