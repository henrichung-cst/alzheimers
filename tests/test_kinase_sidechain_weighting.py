from __future__ import annotations

import subprocess
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from alz.cross_reference import kinase_kinase_edges as edges
from alz.cross_reference import kinase_incytr_bridge as bridge


ROOT = Path(__file__).resolve().parents[1]
SIDECHAINS_JS = (
    ROOT / "alz" / "viewer_shared" / "template" / "js" / "tabs" /
    "incytr_sidechains.js"
)


class KinaseSidechainBackendTests(unittest.TestCase):
    def test_substrate_bridge_counts_distinct_leading_motifs_per_gene(self) -> None:
        mea = pd.DataFrame(
            {
                "kinase": ["K1"],
                "contrast": ["d13_d2"],
                "track": ["st"],
                "NES": [2.5],
                "FDR": [0.01],
                # Repeating MOTIFA must not inflate the count; both distinct
                # motifs map to GENE_A, while only one maps to GENE_B.
                "Leading substrates": ["_MOTIFA_;_MOTIFB_;_MOTIFA_"],
            }
        )
        stoich = pd.DataFrame(
            {
                "motif": ["MOTIFA", "MOTIFA", "MOTIFB", "MOTIFB"],
                "gene_symbol": ["GENE_A", "GENE_A", "GENE_A", "GENE_B"],
            }
        )

        result = bridge.build_substrate_bridge(mea, stoich).set_index("gene_symbol")

        self.assertEqual(list(result.columns), ["kinase", "contrast", "channel", "NES", "FDR", "n_sites"])
        self.assertEqual(result.loc["GENE_A", "n_sites"], 2)
        self.assertEqual(result.loc["GENE_B", "n_sites"], 1)

    def test_terminal_and_interactome_motif_weights_use_abs_nes(self) -> None:
        motif = pd.DataFrame(
            {
                "kinase_gene": ["LOW", "HIGH", "LOW", "HIGH"],
                "target_gene": ["NODE", "NODE", "HIGH", "HIGH"],
                "role": ["Receptor", "Receptor", "Receptor", "Receptor"],
                "contrast": ["d13_d2"] * 4,
                "best_abs_pds": [100.0, 1.0, 100.0, 1.0],
                "best_abs_nes": [1.0, 4.0, 1.0, 4.0],
                "signed_nes": [-1.0, 4.0, -1.0, 4.0],
                "best_fdr": [0.01, 0.01, 0.01, 0.01],
                "n_sites": [1, 3, 1, 3],
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
                "best_abs_nes", "signed_nes", "best_fdr", "n_sites",
            ],
        )
        self.assertEqual(by_kinase.loc["LOW", "best_abs_pds"], 100.0)
        self.assertEqual(by_kinase.loc["HIGH", "best_abs_pds"], 1.0)
        self.assertEqual(by_kinase.loc["LOW", "weight_motif"], 0.25)
        self.assertEqual(by_kinase.loc["HIGH", "weight_motif"], 1.0)
        self.assertEqual(by_kinase.loc["LOW", "n_sites"], 1)
        self.assertEqual(by_kinase.loc["HIGH", "n_sites"], 3)

        interactome = edges.build_interactome(motif.iloc[2:], psp)
        self.assertEqual(interactome.iloc[0]["source_gene"], "HIGH")
        self.assertEqual(interactome.iloc[0]["weight_motif"], 1.0)

    def test_load_motif_edges_carries_arg_max_aligned_nes_and_sites(self) -> None:
        bridge_root = self._tmp_path()
        source = bridge_root / "cohort" / "kinase_node_hits.parquet"
        source.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "kinase": ["K1", "K1", "K1"],
                "gene_symbol": ["GENE", "GENE", "GENE"],
                "role": ["Receptor", "Receptor", "Receptor"],
                "contrast": ["d13_d2", "d13_d2", "d13_d2"],
                "channel": ["st", "st", "py"],
                "best_abs_pds": [2.0, 3.0, 1.0],
                "NES": [-2.5, 1.5, 2.5],
                "FDR": [0.02, 0.01, 0.03],
                "n_sites": [4, 1, 9],
                "celltype_match": [False, True, True],
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
                "best_abs_nes", "signed_nes", "best_fdr", "n_sites", "celltype_match",
            ],
        )
        row = result.iloc[0]
        self.assertEqual(row["best_abs_pds"], 3.0)
        self.assertEqual(row["best_abs_nes"], 2.5)
        self.assertEqual(row["signed_nes"], -2.5)
        self.assertEqual(abs(row["signed_nes"]), row["best_abs_nes"])
        self.assertEqual(row["n_sites"], 4)
        self.assertEqual(row["best_fdr"], 0.01)
        self.assertTrue(row["celltype_match"])

    @staticmethod
    def _tmp_path() -> Path:
        import tempfile

        return Path(tempfile.mkdtemp(prefix="kinase-sidechain-test-"))


class KinaseSidechainViewerTests(unittest.TestCase):
    def test_viewer_draws_all_edges_and_encodes_nes_by_emphasis(self) -> None:
        script = textwrap.dedent(
            f"""
            const fs = require("fs");
            const vm = require("vm");
            const src = fs.readFileSync({str(SIDECHAINS_JS)!r}, "utf8");
            const ctx = {{window: {{}}}};
            vm.runInNewContext(src, ctx);

            if (!src.includes("requestFullscreen") || !src.includes("fullscreenchange")
                || !src.includes("cy.resize()"))
              throw new Error("fullscreen control does not refit Cytoscape after resizing");
            if (!src.includes("infoPanel.append(filterControls, legend, edgeDetail, nodeRelationDetail)")
                || !src.includes("nodeRelationDetail.replaceChildren(_isNodeRelationTable(evt.target, cy))")
                || !src.includes('nodeRelationDetail.textContent = "Tap a node for its relationships.";')
                || !src.includes("panel.append(graphHost, fullscreenButton, infoPanel)"))
              throw new Error("fullscreen panel does not retain legend and edge evidence");
            if (!src.includes('selector: "edge.is-focus-edge"')
                || !src.includes('keep.edges().addClass("is-focus-edge")'))
              throw new Error("focused connectors do not reset opacity");
            if (!src.includes('focus(edge.closedNeighborhood().union(spine), detail)'))
              throw new Error("edge taps do not focus the selected edge and endpoints");
            if (src.includes('showAllEvidence')
                || !src.includes('showChains.checked = false')
                || !src.includes(`const chains = cy.edges("edge[kind = 'chain-edge']")`)
                || !src.includes('node.addClass("is-node-filtered")'))
              throw new Error("first-order filter does not constrain kinase-chain nodes");
            if (!src.includes("_isLegendSample") || src.includes("Bold ${{_IS_COLORS"))
              throw new Error("legend does not use visual color samples");

            const terminalFields = [
              "source_gene", "target_gene", "role", "contrast", "provenance", "weight",
              "best_abs_nes", "signed_nes", "best_fdr", "n_sites",
            ];
            const terminalRows = [];
            for (let i = 0; i < 12; i++) {{
              terminalRows.push({{
                source_gene: `RK${{i}}`, target_gene: "R", role: "Receptor",
                contrast: "d13_d2", provenance: "motif", weight: i,
                best_abs_nes: 1 + i * 0.3, signed_nes: i % 2 ? -(1 + i * 0.3) : 1 + i * 0.3,
                best_fdr: 0.01, n_sites: 1,
              }});
            }}
            // A high-|NES| edge that also has a poor FDR: under C it is NOT gated —
            // it must still be drawn (FDR no longer selects).
            terminalRows.push({{
              source_gene: "HIF", target_gene: "R", role: "Receptor",
              contrast: "d13_d2", provenance: "motif", weight: 100,
              best_abs_nes: 5, signed_nes: 5, best_fdr: 0.5, n_sites: 1,
            }});
            const terminal = {{}};
            for (const field of terminalFields) terminal[field] = terminalRows.map(row => row[field]);

            const chainRows = [
              {{source_gene: "RK1", target_gene: "RK2", provenance: "motif", weight: 0.5}},
              {{source_gene: "UPSTREAM", target_gene: "RK2", provenance: "motif", weight: 100}},
              {{source_gene: "RK3", target_gene: "OUTSIDE", provenance: "motif", weight: 100}},
            ];
            const interactome = {{}};
            for (const field of ["source_gene", "target_gene", "provenance", "weight"])
              interactome[field] = chainRows.map(row => row[field]);

            const graph = ctx._isGraphForRow(
              {{interactome, terminal_edges: terminal}},
              {{Ligand: "L", Receptor: "R", EM: "E", Target: "T", contrast: "d13_d2"}},
            );
            // No cutoff: all 13 contrast-matched edges drawn, incl. the poor-FDR one.
            if (graph.terminalEdges.length !== 13) throw new Error("terminal edges were filtered");
            if (!graph.terminalEdges.some(edge => edge.source_gene === "HIF"))
              throw new Error("a poor-FDR edge was wrongly dropped");
            if (graph.directKinaseGenes.size !== 13) throw new Error("direct kinase set was pruned");
            if (!graph.kinaseGenes.has("UPSTREAM") || !graph.kinaseGenes.has("OUTSIDE"))
              throw new Error("full view omitted one-hop kinase regulators");
            if (graph.nesMax !== 5) throw new Error(`nesMax was ${{graph.nesMax}}`);
            if (graph.sitesMax !== 1) throw new Error(`sitesMax was ${{graph.sitesMax}}`);
            // Full view adds the one-hop kinase→kinase neighborhood, while the
            // first-order UI filter hides its chain edges and chain-only nodes.
            if (graph.chainEdges.length !== 3)
              throw new Error("full view did not retain the one-hop kinase chains");

            // Emphasis: convex, anchored at the null; monotone in |NES|.
            const near = 1e-9;
            if (ctx._isEmphasis(1.0, 1.0, 5) !== 0) throw new Error("null NES did not map to 0");
            if (ctx._isEmphasis(5, 1, 5) !== 1) throw new Error("max NES did not map to 1");
            if (Math.abs(ctx._isEmphasis(3, 1, 5) - Math.pow(0.5, 3.5)) > near)
              throw new Error("gamma=3.5 midpoint wrong");  // ((3-1)/(5-1))^3.5
            if (!(ctx._isEmphasis(2, 1, 5) < ctx._isEmphasis(4, 1, 5)))
              throw new Error("emphasis not monotone");

            // Positioned elements: the strongest edge is thicker AND more opaque
            // than a null-NES edge (RK0, |NES|=1 → emphasis 0 → floor width/opacity).
            const els = ctx._isPositionedElements(graph, 900, 500);
            const byId = new Map(els.map(el => [el.data.id, el.data]));
            const strong = els.find(el => el.data.kind === "terminal-edge"
              && el.data.source === "kinase:HIF").data;
            const weak = els.find(el => el.data.kind === "terminal-edge"
              && el.data.source === "kinase:RK0").data;
            if (!(strong.width > weak.width) || !(strong.opacity > weak.opacity))
              throw new Error("emphasis did not encode NES into width+opacity");
            if (Math.abs(weak.opacity - 0.03) > near || Math.abs(weak.width - 0.35) > near)
              throw new Error("null-NES edge not floored to faint-but-present");

            // Node prominence: a kinase node's size tracks its strongest |NES|.
            // HIF (|NES|=5 → emphasis 1) is the max-diameter node; RK0 (|NES|=1 →
            // emphasis 0) shrinks to the min-diameter dot and is unlabeled.
            const strongNode = byId.get("kinase:HIF");
            const weakNode = byId.get("kinase:RK0");
            if (!(strongNode.size > weakNode.size))
              throw new Error("node size did not track |NES| emphasis");
            if (Math.abs(strongNode.emphasis - 1) > near || Math.abs(weakNode.emphasis - 0) > near)
              throw new Error("node emphasis not anchored at the null");
            if (Math.abs(strongNode.size - 34) > near || Math.abs(weakNode.size - 9) > near)
              throw new Error("node size not mapped to [9, 34] diameter range");

            // Substrate multiplicity changes terminal-edge emphasis only: equal-NES
            // terminal edges with different counts get different widths, while their
            // kinase-node sizes remain equal because node size reads |NES| alone.
            const multiplicityGraph = ctx._isGraphForRow(
              {{interactome: {{}}, terminal_edges: {{
                source_gene: ["FAN", "FAN"], target_gene: ["R", "E"],
                role: ["Receptor", "EM"], contrast: ["d13_d2", "d13_d2"],
                provenance: ["motif", "motif"], weight: [0, 0],
                best_abs_nes: [5, 5], signed_nes: [5, 5], best_fdr: [0.01, 0.01],
                n_sites: [4, 1],
              }}}},
              {{Ligand: "L", Receptor: "R", EM: "E", Target: "T", contrast: "d13_d2"}},
            );
            const multiplicityEls = ctx._isPositionedElements(multiplicityGraph, 900, 500);
            const multiEdge = multiplicityEls.find(el => el.data.id === "terminal:0").data;
            const singleEdge = multiplicityEls.find(el => el.data.id === "terminal:1").data;
            const fanNode = multiplicityEls.find(el => el.data.id === "kinase:FAN").data;
            if (!(multiEdge.width > singleEdge.width))
              throw new Error("substrate count did not differentiate one kinase fan");
            const flatMultiplicityGraph = ctx._isGraphForRow(
              {{interactome: {{}}, terminal_edges: {{
                source_gene: ["FAN", "FAN"], target_gene: ["R", "E"],
                role: ["Receptor", "EM"], contrast: ["d13_d2", "d13_d2"],
                provenance: ["motif", "motif"], weight: [0, 0],
                best_abs_nes: [5, 5], signed_nes: [5, 5], best_fdr: [0.01, 0.01],
                n_sites: [1, 1],
              }}}},
              {{Ligand: "L", Receptor: "R", EM: "E", Target: "T", contrast: "d13_d2"}},
            );
            const flatFanNode = ctx._isPositionedElements(flatMultiplicityGraph, 900, 500)
              .find(el => el.data.id === "kinase:FAN").data;
            if (Math.abs(fanNode.size - flatFanNode.size) > near)
              throw new Error("substrate count changed kinase node size");

            // Signed NES is retained as raw edge evidence and mapped only to a
            // categorical enriched/depleted direction for the terminal-edge hue.
            const negative = els.find(el => el.data.kind === "terminal-edge"
              && el.data.source === "kinase:RK1").data;
            if (strong.signed_nes !== 5 || strong.nes_direction !== "enriched")
              throw new Error("positive signed NES was not carried onto the edge");
            if (!(negative.signed_nes < 0) || negative.nes_direction !== "depleted")
              throw new Error("negative signed NES was not carried onto the edge");
            if (strong.role !== "Receptor")
              throw new Error("terminal edge did not carry its pathway role");

            // The arc layout assigns each single-target regulator to a deterministic
            // outward wedge of its pathway node rather than alternating up/down lanes.
            const receptorNode = els.find(el => el.data.id === "path:Receptor");
            const receptorKinase = els.find(el => el.data.id === "kinase:RK0");
            const arcCenter = ctx._isArcCenter(900, 500);
            const nodeAngle = Math.atan2(
              receptorNode.position.y - arcCenter.y, receptorNode.position.x - arcCenter.x);
            const kinaseAngle = Math.atan2(
              receptorKinase.position.y - arcCenter.y, receptorKinase.position.x - arcCenter.x);
            const halfWedge = Math.PI * 90 / 180 / (2 * 3);
            if (Math.abs(nodeAngle - kinaseAngle) > halfWedge)
              throw new Error("kinase did not land in its receptor wedge");

            // Node relation tables are direction-aware and independent of the
            // graph's chain visibility filter.
            class FakeElement {{
              constructor(tagName) {{
                this.tagName = tagName.toUpperCase();
                this.children = [];
                this.attributes = {{}};
                this.style = {{}};
                this._text = "";
              }}
              append(...children) {{ this.children.push(...children); }}
              appendChild(child) {{ this.children.push(child); return child; }}
              setAttribute(name, value) {{ this.attributes[name] = String(value); }}
              set textContent(value) {{ this._text = String(value); this.children = []; }}
              get textContent() {{
                return this._text + this.children.map(child => child.textContent || "").join("");
              }}
            }}
            ctx.document = {{
              createElement: tagName => new FakeElement(tagName),
              createTextNode: value => ({{textContent: String(value)}}),
            }};
            const makeNode = (id, kind, label, role = "") => {{
              const node = {{
                id: () => id,
                data: key => ({{id, kind, label, role}})[key],
                connectedEdges: () => node._edges,
                _edges: [],
              }};
              return node;
            }};
            const spineNode = makeNode("path:Receptor", "spine-node", "R", "Receptor");
            const high = makeNode("kinase:HIGH", "kinase-node", "HIGH");
            const medium = makeNode("kinase:MEDIUM", "kinase-node", "MEDIUM");
            const low = makeNode("kinase:LOW", "kinase-node", "LOW");
            const upstream = makeNode("kinase:UPSTREAM", "kinase-node", "UPSTREAM");
            const downstream = makeNode("kinase:DOWNSTREAM", "kinase-node", "DOWNSTREAM");
            const makeEdge = (source, target, data) => {{
              const edge = {{
                source: () => source,
                target: () => target,
                data: key => data[key],
              }};
              source._edges.push(edge);
              target._edges.push(edge);
              return edge;
            }};
            makeEdge(high, spineNode, {{
              kind: "terminal-edge", role: "Receptor", signed_nes: 4,
              nes_direction: "enriched", provenance: "motif",
            }});
            makeEdge(medium, spineNode, {{
              kind: "terminal-edge", role: "Receptor", signed_nes: 2,
              nes_direction: "enriched", provenance: "motif",
            }});
            makeEdge(low, spineNode, {{
              kind: "terminal-edge", role: "Receptor", signed_nes: -1,
              nes_direction: "depleted", provenance: "psp",
            }});
            makeEdge(upstream, high, {{
              kind: "chain-edge", provenance: "psp", weight: 0.8,
            }});
            makeEdge(high, downstream, {{
              kind: "chain-edge", provenance: "both", weight: 1.5,
            }});
            const spineTable = ctx._isNodeRelationTable(spineNode, {{}});
            if (spineTable.tagName !== "TABLE") throw new Error("spine tap did not produce a table");
            if (!spineTable.textContent.includes("3 kinases affecting · 2 enriched · 1 depleted"))
              throw new Error(`unexpected spine summary: ${{spineTable.textContent}}`);
            const spineBody = spineTable.children.find(child => child.tagName === "TBODY");
            if (!spineBody || spineBody.children.length !== 3)
              throw new Error("spine table row count was wrong");
            if (!spineBody.children[0].textContent.includes("HIGH → R")
                || !spineBody.children[1].textContent.includes("MEDIUM → R")
                || !spineBody.children[2].textContent.includes("LOW → R"))
              throw new Error("spine rows were not ordered by absolute NES");
            if (!spineTable.textContent.includes("4.000")
                || !spineTable.textContent.includes("-1.000"))
              throw new Error("spine table omitted signed NES evidence");

            const kinaseTable = ctx._isNodeRelationTable(high, {{}});
            if (kinaseTable.tagName !== "TABLE") throw new Error("kinase tap did not produce a table");
            if (!kinaseTable.textContent.includes("targets 1 nodes · 2 kinases"))
              throw new Error(`unexpected kinase summary: ${{kinaseTable.textContent}}`);
            const kinaseBody = kinaseTable.children.find(child => child.tagName === "TBODY");
            if (!kinaseBody || kinaseBody.children.length !== 3)
              throw new Error("kinase table row count was wrong");
            if (!kinaseBody.children[0].textContent.includes("HIGH → R")
                || !kinaseBody.children[0].textContent.includes("Receptor")
                || !kinaseBody.children[1].textContent.includes("HIGH → DOWNSTREAM")
                || !kinaseBody.children[2].textContent.includes("UPSTREAM → HIGH"))
              throw new Error("kinase rows did not put terminal NES rows before chains");
            if (!kinaseBody.children[1].textContent.includes("1.500")
                || !kinaseBody.children[2].textContent.includes("0.800"))
              throw new Error("chain rows were not ordered by weight");
            """
        )
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
