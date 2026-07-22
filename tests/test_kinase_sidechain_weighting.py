from __future__ import annotations

import subprocess
import textwrap
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alz.cross_reference import kinase_kinase_edges as edges
from alz.cross_reference import kinase_incytr_bridge as bridge


ROOT = Path(__file__).resolve().parents[1]
SIDECHAINS_JS = (
    ROOT / "alz" / "viewer_shared" / "template" / "js" / "tabs" /
    "incytr_sidechains.js"
)


class KinaseSidechainBackendTests(unittest.TestCase):
    def test_substrate_bridge_uses_floor99_motifs_without_fdr_gate(self) -> None:
        mea = pd.DataFrame(
            {
                "kinase": ["K1"],
                "contrast": ["d13_d2"],
                "track": ["st"],
                "NES": [2.5],
                # A poor MEA FDR must not remove an otherwise eligible floor-99 row.
                "FDR": [0.9],
            }
        )
        stoich = pd.DataFrame(
            {
                "site_id": ["S_A", "S_B", "S_C"],
                "motif": ["MOTIFA", "MOTIFB", "MOTIFC"],
                "gene_symbol": ["GENE_A", "GENE_A", "GENE_B"],
            }
        )
        substrate_sets = pd.DataFrame(
            {
                "kinase": ["K1", "K1", "K1", "K1"],
                "contrast": ["d13_d2"] * 4,
                "motif": ["MOTIFA", "MOTIFB", "MOTIFA", "MOTIFC"],
                "kl_percentile": [99.0, 100.0, 98.0, 99.5],
            }
        )

        result = bridge.build_substrate_bridge(mea, stoich, substrate_sets).set_index("gene_symbol")

        self.assertEqual(
            list(result.columns),
            ["kinase", "contrast", "channel", "NES", "FDR", "n_sites", "sites"],
        )
        self.assertEqual(result.loc["GENE_A", "n_sites"], 2)
        self.assertEqual(result.loc["GENE_B", "n_sites"], 1)
        self.assertEqual(result.loc["GENE_A", "FDR"], 0.9)
        sites = json.loads(result.loc["GENE_A", "sites"])
        self.assertEqual([site["site_id"] for site in sites], ["S_B", "S_A"])
        self.assertEqual([site["site_position"] for site in sites], [None, None])
        self.assertEqual([site["motif"] for site in sites], ["MOTIFB", "MOTIFA"])
        self.assertEqual([site["residue_type"] for site in sites], [None, None])
        self.assertEqual([site["kl_percentile"] for site in sites], [100.0, 99.0])

    def test_substrate_bridge_keeps_track_specific_pY_gene_mapping(self) -> None:
        mea = pd.DataFrame(
            {
                "kinase": ["KPY"],
                "contrast": ["d13_d2"],
                "track": ["py"],
                "NES": [-1.75],
                "FDR": [0.8],
            }
        )
        py_stoich = pd.DataFrame(
            {"site_id": ["TYR_Y1"], "motif": ["PYMOTIF"], "gene_symbol": ["TYR_NODE"]}
        )
        substrate_sets = pd.DataFrame(
            {
                "kinase": ["KPY"],
                "contrast": ["d13_d2"],
                "motif": ["PYMOTIF"],
                "kl_percentile": [99.25],
            }
        )

        result = bridge.build_substrate_bridge(mea, py_stoich, substrate_sets)

        self.assertEqual(result.loc[0, "gene_symbol"], "TYR_NODE")
        self.assertEqual(result.loc[0, "channel"], "py")
        self.assertEqual(result.loc[0, "NES"], -1.75)

    def test_terminal_and_interactome_provenance_and_nes(self) -> None:
        motif = pd.DataFrame(
            {
                "kinase": ["LOW", "HIGH", "LOW", "HIGH"],
                "kinase_gene": ["LOW", "HIGH", "LOW", "HIGH"],
                "target_gene": ["NODE", "NODE", "HIGH", "HIGH"],
                "role": ["Receptor", "Receptor", "Receptor", "Receptor"],
                "contrast": ["d13_d2"] * 4,
                "owning_cluster": ["c1"] * 4,
                "best_abs_pds": [100.0, 1.0, 100.0, 1.0],
                "best_abs_nes": [1.0, 4.0, 1.0, 4.0],
                "signed_nes": [-1.0, 4.0, -1.0, 4.0],
                "best_fdr": [0.01, 0.01, 0.01, 0.01],
                "n_sites": [1, 3, 1, 3],
                "n_significant_concordant": [1, 1, 1, 1],
                "sites": [
                    json.dumps([{"site_id": "L1", "delta": -1, "site_significance": 0.01, "concordant": True, "changed": True}]),
                    json.dumps([{"site_id": f"H{i}", "delta": i + 1, "site_significance": 0.01, "concordant": True, "changed": True} for i in range(3)]),
                    json.dumps([{"site_id": "L2", "delta": -1, "site_significance": 0.01, "concordant": True, "changed": True}]),
                    json.dumps([{"site_id": f"H{i}", "delta": i + 1, "site_significance": 0.01, "concordant": True, "changed": True} for i in range(3)]),
                ],
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
                "kinase", "source_gene", "target_gene", "role", "contrast", "owning_cluster",
                "celltype_match", "provenance", "best_abs_pds", "best_abs_nes", "signed_nes",
                "best_fdr", "n_sites", "sites",
                "n_significant_concordant", "edge_delta",
            ],
        )
        # No PSP literature → every terminal edge stays motif-only; no fused weight.
        self.assertTrue((terminal["provenance"] == "motif").all())
        self.assertNotIn("weight", terminal.columns)
        # Rows ordered by |NES| descending, not by any fused weight.
        self.assertEqual(list(terminal["source_gene"]), ["HIGH", "LOW"])
        self.assertEqual(by_kinase.loc["LOW", "best_abs_nes"], 1.0)
        self.assertEqual(by_kinase.loc["HIGH", "best_abs_nes"], 4.0)
        self.assertEqual(by_kinase.loc["LOW", "best_abs_pds"], 100.0)
        self.assertEqual(by_kinase.loc["HIGH", "best_abs_pds"], 1.0)
        self.assertEqual(by_kinase.loc["LOW", "n_sites"], 1)
        self.assertEqual(by_kinase.loc["HIGH", "n_sites"], 3)
        self.assertEqual(by_kinase.loc["HIGH", "edge_delta"], 2.0)

        interactome = edges.build_interactome(motif.iloc[2:], psp)
        self.assertEqual(
            list(interactome.columns),
            [
                "source_gene", "target_gene", "provenance",
                "in_vivo_refs", "in_vitro_refs", "n_motif_contrasts", "motif_contrasts",
            ],
        )
        self.assertTrue((interactome["provenance"] == "motif").all())
        self.assertNotIn("weight", interactome.columns)
        self.assertEqual(interactome.iloc[0]["source_gene"], "HIGH")

    def test_significance_b_is_two_sided_and_bh_corrected_with_robust_bins(self) -> None:
        rng = np.random.default_rng(20260721)
        n_sites = 90
        baseline = 10.0 + rng.uniform(-0.5, 0.5, n_sites)
        delta = rng.normal(0.0, 0.1, n_sites)
        delta[30] = 3.0
        delta[60] = -3.0
        matrix = pd.DataFrame({
            "site_id": [f"S{i}" for i in range(n_sites)],
            "D1_d2": baseline,
            "D1_d13": baseline + delta,
        })

        calls = bridge.compute_site_significance_b(matrix, "D1_d13_vs_d2")
        positive = calls.loc[calls["site_id"].eq("S30")].iloc[0]
        negative = calls.loc[calls["site_id"].eq("S60")].iloc[0]
        self.assertLess(float(positive["site_significance"]), 0.05)
        self.assertLess(float(negative["site_significance"]), 0.05)
        self.assertTrue(bool(positive["site_changed"]))
        self.assertTrue(bool(negative["site_changed"]))
        self.assertTrue((calls["site_significance"].dropna() >= 0).all())
        self.assertTrue((calls["site_significance"].dropna() <= 1).all())

        raw = pd.Series([0.001, 0.02, 0.5])
        self.assertEqual(
            list(bridge._benjamini_hochberg(raw).round(3)),
            [0.003, 0.03, 0.5],
        )

    def test_concordance_uses_delta_and_signed_nes_signs(self) -> None:
        self.assertTrue(bridge._site_concordant(2.0, 1.5))
        self.assertTrue(bridge._site_concordant(-2.0, -1.5))
        self.assertFalse(bridge._site_concordant(2.0, -1.5))
        self.assertFalse(bridge._site_concordant(-2.0, 1.5))

    def test_timecourse_consistency_spans_contrasts_including_mea_ineligible(self) -> None:
        # Site S0 moves +3 in BOTH d13 (MEA-eligible, FDR 0.01) and d15
        # (MEA-ineligible, FDR 0.40).  Only the eligible contrast is emitted as an
        # edge, but the consistency count must see both (stored, not gated).
        rng = np.random.default_rng(11)
        n_sites = 60
        baseline = 8.0 + rng.uniform(-0.5, 0.5, n_sites)
        common = rng.normal(0.0, 0.05, n_sites)
        first = np.zeros(n_sites)
        first[0] = 3.0       # d13 movement of S0
        second = np.zeros(n_sites)
        second[0] = 3.0      # d15 movement of the same site
        matrix = pd.DataFrame({
            "site_id": [f"S{i}" for i in range(n_sites)],
            "D1_d2": baseline,
            "D1_d13": baseline + common + first,
            "D1_d15": baseline + common + second,
        })
        site_json = json.dumps([{
            "site_id": "S0", "site_position": "S10", "motif": "M0",
            "residue_type": "ST", "kl_percentile": 99.0,
        }])
        substrate = pd.DataFrame({
            "kinase": ["K1", "K1"],
            "contrast": ["D1_d13_vs_d2", "D1_d15_vs_d2"],
            "channel": ["st", "st"],
            "NES": [2.0, 2.0],
            "FDR": [0.01, 0.40],   # d13 MEA-eligible; d15 not
            "gene_symbol": ["NODE", "NODE"],
            "n_sites": [1, 1],
            "sites": [site_json, site_json],
        })
        calls = bridge.annotate_tcell_direct_site_changes(substrate, {"st": matrix})

        # Only the MEA-eligible contrast is emitted as an edge row.
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls.iloc[0]["contrast"], "D1_d13_vs_d2")
        self.assertEqual(calls.iloc[0]["n_significant_concordant"], 1)
        site = json.loads(calls.iloc[0]["sites"])[0]
        self.assertTrue(site["concordant"])
        self.assertTrue(site["changed"])
        # ...but consistency counts BOTH contrasts, including the ineligible d15.
        self.assertEqual(site["timecourse_consistency"], 2)
        self.assertEqual(len(bridge.select_significant_concordant_edges(calls)), 1)

    def test_edge_delta_consumes_stored_changed_flag_without_realphaing(self) -> None:
        # edge_delta must average deltas of sites the bridge already flagged
        # changed+concordant, ignoring a concordant-but-not-changed site — proving
        # it does not re-threshold on site_significance with a forked alpha.
        motif = pd.DataFrame({
            "kinase": ["K"], "kinase_gene": ["K"], "target_gene": ["NODE"],
            "role": ["Receptor"], "contrast": ["d13_d2"], "owning_cluster": ["c1"],
            "best_abs_pds": [1.0], "best_abs_nes": [2.0], "signed_nes": [2.0],
            "best_fdr": [0.01], "n_sites": [2], "n_significant_concordant": [1],
            "celltype_match": [True],
            "sites": [json.dumps([
                {"site_id": "A", "delta": 4.0, "site_significance": 0.30, "concordant": True, "changed": True},
                {"site_id": "B", "delta": 9.0, "site_significance": 0.01, "concordant": True, "changed": False},
            ])],
        })
        psp = pd.DataFrame(columns=["source_gene", "target_gene", "in_vivo_refs", "in_vitro_refs"])
        terminal = edges.build_terminal_map(motif, psp)
        # Only site A (changed=True) counts, despite B having a smaller q-value.
        self.assertEqual(terminal.iloc[0]["edge_delta"], 4.0)

    def test_direct_change_edge_selection_drops_zero_count_rows(self) -> None:
        rows = pd.DataFrame({
            "kinase": ["K1", "K1"],
            "n_significant_concordant": [0, 1],
        })
        selected = bridge.select_significant_concordant_edges(rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["n_significant_concordant"], 1)

        motif = pd.DataFrame({
            "kinase": ["K0"], "kinase_gene": ["K0"], "target_gene": ["NODE"],
            "role": ["Receptor"], "contrast": ["d13_d2"], "owning_cluster": ["c1"],
            "best_abs_pds": [1.0], "best_abs_nes": [2.0], "signed_nes": [2.0],
            "best_fdr": [0.01], "n_sites": [1], "sites": ["[]"],
            "n_significant_concordant": [0], "celltype_match": [True],
        })
        psp = pd.DataFrame(columns=["source_gene", "target_gene", "in_vivo_refs", "in_vitro_refs"])
        self.assertTrue(edges.build_terminal_map(motif, psp).empty)

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
                "owning_cluster": ["clusterA", "clusterA", "clusterA"],
                "channel": ["st", "st", "py"],
                "best_abs_pds": [2.0, 3.0, 1.0],
                "NES": [-2.5, 1.5, 2.5],
                "FDR": [0.02, 0.01, 0.03],
                "n_sites": [4, 1, 9],
                "n_significant_concordant": [1, 1, 1],
                "sites": [
                    json.dumps([{"site_id": f"S{i}", "site_position": f"S{i}", "motif": f"M{i}", "residue_type": "ST", "kl_percentile": 99 + i, "delta": 1, "site_significance": 0.01, "concordant": True, "timecourse_consistency": 1} for i in range(4)]),
                    json.dumps([{"site_id": "S4", "site_position": "S4", "motif": "M4", "residue_type": "ST", "kl_percentile": 99, "delta": 1, "site_significance": 0.01, "concordant": True, "timecourse_consistency": 1}]),
                    json.dumps([{"site_id": f"Y{i}", "site_position": f"Y{i}", "motif": f"M{i}", "residue_type": "Y", "kl_percentile": 99 + i, "delta": 1, "site_significance": 0.01, "concordant": True, "timecourse_consistency": 1} for i in range(9)]),
                ],
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
                "kinase", "kinase_gene", "target_gene", "role", "contrast", "owning_cluster",
                "best_abs_pds", "best_abs_nes", "signed_nes", "best_fdr", "n_sites",
                "sites", "n_significant_concordant",
                "celltype_match",
            ],
        )
        row = result.iloc[0]
        self.assertEqual(row["best_abs_pds"], 3.0)
        self.assertEqual(row["best_abs_nes"], 2.5)
        self.assertEqual(row["signed_nes"], -2.5)
        self.assertEqual(abs(row["signed_nes"]), row["best_abs_nes"])
        self.assertEqual(row["n_sites"], 4)
        self.assertEqual(row["n_significant_concordant"], 1)
        self.assertEqual(len(json.loads(row["sites"])), row["n_sites"])
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
            if (!src.includes("infoPanel.append(filterControls, legend, detail)")
                || !src.includes("detail.replaceChildren(_isSelectionDetail(evt.target))")
                || !src.includes('detail.textContent = "Tap a node or edge for details.";')
                || !src.includes("panel.append(graphHost, splitter, infoPanel, fullscreenButton)")
                || !src.includes('splitter.addEventListener("pointerdown", startPanelDrag)')
                || !src.includes("const maxPanelWidthPx = panel.clientWidth - _IS_STYLE.minGraphWidthPx")
                || !src.includes("Math.min(maxPanelWidthPx, width)")
                || !src.includes("display:flex"))
              throw new Error("sidechain panel does not retain unified details and splitter");
            if (!src.includes('selector: "edge.is-focus-edge"')
                || !src.includes('keep.edges().addClass("is-focus-edge")'))
              throw new Error("focused connectors do not reset opacity");
            if (!src.includes('focus(edge.closedNeighborhood().union(spine))'))
              throw new Error("edge taps do not focus the selected edge and endpoints");
            if (src.includes('showAllEvidence')
                || !src.includes('showChains.checked = false')
                || !src.includes(`const chains = cy.edges("edge[kind = 'chain-edge']")`)
                || !src.includes('node.addClass("is-node-filtered")'))
              throw new Error("first-order filter does not constrain kinase-chain nodes");
            if (!src.includes("_isLegendSample") || src.includes("Bold ${{_IS_COLORS"))
              throw new Error("legend does not use visual color samples");
            if (!src.includes("_isTerminalSiteRows") || !src.includes("KL percentile")
                || !src.includes("_isTerminalSiteTable")
                || src.includes("Tap a terminal edge for phosphosite evidence"))
              throw new Error("unified terminal edge site table is not wired");

            const terminalFields = [
              "source_gene", "target_gene", "role", "contrast", "provenance",
              "best_abs_nes", "signed_nes", "best_fdr", "n_sites",
              "n_significant_concordant", "edge_delta",
            ];
            const terminalRows = [];
            for (let i = 0; i < 12; i++) {{
              terminalRows.push({{
                source_gene: `RK${{i}}`, target_gene: "R", role: "Receptor",
                contrast: "d13_d2", provenance: "motif",
                best_abs_nes: 1 + i * 0.3, signed_nes: i % 2 ? -(1 + i * 0.3) : 1 + i * 0.3,
                best_fdr: 0.01, n_sites: 1, n_significant_concordant: 1,
                edge_delta: i * 0.2,
              }});
            }}
            // A high-|NES| edge that also has a poor FDR: under C it is NOT gated —
            // it must still be drawn (FDR no longer selects).
            terminalRows.push({{
              source_gene: "HIF", target_gene: "R", role: "Receptor",
              contrast: "d13_d2", provenance: "motif",
              best_abs_nes: 5, signed_nes: 5, best_fdr: 0.5, n_sites: 1,
              n_significant_concordant: 1, edge_delta: 4,
            }});
            const terminal = {{}};
            for (const field of terminalFields) terminal[field] = terminalRows.map(row => row[field]);

            const chainRows = [
              {{source_gene: "RK1", target_gene: "RK2", provenance: "motif"}},
              {{source_gene: "UPSTREAM", target_gene: "RK2", provenance: "motif"}},
              {{source_gene: "RK3", target_gene: "OUTSIDE", provenance: "motif"}},
            ];
            const interactome = {{}};
            for (const field of ["source_gene", "target_gene", "provenance"])
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
            if (graph.nesMax !== 5) throw new Error(`nesMax was ${{graph.nesMax}}`);
            // Full view adds the one-hop kinase→kinase neighborhood, while the
            // first-order UI filter hides its chain edges and chain-only nodes.
            if (graph.chainEdges.length !== 3)
              throw new Error("full view did not retain the one-hop kinase chains");
            const siteRows = ctx._isTerminalSiteRows({{
              n_sites: 2,
              sites: JSON.stringify([
                {{motif: "LOW", residue_type: "ST", kl_percentile: 99.1}},
                {{site_id: "S2", site_position: "S267", motif: "HIGH", residue_type: "ST", kl_percentile: 100,
                  delta: 2, site_significance: 0.01, concordant: true, timecourse_consistency: 3}},
              ]),
            }});
            if (siteRows.length !== 2 || siteRows[0].motif !== "HIGH")
              throw new Error("terminal site rows were not sorted by KL percentile");
            if (!ctx._isTerminalSiteRows({{n_sites: 2, sites: "[]"}}).error)
              throw new Error("site/count mismatch was not surfaced");

            // Emphasis: convex, anchored at zero measured movement; monotone in |Δ|.
            const near = 1e-9;
            if (ctx._isEmphasis(0, 0, 4) !== 0) throw new Error("zero Δ did not map to 0");
            if (ctx._isEmphasis(4, 0, 4) !== 1) throw new Error("anchor Δ did not map to 1");
            if (Math.abs(ctx._isEmphasis(2, 0, 4) - Math.pow(0.5, 3.5)) > near)
              throw new Error("gamma=3.5 midpoint wrong");  // (2/4)^3.5
            if (!(ctx._isEmphasis(1, 0, 4) < ctx._isEmphasis(3, 0, 4)))
              throw new Error("emphasis not monotone");

            // Positioned elements: the strongest measured-change edge is thicker
            // and more opaque than the zero-Δ edge (RK0 → emphasis 0).
            const els = ctx._isPositionedElements(graph, 900, 500);
            const byId = new Map(els.map(el => [el.data.id, el.data]));
            const strong = els.find(el => el.data.kind === "terminal-edge"
              && el.data.source === "kinase:HIF").data;
            const weak = els.find(el => el.data.kind === "terminal-edge"
              && el.data.source === "kinase:RK0").data;
            if (!(strong.width > weak.width) || !(strong.opacity > weak.opacity))
              throw new Error("emphasis did not encode NES into width+opacity");
            if (Math.abs(weak.opacity - 0.03) > near || Math.abs(weak.width - 0.35) > near)
              throw new Error("zero-Δ edge not floored to faint-but-present");

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

            // Measured edge Δ changes terminal-edge emphasis only: equal-NES
            // terminal edges with different measured changes get different widths,
            // while their kinase-node sizes remain equal because node size reads |NES| alone.
            const multiplicityGraph = ctx._isGraphForRow(
              {{interactome: {{}}, terminal_edges: {{
                source_gene: ["FAN", "FAN"], target_gene: ["R", "E"],
                role: ["Receptor", "EM"], contrast: ["d13_d2", "d13_d2"],
                provenance: ["motif", "motif"],
                best_abs_nes: [5, 5], signed_nes: [5, 5], best_fdr: [0.01, 0.01],
                n_sites: [4, 1], n_significant_concordant: [2, 1], edge_delta: [4, 1],
              }}}},
              {{Ligand: "L", Receptor: "R", EM: "E", Target: "T", contrast: "d13_d2"}},
            );
            const multiplicityEls = ctx._isPositionedElements(multiplicityGraph, 900, 500);
            const multiEdge = multiplicityEls.find(el => el.data.id === "terminal:0").data;
            const singleEdge = multiplicityEls.find(el => el.data.id === "terminal:1").data;
            const fanNode = multiplicityEls.find(el => el.data.id === "kinase:FAN").data;
            if (!(multiEdge.width > singleEdge.width))
              throw new Error("edge Δ did not differentiate one kinase fan");
            const flatMultiplicityGraph = ctx._isGraphForRow(
              {{interactome: {{}}, terminal_edges: {{
                source_gene: ["FAN", "FAN"], target_gene: ["R", "E"],
                role: ["Receptor", "EM"], contrast: ["d13_d2", "d13_d2"],
                provenance: ["motif", "motif"],
                best_abs_nes: [5, 5], signed_nes: [5, 5], best_fdr: [0.01, 0.01],
                n_sites: [1, 1], n_significant_concordant: [1, 1], edge_delta: [1, 1],
              }}}},
              {{Ligand: "L", Receptor: "R", EM: "E", Target: "T", contrast: "d13_d2"}},
            );
            const flatFanNode = ctx._isPositionedElements(flatMultiplicityGraph, 900, 500)
              .find(el => el.data.id === "kinase:FAN").data;
            if (Math.abs(fanNode.size - flatFanNode.size) > near)
              throw new Error("edge Δ changed kinase node size");

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
              createDocumentFragment: () => new FakeElement("fragment"),
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
            const chainEdge = makeEdge(upstream, high, {{
              kind: "chain-edge", provenance: "psp",
            }});
            makeEdge(high, downstream, {{
              kind: "chain-edge", provenance: "both",
            }});
            const spineTable = ctx._isNodeRelationTable(spineNode);
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

            const kinaseTable = ctx._isNodeRelationTable(high);
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
            // Chain rows carry provenance (Evidence), not a fused weight; the table
            // no longer renders a Weight column at all.
            if (kinaseTable.textContent.includes("Weight"))
              throw new Error("relationship table still renders a Weight column");
            if (!kinaseBody.children[2].textContent.includes("psp"))
              throw new Error("chain row dropped its provenance evidence");

            const terminalDetailEdge = makeEdge(high, spineNode, {{
              kind: "terminal-edge", role: "Receptor", contrast: "d13_d2",
              signed_nes: 4, best_abs_nes: 4, best_fdr: 0.01, n_sites: 1,
              n_significant_concordant: 1, edge_delta: 2, provenance: "motif",
              sites: JSON.stringify([{{site_id: "HIGH_SITE", site_position: "S10", motif: "HIGH_MOTIF",
                residue_type: "ST", kl_percentile: 99.5, delta: 2, site_significance: 0.01,
                concordant: true, timecourse_consistency: 2}}]),
            }});
            const terminalDetail = ctx._isSelectionDetail(terminalDetailEdge);
            if (!terminalDetail.textContent.includes("HIGH → R · d13_d2")
                || !terminalDetail.textContent.includes("KL percentile")
                || !terminalDetail.textContent.includes("HIGH_MOTIF"))
              throw new Error("terminal selection detail omitted motif site evidence");
            const chainDetail = ctx._isSelectionDetail(chainEdge);
            const chainNoteCount = chainDetail.textContent.split(
              "Per-site motif detail is available for kinase→pathway-gene edges only.").length - 1;
            if (!chainDetail.textContent.includes("psp")
                || chainDetail.textContent.includes("Weight")
                || chainNoteCount !== 1)
              throw new Error("chain selection detail lost provenance, kept a weight, or duplicated the site-gap note");
            const kinaseDetail = ctx._isSelectionDetail(high);
            const kinaseNoteCount = kinaseDetail.textContent.split(
              "Per-site motif detail is available for kinase→pathway-gene edges only.").length - 1;
            if (kinaseNoteCount !== 1)
              throw new Error("kinase node detail repeated the site-gap note per chain edge");
            """
        )
        result = subprocess.run(
            ["node", "-e", script], cwd=ROOT, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
