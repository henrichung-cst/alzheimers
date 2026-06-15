from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import alz.build_unified_viewer as viewer
from alz.ingest import fivexfad


class FiveXFADTests(unittest.TestCase):
    def test_parse_raw_run_cortex_imac_trailing_sample(self) -> None:
        row = fivexfad.parse_raw_run(
            "cortex", "imac", "102325_LD_3mon_CTX_IMAC_DIA_10.raw"
        )

        self.assertEqual(row["age_months"], 3)
        self.assertEqual(row["duplicate_group"], "10")
        self.assertEqual(row["genotype"], "TG")
        self.assertEqual(row["biological_sample_id"], "cortex_3mo_TG_10")
        self.assertEqual(row["analysis_action"], "primary")

    def test_parse_raw_run_hippocampus_py_uses_biological_sample_not_injection(
        self,
    ) -> None:
        row = fivexfad.parse_raw_run(
            "hippocampus", "py", "052325_5x_3mth_11_pY_1.raw"
        )

        self.assertEqual(row["age_months"], 3)
        self.assertEqual(row["duplicate_group"], "11")
        self.assertEqual(row["genotype"], "TG")
        self.assertEqual(row["biological_sample_id"], "hippocampus_3mo_TG_11")

    def test_delivered_docx_genotype_corrections(self) -> None:
        cases = [
            ("cortex", "total", "260203_LD_CTX_M6_15_TP_DIA.raw", "WT", "cortex_6mo_WT_15"),
            ("hippocampus", "py", "052325_5x_6mth_15_pY_1.raw", "WT", "hippocampus_6mo_WT_15"),
            ("cortex", "imac", "102425_CTX_IMAC_DIA_12mon_6.raw", "TG", "cortex_12mo_TG_6"),
            ("hippocampus", "imac", "102025_Hippo_IMAC_DIA_12mon_6.raw", "TG", "hippocampus_12mo_TG_6"),
            ("cortex", "total", "260203_LD_CTX_M12_10_TP_DIA.raw", "WT", "cortex_12mo_WT_10"),
        ]

        for tissue, assay, raw_run, genotype, bio_id in cases:
            with self.subTest(raw_run=raw_run):
                row = fivexfad.parse_raw_run(tissue, assay, raw_run)
                self.assertEqual(row["genotype"], genotype)
                self.assertEqual(row["biological_sample_id"], bio_id)
                self.assertEqual(row["genotype_source"], "delivered_lucie_proteomics_docx_sample_lists")

    def test_pool_exclusion_and_hippocampus_imac_duplicate_group(self) -> None:
        pool = fivexfad.parse_raw_run(
            "cortex", "py", "011626_LD_Cort_M6_pool_pY.raw"
        )
        dup = fivexfad.parse_raw_run(
            "hippocampus", "imac", "101525_Hippo_IMAC_3mon_1a.raw"
        )

        self.assertIs(pool["pool"], True)
        self.assertEqual(pool["analysis_action"], "exclude_pool")
        self.assertEqual(dup["duplicate_group"], "1")
        self.assertEqual(dup["biological_sample_id"], "hippocampus_3mo_WT_1")

    def test_contrast_group_counts_are_audit_only(self) -> None:
        mapping = pd.DataFrame(
            {
                "biological_sample_id": ["s1", "s2", "s3", "s4"],
                "age_months": [3, 3, 3, 3],
                "genotype": ["WT", "WT", "TG", "TG"],
            }
        )
        y = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, np.nan, 3.0, 4.0],
            ]
        )

        wt, tg = fivexfad._contrast_group_counts(
            y, mapping, ["s1", "s2", "s3", "s4"], 3
        )
        self.assertEqual(wt.tolist(), [2, 1])
        self.assertEqual(tg.tolist(), [2, 2])

    def test_supporting_5xfad_payload_tissue_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kinase_attribution_5xfad"
            out.mkdir()
            detail_dir = Path(tmp) / "viewer" / "fivexfad_detail"
            detail_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "tissue": ["cortex"],
                    "track": ["st"],
                    "contrast": ["TG_vs_WT_3mo"],
                    "age_months": [3],
                    "n_wt": [3],
                    "n_tg": [4],
                    "contrast_status": ["primary"],
                }
            ).to_csv(out / "cortex_st_contrast_qc.csv", index=False)
            pd.DataFrame(
                {
                    "kinase": ["AKT1"],
                    "ES": [0.2],
                    "NES": [1.5],
                    "p-value": [0.01],
                    "FDR": [0.05],
                    "Subs fraction": ["3/100"],
                    "Leading substrates": ["A;B"],
                    "contrast": ["TG_vs_WT_3mo"],
                    "residue_type": ["ST"],
                    "track": ["st"],
                }
            ).to_csv(out / "cortex_st_mea_stoichiometry.csv", index=False)
            pd.DataFrame(
                {
                    "kinase": ["AKT1"],
                    "contrast": ["TG_vs_WT_3mo"],
                    "analysis_track": ["stoichiometry"],
                    "motif": ["A"],
                    "kl_percentile": [91.2],
                }
            ).to_csv(out / "cortex_st_mea_substrate_sets.csv", index=False)
            long_site_id = (
                ">ENSMUSP00000079691.7 pep chromosome:GRCm39:7:24677592:24705383:-1 "
                "gene:ENSMUSG00000040907.16 transcript:ENSMUST00000080882.11 "
                "gene_biotype:protein_coding transcript_biotype:protein_coding "
                "gene_symbol:Atp1a3 description:ATPase Na+/K+ transporting alpha 3 polypeptide "
                "[Source:MGI SymbolAcc:MGI:88107]_S456_M1_0"
            )
            matrix = pd.DataFrame(
                {
                    "site_id": [long_site_id],
                    "gene_symbol": ["Atp1a3"],
                    "motif": ["A"],
                    "site_position": [456],
                    "residue_type": ["S"],
                    "matched_protein": ["Prot1"],
                    "cortex_3mo_WT_1": [2.0],
                }
            )
            matrix.to_csv(out / "cortex_st_raw_phospho_normalized.csv", index=False)
            matrix.to_csv(out / "cortex_st_matched_total_protein.csv", index=False)
            matrix.to_csv(out / "cortex_st_stoichiometry_matrix.csv", index=False)
            pd.DataFrame(
                {
                    "site_id": [long_site_id],
                    "gene_symbol": ["Atp1a3"],
                    "n_obs_stoich": [1],
                    "n_obs_raw": [1],
                    "stoich_lfc_TG_vs_WT_3mo": [0.5],
                    "stoich_pval_TG_vs_WT_3mo": [0.01],
                    "stoich_fdr_TG_vs_WT_3mo": [0.05],
                    "stoich_n_wt_TG_vs_WT_3mo": [1],
                    "stoich_n_tg_TG_vs_WT_3mo": [1],
                }
            ).to_csv(out / "cortex_st_site_level_ols.csv", index=False)
            pd.DataFrame(
                {
                    "tissue": ["cortex"],
                    "assay": ["imac"],
                    "analysis_action": ["primary"],
                    "analysis_scope": ["kinase_mea_v1"],
                    "biological_sample_id": ["cortex_3mo_WT_1"],
                    "age_months": [3],
                    "age": ["3mo"],
                    "genotype": ["WT"],
                }
            ).to_csv(out / "sample_manifest.csv", index=False)

            old_dir = viewer.FIVEXFAD_KINASE_DIR
            old_detail_dir = viewer.FIVEXFAD_DETAIL_DIR
            viewer.FIVEXFAD_KINASE_DIR = str(out)
            viewer.FIVEXFAD_DETAIL_DIR = str(detail_dir)
            try:
                payload = viewer.build_supporting_5xfad_slice()
                shard = next(detail_dir.glob("AKT1_cortex_IMAC_stoichiometry.json"), None)
                self.assertIsNotNone(shard)
                assert shard is not None
                detail = json.loads(shard.read_text())
                trace_age = detail["measurement_trace"][0]["age_months"]
                trace_genotype = detail["measurement_trace"][0]["genotype"]
                trace_kl = detail["measurement_trace"][0]["kl_percentile"]
                trace_site_label = detail["measurement_trace"][0]["site_label"]
                stats_site_label = detail["site_stats"][0]["site_label"]
            finally:
                viewer.FIVEXFAD_KINASE_DIR = old_dir
                viewer.FIVEXFAD_DETAIL_DIR = old_detail_dir

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["filters"]["tissue"], ["cortex", "hippocampus"])
        self.assertNotIn("assay", payload["filters"])
        self.assertNotIn("analysis_track", payload["filters"])
        self.assertEqual(payload["rows"][0]["tissue"], "cortex")
        self.assertEqual(payload["rows"][0]["analysis_track"], "stoichiometry")
        self.assertEqual(payload["rows"][0]["substrate_hits"], 3)
        self.assertEqual(payload["rows"][0]["substrate_universe"], 100)
        self.assertEqual(payload["rows"][0]["n_wt"], 3)
        self.assertIn("detail_shards", payload)
        self.assertEqual(trace_age, 3)
        self.assertEqual(trace_genotype, "WT")
        self.assertEqual(trace_kl, 91.2)
        self.assertEqual(trace_site_label, "Atp1a3_S456")
        self.assertEqual(stats_site_label, "Atp1a3_S456")

    def test_fivexfad_viewer_matches_single_kinase_tab_pattern(self) -> None:
        root = Path(__file__).resolve().parents[2]
        body = (root / "alz/viewer/template/body.html").read_text()
        state = (root / "alz/viewer/template/js/01_state.js").read_text()
        manifest = (root / "alz/viewer/template/js/02_ui_chrome.js").read_text()
        boot = (root / "alz/viewer_shared/template/js/boot.js").read_text()
        header = (root / "alz/viewer_shared/template/js/05_header.js").read_text()
        tab_js = (root / "alz/viewer/template/js/tabs/kinase_fivexfad.js").read_text()

        self.assertIn('data-mode="fivexfad"', body)
        self.assertIn('id="tab-fivexfadkinase"', body)
        self.assertNotIn("tab-fivexfadqc", body)
        self.assertNotIn("tab-fivexfadmethods", body)
        self.assertIn("kinaseFiveXFAD:null", state)
        self.assertIn('fivexfadkinase: {\n    group: "drilldown", label: "Kinase"', manifest)
        self.assertIn('selection: ["kinaseFiveXFAD"]', manifest)
        self.assertIn("updateFiveXFADKinaseSelection", manifest)
        self.assertIn("kinaseFiveXFADSelChanged", boot)
        self.assertIn('key:"kinaseFiveXFAD"', header)
        self.assertNotIn("5xFAD QC", manifest)
        self.assertNotIn("5xFAD Methods", manifest)
        self.assertIn("nes-profile-cell", tab_js)
        self.assertIn("kinase-audit-tabs", tab_js)
        self.assertIn('{id: "attribution", label: "Attribution"}', tab_js)
        self.assertNotIn("No attribution rows are packaged", tab_js)
        self.assertIn("measurement_trace", tab_js)
        self.assertIn("matched_total_protein", tab_js)
        self.assertIn("kl_percentile", tab_js)
        self.assertIn("SequenceLogo.buildBlock", tab_js)
        self.assertIn("function _f5SiteLabel", tab_js)
        self.assertIn("function _f5SiteCell", tab_js)
        self.assertIn("gene_symbol:([A-Za-z0-9_.-]+)", tab_js)
        self.assertIn('label: "Site", fmt: _f5SiteCell, html: true', tab_js)
        self.assertIn("detail_shards", tab_js)
        self.assertIn("prepared_mea_input", tab_js)
        self.assertIn("running_enrichment", tab_js)
        self.assertIn("Running enrichment for", tab_js)
        self.assertIn("Stoichiometry vs raw phospho", tab_js)
        self.assertIn("const F5_ATTR_COLS", tab_js)
        self.assertIn("attr-verdict-table", tab_js)
        self.assertIn("attr-verdict-supergroup", tab_js)
        self.assertIn("attr-verdict-toggle", tab_js)
        self.assertIn("attr-explainer", tab_js)
        self.assertIn("_attrConfidenceClass", tab_js)
        self.assertIn("_msTierBadge", tab_js)
        self.assertIn("_wmbTierBadge", tab_js)
        self.assertIn("_attrLfcColor", tab_js)
        self.assertIn("f5-attr-drawer", tab_js)
        attr_renderer = tab_js[
            tab_js.index("function _f5RenderAttribution("):
            tab_js.index("function _f5RenderAttributionDrawer(")
        ]
        self.assertNotIn("_f5SmallTable", attr_renderer)
        self.assertNotIn('label: "Basis"', attr_renderer)
        self.assertNotIn('label: "Song tier"', attr_renderer)
        self.assertNotIn('label: "Human tier"', attr_renderer)
        self.assertIn("nes-profile-age-labels", tab_js)
        self.assertNotIn("f5-age-cell", tab_js)
        self.assertNotIn('data-col="slice"', body)
        self.assertNotIn('data-col="substrateHits"', body)
        self.assertNotIn("<th>Subs</th>", tab_js)
        self.assertNotIn("<td>${_f5Esc(r.slice)}</td>", tab_js)
        self.assertNotIn("f5-audit-slice", tab_js)
        self.assertNotIn("Slice <select", tab_js)
        self.assertNotIn("f5-filter-assay", body)
        self.assertNotIn("f5-filter-analysis", body)
        self.assertNotIn("f5-audit-assay", tab_js)
        self.assertNotIn("f5-audit-track", tab_js)
        self.assertNotIn("Contrast evidence", tab_js)
        self.assertNotIn("Sample counts", tab_js)
        self.assertNotIn("Packaged source files", tab_js)
        self.assertNotIn("does not currently embed per-site measurement matrices", tab_js)
        self.assertNotIn("QC status", tab_js)
        self.assertNotIn("<th>QC</th>", tab_js)


if __name__ == "__main__":
    unittest.main()
