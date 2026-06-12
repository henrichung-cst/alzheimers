from __future__ import annotations

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

    def test_contrast_group_counts_and_replicated_mask(self) -> None:
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
        lfc, pval, fdr = fivexfad._mask_nonestimable(
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
            np.array([0.1, 0.2]),
            wt,
            tg,
        )

        self.assertEqual(wt.tolist(), [2, 1])
        self.assertEqual(tg.tolist(), [2, 2])
        self.assertEqual(lfc[0], 1.0)
        self.assertTrue(np.isnan(lfc[1]))
        self.assertTrue(np.isnan(pval[1]))
        self.assertTrue(np.isnan(fdr[1]))

    def test_supporting_5xfad_payload_tissue_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kinase_attribution_5xfad"
            out.mkdir()
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
                    "tissue": ["cortex"],
                    "assay": ["imac"],
                    "analysis_action": ["primary"],
                    "analysis_scope": ["kinase_mea_v1"],
                    "biological_sample_id": ["cortex_3mo_WT_1"],
                    "age": ["3mo"],
                    "genotype": ["WT"],
                }
            ).to_csv(out / "sample_manifest.csv", index=False)

            old_dir = viewer.FIVEXFAD_KINASE_DIR
            viewer.FIVEXFAD_KINASE_DIR = str(out)
            try:
                payload = viewer.build_supporting_5xfad_slice()
            finally:
                viewer.FIVEXFAD_KINASE_DIR = old_dir

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["filters"]["tissue"], ["cortex", "hippocampus"])
        self.assertEqual(payload["rows"][0]["tissue"], "cortex")
        self.assertEqual(payload["rows"][0]["analysis_track"], "stoichiometry")
        self.assertEqual(payload["rows"][0]["substrate_hits"], 3)
        self.assertEqual(payload["rows"][0]["substrate_universe"], 100)
        self.assertEqual(payload["rows"][0]["n_wt"], 3)
        self.assertIn("detail_shards", payload)

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
        self.assertIn("hasAttribution", tab_js)
        self.assertNotIn("No attribution rows are packaged", tab_js)
        self.assertIn("measurement_trace", tab_js)
        self.assertIn("matched_total_protein", tab_js)
        self.assertIn("detail_shards", tab_js)
        self.assertIn("prepared_mea_input", tab_js)
        self.assertIn("running_enrichment", tab_js)
        self.assertIn("Running enrichment for", tab_js)
        self.assertIn("Stoichiometry vs raw phospho", tab_js)
        self.assertIn("nes-profile-age-labels", tab_js)
        self.assertNotIn("f5-age-cell", tab_js)
        self.assertNotIn('data-col="slice"', body)
        self.assertNotIn('data-col="substrateHits"', body)
        self.assertNotIn("<th>Subs</th>", tab_js)
        self.assertNotIn("<td>${_f5Esc(r.slice)}</td>", tab_js)
        self.assertNotIn("f5-audit-slice", tab_js)
        self.assertNotIn("Slice <select", tab_js)
        self.assertNotIn("Contrast evidence", tab_js)
        self.assertNotIn("Sample counts", tab_js)
        self.assertNotIn("Packaged source files", tab_js)
        self.assertNotIn("does not currently embed per-site measurement matrices", tab_js)
        self.assertNotIn("QC status", tab_js)
        self.assertNotIn("<th>QC</th>", tab_js)


if __name__ == "__main__":
    unittest.main()
