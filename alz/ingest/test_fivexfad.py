from __future__ import annotations

import json
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import alz.build_unified_viewer as viewer
from alz.viewer.cohorts import fivexfad as f5viewer
from alz.cohorts.fivexfad import ingest as fivexfad
from alz.cohorts.fivexfad import celltype_mea as fivexfad_celltype_mea


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

    def test_celltype_mea_loader_excludes_unnamed_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pb_path = tmp_path / "pb.csv"
            counts_path = tmp_path / "counts.csv"
            gene_map_path = tmp_path / "gene_map.csv"
            pd.DataFrame(
                {
                    "tissue": ["cortex", "cortex"],
                    "age_months": [3, 3],
                    "genotype": ["WT", "WT"],
                    "sample_id": ["s1", "s1"],
                    "cell_type": ["Astrocytes", "cluster-27"],
                    "n_cells": [10, 5],
                    "Mapk1": [1.0, 2.0],
                }
            ).to_csv(pb_path, index=False)
            pd.DataFrame(
                {
                    "tissue": ["cortex", "cortex"],
                    "age_months": [3, 3],
                    "genotype": ["WT", "WT"],
                    "sample_id": ["s1", "s1"],
                    "cell_type": ["Astrocytes", "cluster-27"],
                    "n_cells": [10, 5],
                }
            ).to_csv(counts_path, index=False)
            pd.DataFrame(
                {"gene_symbol": ["MAPK1"], "matched_gene": ["Mapk1"]}
            ).to_csv(gene_map_path, index=False)

            old_pb = fivexfad_celltype_mea.PSEUDOBULK_PATH
            old_counts = fivexfad_celltype_mea.COUNTS_PATH
            old_gene_map = fivexfad_celltype_mea.GENE_MAP_PATH
            fivexfad_celltype_mea.PSEUDOBULK_PATH = pb_path
            fivexfad_celltype_mea.COUNTS_PATH = counts_path
            fivexfad_celltype_mea.GENE_MAP_PATH = gene_map_path
            try:
                pb, gene_map, counts = fivexfad_celltype_mea._load_pseudobulk()
            finally:
                fivexfad_celltype_mea.PSEUDOBULK_PATH = old_pb
                fivexfad_celltype_mea.COUNTS_PATH = old_counts
                fivexfad_celltype_mea.GENE_MAP_PATH = old_gene_map

        self.assertEqual(pb["cell_type"].tolist(), ["Astrocytes"])
        self.assertEqual(counts["cell_type"].tolist(), ["Astrocytes"])
        self.assertEqual(gene_map["gene_symbol"].tolist(), ["MAPK1"])

    def test_supporting_5xfad_payload_tissue_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = tmp_path / "kinase_attribution_5xfad"
            out.mkdir()
            detail_dir = Path(tmp) / "viewer" / "fivexfad_detail"
            detail_dir.mkdir(parents=True)
            celltype_dir = out / "celltype_mea"
            celltype_dir.mkdir()
            celltype_mea_shard_dir = Path(tmp) / "viewer" / "fivexfad_celltype_mea"
            celltype_ols_dir = Path(tmp) / "viewer" / "fivexfad_celltype_ols"
            attribution_dir = Path(tmp) / "viewer" / "fivexfad_attribution"
            index_dir = Path(tmp) / "viewer" / "edge_slices"
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
                    "FDR": [0.01],
                    "Subs fraction": ["3/100"],
                    "Leading substrates": ["A;B"],
                    "contrast": ["TG_vs_WT_3mo"],
                    "residue_type": ["ST"],
                    "track": ["st"],
                }
            ).to_csv(out / "cortex_st_mea_stoichiometry.csv", index=False)
            pd.DataFrame(
                {
                    "tissue": ["cortex", "cortex"],
                    "track": ["st", "st"],
                    "cell_type": ["Astrocytes", "cluster-27"],
                    "kinase": ["AKT1", "AKT1"],
                    "contrast": ["TG_vs_WT_3mo", "TG_vs_WT_3mo"],
                    "NES": [1.2, -2.1],
                    "FDR": [0.01, 0.01],
                    "ES": [0.3, -0.4],
                    "p-value": [0.002, 0.003],
                    "Subs fraction": ["2/20", "3/20"],
                    "Leading substrates": ["large;unused;field", "also;unused"],
                    "residue_type": ["ST", "ST"],
                }
            ).to_parquet(celltype_dir / "fivexfad_celltype_mea.parquet", index=False)
            # Direction CSV: LFC + cell support only (specificity is contrast-invariant
            # and lives in fivexfad_expression_specificity.csv, joined by the consumer).
            pd.DataFrame(
                {
                    "kinase": ["AKT1", "AKT1"],
                    "gene_symbol": ["AKT1", "AKT1"],
                    "matched_gene": ["AKT1_MAPPED", "AKT1_MAPPED"],
                    "tissue": ["cortex", "cortex"],
                    "age_months": [3, 3],
                    "cell_type": ["Astrocytes", "Microglia"],
                    "fivexfad_lfc": [0.4, -0.1],
                    "fivexfad_pval": [0.01, 0.6],
                    "fivexfad_fdr": [0.02, 0.8],
                    "n_snrna_samples_wt": [2, 2],
                    "n_snrna_samples_tg": [2, 2],
                    "n_cells_wt": [100, 50],
                    "n_cells_tg": [110, 55],
                    "cluster_source": ["new_clusters", "new_clusters"],
                }
            ).to_csv(out / "fivexfad_snrna_attribution.csv", index=False)
            # Standard detection-metric specificity (from specificity.compute).
            pd.DataFrame(
                {
                    "kinase": ["AKT1", "AKT1"],
                    "gene_symbol": ["AKT1", "AKT1"],
                    "matched_gene": ["AKT1_MAPPED", "AKT1_MAPPED"],
                    "tissue": ["cortex", "cortex"],
                    "cell_type": ["Astrocytes", "Microglia"],
                    "fivexfad_detected": [True, False],
                    "fivexfad_fraction_cells_expressing": [0.8, 0.02],
                    "fivexfad_concentration": [0.9, 0.0],
                    "fivexfad_concentration_of_total": [0.75, 0.01],
                    "fivexfad_concentration_tier": [10, 0],
                    "fivexfad_effective_n": [1.2, 1.2],
                    "fivexfad_top_celltype": ["Astrocytes", "Astrocytes"],
                    "fivexfad_n_celltypes_detected": [1, 1],
                }
            ).to_csv(out / "fivexfad_expression_specificity.csv", index=False)
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

            old_dir = f5viewer.FIVEXFAD_KINASE_DIR
            old_detail_dir = f5viewer.FIVEXFAD_DETAIL_DIR
            old_celltype_dir = f5viewer.FIVEXFAD_CELLTYPE_DIR
            old_celltype_mea_dir = f5viewer.FIVEXFAD_CELLTYPE_MEA_DIR
            old_celltype_ols_dir = f5viewer.FIVEXFAD_CELLTYPE_OLS_DIR
            old_attribution_dir = f5viewer.FIVEXFAD_ATTRIBUTION_DIR
            old_index_dir = f5viewer.FIVEXFAD_INDEX_DIR
            old_mapping_cache = f5viewer.config.MAPPING_CACHE_FILE
            mapping_cache = tmp_path / "kinase_to_gene_mapping.csv"
            pd.DataFrame(
                {"kinase_abbreviation": ["AKT1"], "gene_symbol": ["AKT1_MAPPED"]}
            ).to_csv(mapping_cache, index=False)
            f5viewer.FIVEXFAD_KINASE_DIR = str(out)
            f5viewer.FIVEXFAD_DETAIL_DIR = str(detail_dir)
            f5viewer.FIVEXFAD_CELLTYPE_DIR = str(celltype_dir)
            f5viewer.FIVEXFAD_CELLTYPE_MEA_DIR = str(celltype_mea_shard_dir)
            f5viewer.FIVEXFAD_CELLTYPE_OLS_DIR = str(celltype_ols_dir)
            f5viewer.FIVEXFAD_ATTRIBUTION_DIR = str(attribution_dir)
            f5viewer.FIVEXFAD_INDEX_DIR = str(index_dir)
            f5viewer.config.MAPPING_CACHE_FILE = str(mapping_cache)
            try:
                payload = f5viewer.build_supporting_5xfad_slice()
                shard = next(detail_dir.glob("AKT1.json.gz"), None)
                mea_shard = next(celltype_mea_shard_dir.glob("AKT1.json"), None)
                attr_shard = next(attribution_dir.glob("AKT1.json"), None)
                attr_summary_shard = index_dir / "fivexfad_attribution_summary.json.gz"
                mea_index_shard = index_dir / "fivexfad_celltype_mea_index.json.gz"
                self.assertIsNotNone(shard)
                self.assertIsNotNone(mea_shard)
                self.assertIsNotNone(attr_shard)
                self.assertTrue(attr_summary_shard.exists())
                self.assertTrue(mea_index_shard.exists())
                assert shard is not None
                assert mea_shard is not None
                assert attr_shard is not None
                with gzip.open(shard, "rt", encoding="utf-8") as f:
                    detail_bundle = json.load(f)
                detail = detail_bundle["details"]["AKT1|cortex|IMAC|stoichiometry"]
                mea_detail = json.loads(mea_shard.read_text())
                attr_detail = json.loads(attr_shard.read_text())
                with gzip.open(attr_summary_shard, "rt", encoding="utf-8") as f:
                    attr_summary_rows = json.load(f)["rows"]
                with gzip.open(mea_index_shard, "rt", encoding="utf-8") as f:
                    mea_index_rows = json.load(f)["rows"]
                trace_age = detail["measurement_trace"][0]["age_months"]
                trace_genotype = detail["measurement_trace"][0]["genotype"]
                trace_kl = detail["measurement_trace"][0]["kl_percentile"]
                trace_site_label = detail["measurement_trace"][0]["site_label"]
                stats_site_label = detail["site_stats"][0]["site_label"]
            finally:
                f5viewer.FIVEXFAD_KINASE_DIR = old_dir
                f5viewer.FIVEXFAD_DETAIL_DIR = old_detail_dir
                f5viewer.FIVEXFAD_CELLTYPE_DIR = old_celltype_dir
                f5viewer.FIVEXFAD_CELLTYPE_MEA_DIR = old_celltype_mea_dir
                f5viewer.FIVEXFAD_CELLTYPE_OLS_DIR = old_celltype_ols_dir
                f5viewer.FIVEXFAD_ATTRIBUTION_DIR = old_attribution_dir
                f5viewer.FIVEXFAD_INDEX_DIR = old_index_dir
                f5viewer.config.MAPPING_CACHE_FILE = old_mapping_cache

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["filters"]["tissue"], ["cortex", "hippocampus"])
        self.assertNotIn("assay", payload["filters"])
        self.assertNotIn("analysis_track", payload["filters"])
        self.assertEqual(payload["rows"][0]["tissue"], "cortex")
        self.assertEqual(payload["rows"][0]["gene_symbol"], "AKT1_MAPPED")
        self.assertEqual(payload["rows"][0]["analysis_track"], "stoichiometry")
        self.assertEqual(payload["rows"][0]["substrate_hits"], 3)
        self.assertEqual(payload["rows"][0]["substrate_universe"], 100)
        self.assertNotIn("leading_substrates", payload["rows"][0])
        self.assertNotIn("leading_substrate_count", payload["rows"][0])
        self.assertEqual(payload["rows"][0]["n_wt"], 3)
        self.assertIn("detail_shards", payload)
        self.assertIn("celltype_agreement_index", payload)
        self.assertIn("celltype_mea_shards", payload)
        self.assertIn("celltype_mea_plot_index_shard", payload)
        self.assertIn("celltype_attribution_summary_shard", payload)
        self.assertIn("celltype_attribution_shards", payload)
        # The large indexes (P1/P2) are now fetched-on-demand sidecars, not inlined.
        self.assertNotIn("celltype_mea_plot_index", payload)
        self.assertNotIn("celltype_attribution_summary_index", payload)
        self.assertNotIn("celltype_mea_index", payload)
        self.assertNotIn("attribution_rows", payload)
        self.assertEqual(attr_summary_rows[0]["top_cell_type"], "Astrocytes")
        self.assertEqual(attr_summary_rows[0]["high_moderate_celltype_count"], 1)
        self.assertEqual(attr_summary_rows[0]["celltypes"][0]["cell_type"], "Astrocytes")
        self.assertEqual(attr_summary_rows[0]["celltypes"][1]["cell_type"], "Microglia")
        self.assertNotIn("confidence_basis", attr_summary_rows[0]["celltypes"][0])
        self.assertEqual(mea_index_rows[0]["cell_type"], "Astrocytes")
        self.assertNotIn("ES", mea_index_rows[0])
        self.assertEqual(attr_detail["rows"][0]["confidence_tier"], "very_high")
        self.assertEqual(attr_detail["rows"][0]["confidence_basis"], "5xFAD snRNA high + decomp agreement")
        self.assertTrue(attr_detail["rows"][0]["decomp_agrees_bulk"])
        self.assertEqual(payload["celltype_agreement_index"][0]["agreement_state"], "agree")
        self.assertEqual(payload["celltype_agreement_index"][0]["decomp_sig_celltype_count"], 1)
        self.assertNotIn("Leading substrates", payload["celltype_agreement_index"][0])
        self.assertEqual(len(mea_detail["rows"]), 1)
        self.assertEqual(mea_detail["rows"][0]["cell_type"], "Astrocytes")
        self.assertNotIn("Leading substrates", mea_detail["rows"][0])
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
        self.assertIn('data-col="agreement_profile"', body)
        self.assertIn("agreement-profile-cell f5-agreement-profile-cell", tab_js)
        self.assertIn("function _f5AgreementState", tab_js)
        self.assertIn("function _f5DisagreeCountScoped", tab_js)
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
        self.assertIn("Per-cell-type decomposition for TG_vs_WT_", tab_js)
        self.assertIn("f5-mea-decomp", tab_js)
        self.assertIn("_renderKinaseDecompBars", tab_js)
        self.assertIn("function _f5CelltypeMeaRowsForGroup", tab_js)
        self.assertIn("const F5_ATTR_COLS", tab_js)
        self.assertIn("attr-verdict-table", tab_js)
        self.assertIn("attr-verdict-supergroup", tab_js)
        self.assertIn("attr-verdict-toggle", tab_js)
        self.assertIn("attr-explainer", tab_js)
        self.assertIn("_attrConfidenceClass", tab_js)
        self.assertIn("_f5NativeTierBadge", tab_js)
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
        self.assertNotIn("song_specificity", attr_renderer)
        self.assertIn('key: "fivexfad_concentration_tier", label: "Conc"', tab_js)

        self.assertIn('label: "Decomp NES"', tab_js)
        self.assertIn('label: "Decomp FDR"', tab_js)
        self.assertNotIn("celltype_mea_index", tab_js)
        self.assertIn("celltype_agreement_index", tab_js)
        self.assertIn("celltype_mea_shards", tab_js)
        self.assertIn("celltype_mea_plot_index_shard", tab_js)
        self.assertIn("celltype_attribution_summary_shard", tab_js)
        self.assertIn("celltype_attribution_shards", tab_js)
        self.assertIn("function _f5EnsureShardData", tab_js)
        self.assertIn("function _f5LoadAttribution", tab_js)
        self.assertIn("No native 5xFAD snRNA attribution row is available", tab_js)
        self.assertIn("_decomp_only", tab_js)
        self.assertNotIn("block.attribution_rows", tab_js)
        self.assertIn("_F5CelltypeMeaRowsByAgeKey.get", tab_js)
        self.assertNotIn("for (const r of index.values())", tab_js)
        self.assertIn("celltype_ols_shards", tab_js)
        self.assertIn("new_clusters", attr_renderer)
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

    def test_fivexfad_snrna_uses_standard_detection_metric(self) -> None:
        """5xFAD within-cohort specificity is the repo-wide detection metric.

        The producer exports detection + mean log2(x+1); the Python step runs
        specificity.compute; the viewer and tab render detection/concentration —
        the legacy share/τ localizer is gone everywhere (anti-shim).
        """
        root = Path(__file__).resolve().parents[2]
        script = (root / "alz/ingest/build_5xfad_snrna_attribution.R").read_text()
        spec_step = (root / "alz/cohorts/fivexfad/snrna_specificity.py").read_text()
        viewer_py = (root / "alz/viewer/cohorts/fivexfad.py").read_text()
        tab_js = (root / "alz/viewer/template/js/tabs/kinase_fivexfad.js").read_text()

        # Producer: count-based detection + log2 expression per (tissue, cell type),
        # written to the expression export. No share/τ math remains.
        self.assertIn("det_group <- paste(md$tissue, md$new_clusters", script)
        self.assertIn("fraction_cells_expressing", script)
        self.assertIn("mean_log2_expression = as.numeric(mean_ln_tc) / log(2)", script)
        self.assertIn("fivexfad_snrna_expression.csv", script)
        self.assertNotIn("specificity_tau", script)
        self.assertNotIn("location_tier", script)
        self.assertNotIn("fivexfad_fold_over_uniform", script)

        # Python step: the shared standard metric, not a 5xFAD reimplementation.
        self.assertIn("from alz.cross_reference import specificity", spec_step)
        self.assertIn("specificity.compute(", spec_step)
        self.assertIn("fivexfad_expression_specificity.csv", spec_step)

        # Consumer: joins the specificity CSV and gates on detection + concentration.
        self.assertIn("fivexfad_expression_specificity.csv", viewer_py)
        self.assertIn("fivexfad_concentration_tier", viewer_py)
        self.assertIn("fivexfad_detected", viewer_py)
        self.assertIn("wmb_concentration_tier", viewer_py)
        self.assertNotIn("fivexfad_fold_over_uniform", viewer_py)
        self.assertNotIn("fivexfad_specificity", viewer_py)
        self.assertIn("_F5_MIN_CELLS_PER_CONTRAST = 3", viewer_py)
        self.assertIn("config.SONG_LFC_MIN", viewer_py)
        self.assertIn("≥2× concentration over even share", viewer_py)

        # Tab: shared detection-cell / concentration-tier widgets, no share pills.
        self.assertIn("_detGateCell(", tab_js)
        self.assertIn("_concTierCell(", tab_js)
        self.assertIn("fivexfad_concentration_tier", tab_js)
        self.assertNotIn("fivexfad_specificity", tab_js)
        self.assertNotIn("fivexfad_fold_over_uniform", tab_js)


if __name__ == "__main__":
    unittest.main()
