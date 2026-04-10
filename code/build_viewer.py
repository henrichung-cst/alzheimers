#!/usr/bin/env python3
"""Generate an interactive HTML viewer for kinase attribution results.

Reads the 4 hypothesis-generation tables, MEA stoichiometry, site-level OLS,
Pre-computes clustering and kinase family
annotations, then embeds everything as JSON into a single self-contained HTML
file that can be opened in any browser.

Usage:
    python code/build_viewer.py                     # default output path
    python code/build_viewer.py --output path.html  # custom output path
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from kinase_library.modules import data as kl_data
from kinase_library.utils._global_vars import family_colors as KL_FAMILY_COLORS

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTRASTS = [
    "App_2mo", "App_4mo", "App_6mo",
    "Tau_2mo", "Tau_4mo", "Tau_6mo",
    "ApTt_2mo", "ApTt_4mo", "ApTt_6mo",
]

DISEASE_GROUPS = ["App", "Tau", "ApTt"]
TIMEPOINTS = ["2mo", "4mo", "6mo"]
DISEASE_COLORS = {"App": "#c62828", "Tau": "#1565c0", "ApTt": "#6a1b9a"}

TISSUE_ORDER = [
    "Excitatory neurons", "Interneurons", "Astrocytes",
    "Oligodendrocytes", "OPCs", "Microglia", "Endothelial cells",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load all required CSV files."""
    ar_dir = config.ATTRIBUTION_RECOVERY_OUTPUT_DIR
    ka_dir = config.KINASE_ATTRIBUTION_OUTPUT_DIR

    t1 = pd.read_csv(os.path.join(ar_dir, "kinase_activity_matrix.csv"))
    t2 = pd.read_csv(os.path.join(ar_dir, "celltype_evidence_table.csv"))
    t3 = pd.read_csv(os.path.join(ar_dir, "kinase_hypothesis_table.csv"))
    t4 = pd.read_csv(os.path.join(ar_dir, "celltype_kinase_profiles.csv"))

    mea = pd.read_csv(os.path.join(ka_dir, "mea_stoichiometry.csv"))
    # Drop the huge Leading substrates column
    mea = mea[["kinase", "NES", "FDR", "contrast"]].copy()

    return t1, t2, t3, t4, mea


def resolve_families(kinases):
    """Get kinase family map and colors."""
    fam_series = kl_data.get_kinase_family(kinases)
    fam_map = fam_series.to_dict()
    # Use the canonical family groups for coloring
    fam_colors = dict(KL_FAMILY_COLORS)
    return fam_map, fam_colors


def compute_clustering(t1, t3):
    """Pre-compute hierarchical clustering order per tissue category."""
    # Build tissue map from t3
    t3_with_tissue = t3[t3["top_celltype_1"].notna()].copy()
    t3_with_tissue["tissue_category"] = t3_with_tissue["top_celltype_1"].map(
        config.SUBCLASS_TO_TISSUE_CATEGORY
    ).fillna("Other")

    nes_cols = [f"{c}_NES" for c in CONTRASTS]

    cluster_orders = {}
    for tissue in TISSUE_ORDER + ["Other"]:
        kinases_in_tissue = t3_with_tissue[
            t3_with_tissue["tissue_category"] == tissue
        ]["kinase"].tolist()
        if not kinases_in_tissue:
            continue

        sub = t1[t1["kinase"].isin(kinases_in_tissue)].set_index("kinase")
        nes_matrix = sub[nes_cols].fillna(0)

        if len(nes_matrix) <= 2:
            cluster_orders[tissue] = list(nes_matrix.index)
            continue

        dist = pdist(nes_matrix.values, metric="correlation")
        dist = np.nan_to_num(dist, nan=0.0)
        Z = linkage(dist, method="average")
        order = leaves_list(Z)
        cluster_orders[tissue] = [nes_matrix.index[i] for i in order]

    return cluster_orders



# ---------------------------------------------------------------------------
# JSON payload
# ---------------------------------------------------------------------------

def _round_floats(obj, decimals=4):
    """Recursively round floats in nested dicts/lists."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, decimals) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(float(obj), decimals)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def build_payload(t1, t2, t3, t4, mea, fam_map, fam_colors, cluster_orders):
    """Build the complete JSON-serializable data payload."""
    payload = {
        "kinaseActivity": t1.to_dict(orient="records"),
        "celltypeEvidence": t2.to_dict(orient="records"),
        "kinaseHypothesis": t3.to_dict(orient="records"),
        "celltypeProfiles": t4.to_dict(orient="records"),
        "meaStoichiometry": mea.to_dict(orient="records"),
        "clusterOrders": cluster_orders,
        "familyMap": fam_map,
        "familyColors": fam_colors,
        "config": {
            "fdrThreshDefault": config.MEA_FDR_THRESH,
            "specificityHigh": round(config.SPECIFICITY_HIGH, 4),
            "specificityLow": round(config.SPECIFICITY_LOW, 4),
            "seaAdLfcMin": config.SEA_AD_LFC_MIN,
            "subclassToTissue": dict(config.SUBCLASS_TO_TISSUE_CATEGORY),
            "tissueOrder": TISSUE_ORDER,
            "diseaseGroups": DISEASE_GROUPS,
            "timepoints": TIMEPOINTS,
            "diseaseColors": DISEASE_COLORS,
            "contrasts": CONTRASTS,
        },
    }
    return _round_floats(payload)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Kinase Attribution Viewer &mdash; AD Phosphoproteomics</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
:root {
  --app-red: #c62828; --tau-blue: #1565c0; --aptt-purple: #6a1b9a;
  --bg: #fafafa; --card-bg: #ffffff; --border: #e0e0e0;
  --text: #212121; --text-muted: #757575;
  --near-miss-bg: #fff8e1; --sub-thresh-bg: #f5f5f5;
  --selected-border: #1976d2;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--text); min-width: 1200px; }
header { background: #263238; color: white; padding: 12px 24px; display: flex;
         align-items: center; gap: 32px; flex-wrap: wrap; }
header h1 { font-size: 18px; font-weight: 600; white-space: nowrap; }
.global-controls { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
.global-controls label { font-size: 13px; display: flex; align-items: center; gap: 6px; }
.global-controls input[type=range] { width: 140px; }
.global-controls .val { font-weight: 700; min-width: 40px; }
nav#tab-bar { display: flex; background: #37474f; overflow-x: auto; }
nav#tab-bar button { background: none; border: none; color: #b0bec5; padding: 10px 20px;
  font-size: 13px; font-weight: 500; cursor: pointer; white-space: nowrap;
  border-bottom: 3px solid transparent; transition: all 0.15s; }
nav#tab-bar button:hover { color: white; background: rgba(255,255,255,0.05); }
nav#tab-bar button.active { color: white; border-bottom-color: #42a5f5; }
main { padding: 16px 24px; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
/* Tables */
.data-table-wrap { max-height: 70vh; overflow: auto; border: 1px solid var(--border);
  border-radius: 4px; background: var(--card-bg); }
table.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
table.data-table th { position: sticky; top: 0; background: #eceff1; padding: 6px 8px;
  text-align: left; font-weight: 600; cursor: pointer; user-select: none;
  border-bottom: 2px solid var(--border); white-space: nowrap; z-index: 2; }
table.data-table th:hover { background: #cfd8dc; }
table.data-table td { padding: 5px 8px; border-bottom: 1px solid #f0f0f0; white-space: nowrap; }
table.data-table tr:nth-child(even) { background: #fafafa; }
table.data-table tr:hover { background: #e3f2fd; }
table.data-table tr.selected { background: #bbdefb !important; }
table.data-table tr.near-miss { background: var(--near-miss-bg) !important; }
table.data-table tr.sub-thresh td { color: var(--text-muted); opacity: 0.6; }
table.data-table tr.sub-thresh td:first-child { opacity: 1; }
.badge { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px;
  font-weight: 600; margin-left: 4px; }
.badge-near-miss { background: #fff3e0; color: #e65100; }
.badge-high { background: #e8f5e9; color: #2e7d32; }
.badge-low { background: #fce4ec; color: #c62828; }
.fam-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 4px; vertical-align: middle; }
/* Explorer layout */
.explorer-layout { display: grid; grid-template-columns: 1fr 380px; gap: 16px; }
.detail-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 16px; position: sticky; top: 16px; max-height: 85vh; overflow-y: auto; }
.detail-card h3 { font-size: 15px; margin-bottom: 12px; }
.detail-card .meta { font-size: 12px; color: var(--text-muted); margin-bottom: 8px; }
/* Filters */
.filter-bar { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }
.filter-bar input, .filter-bar select { font-size: 12px; padding: 4px 8px;
  border: 1px solid var(--border); border-radius: 4px; }
.filter-bar input[type=text] { width: 200px; }
/* Heatmap controls */
.heatmap-controls { display: flex; gap: 16px; margin-bottom: 12px; align-items: center; flex-wrap: wrap; }
.heatmap-controls label, .heatmap-controls select, .heatmap-controls button {
  font-size: 12px; }
.heatmap-controls button { padding: 4px 12px; border: 1px solid var(--border);
  border-radius: 4px; background: var(--card-bg); cursor: pointer; }
.heatmap-controls button.active { background: #42a5f5; color: white; border-color: #42a5f5; }
/* Sparkline */
.sparkline { display: inline-flex; align-items: center; gap: 1px; height: 24px; vertical-align: middle;
  position: relative; }
.sparkline .bar { width: 5px; min-height: 1px; position: relative; }
.sparkline .bar.pos { align-self: flex-end; border-radius: 1px 1px 0 0; margin-bottom: 0; }
.sparkline .bar.neg { align-self: flex-start; border-radius: 0 0 1px 1px; margin-top: 0; }
.sparkline-wrap { display: inline-flex; align-items: stretch; gap: 1px; height: 24px; vertical-align: middle; }
/* Cell-type evidence list */
.evidence-list { font-size: 11px; }
.evidence-list table { width: 100%; }
.evidence-list th { font-size: 10px; background: #f5f5f5; }
.evidence-list td { padding: 3px 6px; }
/* Tab-specific controls */
.tab-controls { margin-bottom: 12px; display: flex; gap: 12px; align-items: center; }
.tab-controls label { font-size: 12px; }
.tab-controls select { font-size: 12px; padding: 3px 6px; }
/* Legend */
.legend-box { background: #f5f5f5; border: 1px solid var(--border); border-radius: 4px;
  padding: 8px 12px; font-size: 11px; margin-bottom: 12px; display: flex; gap: 16px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; }
/* Glossary panel */
/* Synced kinase search bar (per-tab) */
.kinase-search-group { display: inline-flex; align-items: center; gap: 6px; }
.kinase-search { position: relative; display: inline-block; }
.kinase-search input { font-size: 12px; padding: 4px 8px 4px 26px; border: 1px solid var(--border);
  border-radius: 4px; width: 220px; outline: none; }
.kinase-search input:focus { border-color: #42a5f5; }
.kinase-search .search-icon { position: absolute; left: 7px; top: 50%; transform: translateY(-50%);
  color: #90a4ae; font-size: 12px; pointer-events: none; }
.kinase-search .search-dropdown { display: none; position: absolute; top: 100%; left: 0; width: 340px;
  background: #fff; border: 1px solid var(--border); border-radius: 0 0 6px 6px;
  max-height: 280px; overflow-y: auto; z-index: 200; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.kinase-search .search-dropdown.open { display: block; }
.kinase-search .search-item { padding: 5px 10px; cursor: pointer; font-size: 12px; color: #263238;
  display: flex; justify-content: space-between; align-items: center; }
.kinase-search .search-item:hover, .kinase-search .search-item.active { background: #e3f2fd; }
.kinase-search .search-item .kinase-name { font-weight: 600; }
.kinase-search .search-item .gene-name { color: #78909c; font-size: 11px; }
.kinase-search .search-item .search-meta { color: #90a4ae; font-size: 10px; }
.kinase-search .search-empty { padding: 6px 10px; font-size: 12px; color: #90a4ae; font-style: italic; }
/* Search mode toggle */
.search-mode-toggle { display: inline-flex; border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.search-mode-toggle button { font-size: 11px; padding: 3px 10px; border: none; background: #fff;
  cursor: pointer; color: var(--text-muted); white-space: nowrap; }
.search-mode-toggle button.active { background: #42a5f5; color: #fff; }
.search-mode-toggle button:not(:last-child) { border-right: 1px solid var(--border); }
/* Search highlight for table rows */
table.data-table tr.search-hit { background: #fff9c4 !important; }
table.data-table tr.search-hit td { font-weight: 600; }
.glossary-toggle { background: none; border: 1px solid rgba(255,255,255,0.3); color: #b0bec5;
  padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; white-space: nowrap; }
.glossary-toggle:hover { color: white; border-color: rgba(255,255,255,0.6); }
.glossary-panel { display: none; background: #fff; border-bottom: 2px solid var(--border);
  padding: 16px 24px; max-height: 70vh; overflow-y: auto; }
.glossary-panel.open { display: block; }
.glossary-panel h2 { font-size: 15px; margin-bottom: 12px; color: #263238; }
.glossary-columns { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 32px; }
@media (max-width: 1400px) { .glossary-columns { grid-template-columns: 1fr; } }
.glossary-section { margin-bottom: 12px; }
.glossary-section h3 { font-size: 12px; text-transform: uppercase; color: var(--text-muted);
  letter-spacing: 0.5px; margin-bottom: 6px; border-bottom: 1px solid var(--border); padding-bottom: 3px; }
.glossary-dl { font-size: 12px; line-height: 1.5; }
.glossary-dl dt { font-weight: 600; color: var(--text); margin-top: 4px; }
.glossary-dl dd { margin-left: 12px; color: #555; margin-bottom: 2px; }
/* Tooltip styling for th elements */
th[title], label[title], .has-tip[title] { cursor: help; }
th[title]::after { content: " \u24D8"; font-size: 9px; color: #90a4ae; vertical-align: super; }
/* Hierarchy dropdown (tissue > cell type) */
.hierarchy-select { min-width: 200px; }
.hierarchy-select optgroup { font-weight: 600; font-style: normal; color: #263238; }
.hierarchy-select option { font-weight: 400; color: var(--text); }
.hierarchy-select option.tissue-option { font-weight: 600; }
/* Tab description */
.tab-desc { font-size: 12px; color: var(--text-muted); margin-bottom: 12px;
  padding: 8px 12px; background: #f5f5f5; border-radius: 4px; border-left: 3px solid #42a5f5; }
/* Checkbox dropdown (multi-select) */
.cb-dropdown { position: relative; display: inline-block; }
.cb-toggle { font-size: 12px; padding: 2px 8px; border: 1px solid rgba(255,255,255,0.3);
  border-radius: 4px; background: none; color: #b0bec5; cursor: pointer; white-space: nowrap; }
.cb-toggle:hover { color: white; border-color: rgba(255,255,255,0.6); }
.cb-toggle .cb-count { background: #42a5f5; color: #fff; border-radius: 8px; padding: 0 5px;
  font-size: 10px; font-weight: 700; margin-left: 4px; }
.cb-menu { display: none; position: absolute; top: 100%; left: 0; min-width: 200px;
  background: #fff; border: 1px solid var(--border); border-radius: 0 0 6px 6px;
  max-height: 300px; overflow-y: auto; z-index: 200; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
.cb-menu.open { display: block; }
.cb-actions { display: flex; gap: 4px; padding: 4px 8px; border-bottom: 1px solid var(--border); }
.cb-actions button { font-size: 10px; padding: 1px 6px; border: 1px solid var(--border);
  border-radius: 3px; background: #fafafa; cursor: pointer; color: var(--text-muted); }
.cb-actions button:hover { background: #e3f2fd; }
.cb-item { display: flex; align-items: center; gap: 5px; padding: 3px 8px; font-size: 12px;
  cursor: pointer; color: var(--text); }
.cb-item:hover { background: #e3f2fd; }
.cb-item input { margin: 0; }
/* Score Builder */
.score-builder-toggle { background: none; border: 1px solid rgba(255,255,255,0.3); color: #b0bec5;
  padding: 4px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; white-space: nowrap; }
.score-builder-toggle:hover { color: white; border-color: rgba(255,255,255,0.6); }
.score-builder-toggle.active { color: #fff; border-color: #42a5f5; background: rgba(66,165,245,0.2); }
.score-builder-panel { display: none; background: #fff; border-bottom: 2px solid var(--border);
  padding: 14px 24px; }
.score-builder-panel.open { display: block; }
.score-builder-panel h2 { font-size: 14px; margin-bottom: 10px; color: #263238; }
.score-builder-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px 18px; }
.score-dim { display: flex; flex-direction: column; gap: 3px; }
.score-dim label { font-size: 11px; font-weight: 600; color: var(--text); display: flex; align-items: center; gap: 5px; }
.score-dim .dim-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
.score-dim input[type=range] { width: 100%; }
.score-dim .dim-val { font-size: 11px; color: var(--text-muted); text-align: center; }
.score-preset-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.score-preset-row label { font-size: 12px; font-weight: 600; }
.score-preset-row select { font-size: 12px; padding: 2px 6px; }
.score-legend { display: flex; gap: 14px; margin-top: 8px; font-size: 10px; color: var(--text-muted); }
.score-legend-item { display: flex; align-items: center; gap: 3px; }
.score-bar-wrap { display: inline-flex; align-items: center; gap: 4px; }
.score-bar { display: inline-flex; height: 12px; width: 70px; border-radius: 2px; overflow: hidden; }
.score-bar span { display: block; height: 100%; min-width: 0; }
.score-val { font-size: 11px; font-weight: 600; min-width: 22px; text-align: right; }
</style>
</head>
<body>

<header>
  <h1>Kinase Attribution Viewer</h1>
  <div class="global-controls">
    <label title="False Discovery Rate: the expected proportion of false positives among significant results. Lower values are more stringent. Default 0.25 is standard for GSEA-based enrichment. Adjusting this slider re-computes significance across all views.">FDR threshold:
      <input type="range" id="fdr-slider" min="0.05" max="0.50" step="0.05" value="0.25">
      <span class="val" id="fdr-value">0.25</span>
    </label>
    <label title="Whole Mouse Brain atlas expression fold: how specifically a kinase gene is expressed in a cell type compared to uniform expression across all 24 cell types. 1.0&times; = no filtering. 2.0&times; = kinase must be expressed at least 2&times; more than the average cell type.">WMB fold min:
      <input type="range" id="wmb-slider" min="1.0" max="5.0" step="0.5" value="1.0">
      <span class="val" id="wmb-value">1.0&times;</span>
    </label>
    <div class="cb-dropdown" id="family-filter" title="Filter all views to kinases in selected evolutionary families. Check multiple to see their union.">
      <button class="cb-toggle">Family</button>
      <div class="cb-menu"></div>
    </div>
    <div class="cb-dropdown" id="trajectory-filter" title="Filter all views by temporal significance pattern. Check multiple to see kinases matching any selected pattern.">
      <button class="cb-toggle">Significant in</button>
      <div class="cb-menu"></div>
    </div>
    <button class="glossary-toggle" id="glossary-btn" title="Open terminology glossary">&#9432; Glossary &amp; Help</button>
    <button class="score-builder-toggle" id="score-btn" title="Configure composite ranking score">&#9881; Score Builder</button>
  </div>
</header>

<!-- Collapsible Score Builder -->
<div class="score-builder-panel" id="score-panel">
  <h2>Composite Score Builder</h2>
  <div class="score-preset-row">
    <label>Preset:</label>
    <select id="score-preset">
      <option value="balanced">Balanced</option>
      <option value="consistency">Consistency-first</option>
      <option value="effect">Effect-size-first</option>
      <option value="custom">Custom</option>
    </select>
    <span style="font-size:11px; color:#757575; margin-left:8px;">Adjust weights to define how kinases are ranked. The Score column in explorer tables updates live.</span>
  </div>
  <div class="score-builder-grid">
    <div class="score-dim">
      <label><span class="dim-dot" style="background:#1976d2;"></span>Sig. contrasts (of 9)</label>
      <input type="range" id="sw-consistency" min="0" max="100" step="5" value="15">
      <div class="dim-val" id="sv-consistency">15</div>
    </div>
    <div class="score-dim">
      <label><span class="dim-dot" style="background:#d32f2f;"></span>Peak |NES|</label>
      <input type="range" id="sw-magnitude" min="0" max="100" step="5" value="15">
      <div class="dim-val" id="sv-magnitude">15</div>
    </div>
    <div class="score-dim">
      <label><span class="dim-dot" style="background:#388e3c;"></span>Multi-timepoint signal</label>
      <input type="range" id="sw-temporal" min="0" max="100" step="5" value="15">
      <div class="dim-val" id="sv-temporal">15</div>
    </div>
    <div class="score-dim">
      <label><span class="dim-dot" style="background:#f57c00;"></span>WMB fold (top cell type)</label>
      <input type="range" id="sw-specificity" min="0" max="100" step="5" value="15">
      <div class="dim-val" id="sv-specificity">15</div>
    </div>
    <div class="score-dim">
      <label><span class="dim-dot" style="background:#7b1fa2;"></span>|SEA-AD LFC| (top cell type)</label>
      <input type="range" id="sw-concordance" min="0" max="100" step="5" value="10">
      <div class="dim-val" id="sv-concordance">10</div>
    </div>
    <div class="score-dim">
      <label><span class="dim-dot" style="background:#00897b;"></span>|Song LFC| (within-cohort)</label>
      <input type="range" id="sw-songConcordance" min="0" max="100" step="5" value="30">
      <div class="dim-val" id="sv-songConcordance">30</div>
    </div>
  </div>
  <div class="score-legend">
    <span class="score-legend-item"><span class="dim-dot" style="background:#1976d2; width:8px; height:8px; border-radius:50%; display:inline-block;"></span> Sig. contrasts: fraction of 9 contrasts with FDR &lt; threshold</span>
    <span class="score-legend-item"><span class="dim-dot" style="background:#d32f2f; width:8px; height:8px; border-radius:50%; display:inline-block;"></span> Peak |NES|: strongest effect, normalized to dataset max</span>
    <span class="score-legend-item"><span class="dim-dot" style="background:#388e3c; width:8px; height:8px; border-radius:50%; display:inline-block;"></span> Trend: disease models with &ge;2 sig. timepoints (0/3 = 0, 3/3 = 1.0)</span>
    <span class="score-legend-item"><span class="dim-dot" style="background:#f57c00; width:8px; height:8px; border-radius:50%; display:inline-block;"></span> WMB fold: top cell-type specificity, normalized to dataset max</span>
    <span class="score-legend-item"><span class="dim-dot" style="background:#7b1fa2; width:8px; height:8px; border-radius:50%; display:inline-block;"></span> |SEA-AD LFC|: top cell-type AD expression change, normalized to dataset max</span>
    <span class="score-legend-item"><span class="dim-dot" style="background:#00897b; width:8px; height:8px; border-radius:50%; display:inline-block;"></span> |Song LFC|: within-cohort snRNA-seq concordance (paired animals), normalized to dataset max</span>
  </div>
</div>

<!-- Collapsible glossary -->
<div class="glossary-panel" id="glossary-panel">
  <h2>Terminology &amp; Glossary</h2>
  <div class="glossary-columns">
    <div>
      <div class="glossary-section">
        <h3>Core Metrics</h3>
        <dl class="glossary-dl">
          <dt>NES (Normalized Enrichment Score)</dt>
          <dd>Measures whether a kinase's known substrate phosphorylation sites are collectively shifted up or down in stoichiometry. Positive NES = increased kinase activity; negative = decreased. Computed via GSEA on stoichiometry-corrected phosphoproteomics data. Values typically range from &minus;3 to +3.</dd>
          <dt>FDR (False Discovery Rate)</dt>
          <dd>Statistical correction for multiple testing. An FDR of 0.25 means up to 25% of results called "significant" may be false positives. This is the standard threshold for GSEA/MEA enrichment analysis. The slider lets you explore stricter (0.05) or more permissive (0.50) cutoffs.</dd>
          <dt>Stoichiometry</dt>
          <dd>log2(phosphorylation intensity) &minus; log2(parent protein abundance). Removes the effect of protein abundance changes, isolating activity-driven phosphorylation signals.</dd>
          <dt>SEA-AD LFC (Log Fold Change)</dt>
          <dd>Differential gene expression of the kinase gene in human Alzheimer's disease brain tissue (Seattle Alzheimer's Disease Brain Cell Atlas, Allen Institute). Positive = upregulated in AD; negative = downregulated. Measured in specific cell types from human postmortem brain snRNA-seq.</dd>
          <dt>Composite Score (Score Builder)</dt>
          <dd>User-configurable ranking from 0&ndash;100 combining 6 normalized dimensions: (1) Sig. contrasts &mdash; fraction of 9 contrasts with FDR &lt; threshold, (2) Peak |NES| &mdash; strongest effect size normalized to the dataset maximum, (3) Multi-timepoint signal &mdash; count of disease models with &ge;2 significant timepoints (0/3=0, 3/3=1.0), (4) WMB fold &mdash; top cell-type expression specificity normalized to dataset max, (5) |SEA-AD LFC| &mdash; top cell-type human AD expression change normalized to dataset max, (6) |Song LFC| &mdash; within-cohort snRNA-seq concordance from paired animals (default weight 0). Each dimension is weighted by the Score Builder sliders. The stacked bar shows which dimensions contribute to each kinase's score.</dd>
        </dl>
      </div>
      <div class="glossary-section">
        <h3>Cell-Type Attribution</h3>
        <dl class="glossary-dl">
          <dt>WMB (Whole Mouse Brain) Expression</dt>
          <dd>Expression specificity of each kinase gene across 24 cell types, from the Allen Institute Whole Mouse Brain atlas (10Xv3 scRNA-seq). Used as a biological gate: a kinase must be expressed in a cell type to be a plausible candidate for that cell type.</dd>
          <dt>WMB Fold</dt>
          <dd>How many times more specifically a kinase is expressed in a given cell type compared to uniform distribution (1/24 &asymp; 4.2%). A fold of 6.0&times; means the kinase is expressed 6&times; more in that cell type than the average across all 24 types.</dd>
          <dt>WMB Tier</dt>
          <dd><strong>HIGH</strong>: WMB specificity &ge; 2&times; uniform (&ge;8.3% of total expression in this cell type). <strong>Low</strong>: between 1&times; and 2&times; uniform &mdash; kinase is expressed above average but not strongly specific.</dd>
          <dt>Confidence (HIGH / low)</dt>
          <dd><strong>HIGH</strong>: At least one cell type where WMB tier is "high" AND concordance evidence from Song within-cohort snRNA-seq (|Song LFC| &gt; 0.1) or SEA-AD cross-species data (|SEA-AD LFC| &gt; 0.1). <strong>Low</strong>: Has cell-type candidates but none meet both criteria simultaneously.</dd>
          <dt>Song LFC (Within-Cohort)</dt>
          <dd>Differential expression of the kinase gene in the paired snRNA-seq data from the same mouse cohort. Computed via factorial OLS on pseudobulk (males only, pooled across timepoints). Unlike SEA-AD (cross-species human reference), this is same-species, same-cohort evidence. Direction is pathway-matched: App, Tau, or ApTt.</dd>
          <dt>Tissue Category</dt>
          <dd>The 24 cell subclasses are grouped into 7 tissue categories: Excitatory neurons (9 subtypes, e.g., L2/3 IT, L5 ET), Interneurons (9 subtypes, e.g., Pvalb, Sst, Vip), Astrocytes, Oligodendrocytes, OPCs, Microglia, Endothelial cells.</dd>
        </dl>
      </div>
    </div>
    <div>
      <div class="glossary-section">
        <h3>Significant In</h3>
        <dl class="glossary-dl">
          <dt>Significant in column</dt>
          <dd>Shows significance and direction at each timepoint (2mo/4mo/6mo) per disease model. <strong>&uarr;</strong> = significant upregulated NES, <strong>&darr;</strong> = significant downregulated NES, <strong>&mdash;</strong> = not significant at the current FDR threshold. Disease models are color-coded: <span style="color:#c62828;">APP</span> (amyloid), <span style="color:#1565c0;">Tau</span>, <span style="color:#6a1b9a;">A&times;T</span> (double transgenic). Updates when you adjust the FDR slider.</dd>
          <dt>Significant in filter (header)</dt>
          <dd>Multi-select checkbox filter. Options: <strong>Sig. in App/Tau/A&times;T</strong> &mdash; kinase has &ge;1 significant contrast in that disease model. <strong>Early (2mo only)</strong> &mdash; significant at 2mo but not 4mo/6mo in at least one condition. <strong>Late (6mo only)</strong> &mdash; significant at 6mo but not 2mo/4mo. <strong>Multi-timepoint</strong> &mdash; &ge;2 significant timepoints in at least one condition. Checking multiple items shows kinases matching <em>any</em> checked pattern.</dd>
          <dt>Legacy trend label (detail card)</dt>
          <dd>The kinase detail card shows a single trend label (e.g., progressive, peaked) computed from the peak disease condition only. These are a backward-compatible summary; the significant in column provides more complete per-disease-model information.</dd>
        </dl>
      </div>
      <div class="glossary-section">
        <h3>Experimental Design</h3>
        <dl class="glossary-dl">
          <dt>9 Contrasts (disease vs WT)</dt>
          <dd>Each contrast compares one disease model to wild-type controls at a specific timepoint. 3 disease models &times; 3 timepoints = 9 comparisons. Disease models: <strong>APP</strong> (amyloid pathology), <strong>Tau</strong> (tau pathology), <strong>A&times;T</strong> (double transgenic, both pathologies). Timepoints: 2, 4, and 6 months. "Sig. vs WT" counts how many of these 9 comparisons show significant kinase activity change.</dd>
          <dt>Kinase Family</dt>
          <dd>Evolutionary classification of kinases (colored dots). Major groups: AGC, CAMK, CMGC, STE, TKL, CK1. Each family shares structural features and often targets related substrates.</dd>
          <dt>MEA (Motif Enrichment Analysis)</dt>
          <dd>GSEA-based method that tests whether known substrates of each kinase are collectively shifted in stoichiometry. Produces NES and FDR per kinase per contrast.</dd>
          <dt>Additivity (A&times;T predicted vs observed)</dt>
          <dd>Predicted A&times;T NES = APP NES + Tau NES. Observed = actual A&times;T NES. Points above the diagonal are synergistic (combined effect exceeds sum of parts). Points below are sub-additive or antagonistic (combined effect is weaker than expected). Biologically, co-expression of both AD transgenes is expected to produce synergy.</dd>
        </dl>
      </div>
      <div class="glossary-section">
        <h3>Visual Cues</h3>
        <dl class="glossary-dl">
          <dt><span class="badge badge-near-miss">near-miss</span></dt>
          <dd>Kinase has 0 significant contrasts at the current FDR threshold, but would gain significance if the threshold were relaxed by 0.05. These are threshold-adjacent results worth examining.</dd>
          <dt>Sub-threshold rows (grayed out)</dt>
          <dd>Rows that fall below the current WMB fold or FDR threshold. They remain visible (not deleted) so you can see what was filtered and adjust thresholds to include them.</dd>
          <dt>Heatmap colors (brown &harr; teal)</dt>
          <dd>Brown = negative NES (decreased kinase activity). Teal = positive NES (increased activity). White = near zero. This scale intentionally avoids the red/blue/purple disease-model colors used elsewhere in the viewer.</dd>
          <dt>* in heatmap cells</dt>
          <dd>Indicates the NES value is statistically significant (FDR &lt; current threshold).</dd>
          <dt>Black border (bubble mode)</dt>
          <dd>Triangle markers with black borders are statistically significant. Borderless markers are not.</dd>
          <dt>Disease colors</dt>
          <dd><span style="color:#c62828;">&block;</span> Red = APP (amyloid), <span style="color:#1565c0;">&block;</span> Blue = Tau, <span style="color:#6a1b9a;">&block;</span> Purple = A&times;T (double). Used in bar charts, sparklines, and scatter plots &mdash; but <em>not</em> in the heatmap, where color encodes NES direction instead.</dd>
          <dt>Sparkline direction</dt>
          <dd>In the NES profile column, bars extend <strong>up</strong> from center for positive NES (increased activity) and <strong>down</strong> for negative NES (decreased activity). Bar color = disease model. Opaque = significant; faded = not.</dd>
          <dt>Significance arrows (&uarr; &darr; &mdash;)</dt>
          <dd>In the significant in column: <strong>&uarr;</strong> = significant positive NES (upregulated kinase activity), <strong>&darr;</strong> = significant negative NES (downregulated), <strong>&mdash;</strong> = not significant at the current FDR threshold. Three positions per disease model represent 2mo/4mo/6mo.</dd>
        </dl>
      </div>
    </div>
  </div>
</div>

<nav id="tab-bar">
  <button class="active" data-tab="kinase-explorer">Kinase Explorer</button>
  <button data-tab="celltype-explorer">Cell-Type Explorer</button>
  <button data-tab="heatmap">NES Heatmap</button>
  <button data-tab="direction">Direction Over Time</button>
  <button data-tab="additivity">Additivity Scatter</button>
</nav>

<main>
  <!-- Tab 1: Kinase Explorer -->
  <div id="tab-kinase-explorer" class="tab-panel active">
    <div class="tab-desc">Browse all 311 kinases ranked by activity. Click any row to see its full NES profile across 9 contrasts and cell-type evidence. Use the FDR slider above to explore what becomes significant at different thresholds.</div>
    <div class="filter-bar">
      <div class="kinase-search-group"><div class="kinase-search"><span class="search-icon">&#128269;</span><input type="text" class="ks-input" placeholder="Search kinase or gene..." autocomplete="off"><div class="search-dropdown ks-dropdown"></div></div><div class="search-mode-toggle" title="Filter: hide non-matching items. Highlight: show all items, visually emphasize matches."><button class="ks-mode" data-mode="filter">Filter</button><button class="ks-mode active" data-mode="highlight">Highlight</button></div></div>
    </div>
    <div class="explorer-layout">
      <div class="data-table-wrap"><table class="data-table" id="ke-table">
        <thead><tr>
          <th data-col="kinase" title="Kinase name.">Kinase</th>
          <th data-col="_fam" title="Evolutionary kinase family (AGC, CAMK, CMGC, STE, TKL, CK1, etc.). Use the Family filter in the header to restrict all views to one family.">Family</th>
          <th data-col="gene_symbol" title="Gene symbol encoding this kinase protein.">Gene</th>
          <th data-col="n_sig_contrasts" data-type="num" title="Number of the 9 disease-vs-WT comparisons (3 disease models &#215; 3 timepoints: e.g., APP vs WT at 2mo, Tau vs WT at 4mo, etc.) where this kinase shows significant activity change (FDR below the current threshold). A higher count means dysregulation is more consistent across conditions. Updates when you adjust the FDR slider.">Sig. vs WT</th>
          <th data-col="peak_NES" data-type="num" title="Largest absolute NES value across all 9 contrasts. Indicates the strongest kinase activity change observed in any condition/timepoint.">Peak NES</th>
          <th title="Significance and direction at each timepoint per disease model. &#8593; = significant upregulated, &#8595; = significant downregulated, &#8212; = not significant. Positions: 2mo/4mo/6mo. Updates with FDR slider.">Significant in</th>
          <th data-col="top_celltype_1" title="The cell type where this kinase gene is most specifically expressed (highest WMB fold), among those passing the WMB expression gate.">Top cell type</th>
          <th data-col="top_celltype_1_wmb_fold" data-type="num" title="How many times more specifically this kinase is expressed in its top cell type vs. uniform distribution across all 24 types. Higher = more cell-type-specific expression.">WMB fold</th>
          <th data-col="has_high_conf_attribution" title="Attribution confidence. HIGH = at least one cell type with strong WMB expression specificity (&#8805;2x uniform) AND concordance evidence from Song within-cohort or SEA-AD cross-species data (|LFC| > 0.1). Low = has cell-type candidates but evidence is weaker.">Conf.</th>
          <th data-col="_score" data-type="num" title="Composite ranking score (0&ndash;100) combining 6 weighted dimensions: consistency, effect magnitude, temporal coherence, cell-type specificity, SEA-AD concordance, and Song concordance. Configure weights in the Score Builder panel above.">Score</th>
        </tr></thead>
        <tbody></tbody>
      </table></div>
      <div class="detail-card" id="ke-detail">
        <h3>Select a kinase</h3>
        <p class="meta">Click a row to see details</p>
        <div id="ke-detail-nes" style="height:140px;"></div>
        <div id="ke-detail-evidence" class="evidence-list" style="margin-top:12px;"></div>
      </div>
    </div>
  </div>

  <!-- Tab 2: Cell-Type Explorer -->
  <div id="tab-celltype-explorer" class="tab-panel">
    <div class="tab-desc">Select a cell type to see all kinases expressed there. Sorted by WMB expression specificity. Use the WMB fold slider above to dim kinases with weak cell-type specificity.</div>
    <div class="tab-controls">
      <div class="kinase-search-group"><div class="kinase-search"><span class="search-icon">&#128269;</span><input type="text" class="ks-input" placeholder="Search kinase or gene..." autocomplete="off"><div class="search-dropdown ks-dropdown"></div></div><div class="search-mode-toggle" title="Filter: hide non-matching items. Highlight: show all items, visually emphasize matches."><button class="ks-mode" data-mode="filter">Filter</button><button class="ks-mode active" data-mode="highlight">Highlight</button></div></div>
      <label title="Select a cell type or tissue category to view kinases expressed there. Tissue categories group related cell types (e.g., Interneurons includes Pvalb, Sst, Vip, etc.).">View: <select id="ct-select" class="hierarchy-select"></select></label>
    </div>
    <div id="ct-search-notice" style="display:none; padding:8px 12px; margin-bottom:8px; background:#fff3e0; border:1px solid #ffe0b2; border-radius:4px; font-size:12px; color:#e65100;"></div>
    <div class="explorer-layout">
      <div class="data-table-wrap"><table class="data-table" id="ct-table">
        <thead><tr>
          <th data-col="kinase" title="Kinase name.">Kinase</th>
          <th data-col="_fam" title="Evolutionary kinase family.">Family</th>
          <th data-col="gene_symbol" title="Gene symbol encoding this kinase protein.">Gene</th>
          <th data-col="cell_type" title="Cell type (subclass) this row represents. Visible when viewing a tissue category." class="ct-col-celltype" style="display:none;">Cell type</th>
          <th data-col="wmb_fold_over_uniform" data-type="num" title="Expression specificity in the selected cell type: fold enrichment over uniform (1/24). Higher = more specifically expressed here vs. other cell types.">WMB fold</th>
          <th data-col="sea_ad_lfc" data-type="num" title="Log fold change of this kinase gene in human AD brain (SEA-AD snRNA-seq) in this specific cell type. Positive = upregulated in AD; negative = downregulated.">SEA-AD LFC</th>
          <th data-col="song_lfc" data-type="num" title="Log fold change of this kinase gene in paired within-cohort snRNA-seq. Same-species, same-cohort evidence from factorial OLS on pseudobulk (males only).">Song LFC</th>
          <th title="Significance and direction per disease model at each timepoint. &#8593;/&#8595;/&#8212; = up/down/not sig. Positions: 2mo/4mo/6mo.">Significant in</th>
          <th data-col="n_sig_contrasts" data-type="num" title="Number of the 9 disease-vs-WT comparisons where this kinase shows significant activity change at the current FDR threshold.">Sig. vs WT</th>
          <th title="Mini bar chart showing NES values across all 9 contrasts. Bar height = |NES|, color = disease model (red=APP, blue=Tau, purple=AxT). Opaque bars are significant; faded bars are not.">NES profile</th>
          <th data-col="_score" data-type="num" title="Composite ranking score (0&ndash;100). Configure weights in the Score Builder panel above.">Score</th>
        </tr></thead>
        <tbody></tbody>
      </table></div>
      <div class="detail-card" id="ct-detail">
        <h3>Select a kinase</h3>
        <p class="meta">Click a row to see details</p>
        <div id="ct-detail-nes" style="height:140px;"></div>
        <div id="ct-detail-evidence" class="evidence-list" style="margin-top:12px;"></div>
      </div>
    </div>
  </div>

  <!-- Tab 3: NES Heatmap -->
  <div id="tab-heatmap" class="tab-panel">
    <div class="tab-desc">NES activity heatmap for kinases attributed to each tissue category. Color encodes direction and magnitude of kinase activity change. * = statistically significant at the current FDR threshold. Toggle to Bubble mode for a direction-focused view with triangle markers.</div>
    <div class="heatmap-controls">
      <div class="kinase-search-group"><div class="kinase-search"><span class="search-icon">&#128269;</span><input type="text" class="ks-input" placeholder="Search kinase or gene..." autocomplete="off"><div class="search-dropdown ks-dropdown"></div></div><div class="search-mode-toggle" title="Filter: hide non-matching items. Highlight: show all items, visually emphasize matches."><button class="ks-mode" data-mode="filter">Filter</button><button class="ks-mode active" data-mode="highlight">Highlight</button></div></div>
      <label title="Filter to kinases attributed to this tissue category or specific cell type.">View: <select id="hm-tissue" class="hierarchy-select"></select></label>
      <label title="Show only the top N kinases by strongest significant |NES| value.">Top N:
        <select id="hm-topn">
          <option value="10">10</option>
          <option value="25">25</option>
          <option value="50" selected>50</option>
          <option value="">All</option>
        </select>
      </label>
      <label>Sort:
        <select id="hm-sort">
          <option value="clustered">Clustered</option>
          <option value="alpha">Alphabetical</option>
          <option value="peak">By peak |NES|</option>
          <option value="family">By family</option>
        </select>
      </label>
      <span style="margin-left:8px;">Mode:</span>
      <button id="hm-mode-heatmap" class="active">Heatmap</button>
      <button id="hm-mode-bubble">Bubble</button>
    </div>
    <div id="hm-search-notice" style="display:none; padding:8px 12px; margin-bottom:8px; background:#fff3e0; border:1px solid #ffe0b2; border-radius:4px; font-size:12px; color:#e65100;"></div>
    <div id="hm-plot" style="width:100%; max-height:75vh; overflow-y:auto;"></div>
  </div>

  <!-- Tab 4: Direction Over Time -->
  <div id="tab-direction" class="tab-panel">
    <div class="tab-desc">How many kinases are significantly up- or down-regulated at each timepoint, grouped by disease model and tissue. Bars above zero = upregulated kinase activity; below zero = downregulated. Hover over bars to see which kinases are counted. Counts update with the FDR slider.</div>
    <div class="tab-controls">
      <div class="kinase-search-group"><div class="kinase-search"><span class="search-icon">&#128269;</span><input type="text" class="ks-input" placeholder="Search kinase or gene..." autocomplete="off"><div class="search-dropdown ks-dropdown"></div></div><div class="search-mode-toggle" title="Filter: hide non-matching items. Highlight: show all items, visually emphasize matches."><button class="ks-mode" data-mode="filter">Filter</button><button class="ks-mode active" data-mode="highlight">Highlight</button></div></div>
      <label title="Show direction counts for a specific tissue category, cell type, or all tissues combined.">View: <select id="dir-tissue" class="hierarchy-select"></select></label>
    </div>
    <div id="dir-plot" style="width:100%;"></div>
  </div>

  <!-- Tab 5: Additivity Scatter -->
  <div id="tab-additivity" class="tab-panel">
    <div class="tab-desc"><strong>Reading this plot:</strong> Each point is a kinase. The x-axis shows the <em>predicted</em> A&times;T effect (APP NES + Tau NES). The y-axis shows the <em>observed</em> A&times;T NES. Points on the dashed diagonal have perfectly additive effects. Points <strong>above</strong> the diagonal are synergistic (A&times;T exceeds prediction). Points <strong>below</strong> are sub-additive. Colors indicate which disease models show significant activity for that kinase. Gray points have no significant contrasts. Hover for kinase names.</div>
    <div class="info-banner" style="padding:8px 14px; margin-bottom:10px; background:#e3f2fd; border:1px solid #90caf9; border-radius:4px; font-size:12px; color:#1565c0; line-height:1.5;"><strong>Sanity check:</strong> Prior analysis found sub-additive behavior in the combined genotype (A&times;T response weaker than App or Tau alone). Biologically, co-expression of both AD transgenes is expected to produce synergistic effects. Points above the diagonal indicate synergy; points below indicate antagonism or interference.</div>
    <div class="tab-controls">
      <div class="kinase-search-group"><div class="kinase-search"><span class="search-icon">&#128269;</span><input type="text" class="ks-input" placeholder="Search kinase or gene..." autocomplete="off"><div class="search-dropdown ks-dropdown"></div></div><div class="search-mode-toggle" title="Filter: hide non-matching items. Highlight: show all items, visually emphasize matches."><button class="ks-mode" data-mode="filter">Filter</button><button class="ks-mode active" data-mode="highlight">Highlight</button></div></div>
      <label title="Filter to kinases attributed to this tissue category or specific cell type.">View: <select id="add-tissue" class="hierarchy-select"></select></label>
    </div>
    <div id="add-plot" style="width:100%;"></div>
  </div>

</main>

<script>
// =========================================================================
// Embedded data
// =========================================================================
const DATA = __DATA_JSON__;

// =========================================================================
// State
// =========================================================================
const state = {
  fdrThreshold: DATA.config.fdrThreshDefault,
  wmbFoldMin: 1.0,
  selectedKinase: null,
  searchQuery: "",       // synced kinase search across tabs
  searchMatches: new Set(), // kinase names matching current query
  searchMode: "highlight", // "filter" or "highlight"
  familyFilter: new Set(),       // empty = all pass; non-empty = union filter
  trajectoryFilter: new Set(),   // empty = all pass; predicate-based filter
  tissueCategory: DATA.config.tissueOrder[0],
  heatmapMode: "heatmap",
  sortMode: "clustered",
  topN: 50,
  ctCellType: null,
  stale: {}, // track which tabs need re-render
  scoreWeights: { consistency: 15, magnitude: 15, temporal: 15, specificity: 15, concordance: 10, songConcordance: 30 },
  scorePreset: "balanced",
};

// =========================================================================
// Derived lookups (built once)
// =========================================================================
const CONTRASTS = DATA.config.contrasts;
const NES_COLS = CONTRASTS.map(c => c + "_NES");
const FDR_COLS = CONTRASTS.map(c => c + "_FDR");

// Build kinase → T1 record lookup
const t1Map = {};
DATA.kinaseActivity.forEach(r => { t1Map[r.kinase] = r; });

// Build kinase → T3 record lookup
const t3Map = {};
DATA.kinaseHypothesis.forEach(r => { t3Map[r.kinase] = r; });

// Build kinase → [T2 records] lookup
const t2ByKinase = {};
DATA.celltypeEvidence.forEach(r => {
  if (!t2ByKinase[r.kinase]) t2ByKinase[r.kinase] = [];
  t2ByKinase[r.kinase].push(r);
});

// Build cellType → [T4 records] lookup
const t4ByCellType = {};
DATA.celltypeProfiles.forEach(r => {
  if (!t4ByCellType[r.cell_type]) t4ByCellType[r.cell_type] = [];
  t4ByCellType[r.cell_type].push(r);
});

// Build MEA lookup: kinase → {contrast → {NES, FDR}}
const meaLookup = {};
DATA.meaStoichiometry.forEach(r => {
  if (!meaLookup[r.kinase]) meaLookup[r.kinase] = {};
  meaLookup[r.kinase][r.contrast] = { NES: r.NES, FDR: r.FDR };
});

// Tissue map from config
const subclassToTissue = DATA.config.subclassToTissue;

// Build reverse map: tissue → [cell types] (ordered)
const tissueToSubclasses = {};
DATA.config.tissueOrder.forEach(t => { tissueToSubclasses[t] = []; });
Object.entries(subclassToTissue).forEach(([ct, tissue]) => {
  if (!tissueToSubclasses[tissue]) tissueToSubclasses[tissue] = [];
  // Only include cell types that actually have T4 data
  if (t4ByCellType[ct]) tissueToSubclasses[tissue].push(ct);
});
Object.values(tissueToSubclasses).forEach(arr => arr.sort());

function getTissue(kinase) {
  const t3 = t3Map[kinase];
  if (!t3 || !t3.top_celltype_1) return null;
  return subclassToTissue[t3.top_celltype_1] || "Other";
}

// -------------------------------------------------------------------------
// Hierarchical dropdown helpers (tissue category > cell type)
// -------------------------------------------------------------------------
// Values: "all" | "tissue:Excitatory neurons" | "ct:L2/3 IT"
function parseSelection(val) {
  if (!val || val === "all") return { level: "all", name: null };
  if (val.startsWith("tissue:")) return { level: "tissue", name: val.slice(7) };
  if (val.startsWith("ct:")) return { level: "ct", name: val.slice(3) };
  return { level: "all", name: null };
}

function buildHierarchyDropdown(sel, opts) {
  // opts: { includeAll: bool, allLabel: string }
  sel.innerHTML = "";
  if (opts && opts.includeAll) {
    const o = document.createElement("option");
    o.value = "all"; o.textContent = opts.allLabel || "All";
    sel.appendChild(o);
  }
  DATA.config.tissueOrder.forEach(tissue => {
    const subclasses = tissueToSubclasses[tissue] || [];
    if (subclasses.length === 0 && !DATA.clusterOrders[tissue]) return;
    const grp = document.createElement("optgroup");
    grp.label = tissue;
    // Tissue-level option
    const tOpt = document.createElement("option");
    tOpt.value = "tissue:" + tissue;
    tOpt.textContent = "\u25B8 All " + tissue;
    tOpt.className = "tissue-option";
    grp.appendChild(tOpt);
    // Individual cell types
    subclasses.forEach(ct => {
      const cOpt = document.createElement("option");
      cOpt.value = "ct:" + ct;
      cOpt.textContent = "    " + ct;
      grp.appendChild(cOpt);
    });
    sel.appendChild(grp);
  });
}

// Get cell types for a selection
function getCellTypesForSelection(sel) {
  const s = parseSelection(sel);
  if (s.level === "all") return Object.keys(t4ByCellType);
  if (s.level === "tissue") return tissueToSubclasses[s.name] || [];
  if (s.level === "ct") return [s.name];
  return [];
}

// Get tissue categories for a selection
function getTissuesForSelection(sel) {
  const s = parseSelection(sel);
  if (s.level === "all") return DATA.config.tissueOrder.filter(t =>
    Object.values(subclassToTissue).includes(t));
  if (s.level === "tissue") return [s.name];
  if (s.level === "ct") {
    const tissue = subclassToTissue[s.name];
    return tissue ? [tissue] : [];
  }
  return [];
}

// Get kinases attributed to a selection (for heatmap/direction)
function getKinasesForSelection(sel) {
  const s = parseSelection(sel);
  if (s.level === "tissue") {
    return DATA.clusterOrders[s.name] ? [...DATA.clusterOrders[s.name]] : [];
  }
  if (s.level === "ct") {
    // Kinases whose top_celltype_1 is this specific cell type
    return DATA.kinaseHypothesis
      .filter(r => r.top_celltype_1 === s.name)
      .map(r => r.kinase);
  }
  // "all" — union of all tissue orders
  const all = [];
  DATA.config.tissueOrder.forEach(t => {
    if (DATA.clusterOrders[t]) all.push(...DATA.clusterOrders[t]);
  });
  return all;
}

// =========================================================================
// Utility functions
// =========================================================================
function countSig(kinase, fdr) {
  const t1 = t1Map[kinase];
  if (!t1) return 0;
  return FDR_COLS.reduce((n, c) => n + (t1[c] != null && t1[c] < fdr ? 1 : 0), 0);
}

function isNearMiss(kinase, fdr) {
  // Kinase has 0 sig at current threshold but >0 at threshold + 0.05
  return countSig(kinase, fdr) === 0 && countSig(kinase, fdr + 0.05) > 0;
}

function nesColor(nes) {
  // BrBG-inspired: brown (negative NES) → white → teal (positive NES)
  // Avoids collision with disease colors (red=APP, blue=Tau, purple=AxT)
  if (nes == null) return "#f5f5f5";
  const t = Math.max(-2, Math.min(2, nes)) / 2; // [-1, 1]
  if (t >= 0) {
    // white → teal (#01665e)
    const r = Math.round(245 - 244 * t);
    const g = Math.round(245 - 143 * t);
    const b = Math.round(245 - 151 * t);
    return `rgb(${r},${g},${b})`;
  } else {
    // white → brown (#8c510a)
    const s = -t;
    const r = Math.round(245 - 105 * s);
    const g = Math.round(245 - 164 * s);
    const b = Math.round(245 - 235 * s);
    return `rgb(${r},${g},${b})`;
  }
}

function makeSparkline(kinase) {
  // Diverging sparkline: bars go up for positive NES, down for negative NES
  // Center baseline = zero; height proportional to |NES|; color = disease
  const t1 = t1Map[kinase];
  if (!t1) return "";
  const maxAbs = 2;
  const halfH = 11; // max pixels above or below center
  let html = '<span style="display:inline-flex;align-items:center;height:24px;vertical-align:middle;">';
  // Container is a 2-row CSS grid: top row for positive, bottom for negative
  html += '<span style="display:inline-grid;grid-template-rows:' + halfH + 'px ' + halfH + 'px;gap:0;align-items:end;">';
  // Top row (positive bars, aligned to bottom of their cell)
  html += '<span style="display:inline-flex;align-items:flex-end;gap:1px;height:' + halfH + 'px;">';
  CONTRASTS.forEach(c => {
    const nes = t1[c + "_NES"];
    const disease = c.split("_")[0];
    const color = DATA.config.diseaseColors[disease] || "#999";
    const fdr = t1[c + "_FDR"];
    const sig = isSig(fdr);
    const opacity = sig ? 1 : 0.3;
    if (nes != null && nes > 0) {
      const h = Math.max(1, Math.round(Math.min(nes / maxAbs, 1) * halfH));
      html += `<span style="width:5px;height:${h}px;background:${color};opacity:${opacity};border-radius:1px 1px 0 0;" title="${c}: ${nes.toFixed(2)}${sig?'*':''}"></span>`;
    } else {
      html += '<span style="width:5px;height:0;"></span>';
    }
  });
  html += '</span>';
  // Bottom row (negative bars, aligned to top of their cell)
  html += '<span style="display:inline-flex;align-items:flex-start;gap:1px;height:' + halfH + 'px;border-top:1px solid #ccc;">';
  CONTRASTS.forEach(c => {
    const nes = t1[c + "_NES"];
    const disease = c.split("_")[0];
    const color = DATA.config.diseaseColors[disease] || "#999";
    const fdr = t1[c + "_FDR"];
    const sig = isSig(fdr);
    const opacity = sig ? 1 : 0.3;
    if (nes != null && nes < 0) {
      const h = Math.max(1, Math.round(Math.min(Math.abs(nes) / maxAbs, 1) * halfH));
      html += `<span style="width:5px;height:${h}px;background:${color};opacity:${opacity};border-radius:0 0 1px 1px;" title="${c}: ${nes.toFixed(2)}${sig?'*':''}"></span>`;
    } else {
      html += '<span style="width:5px;height:0;"></span>';
    }
  });
  html += '</span>';
  html += '</span></span>';
  return html;
}

function trajectoryLabel(kinase) {
  // Append direction arrow to trajectory label based on peak NES sign
  const t3 = t3Map[kinase];
  if (!t3 || !t3.trajectory_label) return "";
  const label = t3.trajectory_label;
  if (label === "none" || label === "None") return label;
  const peak = t3.peak_NES;
  if (peak == null) return label;
  return label + (peak > 0 ? " \u2191" : " \u2193"); // ↑ or ↓
}

function isSig(fdr) { return fdr != null && fdr < state.fdrThreshold; }

function passesFamily(kinase) {
  if (state.familyFilter.size === 0) return true;
  return state.familyFilter.has(DATA.familyMap[kinase] || "Other");
}

// Per-disease significant timepoints for a kinase (returns array of "2mo"/"4mo"/"6mo")
function getSigTimepoints(kinase, disease) {
  const t1 = t1Map[kinase];
  if (!t1) return [];
  return DATA.config.timepoints.filter(tp => isSig(t1[disease + "_" + tp + "_FDR"]));
}

// Trajectory filter predicates — each maps to a function(kinase) → bool
const TRAJ_PREDICATES = {
  "Sig. in App":      k => getSigTimepoints(k, "App").length > 0,
  "Sig. in Tau":      k => getSigTimepoints(k, "Tau").length > 0,
  "Sig. in A\u00d7T": k => getSigTimepoints(k, "ApTt").length > 0,
  "Early (2mo only)": k => DATA.config.diseaseGroups.some(d => { const s = getSigTimepoints(k, d); return s.length === 1 && s[0] === "2mo"; }),
  "Late (6mo only)":  k => DATA.config.diseaseGroups.some(d => { const s = getSigTimepoints(k, d); return s.length === 1 && s[0] === "6mo"; }),
  "Multi-timepoint":  k => DATA.config.diseaseGroups.some(d => getSigTimepoints(k, d).length >= 2),
};

function passesTrajectory(kinase) {
  if (state.trajectoryFilter.size === 0) return true;
  // Union: passes if kinase matches ANY checked predicate
  for (const label of state.trajectoryFilter) {
    if (TRAJ_PREDICATES[label] && TRAJ_PREDICATES[label](kinase)) return true;
  }
  return false;
}

function passesWmb(kinase) {
  if (state.wmbFoldMin <= 1.0) return true;
  const t3 = t3Map[kinase];
  return t3 && t3.top_celltype_1_wmb_fold != null && t3.top_celltype_1_wmb_fold >= state.wmbFoldMin;
}

function belowWmb(kinase) {
  if (state.wmbFoldMin <= 1.0) return false;
  const t3 = t3Map[kinase];
  return !t3 || t3.top_celltype_1_wmb_fold == null || t3.top_celltype_1_wmb_fold < state.wmbFoldMin;
}

function famDot(kinase) {
  const fam = DATA.familyMap[kinase] || "Other";
  const color = DATA.familyColors[fam] || "#8b4513";
  return `<span class="fam-dot" style="background:${color};" title="${fam}"></span>`;
}

// =========================================================================
// Temporal pattern display
// =========================================================================
const DISEASE_DISPLAY = [["App","APP","#c62828"], ["Tau","Tau","#1565c0"], ["ApTt","A\u00d7T","#6a1b9a"]];

function renderTemporalPattern(kinase) {
  const t1 = t1Map[kinase];
  if (!t1) return "";
  return DISEASE_DISPLAY.map(([key, label, color]) => {
    const arrows = DATA.config.timepoints.map(tp => {
      const nes = t1[key + "_" + tp + "_NES"];
      if (!isSig(t1[key + "_" + tp + "_FDR"])) return '<span style="color:#ccc;">\u2014</span>';
      return nes > 0 ? '<span style="color:#2e7d32;">\u2191</span>' : '<span style="color:#c62828;">\u2193</span>';
    }).join("");
    return `<span title="${label} 2mo/4mo/6mo"><span style="color:${color};font-weight:600;">${label}</span>\u2009${arrows}</span>`;
  }).join(" ");
}

// =========================================================================
// Composite Score computation
// =========================================================================
const SCORE_DIMS = ["consistency", "magnitude", "temporal", "specificity", "concordance", "songConcordance"];
const SCORE_DIM_COLORS = {
  consistency: "#1976d2", magnitude: "#d32f2f", temporal: "#388e3c",
  specificity: "#f57c00", concordance: "#7b1fa2", songConcordance: "#00897b"
};

function trajectoryOrdinal(kinase) {
  // Count disease models with ≥2 significant timepoints (0-3) → 0, 0.33, 0.67, 1.0
  const count = DATA.config.diseaseGroups.filter(d => getSigTimepoints(kinase, d).length >= 2).length;
  return count / 3;
}

// Score map: kinase → { composite, dims: { name: { raw, norm, weighted } } }
let scoreMap = {};

function computeScores() {
  const w = state.scoreWeights;
  const wTotal = w.consistency + w.magnitude + w.temporal + w.specificity + w.concordance + w.songConcordance;

  // Collect raw values per kinase
  const raw = {};
  DATA.kinaseHypothesis.forEach(r => {
    raw[r.kinase] = {
      consistency: countSig(r.kinase, state.fdrThreshold) / 9,
      magnitude: Math.abs(r.peak_NES || 0),
      temporal: trajectoryOrdinal(r.kinase),
      specificity: r.top_celltype_1_wmb_fold || 0,
      concordance: Math.abs(r.top_celltype_1_sea_ad_lfc || 0),
      songConcordance: Math.abs(r.top_celltype_1_song_lfc || 0),
    };
  });

  // Find maxima for normalization
  const maxMag = Math.max(0.001, ...Object.values(raw).map(r => r.magnitude));
  const maxSpec = Math.max(0.001, ...Object.values(raw).map(r => r.specificity));
  const maxConc = Math.max(0.001, ...Object.values(raw).map(r => r.concordance));
  const maxSong = Math.max(0.001, ...Object.values(raw).map(r => r.songConcordance));

  scoreMap = {};
  Object.entries(raw).forEach(([kinase, vals]) => {
    const norm = {
      consistency: vals.consistency,  // already 0–1
      magnitude: vals.magnitude / maxMag,
      temporal: vals.temporal,        // already 0–1
      specificity: vals.specificity / maxSpec,
      concordance: vals.concordance / maxConc,
      songConcordance: vals.songConcordance / maxSong,
    };
    let composite = 0;
    const dims = {};
    SCORE_DIMS.forEach(d => {
      const weighted = wTotal > 0 ? (w[d] / wTotal) * norm[d] : 0;
      composite += weighted;
      dims[d] = { raw: vals[d], norm: norm[d], weighted };
    });
    scoreMap[kinase] = { composite: Math.round(composite * 100), dims };
  });
}

function renderScoreBar(kinase) {
  const s = scoreMap[kinase];
  if (!s) return '<span class="score-bar-wrap"><span class="score-bar"></span><span class="score-val">-</span></span>';
  const total = s.composite;
  const DIM_LABELS = {
    consistency: "Sig. contrasts", magnitude: "Peak |NES|", temporal: "Multi-timepoint",
    specificity: "WMB fold", concordance: "|SEA-AD LFC|", songConcordance: "|Song LFC|"
  };
  const parts = SCORE_DIMS.map(d => {
    const pct = total > 0 ? (s.dims[d].weighted / (total / 100)) : 0;
    const title = DIM_LABELS[d] + ": " + s.dims[d].raw.toFixed(2) + " (normalized " + s.dims[d].norm.toFixed(2) + "), weight " + state.scoreWeights[d];
    return `<span style="width:${pct.toFixed(1)}%;background:${SCORE_DIM_COLORS[d]};" title="${title}"></span>`;
  }).join("");
  return `<span class="score-bar-wrap"><span class="score-bar">${parts}</span><span class="score-val">${total}</span></span>`;
}

// Initial computation
computeScores();

// =========================================================================
// Dynamic dropdown label updates (hierarchy-aware)
// =========================================================================

function getFilteredKinasesForSelection(val) {
  let kinases = getKinasesForSelection(val);
  if (state.familyFilter.size > 0) kinases = kinases.filter(k => passesFamily(k));
  if (state.trajectoryFilter.size > 0) kinases = kinases.filter(k => passesTrajectory(k));
  if (state.wmbFoldMin > 1.0) kinases = kinases.filter(k => passesWmb(k));
  if (state.searchMatches.size > 0 && state.searchMode === "filter") {
    kinases = kinases.filter(k => state.searchMatches.has(k));
  }
  return kinases;
}

function getFilteredCtRowsForSelection(val) {
  const cellTypes = getCellTypesForSelection(val);
  let rows = [];
  cellTypes.forEach(ct => { rows = rows.concat(t4ByCellType[ct] || []); });
  if (state.familyFilter.size > 0) rows = rows.filter(r => passesFamily(r.kinase));
  if (state.trajectoryFilter.size > 0) rows = rows.filter(r => passesTrajectory(r.kinase));
  if (state.wmbFoldMin > 1.0) rows = rows.filter(r => r.wmb_fold_over_uniform >= state.wmbFoldMin);
  if (state.searchMatches.size > 0 && state.searchMode === "filter") {
    rows = rows.filter(r => state.searchMatches.has(r.kinase));
  }
  return rows;
}

function updateHierarchyDropdown(sel, countFn) {
  const hasFilter = state.familyFilter.size > 0 || state.trajectoryFilter.size > 0 || (state.searchMatches.size > 0 && state.searchMode === "filter");
  Array.from(sel.options).forEach(opt => {
    const val = opt.value;
    if (val === "all") return; // "All" option doesn't need counts
    const s = parseSelection(val);
    const filtered = countFn(val);
    const total = countFn(val, true); // ignoreFilters
    const label = s.level === "tissue" ? ("\u25B8 All " + s.name) : ("    " + s.name);
    if (hasFilter) {
      opt.textContent = label + " (" + filtered + "/" + total + ")";
      opt.style.fontWeight = filtered > 0 ? "bold" : "normal";
      opt.style.color = filtered > 0 ? "" : "#999";
    } else {
      opt.textContent = label + " (" + total + ")";
      opt.style.fontWeight = "";
      opt.style.color = "";
    }
  });
}

function updateHmTissueDropdown() {
  updateHierarchyDropdown(document.getElementById("hm-tissue"), (val, raw) => {
    if (raw) return getKinasesForSelection(val).length;
    return getFilteredKinasesForSelection(val).length;
  });
}

function updateCtCellTypeDropdown() {
  updateHierarchyDropdown(document.getElementById("ct-select"), (val, raw) => {
    if (raw) {
      const cts = getCellTypesForSelection(val);
      let n = 0; cts.forEach(ct => { n += (t4ByCellType[ct] || []).length; }); return n;
    }
    return getFilteredCtRowsForSelection(val).length;
  });
}

function updateDirTissueDropdown() {
  // Direction Over Time also uses hierarchy dropdown
  updateHierarchyDropdown(document.getElementById("dir-tissue"), (val, raw) => {
    if (raw) return getKinasesForSelection(val).length;
    return getFilteredKinasesForSelection(val).length;
  });
}

function updateAddTissueDropdown() {
  updateHierarchyDropdown(document.getElementById("add-tissue"), (val, raw) => {
    if (raw) return getKinasesForSelection(val).length;
    return getFilteredKinasesForSelection(val).length;
  });
}

function updateAllDropdowns() {
  updateHmTissueDropdown();
  updateCtCellTypeDropdown();
  updateDirTissueDropdown();
  updateAddTissueDropdown();
}

// =========================================================================
// Tab switching
// =========================================================================
const tabs = document.querySelectorAll("#tab-bar button");
const panels = document.querySelectorAll(".tab-panel");

tabs.forEach(btn => {
  btn.addEventListener("click", () => {
    tabs.forEach(b => b.classList.remove("active"));
    panels.forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const tabId = "tab-" + btn.dataset.tab;
    document.getElementById(tabId).classList.add("active");
    renderActiveTab();
  });
});

function getActiveTab() {
  const active = document.querySelector("#tab-bar button.active");
  return active ? active.dataset.tab : "kinase-explorer";
}

// =========================================================================
// Global controls
// =========================================================================
const fdrSlider = document.getElementById("fdr-slider");
const fdrValue = document.getElementById("fdr-value");
const wmbSlider = document.getElementById("wmb-slider");
const wmbValue = document.getElementById("wmb-value");

fdrSlider.addEventListener("input", () => {
  state.fdrThreshold = parseFloat(fdrSlider.value);
  fdrValue.textContent = state.fdrThreshold.toFixed(2);
  computeScores(); // consistency dimension depends on FDR threshold
  Object.keys(renderers).forEach(k => { state.stale[k] = true; });
  updateAllDropdowns();
  renderActiveTab();
});

wmbSlider.addEventListener("input", () => {
  state.wmbFoldMin = parseFloat(wmbSlider.value);
  wmbValue.innerHTML = state.wmbFoldMin.toFixed(1) + "&times;";
  Object.keys(renderers).forEach(k => { state.stale[k] = true; });
  updateAllDropdowns();
  renderActiveTab();
});

// =========================================================================
// Checkbox dropdown builder (multi-select)
// =========================================================================
function buildCheckboxDropdown(containerId, items, stateKey, label) {
  const container = document.getElementById(containerId);
  const toggle = container.querySelector(".cb-toggle");
  const menu = container.querySelector(".cb-menu");
  toggle.textContent = label;

  let html = '<div class="cb-actions"><button class="cb-all">All</button><button class="cb-none">None</button></div>';
  items.forEach(item => {
    html += `<label class="cb-item"><input type="checkbox" value="${item}" checked> ${item}</label>`;
  });
  menu.innerHTML = html;

  function updateState() {
    const checked = menu.querySelectorAll('input:checked');
    const unchecked = menu.querySelectorAll('input:not(:checked)');
    if (unchecked.length === 0) {
      state[stateKey] = new Set(); // all checked = no filter
      toggle.innerHTML = label;
    } else {
      state[stateKey] = new Set([...checked].map(cb => cb.value));
      toggle.innerHTML = label + ' <span class="cb-count">' + checked.length + '</span>';
    }
    Object.keys(renderers).forEach(k => { state.stale[k] = true; });
    updateAllDropdowns();
    renderActiveTab();
  }

  menu.querySelectorAll('input').forEach(cb => cb.addEventListener("change", updateState));
  menu.querySelector(".cb-all").addEventListener("click", () => {
    menu.querySelectorAll('input').forEach(cb => { cb.checked = true; });
    updateState();
  });
  menu.querySelector(".cb-none").addEventListener("click", () => {
    menu.querySelectorAll('input').forEach(cb => { cb.checked = false; });
    updateState();
  });

  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    // Close other open menus
    document.querySelectorAll(".cb-menu.open").forEach(m => { if (m !== menu) m.classList.remove("open"); });
    menu.classList.toggle("open");
  });
  menu.addEventListener("click", (e) => e.stopPropagation());
}

// Close all checkbox menus when clicking outside
document.addEventListener("click", () => {
  document.querySelectorAll(".cb-menu.open").forEach(m => m.classList.remove("open"));
});

// =========================================================================
// Score Builder controls
// =========================================================================
const scoreBtn = document.getElementById("score-btn");
const scorePanel = document.getElementById("score-panel");
scoreBtn.addEventListener("click", () => {
  scorePanel.classList.toggle("open");
  scoreBtn.classList.toggle("active");
});

const SCORE_PRESETS = {
  balanced:    { consistency: 15, magnitude: 15, temporal: 15, specificity: 15, concordance: 10, songConcordance: 30 },
  consistency: { consistency: 30, magnitude: 15, temporal: 15, specificity: 10, concordance: 5, songConcordance: 25 },
  effect:      { consistency: 10, magnitude: 30, temporal: 10, specificity: 15, concordance: 10, songConcordance: 25 },
};

const scorePresetSel = document.getElementById("score-preset");

function applyScoreWeights(weights) {
  SCORE_DIMS.forEach(d => {
    document.getElementById("sw-" + d).value = weights[d];
    document.getElementById("sv-" + d).textContent = weights[d];
    state.scoreWeights[d] = weights[d];
  });
  computeScores();
  Object.keys(renderers).forEach(k => { state.stale[k] = true; });
  renderActiveTab();
}

scorePresetSel.addEventListener("change", () => {
  const preset = scorePresetSel.value;
  state.scorePreset = preset;
  if (SCORE_PRESETS[preset]) applyScoreWeights(SCORE_PRESETS[preset]);
});

SCORE_DIMS.forEach(d => {
  const slider = document.getElementById("sw-" + d);
  const valEl = document.getElementById("sv-" + d);
  slider.addEventListener("input", () => {
    state.scoreWeights[d] = parseInt(slider.value);
    valEl.textContent = slider.value;
    scorePresetSel.value = "custom";
    state.scorePreset = "custom";
    computeScores();
    Object.keys(renderers).forEach(k => { state.stale[k] = true; });
    renderActiveTab();
  });
});

// =========================================================================
// Synced per-tab kinase search
// =========================================================================

// Build search index once
const ksIndex = DATA.kinaseHypothesis.map(r => ({
  kinase: r.kinase,
  gene: r.gene_symbol || "",
  family: DATA.familyMap[r.kinase] || "Other",
  tissue: getTissue(r.kinase) || "",
  trajectory: trajectoryLabel(r.kinase),
  searchText: ((r.kinase || "") + " " + (r.gene_symbol || "")).toLowerCase(),
}));

// All search inputs and dropdowns (one per tab)
const ksInputs = document.querySelectorAll(".ks-input");
const ksDropdowns = document.querySelectorAll(".ks-dropdown");
let ksActiveIdx = -1;
let ksActiveDropdown = null;

function ksUpdateMatches() {
  const q = state.searchQuery.toLowerCase();
  state.searchMatches = new Set();
  if (!q) return;
  ksIndex.forEach(r => {
    if (r.searchText.includes(q)) state.searchMatches.add(r.kinase);
  });
}

function ksRenderDropdown(dropdown, query) {
  if (!query) { dropdown.classList.remove("open"); return; }
  const q = query.toLowerCase();
  const scored = ksIndex
    .filter(r => r.searchText.includes(q))
    .map(r => {
      let score = 0;
      if (r.kinase.toLowerCase().startsWith(q)) score = 3;
      else if (r.gene.toLowerCase().startsWith(q)) score = 2;
      else if (r.kinase.toLowerCase().includes(q)) score = 1;
      return { ...r, score };
    })
    .sort((a, b) => b.score - a.score || a.kinase.localeCompare(b.kinase))
    .slice(0, 15);

  if (scored.length === 0) {
    dropdown.innerHTML = '<div class="search-empty">No kinases match &ldquo;' + query.replace(/</g, "&lt;") + '&rdquo;</div>';
  } else {
    dropdown.innerHTML = scored.map((r, i) =>
      `<div class="search-item" data-kinase="${r.kinase}">
        <span><span class="kinase-name">${famDot(r.kinase)}${r.kinase}</span> <span class="gene-name">${r.gene}</span></span>
        <span class="search-meta">${r.tissue}${r.trajectory ? ' \u00b7 ' + r.trajectory : ''}</span>
      </div>`
    ).join("");
  }
  dropdown.classList.add("open");
  ksActiveIdx = -1;
  ksActiveDropdown = dropdown;
}

function ksSyncInputs(sourceInput) {
  ksInputs.forEach(inp => { if (inp !== sourceInput) inp.value = state.searchQuery; });
}

function ksApplySearch(query, sourceInput) {
  state.searchQuery = query;
  ksUpdateMatches();
  ksSyncInputs(sourceInput);
  // Re-render active tab with highlights
  renderActiveTab();
}

// Wire up each search input
ksInputs.forEach((inp, idx) => {
  const dropdown = ksDropdowns[idx];

  inp.addEventListener("input", () => {
    state.searchQuery = inp.value;
    ksUpdateMatches();
    ksSyncInputs(inp);
    ksRenderDropdown(dropdown, inp.value);
    updateAllDropdowns();
    renderActiveTab();
  });

  inp.addEventListener("keydown", (e) => {
    const items = dropdown.querySelectorAll(".search-item");
    if (e.key === "Escape") { dropdown.classList.remove("open"); inp.blur(); return; }
    if (!items.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      ksActiveIdx = Math.min(ksActiveIdx + 1, items.length - 1);
      items.forEach((el, i) => el.classList.toggle("active", i === ksActiveIdx));
      items[ksActiveIdx].scrollIntoView({ block: "nearest" });
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      ksActiveIdx = Math.max(ksActiveIdx - 1, 0);
      items.forEach((el, i) => el.classList.toggle("active", i === ksActiveIdx));
      items[ksActiveIdx].scrollIntoView({ block: "nearest" });
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (ksActiveIdx >= 0 && items[ksActiveIdx]) {
        // Select specific kinase from dropdown
        const kinase = items[ksActiveIdx].dataset.kinase;
        inp.value = kinase;
        state.searchQuery = kinase;
        ksUpdateMatches();
        ksSyncInputs(inp);
        dropdown.classList.remove("open");
        updateAllDropdowns();
        renderActiveTab();
      } else {
        // Just close dropdown and apply current text as search
        dropdown.classList.remove("open");
      }
    }
  });

  dropdown.addEventListener("click", (e) => {
    const item = e.target.closest(".search-item");
    if (!item) return;
    const kinase = item.dataset.kinase;
    inp.value = kinase;
    state.searchQuery = kinase;
    ksUpdateMatches();
    ksSyncInputs(inp);
    dropdown.classList.remove("open");
    updateAllDropdowns();
    renderActiveTab();
  });
});

// Close all dropdowns when clicking outside
document.addEventListener("click", (e) => {
  if (!e.target.closest(".kinase-search")) {
    ksDropdowns.forEach(d => d.classList.remove("open"));
  }
});

// Search mode toggle (filter vs highlight), synced across all tabs
const ksModeButtons = document.querySelectorAll(".ks-mode");
ksModeButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    state.searchMode = btn.dataset.mode;
    // Sync all toggle buttons across tabs
    ksModeButtons.forEach(b => b.classList.toggle("active", b.dataset.mode === state.searchMode));
    Object.keys(renderers).forEach(k => { state.stale[k] = true; });
    updateAllDropdowns();
    renderActiveTab();
  });
});

// =========================================================================
// Tab 1: Kinase Explorer
// =========================================================================
let keSortCol = "n_sig_contrasts";
let keSortAsc = false;

function renderKinaseExplorer() {
  const tbody = document.querySelector("#ke-table tbody");
  const hasSearch = state.searchMatches.size > 0;

  let rows = DATA.kinaseHypothesis.map(r => {
    const nSig = countSig(r.kinase, state.fdrThreshold);
    const tissue = getTissue(r.kinase);
    const fam = DATA.familyMap[r.kinase] || "Other";
    const nm = isNearMiss(r.kinase, state.fdrThreshold);
    const hit = hasSearch && state.searchMatches.has(r.kinase);
    return { ...r, _nSig: nSig, _tissue: tissue, _fam: fam, _nearMiss: nm, _hit: hit };
  });

  // Filters
  if (hasSearch && state.searchMode === "filter") rows = rows.filter(r => r._hit);
  if (state.familyFilter.size > 0) rows = rows.filter(r => state.familyFilter.has(r._fam));
  if (state.trajectoryFilter.size > 0) rows = rows.filter(r => passesTrajectory(r.kinase));

  // Sort
  const col = keSortCol;
  rows.sort((a, b) => {
    let va, vb;
    if (col === "n_sig_contrasts") { va = a._nSig; vb = b._nSig; }
    else if (col === "top_celltype_1_wmb_fold") { va = a.top_celltype_1_wmb_fold || 0; vb = b.top_celltype_1_wmb_fold || 0; }
    else if (col === "peak_NES") { va = Math.abs(a.peak_NES || 0); vb = Math.abs(b.peak_NES || 0); }
    else if (col === "has_high_conf_attribution") { va = a.has_high_conf_attribution ? 1 : 0; vb = b.has_high_conf_attribution ? 1 : 0; }
    else if (col === "_score") { va = (scoreMap[a.kinase] || {}).composite || 0; vb = (scoreMap[b.kinase] || {}).composite || 0; }
    else { va = (a[col] || "").toString(); vb = (b[col] || "").toString(); return keSortAsc ? va.localeCompare(vb) : vb.localeCompare(va); }
    return keSortAsc ? va - vb : vb - va;
  });

  let html = "";
  rows.forEach(r => {
    const cls = [];
    if (state.selectedKinase === r.kinase) cls.push("selected");
    if (r._hit && state.searchMode === "highlight") cls.push("search-hit");
    if (r._nSig === 0 && !r._nearMiss) cls.push("sub-thresh");
    if (belowWmb(r.kinase)) cls.push("sub-thresh");
    if (r._nearMiss) cls.push("near-miss");
    const confBadge = r.has_high_conf_attribution
      ? '<span class="badge badge-high">HIGH</span>'
      : (r.n_celltype_candidates > 0 ? '<span class="badge badge-low">low</span>' : '');
    const nmBadge = r._nearMiss ? ' <span class="badge badge-near-miss">near-miss</span>' : '';
    html += `<tr class="${cls.join(" ")}" data-kinase="${r.kinase}">
      <td>${famDot(r.kinase)}${r.kinase}${nmBadge}</td>
      <td>${famDot(r.kinase)}${r._fam}</td>
      <td>${r.gene_symbol || ""}</td>
      <td>${r._nSig}</td>
      <td>${r.peak_NES != null ? r.peak_NES.toFixed(2) : ""}</td>
      <td style="font-family:monospace;font-size:11px;letter-spacing:-0.5px;">${renderTemporalPattern(r.kinase)}</td>
      <td>${r.top_celltype_1 || ""}</td>
      <td>${r.top_celltype_1_wmb_fold != null ? r.top_celltype_1_wmb_fold.toFixed(1) + "\u00d7" : ""}</td>
      <td>${confBadge}</td>
      <td>${renderScoreBar(r.kinase)}</td>
    </tr>`;
  });
  tbody.innerHTML = html;

  // Click handler
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => {
      state.selectedKinase = tr.dataset.kinase;
      renderKinaseDetail(state.selectedKinase);
      tbody.querySelectorAll("tr").forEach(r => r.classList.remove("selected"));
      tr.classList.add("selected");
    });
  });

  if (state.selectedKinase) renderKinaseDetail(state.selectedKinase);
}

function renderKinaseDetail(kinase) {
  const card = document.getElementById("ke-detail");
  const t3 = t3Map[kinase];
  const t1 = t1Map[kinase];
  if (!t3) { card.querySelector("h3").textContent = "Not found"; return; }

  const fam = DATA.familyMap[kinase] || "Other";
  const tissue = getTissue(kinase);
  card.querySelector("h3").innerHTML = `${famDot(kinase)}<strong>${kinase}</strong> (${t3.gene_symbol || "?"})`;
  card.querySelector(".meta").innerHTML =
    `Family: ${fam} &bull; Tissue: ${tissue || "none"} &bull; Trend: ${trajectoryLabel(kinase)}<br>` +
    `Sig. contrasts: ${countSig(kinase, state.fdrThreshold)} (at FDR &lt; ${state.fdrThreshold.toFixed(2)})`;

  // NES bar chart
  if (t1) {
    const vals = CONTRASTS.map(c => t1[c + "_NES"]);
    const fdrs = CONTRASTS.map(c => t1[c + "_FDR"]);
    const colors = CONTRASTS.map(c => DATA.config.diseaseColors[c.split("_")[0]]);
    const borders = fdrs.map(f => isSig(f) ? "black" : "rgba(0,0,0,0)");
    const labels = CONTRASTS.map(c => {
      const [d, t] = c.split("_");
      return {"App":"APP","Tau":"Tau","ApTt":"A\u00d7T"}[d] + " " + t;
    });

    Plotly.react("ke-detail-nes", [{
      type: "bar", x: labels, y: vals,
      marker: { color: colors, line: { color: borders, width: fdrs.map(f => isSig(f) ? 2 : 0) } },
      text: fdrs.map(f => isSig(f) ? "*" : ""),
      textposition: "outside", textfont: { size: 14, color: "black" },
    }], {
      height: 130, margin: { l: 35, r: 10, t: 5, b: 40 },
      yaxis: { title: "NES", range: [-3, 3], zeroline: true },
      xaxis: { tickfont: { size: 9 } },
      plot_bgcolor: "white", paper_bgcolor: "white",
    }, { displayModeBar: false, responsive: true });
  }

  // Evidence table
  const evRows = t2ByKinase[kinase] || [];
  let evHtml = "<table><thead><tr><th>Cell type</th><th>WMB fold</th><th>SEA-AD LFC</th><th>Song LFC</th><th>Tier</th></tr></thead><tbody>";
  evRows.sort((a, b) => (b.wmb_fold_over_uniform || 0) - (a.wmb_fold_over_uniform || 0));
  evRows.forEach(r => {
    const muted = r.wmb_fold_over_uniform < state.wmbFoldMin ? ' style="opacity:0.4;"' : "";
    const songLfc = r.song_lfc != null && isFinite(r.song_lfc) ? r.song_lfc.toFixed(2) : "\u2014";
    evHtml += `<tr${muted}><td>${r.cell_type}</td><td>${(r.wmb_fold_over_uniform || 0).toFixed(1)}\u00d7</td>` +
      `<td>${(r.sea_ad_lfc || 0).toFixed(2)}</td>` +
      `<td>${songLfc}</td>` +
      `<td>${r.wmb_tier === "high" ? '<span class="badge badge-high">high</span>' : "low"}</td></tr>`;
  });
  evHtml += "</tbody></table>";
  if (evRows.length === 0) evHtml = "<p style='color:#999;font-size:11px;'>No WMB-gated cell types</p>";
  document.getElementById("ke-detail-evidence").innerHTML = evHtml;
}

// Sort handlers for kinase explorer table
document.querySelectorAll("#ke-table th").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (!col) return;
    if (keSortCol === col) keSortAsc = !keSortAsc;
    else { keSortCol = col; keSortAsc = false; }
    renderKinaseExplorer();
  });
});

// (Trajectory and tissue filters are now global checkbox dropdowns in header)

// =========================================================================
// Tab 2: Cell-Type Explorer
// =========================================================================
let ctSortCol = "wmb_fold_over_uniform";
let ctSortAsc = false;

function renderCelltypeExplorer() {
  const sel = document.getElementById("ct-select");
  const selVal = sel.value;
  if (!selVal) return;
  const parsed = parseSelection(selVal);
  const cellTypes = getCellTypesForSelection(selVal);
  const isTissueView = parsed.level === "tissue";
  const displayLabel = parsed.name || "selection";
  state.ctCellType = parsed.level === "ct" ? parsed.name : null;

  // Show/hide cell type column based on whether we're in tissue view
  document.querySelectorAll(".ct-col-celltype").forEach(el => {
    el.style.display = isTissueView ? "" : "none";
  });

  const tbody = document.querySelector("#ct-table tbody");
  const hasSearch = state.searchMatches.size > 0;
  let rows = [];
  cellTypes.forEach(ct => {
    (t4ByCellType[ct] || []).forEach(r => {
      const nSig = countSig(r.kinase, state.fdrThreshold);
      const nm = isNearMiss(r.kinase, state.fdrThreshold);
      const isBelowWmb = r.wmb_fold_over_uniform < state.wmbFoldMin;
      const hit = hasSearch && state.searchMatches.has(r.kinase);
      const fam = DATA.familyMap[r.kinase] || "Other";
      rows.push({ ...r, _nSig: nSig, _nearMiss: nm, _belowWmb: isBelowWmb, _hit: hit, _fam: fam });
    });
  });

  // Show notification if search is active but no matches in this selection
  const noticeEl = document.getElementById("ct-search-notice");
  if (hasSearch) {
    const anyHit = rows.some(r => r._hit);
    if (!anyHit) {
      const matchList = [...state.searchMatches].slice(0, 5).join(", ");
      const extra = state.searchMatches.size > 5 ? ` and ${state.searchMatches.size - 5} more` : "";
      noticeEl.innerHTML = `<strong>${matchList}${extra}</strong> not found in <strong>${displayLabel}</strong>. ` +
        `Try selecting a different cell type or tissue, or check the Kinase Explorer tab for cell-type attribution.`;
      noticeEl.style.display = "block";
    } else {
      noticeEl.style.display = "none";
    }
  } else {
    noticeEl.style.display = "none";
  }

  if (hasSearch && state.searchMode === "filter") rows = rows.filter(r => r._hit);
  if (state.familyFilter.size > 0) rows = rows.filter(r => passesFamily(r.kinase));
  if (state.trajectoryFilter.size > 0) rows = rows.filter(r => passesTrajectory(r.kinase));

  // Sort
  rows.sort((a, b) => {
    const col = ctSortCol;
    let va, vb;
    if (col === "_score") {
      va = (scoreMap[a.kinase] || {}).composite || 0;
      vb = (scoreMap[b.kinase] || {}).composite || 0;
    } else if (col === "wmb_fold_over_uniform" || col === "sea_ad_lfc" || col === "song_lfc" || col === "n_sig_contrasts") {
      va = col === "n_sig_contrasts" ? a._nSig : (a[col] || 0);
      vb = col === "n_sig_contrasts" ? b._nSig : (b[col] || 0);
      if (col === "sea_ad_lfc" || col === "song_lfc") { va = Math.abs(va); vb = Math.abs(vb); }
    } else {
      va = (a[col] || "").toString(); vb = (b[col] || "").toString();
      return ctSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return ctSortAsc ? va - vb : vb - va;
  });

  let html = "";
  rows.forEach(r => {
    const cls = [];
    if (state.selectedKinase === r.kinase) cls.push("selected");
    if (r._hit && state.searchMode === "highlight") cls.push("search-hit");
    if (r._belowWmb) cls.push("sub-thresh");
    if (r._nearMiss) cls.push("near-miss");
    const nmBadge = r._nearMiss ? ' <span class="badge badge-near-miss">near-miss</span>' : '';
    const ctCell = isTissueView ? `<td class="ct-col-celltype">${r.cell_type || ""}</td>` : "";
    html += `<tr class="${cls.join(" ")}" data-kinase="${r.kinase}">
      <td>${famDot(r.kinase)}${r.kinase}${nmBadge}</td>
      <td>${famDot(r.kinase)}${r._fam}</td>
      <td>${r.gene_symbol || ""}</td>
      ${ctCell}
      <td>${(r.wmb_fold_over_uniform || 0).toFixed(1)}\u00d7</td>
      <td>${(r.sea_ad_lfc || 0).toFixed(2)}</td>
      <td>${r.song_lfc != null && isFinite(r.song_lfc) ? r.song_lfc.toFixed(2) : "\u2014"}</td>
      <td style="font-family:monospace;font-size:11px;letter-spacing:-0.5px;">${renderTemporalPattern(r.kinase)}</td>
      <td>${r._nSig}</td>
      <td>${makeSparkline(r.kinase)}</td>
      <td>${renderScoreBar(r.kinase)}</td>
    </tr>`;
  });
  tbody.innerHTML = html;

  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => {
      state.selectedKinase = tr.dataset.kinase;
      renderCTDetail(state.selectedKinase);
      tbody.querySelectorAll("tr").forEach(r => r.classList.remove("selected"));
      tr.classList.add("selected");
    });
  });

  if (state.selectedKinase) renderCTDetail(state.selectedKinase);
}

function renderCTDetail(kinase) {
  // Reuse same logic as kinase explorer detail but in ct-detail divs
  const card = document.getElementById("ct-detail");
  const t3 = t3Map[kinase];
  const t1 = t1Map[kinase];
  if (!t3 && !t1) { card.querySelector("h3").textContent = kinase || "Not found"; return; }

  const fam = DATA.familyMap[kinase] || "Other";
  card.querySelector("h3").innerHTML = `${famDot(kinase)}<strong>${kinase}</strong> (${(t3 && t3.gene_symbol) || (t1 && t1.gene_symbol) || "?"})`;
  card.querySelector(".meta").innerHTML =
    `Family: ${fam} &bull; Sig. contrasts: ${countSig(kinase, state.fdrThreshold)}`;

  if (t1) {
    const vals = CONTRASTS.map(c => t1[c + "_NES"]);
    const fdrs = CONTRASTS.map(c => t1[c + "_FDR"]);
    const colors = CONTRASTS.map(c => DATA.config.diseaseColors[c.split("_")[0]]);
    const borders = fdrs.map(f => isSig(f) ? "black" : "rgba(0,0,0,0)");
    const labels = CONTRASTS.map(c => {
      const [d, t] = c.split("_");
      return {"App":"APP","Tau":"Tau","ApTt":"A\u00d7T"}[d] + " " + t;
    });
    Plotly.react("ct-detail-nes", [{
      type: "bar", x: labels, y: vals,
      marker: { color: colors, line: { color: borders, width: fdrs.map(f => isSig(f) ? 2 : 0) } },
      text: fdrs.map(f => isSig(f) ? "*" : ""),
      textposition: "outside", textfont: { size: 14, color: "black" },
    }], {
      height: 130, margin: { l: 35, r: 10, t: 5, b: 40 },
      yaxis: { title: "NES", range: [-3, 3], zeroline: true },
      xaxis: { tickfont: { size: 9 } },
      plot_bgcolor: "white", paper_bgcolor: "white",
    }, { displayModeBar: false, responsive: true });
  }

  const evRows = t2ByKinase[kinase] || [];
  let evHtml = "<table><thead><tr><th>Cell type</th><th>WMB fold</th><th>SEA-AD LFC</th><th>Song LFC</th><th>Tier</th></tr></thead><tbody>";
  evRows.sort((a, b) => (b.wmb_fold_over_uniform || 0) - (a.wmb_fold_over_uniform || 0));
  evRows.forEach(r => {
    const muted = r.wmb_fold_over_uniform < state.wmbFoldMin ? ' style="opacity:0.4;"' : "";
    const highlight = r.cell_type === state.ctCellType ? ' style="font-weight:700;"' : muted;
    const songLfc = r.song_lfc != null && isFinite(r.song_lfc) ? r.song_lfc.toFixed(2) : "\u2014";
    evHtml += `<tr${highlight}><td>${r.cell_type}</td><td>${(r.wmb_fold_over_uniform || 0).toFixed(1)}\u00d7</td>` +
      `<td>${(r.sea_ad_lfc || 0).toFixed(2)}</td><td>${songLfc}</td>` +
      `<td>${r.wmb_tier === "high" ? '<span class="badge badge-high">high</span>' : "low"}</td></tr>`;
  });
  evHtml += "</tbody></table>";
  document.getElementById("ct-detail-evidence").innerHTML = evHtml;
}

// CT table sort
document.querySelectorAll("#ct-table th").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (!col) return;
    if (ctSortCol === col) ctSortAsc = !ctSortAsc;
    else { ctSortCol = col; ctSortAsc = false; }
    renderCelltypeExplorer();
  });
});

document.getElementById("ct-select").addEventListener("change", () => renderCelltypeExplorer());

// =========================================================================
// Tab 3: NES Heatmap
// =========================================================================
function renderHeatmap() {
  const selVal = document.getElementById("hm-tissue").value;
  const parsed = parseSelection(selVal);
  const displayLabel = parsed.name || "All";
  const topNVal = document.getElementById("hm-topn").value;
  const topN = topNVal ? parseInt(topNVal) : null;
  const sortMode = document.getElementById("hm-sort").value;
  state.heatmapMode = document.getElementById("hm-mode-heatmap").classList.contains("active") ? "heatmap" : "bubble";

  // Get kinases for this selection
  let kinases = getKinasesForSelection(selVal);
  if (kinases.length === 0) return;

  const noticeEl = document.getElementById("hm-search-notice");

  // Sort
  if (sortMode === "alpha") {
    kinases.sort();
  } else if (sortMode === "peak") {
    kinases.sort((a, b) => {
      const pa = t1Map[a] ? Math.abs(t1Map[a].peak_NES || 0) : 0;
      const pb = t1Map[b] ? Math.abs(t1Map[b].peak_NES || 0) : 0;
      return pb - pa;
    });
  } else if (sortMode === "family") {
    kinases.sort((a, b) => {
      const fa = DATA.familyMap[a] || "zzz";
      const fb = DATA.familyMap[b] || "zzz";
      if (fa !== fb) return fa.localeCompare(fb);
      return a.localeCompare(b);
    });
  }
  // "clustered" keeps the pre-computed order

  // Family filter
  if (state.familyFilter.size > 0) {
    kinases = kinases.filter(k => passesFamily(k));
    if (kinases.length === 0) {
      noticeEl.innerHTML = 'Selected families have no kinases attributed to <strong>' + displayLabel + '</strong>. Check other tissue categories or cell types in the dropdown.';
      noticeEl.style.display = "block";
      Plotly.purge(document.getElementById("hm-plot"));
      return;
    }
  }

  // Trajectory filter
  if (state.trajectoryFilter.size > 0) {
    kinases = kinases.filter(k => passesTrajectory(k));
    if (kinases.length === 0) {
      noticeEl.innerHTML = 'No kinases in <strong>' + displayLabel + '</strong> match the selected significance filter.';
      noticeEl.style.display = "block";
      Plotly.purge(document.getElementById("hm-plot"));
      return;
    }
  }

  // WMB fold filter
  if (state.wmbFoldMin > 1.0) {
    kinases = kinases.filter(k => passesWmb(k));
    if (kinases.length === 0) {
      noticeEl.innerHTML = 'No kinases in <strong>' + displayLabel + '</strong> meet the WMB fold &ge; ' + state.wmbFoldMin.toFixed(1) + '&times; threshold. Try lowering the WMB fold slider.';
      noticeEl.style.display = "block";
      Plotly.purge(document.getElementById("hm-plot"));
      return;
    }
  }

  // Top N
  if (topN && kinases.length > topN) {
    // Keep top N by max |NES| among significant
    const scored = kinases.map(k => {
      const t1 = t1Map[k];
      if (!t1) return { k, score: 0 };
      let maxNes = 0;
      CONTRASTS.forEach(c => {
        const fdr = t1[c + "_FDR"];
        const nes = t1[c + "_NES"];
        if (isSig(fdr) && nes != null)
          maxNes = Math.max(maxNes, Math.abs(nes));
      });
      return { k, score: maxNes };
    });
    scored.sort((a, b) => b.score - a.score);
    const topKinases = new Set(scored.slice(0, topN).map(s => s.k));
    // Maintain original order for top kinases
    kinases = kinases.filter(k => topKinases.has(k));
  }

  // Search filter mode: restrict to matching kinases only
  const hasSearch = state.searchMatches.size > 0;
  if (hasSearch && state.searchMode === "filter") {
    kinases = kinases.filter(k => state.searchMatches.has(k));
    if (kinases.length === 0) {
      // Find which tissues DO have matches for the notification
      const matchTissues = [];
      Object.entries(DATA.clusterOrders).forEach(([t, klist]) => {
        const n = klist.filter(k => state.searchMatches.has(k) && passesFamily(k) && passesTrajectory(k)).length;
        if (n > 0) matchTissues.push(t + " (" + n + ")");
      });
      const matchList = [...state.searchMatches].slice(0, 3).join(", ");
      const hint = matchTissues.length > 0
        ? " Found in: <strong>" + matchTissues.join(", ") + "</strong>."
        : "";
      noticeEl.innerHTML = '<strong>' + matchList + '</strong> not found in <strong>' + displayLabel + '</strong>.' + hint;
      noticeEl.style.display = "block";
      Plotly.purge(document.getElementById("hm-plot"));
      return;
    }
  }
  // Check for highlight mode: notify if matches exist but none in this selection
  if (hasSearch && state.searchMode === "highlight") {
    const anyHere = kinases.some(k => state.searchMatches.has(k));
    if (!anyHere) {
      const matchTissues = [];
      Object.entries(DATA.clusterOrders).forEach(([t, klist]) => {
        const n = klist.filter(k => state.searchMatches.has(k) && passesFamily(k) && passesTrajectory(k)).length;
        if (n > 0) matchTissues.push(t + " (" + n + ")");
      });
      const matchList = [...state.searchMatches].slice(0, 3).join(", ");
      const hint = matchTissues.length > 0
        ? " Found in: <strong>" + matchTissues.join(", ") + "</strong>."
        : "";
      noticeEl.innerHTML = '<strong>' + matchList + '</strong> not in <strong>' + displayLabel + '</strong> (showing all kinases).' + hint;
      noticeEl.style.display = "block";
    } else {
      noticeEl.style.display = "none";
    }
  } else if (!hasSearch || state.searchMode === "filter") {
    // If search matched (filter mode) or no search, hide notice
    noticeEl.style.display = "none";
  }

  // Build data arrays
  const zVals = []; // NES values (kinases x contrasts)
  const fdrVals = [];
  const annotations = [];
  const labels = CONTRASTS.map(c => {
    const [d, t] = c.split("_");
    return {"App":"APP","Tau":"Tau","ApTt":"A\u00d7T"}[d] + "\n" + t;
  });

  kinases.forEach((k, i) => {
    const t1 = t1Map[k];
    const nesRow = [];
    const fdrRow = [];
    CONTRASTS.forEach((c, j) => {
      const nes = t1 ? t1[c + "_NES"] : null;
      const fdr = t1 ? t1[c + "_FDR"] : null;
      nesRow.push(nes);
      fdrRow.push(fdr);
      if (state.heatmapMode === "heatmap" && nes != null) {
        const sig = isSig(fdr);
        annotations.push({
          x: j, y: i, text: nes.toFixed(2) + (sig ? "*" : ""),
          showarrow: false,
          font: { size: 10, color: Math.abs(nes) > 1.2 ? "white" : "black" },
        });
      }
    });
    zVals.push(nesRow);
    fdrVals.push(fdrRow);
  });

  // Y-axis labels with family colors
  const yLabels = kinases.map(k => {
    const fam = DATA.familyMap[k] || "Other";
    const color = DATA.familyColors[fam] || "#333";
    return k; // Plotly doesn't support HTML in tick labels; we'll use a separate approach
  });

  const plotDiv = document.getElementById("hm-plot");

  // Search highlight shapes for matching kinase rows (highlight mode only)
  const searchShapes = [];
  if (state.searchMatches.size > 0 && state.searchMode === "highlight") {
    kinases.forEach((k, i) => {
      if (state.searchMatches.has(k)) {
        searchShapes.push({
          type: "rect", x0: -0.5, x1: 8.5, y0: i - 0.5, y1: i + 0.5,
          fillcolor: "rgba(255,235,59,0.25)", line: { color: "#f57f17", width: 2 },
          layer: "above",
        });
      }
    });
  }

  // Family grouping: separator lines and labels when sorted by family
  const familyShapes = [];
  const familyAnnotations = [];
  if (sortMode === "family" && kinases.length > 1) {
    let prevFam = DATA.familyMap[kinases[0]] || "Other";
    let groupStart = 0;
    for (let i = 1; i <= kinases.length; i++) {
      const curFam = i < kinases.length ? (DATA.familyMap[kinases[i]] || "Other") : null;
      if (curFam !== prevFam) {
        // Draw separator line
        familyShapes.push({
          type: "line", x0: -0.5, x1: 8.5, y0: i - 0.5, y1: i - 0.5,
          line: { color: "#999", width: 1, dash: "dot" }, layer: "above",
        });
        // Family label annotation at right edge
        const midY = (groupStart + i - 1) / 2;
        const famColor = DATA.familyColors[prevFam] || "#666";
        familyAnnotations.push({
          x: 1.02, y: midY, xref: "paper", yref: "y",
          text: "<b>" + prevFam + "</b>", showarrow: false,
          font: { size: 9, color: famColor }, xanchor: "left",
        });
        groupStart = i;
        prevFam = curFam;
      }
    }
  }

  const rowPx = 22;
  const hmHeight = Math.max(400, kinases.length * rowPx + 120);

  // Purge previous plot to guarantee clean re-render when dimensions change
  Plotly.purge(plotDiv);

  let hmMarginR = familyAnnotations.length > 0 ? 120 : 80;
  if (state.heatmapMode === "heatmap") {
    const trace = {
      type: "heatmap",
      z: zVals, x: labels, y: kinases,
      colorscale: [[0,"#8c510a"],[0.25,"#d8b365"],[0.5,"#f5f5f5"],[0.75,"#5ab4ac"],[1,"#01665e"]],
      zmin: -2, zmax: 2,
      hovertemplate: "%{y}<br>%{x}<br>NES: %{z:.2f}<extra></extra>",
      colorbar: { title: "NES", len: 0.5 },
    };
    const layout = {
      height: hmHeight,
      margin: { l: 120, r: hmMarginR, t: 30, b: 60 },
      xaxis: { tickfont: { size: 12 }, side: "bottom" },
      yaxis: { tickfont: { size: 11 }, autorange: "reversed" },
      annotations: [...annotations, ...familyAnnotations],
      plot_bgcolor: "white",
      shapes: [
        { type: "line", x0: 2.5, x1: 2.5, y0: -0.5, y1: kinases.length - 0.5,
          line: { color: "black", width: 2 } },
        { type: "line", x0: 5.5, x1: 5.5, y0: -0.5, y1: kinases.length - 0.5,
          line: { color: "black", width: 2 } },
        ...searchShapes,
        ...familyShapes,
      ],
    };
    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true });
  } else {
    // Bubble mode: scatter with triangle markers
    const xArr = [], yArr = [], colors = [], symbols = [], sizes = [], edges = [], edgeWidths = [];
    const hoverText = [];

    kinases.forEach((k, i) => {
      CONTRASTS.forEach((c, j) => {
        const t1 = t1Map[k];
        const nes = t1 ? t1[c + "_NES"] : null;
        const fdr = t1 ? t1[c + "_FDR"] : null;
        if (nes == null) return;
        xArr.push(j);
        yArr.push(i);
        colors.push(nes);
        symbols.push(nes > 0 ? "triangle-up" : "triangle-down");
        sizes.push(11);
        const sig = isSig(fdr);
        edges.push(sig ? "black" : "rgba(0,0,0,0)");
        edgeWidths.push(sig ? 1.5 : 0);
        hoverText.push(`${k}<br>${c}<br>NES: ${nes.toFixed(2)}<br>FDR: ${fdr != null ? fdr.toFixed(3) : "N/A"}${sig ? " *" : ""}`);
      });
    });

    const trace = {
      type: "scatter", mode: "markers",
      x: xArr, y: yArr,
      marker: {
        color: colors, colorscale: [[0,"#8c510a"],[0.25,"#d8b365"],[0.5,"#f5f5f5"],[0.75,"#5ab4ac"],[1,"#01665e"]],
        cmin: -2, cmax: 2,
        symbol: symbols, size: sizes,
        line: { color: edges, width: edgeWidths },
        colorbar: { title: "NES", len: 0.5 },
      },
      text: hoverText, hoverinfo: "text",
    };
    const layout = {
      height: hmHeight,
      margin: { l: 120, r: hmMarginR, t: 30, b: 60 },
      xaxis: { tickvals: CONTRASTS.map((_, i) => i), ticktext: labels, tickfont: { size: 12 },
               range: [-0.5, 8.5] },
      yaxis: { tickvals: kinases.map((_, i) => i), ticktext: kinases, tickfont: { size: 11 },
               range: [kinases.length - 0.5, -0.5] },
      annotations: familyAnnotations,
      plot_bgcolor: "white",
      shapes: [
        { type: "line", x0: 2.5, x1: 2.5, y0: -0.5, y1: kinases.length - 0.5,
          line: { color: "black", width: 2 } },
        { type: "line", x0: 5.5, x1: 5.5, y0: -0.5, y1: kinases.length - 0.5,
          line: { color: "black", width: 2 } },
        ...searchShapes,
        ...familyShapes,
      ],
    };
    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true });
  }

  // Scroll to first matching kinase in heatmap (highlight mode)
  if (searchShapes.length > 0 && state.searchMode === "highlight") {
    const firstIdx = kinases.findIndex(k => state.searchMatches.has(k));
    if (firstIdx >= 0) {
      Plotly.relayout(plotDiv, { 'yaxis.range': [firstIdx - 2, firstIdx + 12] });
    }
  }
}

// Heatmap control handlers
document.getElementById("hm-tissue").addEventListener("change", () => renderHeatmap());
document.getElementById("hm-topn").addEventListener("change", () => renderHeatmap());
document.getElementById("hm-sort").addEventListener("change", () => renderHeatmap());
document.getElementById("hm-mode-heatmap").addEventListener("click", () => {
  document.getElementById("hm-mode-heatmap").classList.add("active");
  document.getElementById("hm-mode-bubble").classList.remove("active");
  state.heatmapMode = "heatmap";
  renderHeatmap();
});
document.getElementById("hm-mode-bubble").addEventListener("click", () => {
  document.getElementById("hm-mode-bubble").classList.add("active");
  document.getElementById("hm-mode-heatmap").classList.remove("active");
  state.heatmapMode = "bubble";
  renderHeatmap();
});

// =========================================================================
// Tab 4: Direction Over Time
// =========================================================================
function renderDirectionOverTime() {
  const selVal = document.getElementById("dir-tissue").value;
  const parsed = parseSelection(selVal);

  // Get attributed kinases with tissue and cell-type mapping
  const attrKinases = {}; // kinase → { tissue, celltype }
  DATA.kinaseHypothesis.forEach(r => {
    if (!r.top_celltype_1) return;
    const tissue = subclassToTissue[r.top_celltype_1] || "Other";
    attrKinases[r.kinase] = { tissue, celltype: r.top_celltype_1 };
  });

  // Determine panels: each panel is { label, filterFn }
  let panels;
  if (parsed.level === "all") {
    // One panel per tissue that has attributed kinases
    const activeTissues = DATA.config.tissueOrder.filter(t =>
      Object.values(attrKinases).some(a => a.tissue === t));
    panels = activeTissues.map(t => ({
      label: t,
      filterFn: (kinase) => attrKinases[kinase] && attrKinases[kinase].tissue === t,
    }));
  } else if (parsed.level === "tissue") {
    // One panel per cell type within that tissue (that has kinases)
    const subclasses = tissueToSubclasses[parsed.name] || [];
    const activeCts = subclasses.filter(ct =>
      Object.values(attrKinases).some(a => a.celltype === ct));
    if (activeCts.length <= 1) {
      // Single panel for the tissue
      panels = [{ label: parsed.name, filterFn: (k) => attrKinases[k] && attrKinases[k].tissue === parsed.name }];
    } else {
      panels = activeCts.map(ct => ({
        label: ct,
        filterFn: (k) => attrKinases[k] && attrKinases[k].celltype === ct,
      }));
    }
  } else {
    // Single cell type
    panels = [{ label: parsed.name, filterFn: (k) => attrKinases[k] && attrKinases[k].celltype === parsed.name }];
  }

  if (panels.length === 0) {
    Plotly.purge(document.getElementById("dir-plot"));
    return;
  }

  const ncols = Math.min(panels.length, 3);
  const nrows = Math.ceil(panels.length / ncols);

  const traces = [];
  const subplotDefs = [];

  panels.forEach((panel, idx) => {
    const row = Math.floor(idx / ncols) + 1;
    const col = (idx % ncols) + 1;
    const xAxisId = idx === 0 ? "x" : "x" + (idx + 1);
    const yAxisId = idx === 0 ? "y" : "y" + (idx + 1);

    DATA.config.diseaseGroups.forEach((disease) => {
      const upVals = [];
      const downVals = [];
      const upNames = [];
      const downNames = [];
      const upHasHit = [];
      const downHasHit = [];

      DATA.config.timepoints.forEach(tp => {
        const contrast = disease + "_" + tp;
        const upSet = new Set();
        const downSet = new Set();

        const filterSearch = state.searchMatches.size > 0 && state.searchMode === "filter";
        Object.keys(attrKinases).forEach(kinase => {
          if (!panel.filterFn(kinase)) return;
          if (!passesFamily(kinase)) return;
          if (!passesTrajectory(kinase)) return;
          if (!passesWmb(kinase)) return;
          if (filterSearch && !state.searchMatches.has(kinase)) return;
          const mea = meaLookup[kinase];
          if (!mea || !mea[contrast]) return;
          const { NES, FDR } = mea[contrast];
          if (FDR == null || FDR >= state.fdrThreshold) return;
          if (NES > 0) upSet.add(kinase);
          else downSet.add(kinase);
        });

        upVals.push(upSet.size);
        downVals.push(-downSet.size);
        upNames.push([...upSet].join(", ") || "none");
        downNames.push([...downSet].join(", ") || "none");
        const hasHitUp = state.searchMatches.size > 0 && state.searchMode === "highlight" && [...upSet].some(k => state.searchMatches.has(k));
        const hasHitDown = state.searchMatches.size > 0 && state.searchMode === "highlight" && [...downSet].some(k => state.searchMatches.has(k));
        upHasHit.push(hasHitUp);
        downHasHit.push(hasHitDown);
      });

      const color = DATA.config.diseaseColors[disease];
      const displayName = {"App":"APP","Tau":"Tau","ApTt":"A\u00d7T"}[disease];

      traces.push({
        type: "bar", x: DATA.config.timepoints, y: upVals,
        name: displayName, marker: {
          color: color,
          line: { color: upHasHit.map(h => h ? "#fdd835" : "rgba(0,0,0,0)"),
                  width: upHasHit.map(h => h ? 3 : 0) },
        },
        xaxis: xAxisId, yaxis: yAxisId,
        offsetgroup: disease,
        legendgroup: disease, showlegend: idx === 0,
        hovertext: upNames, hovertemplate: "%{x}: %{y} up<br>%{hovertext}<extra>" + displayName + "</extra>",
      });
      traces.push({
        type: "bar", x: DATA.config.timepoints, y: downVals,
        name: displayName + " (down)", marker: {
          color: color, opacity: 0.55,
          line: { color: downHasHit.map(h => h ? "#fdd835" : "rgba(0,0,0,0)"),
                  width: downHasHit.map(h => h ? 3 : 0) },
        },
        xaxis: xAxisId, yaxis: yAxisId,
        offsetgroup: disease,
        legendgroup: disease, showlegend: false,
        hovertext: downNames, hovertemplate: "%{x}: %{y} down<br>%{hovertext}<extra>" + displayName + "</extra>",
      });
    });

    subplotDefs.push({ row, col, label: panel.label, xAxisId, yAxisId });
  });

  // Build layout with subplots
  const layout = {
    height: nrows * 300 + 120,
    barmode: "group",
    bargroupgap: 0.15,
    legend: { orientation: "h", y: -0.03, font: { size: 11 } },
    margin: { l: 60, r: 20, t: 50, b: 50 },
    title: { text: "Kinase Dysregulation Direction Over Time (disease vs WT)", font: { size: 14 } },
    annotations: [],
  };

  // Spacing constants for subplot grid
  const gapX = 0.10;  // horizontal gap between subplots
  const gapY = 0.14;  // vertical gap between rows
  const padL = 0.07;  // left padding for y-axis labels

  subplotDefs.forEach((sp, idx) => {
    const xKey = idx === 0 ? "xaxis" : "xaxis" + (idx + 1);
    const yKey = idx === 0 ? "yaxis" : "yaxis" + (idx + 1);
    const cellW = (1 - padL) / ncols;
    const cellH = 1 / nrows;
    const domainX = [padL + (sp.col - 1) * cellW + gapX / 2, padL + sp.col * cellW - gapX / 2];
    const domainY = [1 - sp.row * cellH + gapY / 2, 1 - (sp.row - 1) * cellH - gapY / 2];
    layout[xKey] = { domain: domainX, title: "", tickfont: { size: 11 }, tickangle: 0 };
    layout[yKey] = {
      domain: domainY,
      title: sp.col === 1 ? "Count" : "",
      titlefont: { size: 10 },
      showticklabels: sp.col === 1,  // only show y tick labels on leftmost column
      zeroline: true, zerolinewidth: 1, zerolinecolor: "#999",
    };
    // Panel label as annotation above each subplot
    layout.annotations.push({
      text: "<b>" + sp.label + "</b>", showarrow: false,
      x: (domainX[0] + domainX[1]) / 2, y: domainY[1] + 0.025,
      xref: "paper", yref: "paper",
      font: { size: 10 },
    });
  });

  Plotly.newPlot("dir-plot", traces, layout, { responsive: true });
}

document.getElementById("dir-tissue").addEventListener("change", () => renderDirectionOverTime());
document.getElementById("add-tissue").addEventListener("change", () => renderAdditivityScatter());

// =========================================================================
// Tab 5: Additivity Scatter
// =========================================================================
function renderAdditivityScatter() {
  const catColors = {
    "None sig": "#cccccc", "App only": "#c62828", "Tau only": "#1565c0",
    "ApTt emergent": "#6a1b9a", "App+Tau (no ApTt)": "#9467bd",
    "App+ApTt": "#2ca02c", "Tau+ApTt": "#8c564b", "All three": "#e377c2",
  };
  const catOrder = ["None sig", "App only", "Tau only", "App+Tau (no ApTt)",
    "App+ApTt", "Tau+ApTt", "All three", "ApTt emergent"];

  const traces = [];

  // Tissue filter from dropdown
  const addTissueVal = document.getElementById("add-tissue").value;
  const addTissueParsed = parseSelection(addTissueVal);
  const addTissueKinases = addTissueParsed.level === "all" ? null : new Set(getKinasesForSelection(addTissueVal));
  function passesAddTissue(kinase) {
    return addTissueKinases === null || addTissueKinases.has(kinase);
  }

  // Pre-scan all timepoints to find which categories actually have data
  const catsWithData = new Set();
  const filterSearch = state.searchMatches.size > 0 && state.searchMode === "filter";
  DATA.config.timepoints.forEach(tp => {
    const appC = "App_" + tp, tauC = "Tau_" + tp, apttC = "ApTt_" + tp;
    DATA.kinaseActivity.forEach(r => {
      if (r[appC + "_NES"] == null || r[tauC + "_NES"] == null || r[apttC + "_NES"] == null) return;
      if (!passesFamily(r.kinase) || !passesTrajectory(r.kinase) || !passesWmb(r.kinase)) return;
      if (!passesAddTissue(r.kinase)) return;
      if (filterSearch && !state.searchMatches.has(r.kinase)) return;
      const sApp = isSig(r[appC + "_FDR"]), sTau = isSig(r[tauC + "_FDR"]), sApTt = isSig(r[apttC + "_FDR"]);
      let cat = "None sig";
      if (sApp && !sTau && !sApTt) cat = "App only";
      else if (!sApp && sTau && !sApTt) cat = "Tau only";
      else if (!sApp && !sTau && sApTt) cat = "ApTt emergent";
      else if (sApp && sTau && !sApTt) cat = "App+Tau (no ApTt)";
      else if (sApp && !sTau && sApTt) cat = "App+ApTt";
      else if (!sApp && sTau && sApTt) cat = "Tau+ApTt";
      else if (sApp && sTau && sApTt) cat = "All three";
      catsWithData.add(cat);
    });
  });
  // Only show legend entries for categories present in the data
  const activeCats = catOrder.filter(c => catsWithData.has(c));
  const legendShown = new Set();  // track which legendgroups already have a legend entry

  DATA.config.timepoints.forEach((tp, tpIdx) => {
    const appC = "App_" + tp, tauC = "Tau_" + tp, apttC = "ApTt_" + tp;
    const xAxisId = tpIdx === 0 ? "x" : "x" + (tpIdx + 1);
    const yAxisId = tpIdx === 0 ? "y" : "y" + (tpIdx + 1);

    // Group kinases by category
    const groups = {};
    catOrder.forEach(c => { groups[c] = { x: [], y: [], names: [] }; });

    DATA.kinaseActivity.forEach(r => {
      const appNes = r[appC + "_NES"], tauNes = r[tauC + "_NES"], apttNes = r[apttC + "_NES"];
      if (appNes == null || tauNes == null || apttNes == null) return;
      if (!passesFamily(r.kinase)) return;
      if (!passesTrajectory(r.kinase)) return;
      if (!passesWmb(r.kinase)) return;
      if (!passesAddTissue(r.kinase)) return;
      if (filterSearch && !state.searchMatches.has(r.kinase)) return;

      const appFdr = r[appC + "_FDR"], tauFdr = r[tauC + "_FDR"], apttFdr = r[apttC + "_FDR"];
      const sApp = isSig(appFdr);
      const sTau = isSig(tauFdr);
      const sApTt = isSig(apttFdr);

      let cat = "None sig";
      if (sApp && !sTau && !sApTt) cat = "App only";
      else if (!sApp && sTau && !sApTt) cat = "Tau only";
      else if (!sApp && !sTau && sApTt) cat = "ApTt emergent";
      else if (sApp && sTau && !sApTt) cat = "App+Tau (no ApTt)";
      else if (sApp && !sTau && sApTt) cat = "App+ApTt";
      else if (!sApp && sTau && sApTt) cat = "Tau+ApTt";
      else if (sApp && sTau && sApTt) cat = "All three";

      const additive = appNes + tauNes;
      groups[cat].x.push(additive);
      groups[cat].y.push(apttNes);
      groups[cat].names.push(r.kinase);
    });

    activeCats.forEach(cat => {
      const g = groups[cat];
      if (g.x.length === 0) return;
      const isNone = cat === "None sig";
      const showLeg = !legendShown.has(cat);
      if (showLeg) legendShown.add(cat);
      traces.push({
        type: "scatter", mode: "markers",
        x: g.x, y: g.y, text: g.names,
        name: cat, legendgroup: cat, showlegend: showLeg,
        marker: {
          color: catColors[cat],
          size: isNone ? 4 : 8,
          opacity: isNone ? 0.25 : 0.75,
          line: isNone ? {} : { color: "white", width: 0.5 },
        },
        hovertemplate: "%{text}<br>Additive (APP+Tau): %{x:.2f}<br>A\u00d7T observed: %{y:.2f}<extra>" + cat + "</extra>",
        xaxis: xAxisId, yaxis: yAxisId,
      });
    });

    // Search highlight: larger markers with yellow ring for matched kinases (highlight mode)
    if (state.searchMatches.size > 0 && state.searchMode === "highlight") {
      const hx = [], hy = [], hnames = [];
      catOrder.forEach(cat => {
        const g = groups[cat];
        g.names.forEach((name, i) => {
          if (state.searchMatches.has(name)) {
            hx.push(g.x[i]); hy.push(g.y[i]); hnames.push(name);
          }
        });
      });
      if (hx.length > 0) {
        traces.push({
          type: "scatter", mode: "markers+text",
          x: hx, y: hy, text: hnames,
          textposition: "top center", textfont: { size: 9, color: "#333" },
          marker: { color: "rgba(255,235,59,0.6)", size: 14,
                    line: { color: "#f57f17", width: 2 }, symbol: "circle" },
          hovertemplate: "%{text}<br>Additive: %{x:.2f}<br>A\u00d7T: %{y:.2f}<extra>Search match</extra>",
          xaxis: xAxisId, yaxis: yAxisId,
          showlegend: false,
        });
      }
    }

    // Diagonal line
    const allX = Object.values(groups).flatMap(g => g.x);
    const allY = Object.values(groups).flatMap(g => g.y);
    const allVals = [...allX, ...allY].filter(v => isFinite(v));
    if (allVals.length > 0) {
      const lo = Math.min(...allVals) - 0.5;
      const hi = Math.max(...allVals) + 0.5;
      traces.push({
        type: "scatter", mode: "lines",
        x: [lo, hi], y: [lo, hi],
        line: { color: "black", dash: "dash", width: 1 },
        showlegend: false, hoverinfo: "skip",
        xaxis: xAxisId, yaxis: yAxisId,
      });
    }

    // Pearson r
    const rX = Object.values(groups).flatMap(g => g.x);
    const rY = Object.values(groups).flatMap(g => g.y);
    if (rX.length > 2) {
      const n = rX.length;
      const mx = rX.reduce((a, b) => a + b, 0) / n;
      const my = rY.reduce((a, b) => a + b, 0) / n;
      let num = 0, dx = 0, dy = 0;
      for (let i = 0; i < n; i++) {
        num += (rX[i] - mx) * (rY[i] - my);
        dx += (rX[i] - mx) ** 2;
        dy += (rY[i] - my) ** 2;
      }
      const r = num / (Math.sqrt(dx) * Math.sqrt(dy));
      traces.push({
        type: "scatter", mode: "text",
        x: [allVals.length > 0 ? Math.min(...allVals) + 0.3 : 0],
        y: [allVals.length > 0 ? Math.max(...allVals) - 0.3 : 0],
        text: ["r = " + r.toFixed(3)],
        textfont: { size: 12, color: "black" },
        showlegend: false, hoverinfo: "skip",
        xaxis: xAxisId, yaxis: yAxisId,
      });
    }
  });

  const layout = {
    height: 550, width: 1100,
    title: { text: "A\u00d7T Observed vs Additive Prediction (APP + Tau)", font: { size: 14 } },
    legend: { orientation: "h", y: -0.12, font: { size: 10 },
              itemsizing: "constant" },
    margin: { l: 65, r: 20, t: 60, b: 80 },
    annotations: [],
  };

  DATA.config.timepoints.forEach((tp, idx) => {
    const xKey = idx === 0 ? "xaxis" : "xaxis" + (idx + 1);
    const yKey = idx === 0 ? "yaxis" : "yaxis" + (idx + 1);
    const xAnchor = idx === 0 ? "y" : "y" + (idx + 1);
    const yAnchor = idx === 0 ? "x" : "x" + (idx + 1);
    const domainX = [idx / 3 + 0.04, (idx + 1) / 3 - 0.02];
    const domainY = [0.12, 0.88];
    layout[xKey] = {
      domain: domainX, anchor: xAnchor,
      title: { text: "Predicted NES (APP + Tau)", font: { size: 10 } },
      zeroline: true, zerolinecolor: "#ccc",
    };
    layout[yKey] = {
      domain: domainY, anchor: yAnchor,
      title: idx === 0 ? { text: "Observed A\u00d7T NES", font: { size: 10 } } : "",
      zeroline: true, zerolinecolor: "#ccc",
    };
    // Timepoint title
    layout.annotations.push({
      text: "<b>" + tp + "</b>", showarrow: false,
      x: (domainX[0] + domainX[1]) / 2, y: 0.94,
      xref: "paper", yref: "paper", font: { size: 13 },
    });
    // Quadrant labels (above/below diagonal meaning)
    const midX = (domainX[0] + domainX[1]) / 2;
    layout.annotations.push({
      text: "Synergistic<br>(A\u00d7T > predicted)", showarrow: false,
      x: domainX[0] + 0.01, y: 0.86, xref: "paper", yref: "paper",
      font: { size: 8, color: "#6a1b9a" }, align: "left",
    });
    layout.annotations.push({
      text: "Sub-additive<br>(A\u00d7T < predicted)", showarrow: false,
      x: domainX[1] - 0.01, y: 0.14, xref: "paper", yref: "paper",
      font: { size: 8, color: "#999" }, align: "right",
    });
  });

  Plotly.react("add-plot", traces, layout, { responsive: true });
}

// =========================================================================
// Render dispatch
// =========================================================================
const renderers = {
  "kinase-explorer": renderKinaseExplorer,
  "celltype-explorer": renderCelltypeExplorer,
  "heatmap": renderHeatmap,
  "direction": renderDirectionOverTime,
  "additivity": renderAdditivityScatter,
};

function renderActiveTab() {
  const tab = getActiveTab();
  if (renderers[tab]) {
    // Use rAF to ensure the tab panel is visible before Plotly measures dimensions
    requestAnimationFrame(() => {
      renderers[tab]();
      state.stale[tab] = false;
    });
  }
}

// =========================================================================
// Initialization
// =========================================================================
function init() {
  // Populate checkbox dropdown filters
  const families = [...new Set(Object.values(DATA.familyMap))].sort();
  buildCheckboxDropdown("family-filter", families, "familyFilter", "Family");

  const trajOptions = Object.keys(TRAJ_PREDICATES);
  buildCheckboxDropdown("trajectory-filter", trajOptions, "trajectoryFilter", "Significant in");

  // Hierarchical dropdowns: tissue > cell type (shared structure)
  const ctSel = document.getElementById("ct-select");
  buildHierarchyDropdown(ctSel, { includeAll: false });
  // Default to first tissue
  if (ctSel.options.length > 0) ctSel.value = ctSel.options[0].value;

  const hmTissueSel = document.getElementById("hm-tissue");
  buildHierarchyDropdown(hmTissueSel, { includeAll: false });
  if (hmTissueSel.options.length > 0) hmTissueSel.value = hmTissueSel.options[0].value;

  const dirTissueSel = document.getElementById("dir-tissue");
  buildHierarchyDropdown(dirTissueSel, { includeAll: true, allLabel: "All tissues" });

  const addTissueSel = document.getElementById("add-tissue");
  buildHierarchyDropdown(addTissueSel, { includeAll: true, allLabel: "All tissues" });

  // Initial dropdown count update
  updateAllDropdowns();

  // Render default tab
  renderKinaseExplorer();
}

// Glossary toggle
document.getElementById("glossary-btn").addEventListener("click", () => {
  const panel = document.getElementById("glossary-panel");
  panel.classList.toggle("open");
  document.getElementById("glossary-btn").textContent =
    panel.classList.contains("open") ? "\u2716 Close Glossary" : "\u24D8 Glossary & Help";
});

document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build interactive kinase viewer")
    parser.add_argument(
        "--output", default=None,
        help="Output HTML path (default: outputs/reports/attribution_recovery/kinase_viewer.html)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        config.ATTRIBUTION_RECOVERY_OUTPUT_DIR, "kinase_viewer.html")

    print("Loading data...")
    t1, t2, t3, t4, mea = load_data()
    print(f"  T1: {len(t1)} kinases, T2: {len(t2)} pairs, "
          f"T3: {len(t3)} kinases, T4: {len(t4)} profiles")

    print("Resolving kinase families...")
    all_kinases = sorted(set(t1["kinase"].tolist()))
    fam_map, fam_colors = resolve_families(all_kinases)
    print(f"  {len(fam_map)} kinases mapped to families")

    print("Computing clustering orders...")
    cluster_orders = compute_clustering(t1, t3)
    for tissue, order in cluster_orders.items():
        print(f"  {tissue}: {len(order)} kinases")

    print("Building JSON payload...")
    payload = build_payload(t1, t2, t3, t4, mea, fam_map, fam_colors,
                            cluster_orders)
    data_json = json.dumps(payload, separators=(",", ":"))
    print(f"  JSON size: {len(data_json) / 1024:.0f} KB")

    print(f"Writing viewer to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB")
    print("Done.")


if __name__ == "__main__":
    main()
