#!/usr/bin/env python3
"""Generate an interactive HTML viewer for significant Incytr pathways.

Reads the backbone-level permutation results (significant backbones that pass
both null tests) and backbone recurrence data, builds a compact JSON payload,
and embeds everything into a single self-contained HTML file.

The viewer provides:
  - Overview heatmap: receiver × contrast with count/direction toggle
  - Sender Matrix: sender → receiver cell-cell signaling heatmap
  - Pathway graph: Cytoscape.js radial DAG (Receptor → EM → Target)
  - Backbone table: filterable/sortable table with direction coloring
  - Cross-contrast view: selected backbone across all 9 contrasts
  - Temporal trajectory: disease progression per genotype
  - Additivity scatter: App+Tau predicted vs ApTt observed
  - Kinase context: attributed kinases for selected contrast+receiver

Usage:
    python code/integration/build_pathway_viewer.py
    python code/integration/build_pathway_viewer.py --output path.html
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import config_integration as icfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTRASTS = list(icfg.FACTORIAL_CONTRASTS.keys())

DISEASE_COLORS = {
    "App": "#c62828",
    "Tau": "#1565c0",
    "ApTt": "#6a1b9a",
}

TISSUE_CATEGORIES = {
    "Excitatory": ["L2/3 IT", "L4 IT", "L5 ET", "L5 IT", "L5/6 NP",
                   "L6 CT", "L6 IT", "L6b"],
    "Inhibitory": ["Chandelier", "Lamp5", "Lamp5 Lhx6", "Pvalb", "Sncg",
                   "Sst", "Sst Chodl", "Vip"],
    "Non-neuronal": ["Astrocyte", "Endothelial", "Microglia-PVM", "OPC",
                     "Oligodendrocyte", "VLMC"],
}

RECEIVER_TO_TISSUE = {}
for tissue, receivers in TISSUE_CATEGORIES.items():
    for r in receivers:
        RECEIVER_TO_TISSUE[r] = tissue


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load significant backbones, recurrence data, and kinase attribution."""
    agg_dir = os.path.join(icfg.FACTORIAL_ALL_PAIRS_DIR, "aggregation")

    sig_path = os.path.join(agg_dir, "backbone_significant_both_nulls.csv")
    bb_path = os.path.join(agg_dir, "backbone_recurrence_by_contrast.csv")

    sig = pd.read_csv(sig_path)
    bb = pd.read_csv(bb_path,
                     usecols=["contrast", "receiver", "Receptor", "EM",
                              "Target", "n_senders", "n_senders_significant",
                              "mean_tpds", "max_abs_tpds", "sender_list"])

    merged = sig.merge(
        bb, on=["contrast", "receiver", "Receptor", "EM", "Target"],
        how="left")

    # Drop per-contrast constants (pi0)
    pi0_by_contrast = {}
    for c in CONTRASTS:
        sub = merged[merged["contrast"] == c]
        if len(sub) > 0:
            pi0_by_contrast[c] = {
                "pi0_null1": round(float(sub["pi0_null1"].iloc[0]), 4),
                "pi0_null2": round(float(sub["pi0_null2"].iloc[0]), 4),
            }
    merged = merged.drop(columns=["pi0_null1", "pi0_null2"])

    # Load kinase attribution
    attr = pd.read_csv(icfg.UNIFIED_ATTRIBUTION_CSV,
                       usecols=["kinase", "gene_symbol", "contrast",
                                "cell_type", "NES", "FDR",
                                "combined_score", "combined_confidence"])

    print(f"  Loaded {len(merged):,} significant backbones")
    print(f"  Loaded {len(attr):,} kinase attribution rows")
    return merged, pi0_by_contrast, attr


# ---------------------------------------------------------------------------
# Payload builder
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


def build_payload(merged, pi0_by_contrast, attr):
    """Build compact JSON payload for embedding."""

    # Sender bitmask encoding
    all_senders = set()
    for sl in merged["sender_list"].dropna():
        all_senders.update(sl.split(","))
    sender_order = sorted(all_senders)
    sender_idx = {s: i for i, s in enumerate(sender_order)}

    sender_masks = []
    for sl in merged["sender_list"].fillna(""):
        mask = 0
        if sl:
            for s in sl.split(","):
                if s in sender_idx:
                    mask |= (1 << sender_idx[s])
        sender_masks.append(mask)

    # Columnar format for main data
    cols_to_embed = [
        "contrast", "receiver", "Receptor", "EM", "Target",
        "observed_score", "n_edges",
        "pval_null1", "pval_null2", "qval_null1", "qval_null2",
        "n_senders", "mean_tpds", "max_abs_tpds",
    ]
    columnar = {}
    for col in cols_to_embed:
        columnar[col] = merged[col].tolist()
    columnar["sender_mask"] = sender_masks

    # Overview summary with direction stats
    overview = {}
    for (c, r), g in merged.groupby(["contrast", "receiver"]):
        key = f"{c}|{r}"
        tpds = g["mean_tpds"].values
        overview[key] = {
            "n": len(g),
            "mean_score": round(float(g["observed_score"].mean()), 4),
            "median_score": round(float(g["observed_score"].median()), 4),
            "mean_senders": round(float(g["n_senders"].mean()), 1),
            "n_up": int((tpds > 0).sum()),
            "n_down": int((tpds < 0).sum()),
            "mean_tpds": round(float(tpds.mean()), 4) if len(tpds) else 0,
        }

    # Cross-contrast index with row indices
    cross_dict = {}
    for idx, row in merged.iterrows():
        key = (row["receiver"], row["Receptor"], row["EM"], row["Target"])
        if key not in cross_dict:
            cross_dict[key] = {}
        cross_dict[key][row["contrast"]] = int(idx)
    cross_list = []
    for (recv, rec, em, tgt), ci in cross_dict.items():
        cross_list.append({
            "r": recv, "R": rec, "E": em, "T": tgt,
            "ci": ci,
        })

    # Kinase attribution indexed by contrast|cell_type
    kinase_attr = {}
    for (c, ct), g in attr.groupby(["contrast", "cell_type"]):
        key = f"{c}|{ct}"
        entries = []
        for _, row in g.iterrows():
            entries.append({
                "k": row["kinase"],
                "g": row["gene_symbol"] if pd.notna(row["gene_symbol"]) else "",
                "n": round(float(row["NES"]), 3) if pd.notna(row["NES"]) else 0,
                "f": round(float(row["FDR"]), 4) if pd.notna(row["FDR"]) else 1,
                "s": round(float(row["combined_score"]), 3) if pd.notna(row["combined_score"]) else 0,
                "c": row["combined_confidence"] if pd.notna(row["combined_confidence"]) else "low",
            })
        kinase_attr[key] = sorted(entries, key=lambda x: -abs(x["s"]))

    payload = {
        "backbones": columnar,
        "nRows": len(merged),
        "overview": overview,
        "crossContrast": cross_list,
        "pi0": pi0_by_contrast,
        "kinaseAttr": kinase_attr,
        "config": {
            "contrasts": CONTRASTS,
            "diseaseColors": DISEASE_COLORS,
            "tissueCategories": TISSUE_CATEGORIES,
            "receiverToTissue": RECEIVER_TO_TISSUE,
            "senderOrder": sender_order,
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
<title>Pathway Viewer &mdash; Incytr Significant Backbones</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<style>
:root {
  --app-red: #c62828; --tau-blue: #1565c0; --aptt-purple: #6a1b9a;
  --up-red: #c62828; --down-blue: #1565c0;
  --bg: #fafafa; --card-bg: #ffffff; --border: #e0e0e0;
  --text: #212121; --text-muted: #757575;
  --selected-border: #1976d2;
  --receptor-color: #43a047; --em-color: #fb8c00; --target-color: #5c6bc0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--text); min-width: 1100px; }
header { background: #263238; color: white; padding: 12px 24px; display: flex;
         align-items: center; gap: 24px; flex-wrap: wrap; }
header h1 { font-size: 18px; font-weight: 600; white-space: nowrap; }
header .subtitle { font-size: 12px; color: #90a4ae; }
.controls-bar { background: #37474f; color: #cfd8dc; padding: 8px 24px;
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.controls-bar label { font-size: 12px; display: flex; align-items: center; gap: 5px; }
.controls-bar select, .controls-bar input[type=range] { font-size: 12px; }
.controls-bar input[type=range] { width: 100px; }
.controls-bar .val { font-weight: 700; min-width: 32px; color: white; }
.controls-bar input[type=text] { font-size: 12px; padding: 3px 8px; border: 1px solid #546e7a;
  border-radius: 3px; background: #455a64; color: white; width: 140px; }
.controls-bar input[type=text]::placeholder { color: #90a4ae; }
nav#tab-bar { display: flex; background: #455a64; overflow-x: auto; }
nav#tab-bar button { background: none; border: none; color: #b0bec5; padding: 10px 16px;
  font-size: 12px; font-weight: 500; cursor: pointer; white-space: nowrap;
  border-bottom: 3px solid transparent; transition: all 0.15s; }
nav#tab-bar button:hover { color: white; background: rgba(255,255,255,0.05); }
nav#tab-bar button.active { color: white; border-bottom-color: #42a5f5; }
main { padding: 16px 24px; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
/* Overview */
#overview-plot { width: 100%; height: 620px; }
.overview-toggle { margin-bottom: 8px; }
.overview-toggle button { padding: 4px 12px; font-size: 12px; border: 1px solid var(--border);
  background: var(--card-bg); cursor: pointer; border-radius: 3px; margin-right: 4px; }
.overview-toggle button.active { background: #1976d2; color: white; border-color: #1976d2; }
/* Sender matrix */
#sender-plot { width: 100%; height: 620px; }
/* Graph */
#graph-trajectory-container { min-height: 400px; }
#graph-container { display: flex; gap: 16px; height: calc(100vh - 200px); min-height: 500px; }
#cy { flex: 1; border: 1px solid var(--border); border-radius: 4px; background: white; }
#graph-sidebar { width: 320px; flex-shrink: 0; overflow-y: auto; }
#graph-info { background: var(--card-bg); border: 1px solid var(--border); border-radius: 4px;
  padding: 12px; font-size: 12px; }
#graph-info h3 { font-size: 14px; margin-bottom: 8px; }
#graph-info .stat { margin: 4px 0; }
#graph-info .stat .label { color: var(--text-muted); }
#graph-legend { margin-top: 12px; background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 12px; font-size: 12px; }
#graph-legend h4 { margin-bottom: 6px; }
.legend-item { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; }
#graph-status { padding: 8px 0; font-size: 12px; color: var(--text-muted); }
#node-detail { margin-top: 12px; background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 12px; font-size: 12px; display: none; }
#node-detail h4 { margin-bottom: 6px; }
#node-detail table { width: 100%; border-collapse: collapse; }
#node-detail td { padding: 2px 4px; border-bottom: 1px solid #f0f0f0; }
#node-detail td:first-child { color: var(--text-muted); width: 40%; }
/* Table */
.data-table-wrap { max-height: 70vh; overflow: auto; border: 1px solid var(--border);
  border-radius: 4px; background: var(--card-bg); }
table.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
table.data-table th { position: sticky; top: 0; background: #eceff1; padding: 6px 8px;
  text-align: left; font-weight: 600; cursor: pointer; user-select: none;
  border-bottom: 2px solid var(--border); white-space: nowrap; z-index: 2; }
table.data-table th:hover { background: #cfd8dc; }
table.data-table th .sort-arrow { font-size: 10px; margin-left: 3px; color: #999; }
table.data-table td { padding: 4px 8px; border-bottom: 1px solid #f5f5f5; white-space: nowrap; }
table.data-table tr:hover { background: #e3f2fd; cursor: pointer; }
table.data-table tr.highlighted { background: #fff3e0; }
.table-status { font-size: 12px; color: var(--text-muted); padding: 8px 0; }
.tpds-up { color: var(--up-red); font-weight: 600; }
.tpds-down { color: var(--down-blue); font-weight: 600; }
/* Cross-contrast */
#cross-contrast-container { display: flex; gap: 16px; }
#cross-plot { flex: 1; height: 400px; }
#cross-info { width: 320px; background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 12px; font-size: 12px; overflow-y: auto; max-height: 500px; }
/* Temporal */
#temporal-controls { margin-bottom: 8px; display: flex; gap: 16px; align-items: center; }
#temporal-controls select { font-size: 12px; }
#temporal-main { width: 100%; height: 350px; }
#temporal-grid { width: 100%; height: 700px; margin-top: 16px; }
/* Additivity */
#additivity-plot { width: 100%; height: 500px; }
#additivity-stats { font-size: 12px; color: var(--text-muted); padding: 8px 0; }
/* Kinase context */
#kinase-ctx-container { display: flex; gap: 16px; }
#kinase-ctx-plot { flex: 1; height: 500px; }
#kinase-ctx-info { width: 320px; background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 12px; font-size: 12px; }
/* Badge */
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px;
  font-weight: 600; }
.badge-exc { background: #e8f5e9; color: #2e7d32; }
.badge-inh { background: #e3f2fd; color: #1565c0; }
.badge-nn { background: #fce4ec; color: #c62828; }
.badge-high { background: #e8f5e9; color: #2e7d32; }
.badge-mod { background: #fff8e1; color: #f57f17; }
.badge-low { background: #f5f5f5; color: #757575; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Pathway Viewer</h1>
    <div class="subtitle">Incytr significant backbones &mdash; dual-null permutation (q &lt; 0.25)</div>
  </div>
  <div class="subtitle" id="header-stats"></div>
</header>

<div class="controls-bar">
  <label>Contrast: <select id="sel-contrast"></select></label>
  <label>Receiver: <select id="sel-receiver"></select></label>
  <label>Sender: <select id="sel-sender"></select></label>
  <label>Direction: <select id="sel-direction">
    <option value="ALL">All</option>
    <option value="up">Up (TPDS &gt; 0)</option>
    <option value="down">Down (TPDS &lt; 0)</option>
  </select></label>
  <label>Min score:
    <input type="range" id="sl-score" min="0" max="3" step="0.05" value="0">
    <span class="val" id="val-score">0</span>
  </label>
  <label>Max q:
    <input type="range" id="sl-qval" min="0.01" max="0.25" step="0.01" value="0.25">
    <span class="val" id="val-qval">0.25</span>
  </label>
  <label>Min senders:
    <input type="range" id="sl-senders" min="1" max="21" step="1" value="1">
    <span class="val" id="val-senders">1</span>
  </label>
  <label>Search:
    <input type="text" id="gene-search" placeholder="Gene name...">
  </label>
</div>

<nav id="tab-bar">
  <button class="active" data-tab="overview">Overview</button>
  <button data-tab="senders">Sender Matrix</button>
  <button data-tab="graph">Pathway Graph</button>
  <button data-tab="table">Backbone Table</button>
  <button data-tab="cross">Cross-Contrast</button>
  <button data-tab="temporal">Temporal</button>
  <button data-tab="additivity">Additivity</button>
  <button data-tab="kinase">Kinase Context</button>
</nav>

<main>
  <div id="tab-overview" class="tab-panel active">
    <div style="background:#f5f5f0;border:1px solid #ddd;border-radius:4px;padding:10px 14px;margin-bottom:8px;font-size:12px;line-height:1.5;color:#444;">
      <strong>How to read this heatmap:</strong> Each cell shows the number of significant <em>backbones</em> for a receiver cell type (row) under a disease contrast (column).
      A <strong>backbone</strong> is a Receptor &rarr; Effector Molecule &rarr; Target signaling path within a receiver cell type, supported by at least one sender.
      Backbones are significant if they pass both enrichment (Null 1) and within-receiver wiring (Null 2) permutation tests at Storey q &lt; 0.25.
      <strong>Count</strong> view shows total significant backbones (log-scaled); <strong>Direction</strong> view shows the net balance of upregulated (red) vs downregulated (blue) backbones based on TPDS sign.
      Click any cell to explore its pathways in the Graph tab.
    </div>
    <div class="overview-toggle">
      <button class="active" onclick="setOverviewMode('count')">Count</button>
      <button onclick="setOverviewMode('direction')">Direction</button>
      <span style="margin-left:16px;color:#888;">Sort:</span>
      <select id="overview-sort" onchange="setOverviewSort(this.value)" style="margin-left:4px;padding:2px 6px;font-size:12px;">
        <option value="tissue" selected>By Tissue</option>
        <option value="cluster">By Cluster</option>
        <option value="total">By Total Count</option>
      </select>
    </div>
    <div id="overview-plot"></div>
  </div>

  <div id="tab-senders" class="tab-panel">
    <div style="background:#f5f5f0;border:1px solid #ddd;border-radius:4px;padding:10px 14px;margin-bottom:8px;font-size:12px;line-height:1.5;color:#444;">
      <strong>Sender &rarr; Receiver Matrix:</strong> Each cell shows how many significant backbones in a receiver (column) are supported by a given sender cell type (row).
      A backbone is "supported" by a sender if that sender expresses the ligand for the backbone's receptor.
      <strong>Count</strong> view shows backbone counts (log-scaled); <strong>Mean TPDS</strong> shows the average direction of dysregulation for those backbones (red = upregulated, blue = downregulated).
      Filtered by the current contrast and other global filters. Click any cell to filter the table to that sender&ndash;receiver pair.
    </div>
    <div class="overview-toggle">
      <button class="active" onclick="setSenderMode('count')">Count</button>
      <button onclick="setSenderMode('tpds')">Mean TPDS</button>
      <span style="margin-left:16px;color:#888;">Sort:</span>
      <select id="sender-sort" onchange="setSenderSort(this.value)" style="margin-left:4px;padding:2px 6px;font-size:12px;">
        <option value="tissue" selected>By Tissue</option>
        <option value="cluster">By Cluster</option>
        <option value="total">By Total</option>
      </select>
    </div>
    <div id="sender-plot"></div>
  </div>

  <div id="tab-graph" class="tab-panel">
    <div style="background:#f5f5f0;border:1px solid #ddd;border-radius:4px;padding:10px 14px;margin-bottom:8px;font-size:12px;line-height:1.5;color:#444;">
      <strong>Effector Molecule Trajectories:</strong> Each row is an <em>effector molecule</em> (EM) &mdash;
      an intracellular signaling gene that connects receptor inputs to downstream targets.
      Columns show the three timepoints (2, 4, 6 months). Color encodes direction:
      <span style="color:#c62828;font-weight:600">red</span> = upregulated,
      <span style="color:#1565c0;font-weight:600">blue</span> = downregulated.
      Intensity reflects the number of significant backbones through that EM.
      Select a genotype to compare temporal signaling patterns. Click an EM row to see its receptor inputs and target outputs.
    </div>
    <div class="overview-toggle">
      <span style="color:#888;">View:</span>
      <select id="graph-view-mode" onchange="setGraphViewMode(this.value)">
        <option value="trajectory" selected>EM Trajectories</option>
        <option value="network">Network Graph</option>
      </select>
      <span id="graph-geno-controls">
        <span style="margin-left:12px;color:#888;">Genotype:</span>
        <select id="graph-genotype" onchange="setGraphGenotype(this.value)">
          <option value="App" selected>App</option>
          <option value="Tau">Tau</option>
          <option value="ApTt">ApTt</option>
          <option value="all">All (side by side)</option>
        </select>
        <span style="margin-left:12px;color:#888;">Metric:</span>
        <select id="graph-metric" onchange="setGraphMetric(this.value)">
          <option value="direction" selected>Direction (net TPDS)</option>
          <option value="count">Backbone count</option>
          <option value="score">Mean score</option>
        </select>
        <span style="margin-left:12px;color:#888;">Top EMs:</span>
        <select id="graph-top-n" onchange="setGraphTopN(+this.value)">
          <option value="20">20</option>
          <option value="30" selected>30</option>
          <option value="50">50</option>
          <option value="100">100</option>
        </select>
      </span>
      <span id="graph-net-controls" style="display:none;">
        <span style="margin-left:12px;color:#888;">Layout:</span>
        <select id="graph-layout" onchange="setGraphLayout(this.value)">
          <option value="concentric" selected>Concentric (by role)</option>
          <option value="flow">Layered Flow (L&rarr;R)</option>
          <option value="force">Force-directed</option>
        </select>
        <span style="margin-left:12px;color:#888;">Min degree:</span>
        <select id="graph-min-degree" onchange="setGraphMinDegree(+this.value)">
          <option value="1" selected>1 (all)</option>
          <option value="2">2</option>
          <option value="5">5</option>
          <option value="10">10</option>
          <option value="20">20</option>
          <option value="50">50</option>
        </select>
        <button id="graph-reset-focus" style="margin-left:12px;display:none;" onclick="resetGraphFocus()">&#x2190; Back to full graph</button>
      </span>
    </div>

    <!-- EM Trajectory view -->
    <div id="graph-trajectory-container">
      <div id="em-trajectory-plot" style="width:100%;"></div>
      <div id="em-detail-panel" style="display:none; margin-top:12px; background:var(--card-bg); border:1px solid var(--border); border-radius:4px; padding:16px;">
        <h4 id="em-detail-title" style="margin-bottom:8px;"></h4>
        <div style="display:flex; gap:16px;">
          <div id="em-detail-receptors" style="flex:1;"></div>
          <div id="em-detail-targets" style="flex:1;"></div>
        </div>
      </div>
    </div>

    <!-- Network graph view (hidden by default) -->
    <div id="graph-network-container" style="display:none;">
      <div id="graph-container">
        <div id="cy"></div>
        <div id="graph-sidebar">
          <div id="graph-status">Select a contrast and receiver to view the pathway graph.</div>
          <div id="graph-info" style="display:none;">
            <h3>Subgraph Summary</h3>
            <div class="stat"><span class="label">Backbones:</span> <span id="gi-backbones"></span></div>
            <div class="stat"><span class="label">Receptors:</span> <span id="gi-receptors"></span></div>
            <div class="stat"><span class="label">EMs:</span> <span id="gi-ems"></span></div>
            <div class="stat"><span class="label">Targets:</span> <span id="gi-targets"></span></div>
            <div class="stat"><span class="label">Median score:</span> <span id="gi-medscore"></span></div>
            <div class="stat"><span class="label">Direction:</span> <span id="gi-direction"></span></div>
          </div>
          <div id="graph-legend">
            <h4>Legend</h4>
            <div class="legend-item"><div class="legend-dot" style="background:var(--receptor-color)"></div> Receptor</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--em-color)"></div> Effector Molecule (EM)</div>
            <div class="legend-item"><div class="legend-dot" style="background:var(--target-color)"></div> Target</div>
            <div style="margin-top:6px; border-top:1px solid #eee; padding-top:6px;">
              <div class="legend-item"><div class="legend-dot" style="background:var(--up-red)"></div> Upregulated (TPDS &gt; 0)</div>
              <div class="legend-item"><div class="legend-dot" style="background:var(--down-blue)"></div> Downregulated (TPDS &lt; 0)</div>
            </div>
          </div>
          <div id="node-detail">
            <h4 id="nd-title"></h4>
            <table id="nd-table"></table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div id="tab-table" class="tab-panel">
    <div class="table-status" id="table-status"></div>
    <div class="data-table-wrap">
      <table class="data-table" id="backbone-table">
        <thead><tr>
          <th data-col="contrast">Contrast</th>
          <th data-col="receiver">Receiver</th>
          <th data-col="Receptor">Receptor</th>
          <th data-col="EM">EM</th>
          <th data-col="Target">Target</th>
          <th data-col="observed_score">Score</th>
          <th data-col="n_edges">Edges</th>
          <th data-col="qval_null1">q(Null1)</th>
          <th data-col="qval_null2">q(Null2)</th>
          <th data-col="n_senders">Senders</th>
          <th data-col="mean_tpds">TPDS</th>
        </tr></thead>
        <tbody id="table-body"></tbody>
      </table>
    </div>
  </div>

  <div id="tab-cross" class="tab-panel">
    <div id="cross-prompt" style="padding:24px; color:var(--text-muted);">
      Click a backbone row in the table or a node in the graph to see its cross-contrast profile.
    </div>
    <div id="cross-contrast-container" style="display:none;">
      <div id="cross-plot"></div>
      <div id="cross-info"></div>
    </div>
  </div>

  <div id="tab-temporal" class="tab-panel">
    <div id="temporal-controls">
      <label>Metric: <select id="sel-temporal-metric">
        <option value="count">Backbone count</option>
        <option value="mean_score">Mean score</option>
        <option value="mean_tpds">Mean |TPDS|</option>
        <option value="pct_up">% upregulated</option>
      </select></label>
    </div>
    <div id="temporal-main"></div>
    <div id="temporal-grid"></div>
  </div>

  <div id="tab-additivity" class="tab-panel">
    <div id="additivity-stats"></div>
    <div id="additivity-plot"></div>
  </div>

  <div id="tab-kinase" class="tab-panel">
    <div id="kinase-ctx-container">
      <div id="kinase-ctx-plot"></div>
      <div id="kinase-ctx-info">
        <h4>Kinase Context</h4>
        <p style="color:var(--text-muted); font-size:12px;">
          Select a contrast and receiver to see attributed kinases.<br>
          These are kinases with significant MEA enrichment attributed to the selected cell type.
        </p>
      </div>
    </div>
  </div>
</main>

<script>
// ============================================================
// DATA
// ============================================================
const DATA = __DATA_PLACEHOLDER__;

// ============================================================
// STATE
// ============================================================
const state = {
  contrast: 'ALL', receiver: 'ALL', sender: 'ALL', direction: 'ALL',
  minScore: 0, maxQval: 0.25, minSenders: 1, search: '',
  sortCol: 'observed_score', sortAsc: false,
  selectedBackbone: null,
  overviewMode: 'count', overviewSort: 'tissue', senderMode: 'count', senderSort: 'tissue',
  graphViewMode: 'trajectory', graphGenotype: 'App', graphMetric: 'direction', graphTopN: 30,
  graphLayout: 'concentric', graphMinDegree: 1, graphFocusNode: null,
  temporalMetric: 'count',
  tabDirty: {overview:true, senders:true, graph:true, table:true, cross:true, temporal:true, additivity:true, kinase:true},
  cyInstance: null,
  backboneIndex: null,  // Map: "recv|Rec|EM|Tgt" -> {contrast: rowIdx}
};

// ============================================================
// HELPERS
// ============================================================
function getFilteredIndices() {
  const d = DATA.backbones;
  const n = DATA.nRows;
  const indices = [];
  const searchLower = state.search.toLowerCase();
  const senderBit = state.sender !== 'ALL'
    ? (1 << DATA.config.senderOrder.indexOf(state.sender)) : 0;
  for (let i = 0; i < n; i++) {
    if (state.contrast !== 'ALL' && d.contrast[i] !== state.contrast) continue;
    if (state.receiver !== 'ALL' && d.receiver[i] !== state.receiver) continue;
    if (senderBit && !(d.sender_mask[i] & senderBit)) continue;
    if (state.direction === 'up' && d.mean_tpds[i] <= 0) continue;
    if (state.direction === 'down' && d.mean_tpds[i] >= 0) continue;
    if (d.observed_score[i] < state.minScore) continue;
    if (d.qval_null1[i] > state.maxQval || d.qval_null2[i] > state.maxQval) continue;
    if (d.n_senders[i] < state.minSenders) continue;
    if (searchLower) {
      const haystack = (d.Receptor[i]+' '+d.EM[i]+' '+d.Target[i]+' '+d.receiver[i]).toLowerCase();
      if (!haystack.includes(searchLower)) continue;
    }
    indices.push(i);
  }
  return indices;
}

function diseaseColor(c) {
  if (c.startsWith('ApTt')) return DATA.config.diseaseColors.ApTt;
  if (c.startsWith('App')) return DATA.config.diseaseColors.App;
  if (c.startsWith('Tau')) return DATA.config.diseaseColors.Tau;
  return '#666';
}

function tpdsColor(v) { return v > 0 ? 'var(--up-red)' : v < 0 ? 'var(--down-blue)' : '#999'; }
function tpdsCls(v) { return v > 0 ? 'tpds-up' : v < 0 ? 'tpds-down' : ''; }

function tissueBadge(r) {
  const t = DATA.config.receiverToTissue[r] || 'Other';
  const cls = t==='Excitatory'?'badge-exc':t==='Inhibitory'?'badge-inh':'badge-nn';
  return `<span class="badge ${cls}">${t.slice(0,3)}</span>`;
}

function fmt(v, d) { return v == null ? '\u2014' : Number(v).toFixed(d); }

function parseContrast(c) {
  const m = c.match(/^(App|Tau|ApTt)_(\d+mo)$/);
  return m ? {geno: m[1], tp: m[2]} : {geno: c, tp: ''};
}

function decodeSenders(mask) {
  const senders = [];
  for (let i = 0; i < DATA.config.senderOrder.length; i++) {
    if (mask & (1 << i)) senders.push(DATA.config.senderOrder[i]);
  }
  return senders;
}

// ============================================================
// OVERVIEW TAB
// ============================================================
function setOverviewMode(mode) {
  state.overviewMode = mode;
  document.querySelectorAll('#tab-overview .overview-toggle button').forEach(b =>
    b.classList.toggle('active', b.textContent.toLowerCase().includes(mode)));
  renderOverview();
}

// Greedy nearest-neighbor leaf ordering on Euclidean distance of profiles.
function clusterOrder(items, profileFn) {
  if (items.length <= 1) return items;
  const n = items.length;
  const profiles = items.map(profileFn);
  const dist = (a, b) => {
    let s = 0; for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; }
    return Math.sqrt(s);
  };
  let cur = 0, maxSum = -1;
  for (let i = 0; i < n; i++) {
    const s = profiles[i].reduce((a,b) => a + b, 0);
    if (s > maxSum) { maxSum = s; cur = i; }
  }
  const used = new Set([cur]);
  const order = [cur];
  while (order.length < n) {
    let best = -1, bestD = Infinity;
    for (let j = 0; j < n; j++) {
      if (used.has(j)) continue;
      const d = dist(profiles[cur], profiles[j]);
      if (d < bestD) { bestD = d; best = j; }
    }
    used.add(best); order.push(best); cur = best;
  }
  return order.map(i => items[i]);
}

function setOverviewSort(mode) {
  state.overviewSort = mode;
  renderOverview();
}

function renderOverview() {
  const cfg = DATA.config;
  const categories = ['Excitatory','Inhibitory','Non-neuronal'];
  const sortMode = state.overviewSort || 'tissue';

  const isDir = state.overviewMode === 'direction';

  // Profile and total functions adapt to current view mode
  const recvProfile = (r) => cfg.contrasts.map(c => {
    const o = DATA.overview[c + '|' + r];
    if (!o) return 0;
    if (isDir) {
      const raw = o.n_up - o.n_down;
      return Math.sign(raw) * Math.log10(1 + Math.abs(raw));
    }
    return Math.log10(1 + o.n);
  });
  const recvTotal = (r) => {
    let s = 0;
    for (const c of cfg.contrasts) {
      const o = DATA.overview[c + '|' + r];
      if (!o) continue;
      s += isDir ? (o.n_up - o.n_down) : o.n;
    }
    return s;
  };

  // All 22 receivers flat
  const allReceivers = [];
  for (const cat of categories)
    for (const r of cfg.tissueCategories[cat] || []) allReceivers.push(r);

  let receivers, groupBounds;

  if (sortMode === 'tissue') {
    // Grouped by tissue category, alphabetical within
    receivers = [];
    groupBounds = [];
    for (const cat of categories) {
      const start = receivers.length;
      const members = (cfg.tissueCategories[cat] || []).slice().sort();
      for (const r of members) receivers.push(r);
      groupBounds.push([start, receivers.length - 1]);
    }
  } else if (sortMode === 'cluster') {
    // Pure clustering across all receivers, no group separators
    receivers = clusterOrder(allReceivers, recvProfile);
    groupBounds = null;
  } else {
    // By total descending: count mode = highest count first; direction mode = most extreme net first
    receivers = allReceivers.slice().sort((a, b) =>
      isDir ? (Math.abs(recvTotal(b)) - Math.abs(recvTotal(a))) : (recvTotal(b) - recvTotal(a)));
    groupBounds = null;
  }

  const z = [], hoverText = [], customdata = [];
  // Build y-axis labels: prefix with tissue abbreviation in tissue mode
  const yLabels = [];
  const catAbbrev = {'Excitatory':'Exc','Inhibitory':'Inh','Non-neuronal':'NN'};
  for (const r of receivers) {
    if (sortMode === 'tissue') {
      const tissue = cfg.receiverToTissue[r] || '';
      yLabels.push(catAbbrev[tissue] + ' | ' + r);
    } else {
      yLabels.push(r);
    }
    const row = [], trow = [], crow = [];
    for (const c of cfg.contrasts) {
      const o = DATA.overview[c + '|' + r];
      if (!o || o.n === 0) { row.push(0); trow.push(`${r} x ${c}<br>0 backbones`); crow.push({contrast:c,receiver:r,n:0}); continue; }
      const raw = isDir ? (o.n_up - o.n_down) : o.n;
      const val = isDir ? Math.sign(raw) * Math.log10(1 + Math.abs(raw)) : Math.log10(1 + raw);
      row.push(val);
      trow.push(`${r} x ${c}<br>${o.n} backbones (${o.n_up} up, ${o.n_down} down)<br>mean TPDS: ${o.mean_tpds.toFixed(4)}`);
      crow.push({contrast:c,receiver:r,n:o.n});
    }
    z.push(row); hoverText.push(trow); customdata.push(crow);
  }
  // Custom tick values for log-scaled colorbar
  const countTicks = [1, 10, 100, 1000, 3000];
  const logTicks = countTicks.map(v => Math.log10(1 + v));
  const tickLabels = countTicks.map(v => v >= 1000 ? (v/1000)+'k' : String(v));
  const trace = {
    z: z, x: cfg.contrasts, y: yLabels, type: 'heatmap',
    colorscale: isDir ? [[0,'#1565c0'],[0.5,'#ffffff'],[1,'#c62828']] : 'YlOrRd',
    zmid: isDir ? 0 : undefined,
    text: hoverText, hoverinfo: 'text', customdata: customdata,
    colorbar: {
      title: isDir ? 'Net up' : 'Count',
      thickness: 15,
      tickvals: isDir ? undefined : logTicks,
      ticktext: isDir ? undefined : tickLabels,
    },
  };
  // Horizontal separator lines between tissue categories (tissue mode only)
  const shapes = [];
  if (groupBounds) {
    for (let gi = 0; gi < groupBounds.length - 1; gi++) {
      const yPos = groupBounds[gi][1] + 0.5;
      shapes.push({
        type: 'line', xref: 'paper', x0: 0, x1: 1, y0: yPos, y1: yPos,
        line: {color: '#333', width: 2},
      });
    }
  }
  const layout = {
    title: isDir ? 'Direction: net upregulated (red) vs downregulated (blue)' : 'Significant Backbone Count (log scale)',
    xaxis: {title:'Contrast', tickangle:-45},
    yaxis: {title:'', automargin:true, dtick:1, tickfont:{size:11}},
    margin: {l:140,r:40,t:50,b:80}, height: 620,
    shapes: shapes,
  };
  Plotly.newPlot('overview-plot', [trace], layout, {responsive:true});
  document.getElementById('overview-plot').on('plotly_click', function(ev) {
    if (!ev.points.length) return;
    const cd = ev.points[0].customdata;
    if (cd && cd.n > 0) {
      document.getElementById('sel-contrast').value = cd.contrast;
      document.getElementById('sel-receiver').value = cd.receiver;
      state.contrast = cd.contrast; state.receiver = cd.receiver;
      markAllDirty(); switchTab('graph');
    }
  });
}

// ============================================================
// SENDER MATRIX TAB
// ============================================================
function setSenderMode(mode) {
  state.senderMode = mode;
  document.querySelectorAll('#tab-senders .overview-toggle button').forEach(b =>
    b.classList.toggle('active', b.textContent.toLowerCase().includes(mode)));
  renderSenderMatrix();
}
function setSenderSort(mode) {
  state.senderSort = mode;
  renderSenderMatrix();
}

function renderSenderMatrix() {
  const indices = getFilteredIndices();
  const d = DATA.backbones;
  const cfg = DATA.config;
  const allSenders = cfg.senderOrder;
  const categories = ['Excitatory','Inhibitory','Non-neuronal'];
  const catAbbrev = {'Excitatory':'Exc','Inhibitory':'Inh','Non-neuronal':'NN'};
  const sortMode = state.senderSort || 'tissue';
  const isTpds = state.senderMode === 'tpds';

  // All receivers flat (tissue order)
  const allReceivers = [];
  for (const cat of categories)
    for (const r of cfg.tissueCategories[cat] || []) allReceivers.push(r);

  // Build raw count and TPDS matrices indexed by sender name -> receiver name
  const rawCount = {}, rawTpds = {};
  for (const s of allSenders) { rawCount[s] = {}; rawTpds[s] = {}; for (const r of allReceivers) { rawCount[s][r] = 0; rawTpds[s][r] = 0; } }
  for (const i of indices) {
    const recv = d.receiver[i];
    if (!rawCount[allSenders[0]] || rawCount[allSenders[0]][recv] === undefined) continue;
    const mask = d.sender_mask[i];
    for (let si = 0; si < allSenders.length; si++) {
      if (mask & (1 << si)) {
        rawCount[allSenders[si]][recv]++;
        rawTpds[allSenders[si]][recv] += d.mean_tpds[i];
      }
    }
  }

  // Sort receivers (columns)
  let receivers, recvGroupBounds;
  const recvProfile = (r) => allSenders.map(s => Math.log10(1 + rawCount[s][r]));
  const recvTotal = (r) => { let s = 0; for (const sn of allSenders) s += rawCount[sn][r]; return s; };

  if (sortMode === 'tissue') {
    receivers = []; recvGroupBounds = [];
    for (const cat of categories) {
      const start = receivers.length;
      for (const r of (cfg.tissueCategories[cat] || []).slice().sort()) receivers.push(r);
      recvGroupBounds.push([start, receivers.length - 1]);
    }
  } else if (sortMode === 'cluster') {
    receivers = clusterOrder(allReceivers, recvProfile);
    recvGroupBounds = null;
  } else {
    receivers = allReceivers.slice().sort((a, b) => recvTotal(b) - recvTotal(a));
    recvGroupBounds = null;
  }

  // Sort senders (rows)
  const senderProfile = (s) => receivers.map(r => Math.log10(1 + rawCount[s][r]));
  const senderTotal = (s) => { let t = 0; for (const r of allReceivers) t += rawCount[s][r]; return t; };

  let senders, senderGroupBounds;
  if (sortMode === 'tissue') {
    senders = []; senderGroupBounds = [];
    for (const cat of categories) {
      const start = senders.length;
      const members = (cfg.tissueCategories[cat] || []).filter(r => allSenders.includes(r)).sort();
      // Also add senders not in any tissue category
      for (const s of members) senders.push(s);
      senderGroupBounds.push([start, senders.length - 1]);
    }
    // Any senders not in tissue categories
    const placed = new Set(senders);
    const extra = allSenders.filter(s => !placed.has(s)).sort();
    if (extra.length) {
      const start = senders.length;
      for (const s of extra) senders.push(s);
      senderGroupBounds.push([start, senders.length - 1]);
    }
  } else if (sortMode === 'cluster') {
    senders = clusterOrder(allSenders.slice(), senderProfile);
    senderGroupBounds = null;
  } else {
    senders = allSenders.slice().sort((a, b) => senderTotal(b) - senderTotal(a));
    senderGroupBounds = null;
  }

  // Build display labels
  const xLabels = receivers.map(r => sortMode === 'tissue' ? (catAbbrev[cfg.receiverToTissue[r]||''] || '') + ' | ' + r : r);
  const yLabels = senders.map(s => sortMode === 'tissue' ? (catAbbrev[cfg.receiverToTissue[s]||''] || '') + ' | ' + s : s);

  // Build z, hover, customdata
  const z = [], hoverText = [], customdata = [];
  // Log-scale tick values for count colorbar
  const countTicks = [1, 10, 100, 500, 1000];
  const logTicks = countTicks.map(v => Math.log10(1 + v));
  const tickLabels = countTicks.map(v => v >= 1000 ? (v/1000)+'k' : String(v));

  for (const s of senders) {
    const row = [], trow = [], crow = [];
    for (const r of receivers) {
      const cnt = rawCount[s][r];
      const avgT = cnt > 0 ? rawTpds[s][r] / cnt : 0;
      const val = isTpds ? avgT : Math.log10(1 + cnt);
      row.push(val);
      trow.push(`${s} → ${r}<br>${cnt} backbones<br>mean TPDS: ${avgT.toFixed(4)}`);
      crow.push({sender: s, receiver: r, n: cnt});
    }
    z.push(row); hoverText.push(trow); customdata.push(crow);
  }

  const trace = {
    z: z, x: xLabels, y: yLabels, type: 'heatmap',
    colorscale: isTpds ? [[0,'#1565c0'],[0.5,'#ffffff'],[1,'#c62828']] : 'YlOrRd',
    zmid: isTpds ? 0 : undefined,
    text: hoverText, hoverinfo: 'text', customdata: customdata,
    colorbar: {
      title: isTpds ? 'TPDS' : 'Count', thickness: 15,
      tickvals: isTpds ? undefined : logTicks,
      ticktext: isTpds ? undefined : tickLabels,
    },
  };

  // Separator lines for tissue mode
  const shapes = [];
  if (recvGroupBounds) {
    for (let gi = 0; gi < recvGroupBounds.length - 1; gi++) {
      const xPos = recvGroupBounds[gi][1] + 0.5;
      shapes.push({type:'line', yref:'paper', y0:0, y1:1, x0:xPos, x1:xPos, line:{color:'#333',width:2}});
    }
  }
  if (senderGroupBounds) {
    for (let gi = 0; gi < senderGroupBounds.length - 1; gi++) {
      const yPos = senderGroupBounds[gi][1] + 0.5;
      shapes.push({type:'line', xref:'paper', x0:0, x1:1, y0:yPos, y1:yPos, line:{color:'#333',width:2}});
    }
  }

  const layout = {
    title: 'Sender → Receiver Backbone Support' + (state.contrast !== 'ALL' ? ` (${state.contrast})` : ''),
    xaxis: {title:'Receiver', tickangle:-45, tickfont:{size:10}},
    yaxis: {title:'Sender', automargin:true, dtick:1, tickfont:{size:10}},
    margin: {l:140,r:40,t:50,b:120}, height: 650,
    shapes: shapes,
  };
  Plotly.newPlot('sender-plot', [trace], layout, {responsive:true});
  document.getElementById('sender-plot').on('plotly_click', function(ev) {
    if (!ev.points.length) return;
    const pt = ev.points[0];
    const cd = pt.customdata;
    if (cd && cd.n > 0) {
      document.getElementById('sel-sender').value = cd.sender;
      document.getElementById('sel-receiver').value = cd.receiver;
      state.sender = cd.sender;
      state.receiver = cd.receiver;
      markAllDirty(); switchTab('table');
    }
  });
}

// ============================================================
// GRAPH TAB — EM Trajectories + Network Graph
// ============================================================

const GENOTYPES = ['App', 'Tau', 'ApTt'];
const TIMEPOINTS = ['2mo', '4mo', '6mo'];
const GENO_COLOR = {App:'#c62828', Tau:'#1565c0', ApTt:'#6a1b9a'};

// --- View mode switching ---
function setGraphViewMode(val) {
  state.graphViewMode = val;
  document.getElementById('graph-trajectory-container').style.display = val === 'trajectory' ? '' : 'none';
  document.getElementById('graph-network-container').style.display = val === 'network' ? '' : 'none';
  document.getElementById('graph-geno-controls').style.display = val === 'trajectory' ? '' : 'none';
  document.getElementById('graph-net-controls').style.display = val === 'network' ? '' : 'none';
  state.tabDirty.graph = true;
  renderGraphTab();
}
function setGraphGenotype(val) { state.graphGenotype = val; renderEMTrajectory(); }
function setGraphMetric(val) { state.graphMetric = val; renderEMTrajectory(); }
function setGraphTopN(val) { state.graphTopN = val; renderEMTrajectory(); }

function renderGraphTab() {
  if (state.graphViewMode === 'trajectory') renderEMTrajectory();
  else renderGraph();
}

// --- EM Trajectory rendering ---
function renderEMTrajectory() {
  const d = DATA.backbones;
  const recv = state.receiver;

  if (recv === 'ALL') {
    Plotly.purge('em-trajectory-plot');
    document.getElementById('em-detail-panel').style.display = 'none';
    const div = document.getElementById('em-trajectory-plot');
    div.innerHTML = '<div style="padding:40px;text-align:center;color:#888;">Select a receiver cell type to view EM trajectories.</div>';
    return;
  }

  const geno = state.graphGenotype;
  const metric = state.graphMetric;
  const topN = state.graphTopN;
  const genotypes = geno === 'all' ? GENOTYPES : [geno];

  // Build EM stats: for each EM, compute per-contrast metrics
  // Only consider backbones matching current receiver + sender filters
  const emStats = {};  // em -> {contrast: {count, tpdsSum, scoreSum}}

  for (let i = 0; i < d.contrast.length; i++) {
    if (recv !== 'ALL' && d.receiver[i] !== recv) continue;
    if (state.sender !== 'ALL') {
      const sIdx = DATA.config.senderOrder.indexOf(state.sender);
      if (sIdx >= 0 && !(d.sender_mask[i] & (1 << sIdx))) continue;
    }
    if (state.direction === 'up' && d.mean_tpds[i] <= 0) continue;
    if (state.direction === 'down' && d.mean_tpds[i] >= 0) continue;

    const em = d.EM[i];
    const contrast = d.contrast[i];
    if (!emStats[em]) emStats[em] = {};
    if (!emStats[em][contrast]) emStats[em][contrast] = {count:0, tpdsSum:0, scoreSum:0};
    emStats[em][contrast].count++;
    emStats[em][contrast].tpdsSum += d.mean_tpds[i];
    emStats[em][contrast].scoreSum += d.observed_score[i];
  }

  // Rank EMs by total backbone count across selected genotype contrasts
  const emList = Object.keys(emStats);
  const emTotal = {};
  for (const em of emList) {
    let total = 0;
    for (const g of genotypes) {
      for (const tp of TIMEPOINTS) {
        const c = g + '_' + tp;
        if (emStats[em][c]) total += emStats[em][c].count;
      }
    }
    emTotal[em] = total;
  }
  emList.sort((a,b) => emTotal[b] - emTotal[a]);
  const topEMs = emList.slice(0, topN);

  if (topEMs.length === 0) {
    Plotly.purge('em-trajectory-plot');
    document.getElementById('em-trajectory-plot').innerHTML =
      '<div style="padding:40px;text-align:center;color:#888;">No EMs found for the current filters.</div>';
    return;
  }

  // Build value function
  function getVal(em, contrast) {
    const s = emStats[em] && emStats[em][contrast];
    if (!s || s.count === 0) return 0;
    if (metric === 'count') return s.count;
    if (metric === 'score') return s.scoreSum / s.count;
    // direction: net TPDS (positive = more up, negative = more down)
    return s.tpdsSum / s.count;
  }

  if (geno === 'all') {
    // Side-by-side subplots: one heatmap per genotype
    const traces = [];
    const annotations = [];
    for (let gi = 0; gi < GENOTYPES.length; gi++) {
      const g = GENOTYPES[gi];
      const z = [], customdata = [];
      for (const em of topEMs) {
        const row = [], cdRow = [];
        for (const tp of TIMEPOINTS) {
          const c = g + '_' + tp;
          const v = getVal(em, c);
          const s = emStats[em] && emStats[em][c];
          row.push(v);
          cdRow.push({em:em, contrast:c, count:s?s.count:0,
            meanTpds: s && s.count ? (s.tpdsSum/s.count) : 0,
            meanScore: s && s.count ? (s.scoreSum/s.count) : 0});
        }
        z.push(row); customdata.push(cdRow);
      }

      const isDir = metric === 'direction';
      const zMax = Math.max(...z.flat().map(Math.abs), 0.01);
      traces.push({
        z: z, x: TIMEPOINTS, y: topEMs.map(e => gi === 0 ? e : ' '.repeat(gi) + e),
        type:'heatmap', xaxis: gi===0?'x':'x'+(gi+1), yaxis:'y',
        customdata: customdata,
        colorscale: isDir ? [[0,'#1565c0'],[0.5,'#ffffff'],[1,'#c62828']] : 'YlOrRd',
        zmid: isDir ? 0 : undefined, zmin: isDir ? -zMax : 0, zmax: zMax,
        showscale: gi === GENOTYPES.length-1,
        colorbar: {title: metric==='count'?'Backbones':metric==='score'?'Mean score':'Net TPDS', len:0.6},
        hovertemplate: '%{customdata.em}<br>%{customdata.contrast}<br>'+
          'Backbones: %{customdata.count}<br>Mean TPDS: %{customdata.meanTpds:.3f}<br>'+
          'Mean score: %{customdata.meanScore:.3f}<extra></extra>',
      });
      // Genotype label annotation
      annotations.push({text:'<b>'+g+'</b>', xref:'x'+(gi===0?'':gi+1), yref:'paper',
        x:1, y:1.06, showarrow:false, font:{size:13, color:GENO_COLOR[g]}});
    }

    const layout = {
      height: Math.max(400, topEMs.length * 18 + 120),
      margin:{l:120,r:60,t:50,b:40},
      annotations: annotations,
      grid:{rows:1, columns:3, pattern:'independent'},
      yaxis:{autorange:'reversed', tickfont:{size:11}},
    };
    for (let gi = 0; gi < GENOTYPES.length; gi++) {
      const xkey = gi===0?'xaxis':'xaxis'+(gi+1);
      layout[xkey] = {tickfont:{size:11}, title:''};
    }
    Plotly.newPlot('em-trajectory-plot', traces, layout, {responsive:true});
  } else {
    // Single genotype heatmap
    const z = [], customdata = [];
    for (const em of topEMs) {
      const row = [], cdRow = [];
      for (const tp of TIMEPOINTS) {
        const c = geno + '_' + tp;
        const v = getVal(em, c);
        const s = emStats[em] && emStats[em][c];
        row.push(v);
        cdRow.push({em:em, contrast:c, count:s?s.count:0,
          meanTpds: s && s.count ? (s.tpdsSum/s.count) : 0,
          meanScore: s && s.count ? (s.scoreSum/s.count) : 0});
      }
      z.push(row); customdata.push(cdRow);
    }

    const isDir = metric === 'direction';
    const zMax = Math.max(...z.flat().map(Math.abs), 0.01);
    const trace = {
      z: z, x: TIMEPOINTS, y: topEMs, type:'heatmap',
      customdata: customdata,
      colorscale: isDir ? [[0,'#1565c0'],[0.5,'#ffffff'],[1,'#c62828']] : 'YlOrRd',
      zmid: isDir ? 0 : undefined, zmin: isDir ? -zMax : 0, zmax: zMax,
      colorbar: {title: metric==='count'?'Backbones':metric==='score'?'Mean score':'Net TPDS'},
      hovertemplate: '%{y}<br>%{x} (%{customdata.contrast})<br>'+
        'Backbones: %{customdata.count}<br>Mean TPDS: %{customdata.meanTpds:.3f}<br>'+
        'Mean score: %{customdata.meanScore:.3f}<extra></extra>',
    };

    const layout = {
      height: Math.max(400, topEMs.length * 18 + 120),
      margin:{l:120,r:60,t:50,b:40},
      title:{text:geno+' — EM Signaling Trajectories ('+recv+')', font:{size:14, color:GENO_COLOR[geno]}},
      yaxis:{autorange:'reversed', tickfont:{size:11}},
      xaxis:{tickfont:{size:11}},
    };
    Plotly.newPlot('em-trajectory-plot', [trace], layout, {responsive:true});
  }

  // Click handler: show EM detail
  document.getElementById('em-trajectory-plot').on('plotly_click', function(ev) {
    if (!ev.points.length) return;
    const pt = ev.points[0];
    const cd = pt.customdata;
    if (cd && cd.em) showEMDetail(cd.em, emStats[cd.em], genotypes);
  });
}

function showEMDetail(em, stats, genotypes) {
  const d = DATA.backbones;
  const recv = state.receiver;
  document.getElementById('em-detail-panel').style.display = '';
  document.getElementById('em-detail-title').textContent = em + ' — Receptor Inputs & Target Outputs';

  // Gather receptors and targets for this EM across relevant contrasts
  const recCounts = {}, tgtCounts = {};
  for (let i = 0; i < d.EM.length; i++) {
    if (d.EM[i] !== em) continue;
    if (recv !== 'ALL' && d.receiver[i] !== recv) continue;
    const c = d.contrast[i];
    // Check this contrast belongs to one of our genotypes
    const cGeno = c.split('_')[0];
    if (!genotypes.includes(cGeno)) continue;

    const rec = d.Receptor[i], tgt = d.Target[i];
    if (!recCounts[rec]) recCounts[rec] = {total:0, byContrast:{}};
    recCounts[rec].total++;
    recCounts[rec].byContrast[c] = (recCounts[rec].byContrast[c]||0) + 1;

    if (!tgtCounts[tgt]) tgtCounts[tgt] = {total:0, byContrast:{}};
    tgtCounts[tgt].total++;
    tgtCounts[tgt].byContrast[c] = (tgtCounts[tgt].byContrast[c]||0) + 1;
  }

  const recSorted = Object.entries(recCounts).sort((a,b) => b[1].total - a[1].total);
  const tgtSorted = Object.entries(tgtCounts).sort((a,b) => b[1].total - a[1].total);

  function buildTable(entries, label) {
    const contrasts = [];
    for (const g of genotypes) for (const tp of TIMEPOINTS) contrasts.push(g+'_'+tp);
    let html = '<div style="font-size:12px;"><strong>' + label + ' (' + entries.length + ')</strong>';
    html += '<table style="width:100%;border-collapse:collapse;margin-top:6px;font-size:11px;">';
    html += '<tr><th style="text-align:left;padding:2px 4px;">Gene</th><th>Total</th>';
    for (const c of contrasts) html += '<th style="padding:2px 3px;font-size:10px;">' + c.replace('_','<br>') + '</th>';
    html += '</tr>';
    for (const [gene, data] of entries.slice(0, 15)) {
      html += '<tr><td style="padding:2px 4px;font-weight:500;">' + gene + '</td>';
      html += '<td style="text-align:center;">' + data.total + '</td>';
      for (const c of contrasts) {
        const n = data.byContrast[c] || 0;
        const bg = n > 0 ? 'rgba(251,140,0,' + Math.min(n/20, 0.8) + ')' : '';
        html += '<td style="text-align:center;padding:2px 3px;background:' + bg + '">' + (n||'') + '</td>';
      }
      html += '</tr>';
    }
    if (entries.length > 15) html += '<tr><td colspan="' + (contrasts.length+2) + '" style="color:#888;padding:4px;">...and ' + (entries.length-15) + ' more</td></tr>';
    html += '</table></div>';
    return html;
  }

  document.getElementById('em-detail-receptors').innerHTML = buildTable(recSorted, 'Receptor Inputs');
  document.getElementById('em-detail-targets').innerHTML = buildTable(tgtSorted, 'Target Outputs');
}

// --- Network graph helpers ---
function setGraphLayout(val) { state.graphLayout = val; state.tabDirty.graph = true; renderGraph(); }
function setGraphMinDegree(val) { state.graphMinDegree = val; state.tabDirty.graph = true; renderGraph(); }
function resetGraphFocus() {
  state.graphFocusNode = null;
  document.getElementById('graph-reset-focus').style.display = 'none';
  state.tabDirty.graph = true; renderGraph();
}

function getGraphLayout(layoutName, nNodes) {
  if (layoutName === 'concentric') {
    return {name:'concentric',
      concentric: function(node){return 3-(node.data('rank')||0);},
      levelWidth: function(){return 1;}, minNodeSpacing: 8, animate: false};
  }
  if (layoutName === 'flow') {
    // First pass: run cose to get good y-positions from connectivity,
    // then snap x to layer bands. We return cose config; the snap happens
    // in a layout 'stop' callback added after cytoscape init.
    return {name:'cose', animate: false, randomize: true,
      nodeRepulsion: function(){return nNodes > 200 ? 80000 : 40000;},
      idealEdgeLength: function(){return nNodes > 200 ? 60 : 80;},
      gravity: 0.3, nodeOverlap: 20,
      _postLayout: 'flow'};  // sentinel for post-layout snap
  }
  // force — unconstrained cose
  return {name:'cose', animate: false, randomize: true,
    nodeRepulsion: function(){return nNodes > 200 ? 80000 : 40000;},
    idealEdgeLength: function(){return nNodes > 200 ? 50 : 70;},
    gravity: 0.25, nodeOverlap: 20};
}

function applyFlowSnap(cy) {
  // Snap x-positions to three vertical bands by node type
  // while preserving y-positions from force layout
  const w = cy.width() || 800;
  const xBand = {Receptor: w*0.15, EM: w*0.50, Target: w*0.85};
  const blend = 0.15; // how much original x leaks through (0=pure column, 1=no snap)
  cy.nodes().forEach(function(node) {
    const t = node.data('type');
    if (!t || !xBand[t]) return;
    const pos = node.position();
    pos.x = xBand[t] + (pos.x - xBand[t]) * blend;
    node.position(pos);
  });
  cy.fit(undefined, 30);
}

// --- Graph data cache (rebuilt on filter change, reused on layout-only change) ---
let graphCache = null;

function buildGraphData() {
  const indices = getFilteredIndices();
  const d = DATA.backbones;
  const nodeDeg = {}, nodeType = {}, edgeScores = {}, edgeTpds = {}, edgeCounts = {};
  const receptors = new Set(), ems = new Set(), targets = new Set();

  for (const i of indices) {
    const rec = 'R:'+d.Receptor[i], em = 'E:'+d.EM[i], tgt = 'T:'+d.Target[i];
    const score = d.observed_score[i], tpds = d.mean_tpds[i];
    nodeType[rec]='Receptor'; nodeType[em]='EM'; nodeType[tgt]='Target';
    nodeDeg[rec]=(nodeDeg[rec]||0)+1; nodeDeg[em]=(nodeDeg[em]||0)+1; nodeDeg[tgt]=(nodeDeg[tgt]||0)+1;
    receptors.add(d.Receptor[i]); ems.add(d.EM[i]); targets.add(d.Target[i]);
    const e1=rec+'>'+em, e2=em+'>'+tgt;
    edgeScores[e1]=Math.max(edgeScores[e1]||0,score); edgeScores[e2]=Math.max(edgeScores[e2]||0,score);
    edgeTpds[e1]=(edgeTpds[e1]||0)+tpds; edgeTpds[e2]=(edgeTpds[e2]||0)+tpds;
    edgeCounts[e1]=(edgeCounts[e1]||0)+1; edgeCounts[e2]=(edgeCounts[e2]||0)+1;
  }
  return {indices, nodeDeg, nodeType, edgeScores, edgeTpds, edgeCounts, receptors, ems, targets};
}

function renderGraph() {
  const d = DATA.backbones;

  if (state.contrast === 'ALL') {
    document.getElementById('graph-status').textContent = 'Select a specific contrast to render the pathway graph.';
    document.getElementById('graph-info').style.display = 'none';
    if (state.cyInstance) { state.cyInstance.destroy(); state.cyInstance = null; }
    return;
  }

  // Build or reuse graph data
  graphCache = buildGraphData();
  const gc = graphCache;

  if (gc.indices.length === 0) {
    document.getElementById('graph-status').textContent = 'No backbones match current filters.';
    document.getElementById('graph-info').style.display = 'none';
    if (state.cyInstance) { state.cyInstance.destroy(); state.cyInstance = null; }
    return;
  }

  // Apply degree filter
  const minDeg = state.graphMinDegree;
  let nodeIds = Object.keys(gc.nodeDeg).filter(n => gc.nodeDeg[n] >= minDeg);

  // If focusing on a node, restrict to its 2-hop neighborhood
  let focusMode = false;
  if (state.graphFocusNode && gc.nodeDeg[state.graphFocusNode]) {
    focusMode = true;
    const focus = state.graphFocusNode;
    const neighbors1 = new Set([focus]);
    // 1-hop: edges incident to focus
    for (const eid of Object.keys(gc.edgeScores)) {
      const [src,tgt] = eid.split('>');
      if (src === focus) neighbors1.add(tgt);
      if (tgt === focus) neighbors1.add(src);
    }
    // 2-hop: edges incident to 1-hop neighbors
    const neighbors2 = new Set(neighbors1);
    for (const eid of Object.keys(gc.edgeScores)) {
      const [src,tgt] = eid.split('>');
      if (neighbors1.has(src)) neighbors2.add(tgt);
      if (neighbors1.has(tgt)) neighbors2.add(src);
    }
    nodeIds = nodeIds.filter(n => neighbors2.has(n));
  }

  // Cap nodes
  const MAX_NODES = 600;
  let capped = false;
  if (nodeIds.length > MAX_NODES) {
    nodeIds.sort((a,b) => gc.nodeDeg[b]-gc.nodeDeg[a]);
    nodeIds = nodeIds.slice(0, MAX_NODES); capped = true;
  }
  const nodeSet = new Set(nodeIds);
  const typeColor = {'Receptor':'#43a047','EM':'#fb8c00','Target':'#5c6bc0'};
  const typeRank = {'Receptor':0,'EM':1,'Target':2};
  const maxDeg = Math.max(...nodeIds.map(n => gc.nodeDeg[n]));

  const elements = [];
  for (const nid of nodeIds) {
    const t = gc.nodeType[nid], deg = gc.nodeDeg[nid];
    elements.push({data:{id:nid,label:nid.slice(2),type:t,deg:deg,
      size:10+30*Math.sqrt(deg/maxDeg),color:typeColor[t],rank:typeRank[t]}});
  }
  const maxScore = Math.max(...Object.values(gc.edgeScores), 0.01);
  for (const [eid,score] of Object.entries(gc.edgeScores)) {
    const [src,tgt] = eid.split('>');
    if (!nodeSet.has(src)||!nodeSet.has(tgt)) continue;
    const avgTpds = gc.edgeTpds[eid] / (gc.edgeCounts[eid]||1);
    const edgeColor = avgTpds > 0 ? '#c62828' : avgTpds < 0 ? '#1565c0' : '#999';
    elements.push({data:{id:eid,source:src,target:tgt,score:score,
      width:0.5+3*(score/maxScore),opacity:0.2+0.6*(score/maxScore),edgeColor:edgeColor}});
  }

  if (state.cyInstance) state.cyInstance.destroy();
  const nNodeCount = nodeIds.length;
  const layoutConfig = getGraphLayout(state.graphLayout, nNodeCount);
  const isFlowLayout = layoutConfig._postLayout === 'flow';
  delete layoutConfig._postLayout;  // clean before passing to cytoscape

  state.cyInstance = cytoscape({
    container: document.getElementById('cy'), elements: elements,
    style: [
      {selector:'node', style:{'label':'data(label)','width':'data(size)','height':'data(size)',
        'background-color':'data(color)','font-size':8,'text-valign':'bottom','text-margin-y':4,
        'color':'#333','text-outline-color':'#fff','text-outline-width':1,'min-zoomed-font-size':6}},
      {selector:'edge', style:{'width':'data(width)','line-color':'data(edgeColor)',
        'target-arrow-color':'data(edgeColor)','target-arrow-shape':'triangle',
        'curve-style':'bezier','opacity':'data(opacity)','arrow-scale':0.6}},
      {selector:'node.highlighted', style:{'border-width':3,'border-color':'#e53935','font-weight':'bold','font-size':10,'z-index':999}},
      {selector:'edge.highlighted', style:{'line-color':'#e53935','target-arrow-color':'#e53935','opacity':1,'width':3,'z-index':999}},
      {selector:'node.faded', style:{'opacity':0.15}},
      {selector:'edge.faded', style:{'opacity':0.05}},
      {selector:'node.focus-center', style:{'border-width':4,'border-color':'#ff6f00','border-style':'double'}},
    ],
    layout: layoutConfig,
    wheelSensitivity: 0.3,
  });

  // For flow layout: snap x-positions to layer bands after cose converges
  if (isFlowLayout) applyFlowSnap(state.cyInstance);

  // Highlight focus node
  if (focusMode && state.graphFocusNode) {
    const focusEl = state.cyInstance.getElementById(state.graphFocusNode);
    if (focusEl.length) focusEl.addClass('focus-center');
  }

  // Single-click: highlight neighborhood
  state.cyInstance.on('tap','node',function(evt){
    const node=evt.target, cy=state.cyInstance;
    cy.elements().removeClass('highlighted faded');
    if (focusMode) cy.getElementById(state.graphFocusNode).addClass('focus-center');
    const connected=node.neighborhood().add(node);
    cy.elements().not(connected).addClass('faded');
    connected.addClass('highlighted');
    showNodeDetail(node.data());
  });

  // Double-click: focus on node's 2-hop neighborhood
  state.cyInstance.on('dbltap','node',function(evt){
    state.graphFocusNode = evt.target.id();
    document.getElementById('graph-reset-focus').style.display = 'inline-block';
    renderGraph();
  });

  state.cyInstance.on('tap',function(evt){
    if(evt.target===state.cyInstance){
      state.cyInstance.elements().removeClass('highlighted faded');
      if (focusMode) state.cyInstance.getElementById(state.graphFocusNode).addClass('focus-center');
      document.getElementById('node-detail').style.display='none';
    }
  });

  // Info panel
  const nUp = gc.indices.filter(i => d.mean_tpds[i] > 0).length;
  const nDown = gc.indices.filter(i => d.mean_tpds[i] < 0).length;
  const totalNodes = Object.keys(gc.nodeDeg).length;
  let statusText = `${nodeIds.length} nodes, ${elements.filter(e=>e.data.source).length} edges`;
  if (capped) statusText = `Top ${MAX_NODES} of ${totalNodes} nodes (by degree)`;
  if (minDeg > 1) statusText += ` | degree \u2265 ${minDeg}`;
  if (focusMode) statusText += ` | focused on ${state.graphFocusNode.slice(2)}`;
  document.getElementById('graph-status').textContent = statusText;
  document.getElementById('graph-info').style.display = 'block';
  document.getElementById('gi-backbones').textContent = gc.indices.length.toLocaleString();
  document.getElementById('gi-receptors').textContent = gc.receptors.size;
  document.getElementById('gi-ems').textContent = gc.ems.size;
  document.getElementById('gi-targets').textContent = gc.targets.size;
  const scores = gc.indices.map(i => d.observed_score[i]).sort((a,b)=>a-b);
  document.getElementById('gi-medscore').textContent = fmt(scores[Math.floor(scores.length/2)], 3);
  document.getElementById('gi-direction').innerHTML =
    `<span class="tpds-up">${nUp} up</span> / <span class="tpds-down">${nDown} down</span>`;
}

function showNodeDetail(nodeData) {
  const panel = document.getElementById('node-detail');
  panel.style.display = 'block';
  document.getElementById('nd-title').textContent = nodeData.label + ' (' + nodeData.type + ')';
  const d = DATA.backbones;
  const indices = getFilteredIndices();
  const field = nodeData.type === 'Receptor' ? 'Receptor' : nodeData.type === 'EM' ? 'EM' : 'Target';
  const matching = indices.filter(i => d[field][i] === nodeData.label);

  let html = `<tr><td>Backbones</td><td>${matching.length}</td></tr>`;
  if (matching.length > 0) {
    const scores = matching.map(i => d.observed_score[i]);
    const nUp = matching.filter(i => d.mean_tpds[i] > 0).length;
    const nDown = matching.filter(i => d.mean_tpds[i] < 0).length;
    html += `<tr><td>Avg score</td><td>${fmt(scores.reduce((a,b)=>a+b,0)/scores.length, 3)}</td></tr>`;
    html += `<tr><td>Direction</td><td><span class="tpds-up">${nUp} up</span> / <span class="tpds-down">${nDown} down</span></td></tr>`;

    // Sender breakdown
    const senderCounts = {};
    for (const i of matching) {
      const slist = decodeSenders(d.sender_mask[i]);
      for (const s of slist) senderCounts[s] = (senderCounts[s]||0) + 1;
    }
    const topSenders = Object.entries(senderCounts).sort((a,b)=>b[1]-a[1]).slice(0,8);
    html += `<tr><td>Top senders</td><td style="white-space:normal; font-size:11px">${
      topSenders.map(([s,c])=>`${s} (${c})`).join(', ')}${Object.keys(senderCounts).length>8?'...':''}</td></tr>`;

    if (nodeData.type === 'EM') {
      const recs = [...new Set(matching.map(i => d.Receptor[i]))].sort();
      const tgts = [...new Set(matching.map(i => d.Target[i]))].sort();
      html += `<tr><td>Receptors (${recs.length})</td><td style="white-space:normal">${recs.slice(0,6).join(', ')}${recs.length>6?'...':''}</td></tr>`;
      html += `<tr><td>Targets (${tgts.length})</td><td style="white-space:normal">${tgts.slice(0,6).join(', ')}${tgts.length>6?'...':''}</td></tr>`;
    } else {
      const connected = [...new Set(matching.map(i => d.EM[i]))].sort();
      html += `<tr><td>EMs (${connected.length})</td><td style="white-space:normal">${connected.slice(0,8).join(', ')}${connected.length>8?'...':''}</td></tr>`;
    }
  }
  document.getElementById('nd-table').innerHTML = html;
}

// ============================================================
// TABLE TAB
// ============================================================
const TABLE_PAGE_SIZE = 500;
let tableIndices = [];

function renderTable() {
  tableIndices = getFilteredIndices();
  const d = DATA.backbones;
  const col = state.sortCol, asc = state.sortAsc ? 1 : -1;
  tableIndices.sort((a,b) => {
    let va=d[col][a], vb=d[col][b];
    if (typeof va==='string') return asc*va.localeCompare(vb);
    return asc*((va||0)-(vb||0));
  });
  const end = Math.min(TABLE_PAGE_SIZE, tableIndices.length);
  let html = '';
  for (let k = 0; k < end; k++) {
    const i = tableIndices[k];
    const tpds = d.mean_tpds[i];
    html += `<tr data-idx="${i}" onclick="selectBackbone(${i})">
      <td style="color:${diseaseColor(d.contrast[i])}">${d.contrast[i]}</td>
      <td>${d.receiver[i]} ${tissueBadge(d.receiver[i])}</td>
      <td>${d.Receptor[i]}</td><td>${d.EM[i]}</td><td>${d.Target[i]}</td>
      <td><b>${fmt(d.observed_score[i],3)}</b></td><td>${d.n_edges[i]}</td>
      <td>${fmt(d.qval_null1[i],4)}</td><td>${fmt(d.qval_null2[i],4)}</td>
      <td>${d.n_senders[i]}</td>
      <td class="${tpdsCls(tpds)}">${fmt(tpds,4)} ${tpds>0?'\u2191':tpds<0?'\u2193':''}</td>
    </tr>`;
  }
  document.getElementById('table-body').innerHTML = html;
  document.getElementById('table-status').textContent =
    `${tableIndices.length.toLocaleString()} backbones (showing first ${end})`;
}

function selectBackbone(idx) {
  const d = DATA.backbones;
  state.selectedBackbone = {receiver:d.receiver[idx],Receptor:d.Receptor[idx],EM:d.EM[idx],Target:d.Target[idx]};
  document.querySelectorAll('#table-body tr').forEach(tr => tr.classList.remove('highlighted'));
  const row = document.querySelector(`#table-body tr[data-idx="${idx}"]`);
  if (row) row.classList.add('highlighted');
  state.tabDirty.cross = true;
  renderCrossContrast();
}

// ============================================================
// CROSS-CONTRAST TAB
// ============================================================
function renderCrossContrast() {
  const bb = state.selectedBackbone;
  if (!bb) { document.getElementById('cross-prompt').style.display='block';
    document.getElementById('cross-contrast-container').style.display='none'; return; }
  document.getElementById('cross-prompt').style.display='none';
  document.getElementById('cross-contrast-container').style.display='flex';

  const d = DATA.backbones, n = DATA.nRows;
  const byContrast = {};
  for (let i = 0; i < n; i++) {
    if (d.receiver[i]===bb.receiver && d.Receptor[i]===bb.Receptor &&
        d.EM[i]===bb.EM && d.Target[i]===bb.Target) {
      byContrast[d.contrast[i]] = {score:d.observed_score[i],n_edges:d.n_edges[i],
        qval1:d.qval_null1[i],qval2:d.qval_null2[i],n_senders:d.n_senders[i],
        mean_tpds:d.mean_tpds[i],sender_mask:d.sender_mask[i]};
    }
  }

  const contrasts = DATA.config.contrasts;
  const scores = contrasts.map(c => byContrast[c] ? byContrast[c].score : 0);
  const colors = contrasts.map(c => {
    const v = byContrast[c]; if (!v) return '#ccc';
    return v.mean_tpds > 0 ? '#c62828' : v.mean_tpds < 0 ? '#1565c0' : '#999';
  });
  const borders = contrasts.map(c => diseaseColor(c));

  const trace = {x:contrasts, y:scores, type:'bar',
    marker:{color:colors, line:{color:borders, width:2}},
    text:scores.map(s => s>0?s.toFixed(3):''), textposition:'outside'};
  const layout = {
    title:`${bb.Receptor} \u2192 ${bb.EM} \u2192 ${bb.Target} (${bb.receiver})`,
    yaxis:{title:'Kinase Support Score'}, xaxis:{tickangle:-45},
    margin:{t:50,b:80,l:60,r:20}, height:400};
  Plotly.newPlot('cross-plot',[trace],layout,{responsive:true});

  // Info panel
  let html = `<h4 style="margin-bottom:8px">Backbone Detail</h4>`;
  html += `<table style="width:100%;font-size:12px;border-collapse:collapse;">`;
  html += `<tr style="border-bottom:2px solid #ddd"><th style="text-align:left;padding:4px">Contrast</th><th>Score</th><th>TPDS</th><th>q\u2082</th><th>Senders</th></tr>`;
  for (const c of contrasts) {
    const v = byContrast[c];
    if (v) {
      html += `<tr style="border-bottom:1px solid #f0f0f0">
        <td style="color:${diseaseColor(c)};padding:3px 4px;font-weight:600">${c}</td>
        <td style="padding:3px 4px">${fmt(v.score,3)}</td>
        <td class="${tpdsCls(v.mean_tpds)}" style="padding:3px 4px">${fmt(v.mean_tpds,4)} ${v.mean_tpds>0?'\u2191':'\u2193'}</td>
        <td style="padding:3px 4px">${fmt(v.qval2,4)}</td>
        <td style="padding:3px 4px">${v.n_senders}</td></tr>`;
    } else {
      html += `<tr style="border-bottom:1px solid #f0f0f0;opacity:0.4"><td style="padding:3px 4px">${c}</td><td colspan="4" style="padding:3px 4px">Not significant</td></tr>`;
    }
  }
  html += `</table>`;

  // Sender list for first available contrast
  const firstV = Object.values(byContrast)[0];
  if (firstV) {
    const sl = decodeSenders(firstV.sender_mask);
    html += `<div style="margin-top:10px;font-size:11px;color:var(--text-muted)"><b>Senders (${sl.length}):</b> ${sl.join(', ')}</div>`;
  }

  const nSig = Object.keys(byContrast).length;
  html += `<div style="margin-top:6px;font-size:12px;color:var(--text-muted)">Significant in <b>${nSig}</b> of 9 contrasts</div>`;
  document.getElementById('cross-info').innerHTML = html;
}

// ============================================================
// TEMPORAL TRAJECTORY TAB
// ============================================================
function renderTemporal() {
  const metric = state.temporalMetric;
  const indices = getFilteredIndices();
  const d = DATA.backbones;
  const timepoints = ['2mo','4mo','6mo'];
  const genotypes = ['App','Tau','ApTt'];
  const genoColors = {App:'#c62828',Tau:'#1565c0',ApTt:'#6a1b9a'};

  // Group by genotype x timepoint
  function aggregate(idxs) {
    const groups = {};
    for (const g of genotypes) for (const t of timepoints) groups[g+'_'+t] = [];
    for (const i of idxs) {
      const p = parseContrast(d.contrast[i]);
      const key = p.geno+'_'+p.tp;
      if (groups[key]) groups[key].push(i);
    }
    return groups;
  }

  function metricVal(idxs) {
    if (idxs.length === 0) return 0;
    if (metric === 'count') return idxs.length;
    if (metric === 'mean_score') return idxs.reduce((s,i)=>s+d.observed_score[i],0)/idxs.length;
    if (metric === 'mean_tpds') return idxs.reduce((s,i)=>s+Math.abs(d.mean_tpds[i]),0)/idxs.length;
    if (metric === 'pct_up') return 100*idxs.filter(i=>d.mean_tpds[i]>0).length/idxs.length;
    return 0;
  }

  const metricLabels = {count:'Backbone Count',mean_score:'Mean Score',mean_tpds:'Mean |TPDS|',pct_up:'% Upregulated'};

  // Main chart (all receivers combined or filtered)
  const groups = aggregate(indices);
  const traces = [];
  for (const g of genotypes) {
    const y = timepoints.map(t => metricVal(groups[g+'_'+t]));
    traces.push({x:timepoints,y:y,name:g,type:'scatter',mode:'lines+markers',
      line:{color:genoColors[g],width:3},marker:{size:10}});
  }
  const mainLayout = {title: metricLabels[metric] + (state.receiver!=='ALL'?' ('+state.receiver+')':' (all receivers)'),
    xaxis:{title:'Timepoint'},yaxis:{title:metricLabels[metric]},
    margin:{t:50,b:50,l:70,r:20},height:350,legend:{x:0.02,y:0.98}};
  Plotly.newPlot('temporal-main',traces,mainLayout,{responsive:true});

  // Small multiples per receiver
  const receivers = [];
  for (const cat of ['Excitatory','Inhibitory','Non-neuronal'])
    for (const r of DATA.config.tissueCategories[cat]||[]) receivers.push(r);
  const ncols = 4, nrows = Math.ceil(receivers.length/ncols);
  const gridTraces = [];
  const annotations = [];

  for (let ri = 0; ri < receivers.length; ri++) {
    const recv = receivers[ri];
    const col = ri % ncols, row = Math.floor(ri / ncols);
    const xaxis = ri===0?'x':'x'+(ri+1), yaxis = ri===0?'y':'y'+(ri+1);
    const recvIdxs = indices.filter(i => d.receiver[i] === recv);
    const rGroups = aggregate(recvIdxs);
    for (const g of genotypes) {
      const y = timepoints.map(t => metricVal(rGroups[g+'_'+t]));
      gridTraces.push({x:timepoints,y:y,name:g,type:'scatter',mode:'lines+markers',
        xaxis:xaxis,yaxis:yaxis,line:{color:genoColors[g],width:2},marker:{size:5},
        showlegend:ri===0});
    }
    const xd = [(col/ncols)+0.02, ((col+1)/ncols)-0.02];
    const yd = [1-((row+1)/nrows)+0.06, 1-(row/nrows)-0.02];
    annotations.push({text:`<b>${recv}</b>`,x:(xd[0]+xd[1])/2,y:yd[1]+0.01,
      xref:'paper',yref:'paper',showarrow:false,font:{size:10}});
  }

  const gridLayout = {height:nrows*160+60, margin:{t:30,b:30,l:50,r:20}, showlegend:true,
    legend:{x:0.02,y:1.02,orientation:'h'}};
  // Build axis configs
  for (let ri = 0; ri < receivers.length; ri++) {
    const col = ri%ncols, row = Math.floor(ri/ncols);
    const xkey = ri===0?'xaxis':'xaxis'+(ri+1);
    const ykey = ri===0?'yaxis':'yaxis'+(ri+1);
    gridLayout[xkey] = {domain:[(col/ncols)+0.02,((col+1)/ncols)-0.02],
      showticklabels:row===nrows-1,tickfont:{size:9}};
    gridLayout[ykey] = {domain:[1-((row+1)/nrows)+0.06,1-(row/nrows)-0.02],
      showticklabels:col===0,tickfont:{size:9}};
  }
  gridLayout.annotations = annotations;
  Plotly.newPlot('temporal-grid',gridTraces,gridLayout,{responsive:true});
}

// ============================================================
// ADDITIVITY TAB
// ============================================================
function renderAdditivity() {
  if (!state.backboneIndex) return;
  const d = DATA.backbones;
  const timepoints = ['2mo','4mo','6mo'];
  const catColors = {
    'All three':'#4caf50','App+ApTt':'#ef5350','Tau+ApTt':'#42a5f5',
    'Both App+Tau':'#ab47bc','App only':'#ffcdd2','Tau only':'#bbdefb',
    'ApTt emergent':'#ce93d8','None sig':'#e0e0e0'
  };
  const catOrder = ['All three','App+ApTt','Tau+ApTt','Both App+Tau','ApTt emergent','App only','Tau only','None sig'];
  const traces = [], statsLines = [];

  for (let ti = 0; ti < 3; ti++) {
    const tp = timepoints[ti];
    const appC = 'App_'+tp, tauC = 'Tau_'+tp, apttC = 'ApTt_'+tp;
    const byCategory = {}; catOrder.forEach(c => byCategory[c] = {x:[],y:[],text:[]});
    let sumXY=0,sumX=0,sumY=0,sumX2=0,sumY2=0,nPts=0;

    for (const [key, ci] of state.backboneIndex) {
      const appIdx = ci[appC], tauIdx = ci[tauC], apttIdx = ci[apttC];
      if (apttIdx === undefined) continue; // need ApTt at minimum
      const appScore = appIdx !== undefined ? d.observed_score[appIdx] : 0;
      const tauScore = tauIdx !== undefined ? d.observed_score[tauIdx] : 0;
      const apttScore = d.observed_score[apttIdx];
      const predicted = appScore + tauScore;

      const hasApp = appIdx !== undefined, hasTau = tauIdx !== undefined, hasApTt = true;
      let cat;
      if (hasApp && hasTau) cat = 'All three';
      else if (hasApp && !hasTau) cat = 'App+ApTt';
      else if (!hasApp && hasTau) cat = 'Tau+ApTt';
      else cat = 'ApTt emergent';

      const parts = key.split('|');
      byCategory[cat].x.push(predicted);
      byCategory[cat].y.push(apttScore);
      byCategory[cat].text.push(`${parts[1]}\u2192${parts[2]}\u2192${parts[3]} (${parts[0]})`);
      sumXY+=predicted*apttScore; sumX+=predicted; sumY+=apttScore;
      sumX2+=predicted*predicted; sumY2+=apttScore*apttScore; nPts++;
    }

    // Pearson r
    let r = 0;
    if (nPts > 2) {
      const num = nPts*sumXY - sumX*sumY;
      const den = Math.sqrt((nPts*sumX2-sumX*sumX)*(nPts*sumY2-sumY*sumY));
      r = den > 0 ? num/den : 0;
    }
    statsLines.push(`${tp}: n=${nPts}, r=${r.toFixed(3)}`);

    const xaxis = ti===0?'x':'x'+(ti+1), yaxis = ti===0?'y':'y'+(ti+1);
    for (const cat of catOrder) {
      const cd = byCategory[cat]; if (cd.x.length === 0) continue;
      const isNone = cat === 'None sig';
      traces.push({x:cd.x,y:cd.y,text:cd.text,name:cat,type:'scatter',mode:'markers',
        xaxis:xaxis,yaxis:yaxis,
        marker:{color:catColors[cat],size:isNone?3:5,opacity:isNone?0.2:0.6},
        hovertemplate:'%{text}<br>Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra>'+cat+'</extra>',
        showlegend:ti===0});
    }
    // Diagonal
    const maxVal = Math.max(3, ...Object.values(byCategory).flatMap(c=>c.x.concat(c.y)));
    traces.push({x:[0,maxVal],y:[0,maxVal],mode:'lines',xaxis:xaxis,yaxis:yaxis,
      line:{color:'#999',dash:'dash',width:1},showlegend:false,hoverinfo:'skip'});
  }

  const layout = {height:500, margin:{t:50,b:60,l:60,r:20},
    title:'Additivity: Predicted (App + Tau) vs Observed (ApTt)' + (state.receiver!=='ALL'?' \u2014 '+state.receiver:'')};
  for (let ti = 0; ti < 3; ti++) {
    const xk = ti===0?'xaxis':'xaxis'+(ti+1), yk = ti===0?'yaxis':'yaxis'+(ti+1);
    layout[xk] = {domain:[ti/3+0.04,(ti+1)/3-0.02],title:ti===1?'Predicted (App + Tau)':''};
    layout[yk] = {domain:[0.08,0.92],title:ti===0?'Observed (ApTt)':'',
      scaleanchor:ti===0?'x':'x'+(ti+1),scaleratio:1};
    layout.annotations = layout.annotations || [];
    layout.annotations.push({text:`<b>${timepoints[ti]}</b>`,x:(ti/3+0.04+(ti+1)/3-0.02)/2,y:1.0,
      xref:'paper',yref:'paper',showarrow:false,font:{size:14}});
  }
  Plotly.newPlot('additivity-plot',traces,layout,{responsive:true});
  document.getElementById('additivity-stats').textContent = statsLines.join('  |  ');
}

// ============================================================
// KINASE CONTEXT TAB
// ============================================================
function renderKinaseContext() {
  const c = state.contrast, r = state.receiver;
  const info = document.getElementById('kinase-ctx-info');
  if (c === 'ALL' || r === 'ALL') {
    info.innerHTML = '<h4>Kinase Context</h4><p style="color:var(--text-muted)">Select a specific contrast and receiver to see attributed kinases.</p>';
    Plotly.purge('kinase-ctx-plot'); return;
  }
  const key = c + '|' + r;
  const kinases = DATA.kinaseAttr[key];
  if (!kinases || kinases.length === 0) {
    info.innerHTML = `<h4>Kinase Context</h4><p style="color:var(--text-muted)">No kinases attributed to ${r} in ${c}.</p>`;
    Plotly.purge('kinase-ctx-plot'); return;
  }

  const names = kinases.map(k => k.k);
  const scores = kinases.map(k => k.s);
  const colors = kinases.map(k => k.n > 0 ? '#c62828' : k.n < 0 ? '#1565c0' : '#999');
  const htext = kinases.map(k => `${k.k} (${k.g})<br>NES: ${k.n}, FDR: ${k.f}<br>Score: ${k.s}, Confidence: ${k.c}`);

  const trace = {y:names.slice().reverse(), x:scores.slice().reverse(), type:'bar', orientation:'h',
    marker:{color:colors.slice().reverse()}, text:htext.slice().reverse(), hoverinfo:'text'};
  const layout = {title:`Kinases attributed to ${r} in ${c}`,
    xaxis:{title:'|Combined Score|'}, yaxis:{automargin:true,dtick:1,tickfont:{size:10}},
    margin:{l:80,r:20,t:50,b:50}, height:Math.max(300, kinases.length*22+100)};
  Plotly.newPlot('kinase-ctx-plot',[trace],layout,{responsive:true});

  let html = `<h4>Kinases (${kinases.length})</h4>`;
  html += `<p style="color:var(--text-muted);margin-bottom:8px;">Attributed to <b>${r}</b> in <b>${c}</b>. Red = positive NES (upregulated), blue = negative.</p>`;
  html += `<table style="width:100%;font-size:11px;border-collapse:collapse;">`;
  html += `<tr style="border-bottom:2px solid #ddd"><th style="text-align:left;padding:3px">Kinase</th><th>NES</th><th>FDR</th><th>Conf.</th></tr>`;
  for (const k of kinases.slice(0,20)) {
    const cls = k.c==='high'?'badge-high':k.c==='moderate'?'badge-mod':'badge-low';
    html += `<tr style="border-bottom:1px solid #f0f0f0">
      <td style="padding:2px 4px;color:${k.n>0?'#c62828':k.n<0?'#1565c0':'#666'};font-weight:600">${k.k}</td>
      <td style="padding:2px 4px">${k.n.toFixed(2)}</td>
      <td style="padding:2px 4px">${k.f.toFixed(4)}</td>
      <td style="padding:2px 4px"><span class="badge ${cls}">${k.c}</span></td></tr>`;
  }
  if (kinases.length > 20) html += `<tr><td colspan="4" style="padding:4px;color:var(--text-muted)">... and ${kinases.length-20} more</td></tr>`;
  html += `</table>`;
  info.innerHTML = html;
}

// ============================================================
// TAB MANAGEMENT
// ============================================================
function switchTab(tabId) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('#tab-bar button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-'+tabId).classList.add('active');
  document.querySelector(`#tab-bar button[data-tab="${tabId}"]`).classList.add('active');
  if (state.tabDirty[tabId]) { state.tabDirty[tabId]=false; renderTab(tabId); }
}

function renderTab(tabId) {
  if (tabId==='overview') renderOverview();
  if (tabId==='senders') renderSenderMatrix();
  if (tabId==='graph') renderGraphTab();
  if (tabId==='table') renderTable();
  if (tabId==='cross') renderCrossContrast();
  if (tabId==='temporal') renderTemporal();
  if (tabId==='additivity') renderAdditivity();
  if (tabId==='kinase') renderKinaseContext();
}

function markAllDirty() {
  for (const k of Object.keys(state.tabDirty)) state.tabDirty[k]=true;
  const active = document.querySelector('#tab-bar button.active').dataset.tab;
  state.tabDirty[active] = false;
  renderTab(active);
}

// ============================================================
// INIT
// ============================================================
function init() {
  // Populate contrast selector
  const selC = document.getElementById('sel-contrast');
  selC.innerHTML = '<option value="ALL">All contrasts</option>';
  for (const c of DATA.config.contrasts) selC.innerHTML += `<option value="${c}">${c}</option>`;

  // Populate receiver selector
  const selR = document.getElementById('sel-receiver');
  selR.innerHTML = '<option value="ALL">All receivers</option>';
  for (const [tissue, receivers] of Object.entries(DATA.config.tissueCategories)) {
    const og = document.createElement('optgroup'); og.label = tissue;
    for (const r of receivers) { const opt = document.createElement('option'); opt.value=r; opt.textContent=r; og.appendChild(opt); }
    selR.appendChild(og);
  }

  // Populate sender selector
  const selS = document.getElementById('sel-sender');
  selS.innerHTML = '<option value="ALL">All senders</option>';
  for (const [tissue, receivers] of Object.entries(DATA.config.tissueCategories)) {
    const og = document.createElement('optgroup'); og.label = tissue;
    for (const r of receivers) {
      if (DATA.config.senderOrder.includes(r)) {
        const opt = document.createElement('option'); opt.value=r; opt.textContent=r; og.appendChild(opt);
      }
    }
    if (og.children.length > 0) selS.appendChild(og);
  }

  document.getElementById('header-stats').textContent =
    `${DATA.nRows.toLocaleString()} significant backbones \u00d7 9 contrasts \u00d7 22 receivers`;

  // Build backbone index
  const d = DATA.backbones;
  state.backboneIndex = new Map();
  for (let i = 0; i < DATA.nRows; i++) {
    const key = d.receiver[i]+'|'+d.Receptor[i]+'|'+d.EM[i]+'|'+d.Target[i];
    if (!state.backboneIndex.has(key)) state.backboneIndex.set(key, {});
    state.backboneIndex.get(key)[d.contrast[i]] = i;
  }

  // Control bindings
  selC.onchange = () => { state.contrast=selC.value; markAllDirty(); };
  selR.onchange = () => { state.receiver=selR.value; markAllDirty(); };
  selS.onchange = () => { state.sender=selS.value; markAllDirty(); };
  document.getElementById('sel-direction').onchange = (e) => { state.direction=e.target.value; markAllDirty(); };

  const slScore = document.getElementById('sl-score');
  slScore.oninput = () => { state.minScore=parseFloat(slScore.value); document.getElementById('val-score').textContent=slScore.value; markAllDirty(); };
  const slQval = document.getElementById('sl-qval');
  slQval.oninput = () => { state.maxQval=parseFloat(slQval.value); document.getElementById('val-qval').textContent=slQval.value; markAllDirty(); };
  const slSenders = document.getElementById('sl-senders');
  slSenders.oninput = () => { state.minSenders=parseInt(slSenders.value); document.getElementById('val-senders').textContent=slSenders.value; markAllDirty(); };

  document.getElementById('sel-temporal-metric').onchange = (e) => {
    state.temporalMetric=e.target.value; state.tabDirty.temporal=true;
    const active = document.querySelector('#tab-bar button.active').dataset.tab;
    if (active==='temporal') { state.tabDirty.temporal=false; renderTemporal(); }
  };

  let searchTimer = null;
  document.getElementById('gene-search').oninput = (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.search=e.target.value; markAllDirty(); }, 250);
  };

  document.querySelectorAll('#tab-bar button').forEach(btn => {
    btn.onclick = () => switchTab(btn.dataset.tab);
  });
  document.querySelectorAll('#backbone-table th[data-col]').forEach(th => {
    th.onclick = () => {
      const col=th.dataset.col;
      if (state.sortCol===col) state.sortAsc=!state.sortAsc;
      else { state.sortCol=col; state.sortAsc=col==='contrast'||col==='receiver'; }
      state.tabDirty.table=true; renderTable();
    };
  });

  renderOverview();
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build interactive pathway viewer HTML")
    parser.add_argument(
        "--output", default=os.path.join(
            icfg.FACTORIAL_ALL_PAIRS_DIR, "aggregation",
            "pathway_viewer.html"),
        help="Output HTML path")
    args = parser.parse_args()

    print("Building pathway viewer...")
    merged, pi0, attr = load_data()
    payload = build_payload(merged, pi0, attr)

    payload_json = json.dumps(payload, separators=(",", ":"))
    print(f"  Payload size: {len(payload_json) / 1024 / 1024:.1f} MB")

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", payload_json)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)

    print(f"  Wrote {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
