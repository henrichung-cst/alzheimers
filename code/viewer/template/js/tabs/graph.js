function _destroyCy() {
  if (_cyInstance) { try { _cyInstance.destroy(); } catch(e) {} _cyInstance = null; }
  _nodeInfo = null;
}

function _graphPlaceholder(msg) {
  const el = document.getElementById("cy");
  if (!el) return;
  el.innerHTML = '<div class="graph-placeholder">' + msg + "</div>";
}

function _buildGraphData(indices, contrast) {
  const BB = PAYLOAD.backbones;
  const scoreCol = BB["observed_score_" + contrast];
  const tpdsCol = BB["mean_tpds_" + contrast];
  const nodeDeg = new Map();
  const nodeType = new Map();
  const nodeInfo = new Map();
  const edgeScores = new Map();
  const edgeTpds = new Map();
  const edgeCounts = new Map();

  for (const i of indices) {
    const bid = BB.id[i];
    const rGene = BB.Receptor[i];
    const emGene = BB.EM[i];
    const tGene = BB.Target[i];
    const rId = "R:" + rGene;
    const eId = "E:" + emGene;
    const tId = "T:" + tGene;
    const score = scoreCol ? scoreCol[i] : null;
    const tpds = tpdsCol ? tpdsCol[i] : null;

    for (const [nid, type] of [[rId, "Receptor"], [eId, "EM"], [tId, "Target"]]) {
      nodeDeg.set(nid, (nodeDeg.get(nid) || 0) + 1);
      if (!nodeType.has(nid)) nodeType.set(nid, type);
      let info = nodeInfo.get(nid);
      if (!info) {
        info = {bbs:[], scoreSum:0, scoreN:0, nUp:0, nDown:0};
        nodeInfo.set(nid, info);
      }
      info.bbs.push(bid);
      if (score != null) { info.scoreSum += score; info.scoreN++; }
      if (tpds != null) { if (tpds > 0) info.nUp++; else if (tpds < 0) info.nDown++; }
    }

    const rek = rId + ">" + eId;
    const etk = eId + ">" + tId;
    const s = (score == null) ? 0 : score;
    const t = (tpds == null) ? 0 : tpds;
    edgeScores.set(rek, Math.max(edgeScores.get(rek) || 0, s));
    edgeScores.set(etk, Math.max(edgeScores.get(etk) || 0, s));
    edgeTpds.set(rek, (edgeTpds.get(rek) || 0) + t);
    edgeTpds.set(etk, (edgeTpds.get(etk) || 0) + t);
    edgeCounts.set(rek, (edgeCounts.get(rek) || 0) + 1);
    edgeCounts.set(etk, (edgeCounts.get(etk) || 0) + 1);
  }

  // Min-degree filter
  const minDeg = Store.state.view.graphMinDegree | 0;
  let keepIds = [...nodeDeg.keys()].filter(id => nodeDeg.get(id) >= minDeg);
  // Node cap (degree-sorted)
  if (keepIds.length > GRAPH_MAX_NODES) {
    keepIds.sort((a,b) => nodeDeg.get(b) - nodeDeg.get(a));
    keepIds = keepIds.slice(0, GRAPH_MAX_NODES);
  }
  const keep = new Set(keepIds);

  // Local |TPDS| threshold and optional top-N edge cap. Threshold drops edges
  // whose mean |TPDS| falls below the user's value; top-N keeps only the
  // strongest |TPDS| edges as a separate rendering safety net.
  const tpdsMin = Math.max(0, Number(Store.state.view.graphTpdsMin) || 0);
  const topN = Number(Store.state.view.graphTopN) || 0;

  const maxDeg = keepIds.reduce((m, id) => Math.max(m, nodeDeg.get(id)), 1);
  const maxScore = [...edgeScores.values()].reduce((m,v) => Math.max(m,v), 0) || 1;

  // Build candidate edges (after min-degree node filter and |TPDS| threshold)
  let candidates = [];
  let tpdsBelowCount = 0;
  for (const [key, score] of edgeScores.entries()) {
    const [src, tgt] = key.split(">");
    if (!keep.has(src) || !keep.has(tgt)) continue;
    const count = edgeCounts.get(key) || 1;
    const avgTpds = (edgeTpds.get(key) || 0) / count;
    if (Math.abs(avgTpds) < tpdsMin) { tpdsBelowCount++; continue; }
    candidates.push({ key, src, tgt, score, avgTpds });
  }
  // Top-N cap by |TPDS| (descending)
  let topNApplied = false;
  if (topN > 0 && candidates.length > topN) {
    candidates.sort((a,b) => Math.abs(b.avgTpds) - Math.abs(a.avgTpds));
    candidates = candidates.slice(0, topN);
    topNApplied = true;
  }

  // Restrict surviving nodes to those touched by surviving edges.
  const nodesUsed = new Set();
  for (const c of candidates) { nodesUsed.add(c.src); nodesUsed.add(c.tgt); }
  const finalIds = keepIds.filter(id => nodesUsed.has(id));

  const nodes = finalIds.map(id => {
    const type = nodeType.get(id);
    const deg = nodeDeg.get(id);
    const sz = 10 + 30 * Math.sqrt(deg / maxDeg);
    const rank = type === "Receptor" ? 0 : type === "EM" ? 1 : 2;
    return { data: {
      id, label: id.slice(2), type, deg, size: sz,
      color: GRAPH_COLORS[type], rank,
    }};
  });

  const edges = candidates.map(c => {
    const w = 0.5 + 3 * (c.score / maxScore);
    const op = 0.2 + 0.6 * (c.score / maxScore);
    const col = c.avgTpds > 0 ? "#c62828"
              : c.avgTpds < 0 ? "#1565c0" : "#999";
    return { data: {
      id: c.key, source: c.src, target: c.tgt,
      score: c.score, width: w, opacity: op, edgeColor: col,
    }};
  });

  const finalInfo = new Map();
  for (const id of finalIds) finalInfo.set(id, nodeInfo.get(id));

  return { nodes, edges, nodeInfo: finalInfo,
           totalNodes: nodeDeg.size, keptNodes: finalIds.length,
           tpdsBelowCount, topNApplied };
}

function _applyFlowSnap(cy) {
  const w = cy.width() || 800;
  const cols = { "Receptor": w * 0.15, "EM": w * 0.50, "Target": w * 0.85 };
  cy.nodes().forEach(n => {
    const xTarget = cols[n.data("type")];
    const xCur = n.position("x");
    n.position("x", xCur * 0.15 + xTarget * 0.85);
  });
}

function _layoutConfig(layoutName, nNodes) {
  if (layoutName === "concentric") {
    return { name:"concentric",
             concentric: node => 3 - (node.data("rank") || 0),
             levelWidth: () => 1,
             minNodeSpacing: 8, animate:false };
  }
  const cose = { name:"cose", animate:false, randomize:true,
                 nodeRepulsion: () => nNodes > 200 ? 80000 : 40000,
                 idealEdgeLength: () => nNodes > 200
                   ? (layoutName === "flow" ? 60 : 50)
                   : (layoutName === "flow" ? 80 : 70),
                 gravity: layoutName === "flow" ? 0.3 : 0.25,
                 nodeOverlap:20 };
  return cose;
}

function _renderNodeDetail(nodeData) {
  const det = document.getElementById("graph-detail");
  if (!det) return;
  const nodeId = nodeData.id;
  const info = (_nodeInfo && _nodeInfo.get(nodeId))
    || {bbs:[], scoreSum:0, scoreN:0, nUp:0, nDown:0};
  const avgScore = info.scoreN ? (info.scoreSum / info.scoreN) : 0;
  det.innerHTML = "<h3>" + nodeData.label
    + ' <span class="meta">(' + nodeData.type + ")</span></h3>"
    + '<div class="meta">Backbones: ' + nodeData.deg
    + " &middot; avg score: " + avgScore.toFixed(3)
    + " &middot; ↑" + info.nUp + " / ↓" + info.nDown + "</div>"
    + '<button id="graph-filter-btn" class="chip" style="margin-top:8px;">'
    + "Filter Pathway Explorer to this gene</button>";
  const btn = document.getElementById("graph-filter-btn");
  if (btn) btn.addEventListener("click", () => {
    const gene = nodeData.label;
    peSearch = gene;
    const search = document.getElementById("pe-search");
    if (search) search.value = gene;
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"pathway"});
  });
}

function _graphActiveContrast() {
  const v = Store.state.view;
  return `${v.graphGenotype}_${v.graphTimepoint}`;
}

function renderGraph() {
  const el = document.getElementById("cy");
  if (!el) return;
  const contrast = _graphActiveContrast();
  // Graph is contrast-driven via its own genotype + timepoint controls; sync
  // the legacy filters.contrast slice so getFilteredIndices uses this snapshot.
  if (Store.state.filters.contrast !== contrast) {
    Store.dispatch({type:"SET_FILTER", key:"contrast", value: contrast});
  }
  el.innerHTML = "";
  const indices = getFilteredIndices();
  const built = _buildGraphData(indices, contrast);
  _nodeInfo = built.nodeInfo;
  const stats = document.getElementById("graph-stats");
  if (stats) {
    let s = `${contrast} · ${built.keptNodes} / ${built.totalNodes} nodes`
      + ` (min-deg ${Store.state.view.graphMinDegree}`;
    if (built.totalNodes > built.keptNodes) s += `, degree-capped at ${GRAPH_MAX_NODES}`;
    s += `), ${built.edges.length} edges`;
    if (built.tpdsBelowCount > 0) s += ` · ${built.tpdsBelowCount} hidden by |TPDS| ≥ ${Store.state.view.graphTpdsMin}`;
    if (built.topNApplied) s += ` · capped at top ${Store.state.view.graphTopN} by |TPDS|`;
    stats.textContent = s;
  }
  if (!built.nodes.length) {
    _destroyCy();
    _graphPlaceholder("No backbones for the current filters.");
    return;
  }

  _destroyCy();
  const layoutName = Store.state.view.graphLayout || "concentric";
  const nNodes = built.nodes.length;
  const layoutCfg = _layoutConfig(layoutName, nNodes);
  _cyInstance = cytoscape({
    container: el,
    elements: { nodes: built.nodes, edges: built.edges },
    style: [
      { selector:"node", style: {
        label:"data(label)", width:"data(size)", height:"data(size)",
        "background-color":"data(color)", "font-size":8,
        "text-valign":"bottom", "text-margin-y":4,
        "text-outline-color":"#fff", "text-outline-width":1,
        "min-zoomed-font-size":6,
      }},
      { selector:"edge", style: {
        width:"data(width)", "line-color":"data(edgeColor)",
        "target-arrow-color":"data(edgeColor)",
        "target-arrow-shape":"triangle", "curve-style":"bezier",
        opacity:"data(opacity)", "arrow-scale":0.6,
      }},
      { selector:"node.highlighted", style: {
        "border-width":3, "border-color":"#e53935",
        "font-weight":"bold", "font-size":10, "z-index":999,
      }},
      { selector:"node.faded", style: { opacity:0.15 } },
      { selector:"edge.faded", style: { opacity:0.05 } },
      { selector:"node.focus-center", style: {
        "border-width":4, "border-color":"#ff6f00", "border-style":"double",
      }},
    ],
    layout: layoutCfg,
    wheelSensitivity: 0.3,
  });
  if (layoutName === "flow") {
    _cyInstance.one("layoutstop", () => _applyFlowSnap(_cyInstance));
  }

  _cyInstance.on("tap", "node", evt => {
    const n = evt.target;
    _cyInstance.elements().removeClass("highlighted faded focus-center");
    const nbh = n.closedNeighborhood();
    _cyInstance.elements().not(nbh).addClass("faded");
    nbh.nodes().addClass("highlighted");
    n.addClass("focus-center");
    _renderNodeDetail(n.data());
  });
  _cyInstance.on("tap", evt => {
    if (evt.target === _cyInstance) {
      _cyInstance.elements().removeClass("highlighted faded focus-center");
      const det = document.getElementById("graph-detail");
      if (det) det.innerHTML = '<div class="muted">Click a node for details.</div>';
    }
  });
}

function wireGraphControls() {
  const v = Store.state.view;
  const genoSel = document.getElementById("graph-genotype");
  if (genoSel) {
    genoSel.value = v.graphGenotype;
    genoSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphGenotype", value: ev.target.value});
    });
  }
  const tpSel = document.getElementById("graph-timepoint");
  if (tpSel) {
    tpSel.value = v.graphTimepoint;
    tpSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphTimepoint", value: ev.target.value});
    });
  }
  const layoutSel = document.getElementById("graph-layout");
  if (layoutSel) {
    layoutSel.value = v.graphLayout;
    layoutSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphLayout", value: ev.target.value});
    });
  }
  const degSel = document.getElementById("graph-min-degree");
  if (degSel) {
    degSel.value = String(v.graphMinDegree);
    degSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphMinDegree",
                      value: parseInt(ev.target.value, 10)});
    });
  }
  const tpdsInp = document.getElementById("graph-tpds-min");
  if (tpdsInp) {
    tpdsInp.value = v.graphTpdsMin;
    tpdsInp.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"graphTpdsMin",
                      value: Math.max(0, parseFloat(ev.target.value) || 0)});
    });
  }
  const topNInp = document.getElementById("graph-top-n");
  if (topNInp) {
    topNInp.value = v.graphTopN == null ? "" : v.graphTopN;
    topNInp.addEventListener("change", ev => {
      const raw = ev.target.value.trim();
      const val = raw === "" ? null : Math.max(0, parseInt(raw, 10) || 0);
      Store.dispatch({type:"SET_VIEW", key:"graphTopN", value: val});
    });
  }
}

function wireGraphKeyboard() {
  const TPS = ["2mo", "4mo", "6mo"];
  document.addEventListener("keydown", ev => {
    if (Store.state.view.activeTab !== "graph") return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    let handled = false;
    if (ev.key === "ArrowLeft" || ev.key === "ArrowRight") {
      const cur = TPS.indexOf(Store.state.view.graphTimepoint);
      const ni = ((cur + (ev.key === "ArrowLeft" ? -1 : 1)) % TPS.length + TPS.length) % TPS.length;
      Store.dispatch({type:"SET_VIEW", key:"graphTimepoint", value: TPS[ni]});
      handled = true;
    }
    if (handled) ev.preventDefault();
  });
}

// ---------------------------------------------------------------------------
// Glossary
