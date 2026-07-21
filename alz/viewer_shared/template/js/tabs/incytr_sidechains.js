// ---------------------------------------------------------------------------
// Incytr pathway sidechains — a row-detail Cytoscape view.
//
// The lazy shard is context-scoped and columnar. This module keeps its own
// small cache because each context has one compact shard shared by every
// pathway row; individual rows only select and walk that loaded graph.
// ---------------------------------------------------------------------------
const IncytrSidechains = (function(){
  let indexCache = { url: null, data: null, error: null, promise: null };
  const shards = new Map();

  async function _fetchResponse(url) {
    let response;
    try {
      response = await fetch(url);
    } catch (err) {
      if (window.location.protocol === "file:") {
        throw new Error("Browser blocked local sidecar fetches under file://. Serve the viewer output directory over HTTP and open that URL.");
      }
      throw err;
    }
    if (!response.ok) throw new Error(`fetch ${url} → ${response.status}`);
    return response;
  }

  async function _fetchJson(url) {
    return (await _fetchResponse(url)).json();
  }

  async function _fetchGzipJson(url) {
    const response = await _fetchResponse(url);
    const raw = await response.arrayBuffer();
    const bytes = new Uint8Array(raw);
    let text;
    if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
      if (typeof DecompressionStream === "undefined") {
        throw new Error("Browser cannot decompress the sidechain shard.");
      }
      const stream = new Response(raw).body.pipeThrough(new DecompressionStream("gzip"));
      text = await new Response(stream).text();
    } else {
      text = new TextDecoder("utf-8").decode(raw);
    }
    return JSON.parse(text);
  }

  function ensureIndex() {
    const url = ViewerPayload.edgeUrl("incytr_sidechains_index");
    if (!url) return Promise.resolve(null);
    if (url !== indexCache.url) {
      indexCache = { url, data: null, error: null, promise: null };
      shards.clear();
    }
    if (indexCache.data) return Promise.resolve(indexCache.data);
    if (indexCache.error) return Promise.reject(new Error(indexCache.error));
    if (indexCache.promise) return indexCache.promise;
    indexCache.promise = _fetchJson(url).then(data => {
      if (!data || data.schema_version !== 1
          || data.slice_type !== "incytr_kinase_sidechains"
          || !data.by_context || typeof data.by_context !== "object") {
        throw new Error("sidechain index has an invalid schema");
      }
      indexCache.data = data;
      return data;
    }).catch(err => {
      indexCache.error = String(err && err.message ? err.message : err);
      throw err;
    }).finally(() => { indexCache.promise = null; });
    return indexCache.promise;
  }

  function activeEntry() {
    const context = ViewerPayload.activeContext();
    return indexCache.data && indexCache.data.by_context
      ? indexCache.data.by_context[context] || null : null;
  }

  function hasActiveSlice() {
    const entry = activeEntry();
    return !!(entry && typeof entry.url === "string");
  }

  async function loadActiveShard() {
    await ensureIndex();
    const context = ViewerPayload.activeContext();
    const entry = activeEntry();
    if (!entry || !entry.url) return null;
    const key = `${context}||${entry.url}`;
    let cache = shards.get(key);
    if (!cache) {
      cache = { url: entry.url, data: null, error: null, promise: null };
      shards.set(key, cache);
    }
    if (cache.data) return cache.data;
    if (cache.error) throw new Error(cache.error);
    if (cache.promise) return cache.promise;
    cache.promise = _fetchGzipJson(entry.url).then(data => {
      if (!data || data.schema_version !== 1
          || data.slice_type !== "incytr_kinase_sidechains"
          || data.context_id !== context) {
        throw new Error("sidechain shard has an invalid schema");
      }
      cache.data = data;
      return data;
    }).catch(err => {
      cache.error = String(err && err.message ? err.message : err);
      throw err;
    }).finally(() => {
      cache.promise = null;
    });
    return cache.promise;
  }

  return { ensureIndex, hasActiveSlice, loadActiveShard };
})();

const _IS_LAYOUT = {
  minimumCenterYPx: 180,
  horizontalMarginPx: 130,
  minimumSpineSpanPx: 480,
  minimumSpineSegmentCount: 1,
  fallbackGraphWidthPx: 900,
  graphHeightPx: 500,
  spineArcRadiusPx: 440,
  spineSweepDeg: 90,
  spineArcCenterBelowGraphPx: 220,
  // Radial fan per spine node: inner arc = strongest kinases (short edges hugging
  // the node), outer arcs = weak (long, faint, angularly spread so edges radiate
  // instead of stacking into a parallel band).
  fanInnerRadiusPx: 64,
  fanRingStepPx: 52,
  fanArcSpacingPx: 46,
  fanAngleMargin: 0.12,
  fitPaddingPx: 48,
  minZoom: 0.08,
  maxZoom: 3,
};
_IS_LAYOUT.halfWedgeRad = _IS_LAYOUT.spineSweepDeg * Math.PI / 180
  / (2 * Math.max(_IS_LAYOUT.minimumSpineSegmentCount, 3));
// Node prominence + focus. A kinase node's size and label visibility track its
// own strongest motif enrichment (|NES| emphasis): strong regulators are large
// and labeled; near-null ones shrink to unlabeled dots and recede. Tap a pathway
// node to pull its fan; hover any kinase to reveal its label.
const _IS_FOCUS = {
  minNodeDiameterPx: 9,
  maxNodeDiameterPx: 34,
  labelEmphasisMin: 0.15,
  focusBorderColor: "#dc2626",
  focusBorderWidthPx: 2,
};
// Edges are never cut. Strength is encoded as width + opacity through a convex
// (gamma) remap that strongly diminishes low signal while preserving strong.
// Terminal (kinase→node) strength is |NES| anchored at 1.0 — the biological null
// (no motif enrichment), an absolute floor, not this pathway's observed minimum —
// so "no enrichment → no ink". Chain (kinase→kinase) strength is the combined
// interactome weight anchored at 0.
const _IS_EMPHASIS = {
  nesNull: 1.0,
  // Presentation-only visibility floor; never exported as analytical evidence.
  siteFloor: 0.4,
  gamma: 3.5,
  minWidthPx: 0.35,
  maxWidthPx: 7.0,
  minOpacity: 0.03,
  maxOpacity: 0.95,
};
const _IS_STYLE = {
  smallTextPx: 11,
  loadingVerticalPaddingPx: 8,
  graphBorderWidthPx: 1,
  graphCornerRadiusPx: 4,
  captionTopMarginPx: 6,
  relationMaxHeightPx: 180,
  infoPanelWidthPx: 340,
  siteMaxHeightPx: 360,
  kinaseNodeDiameterPx: 34,
  kinaseNodeBorderWidthPx: 1,
  spineNodeBorderWidthPx: 2,
  spineNodeWidthPx: 100,
  spineNodeHeightPx: 42,
  labelFontPx: 10,
  labelMaxWidthPx: 96,
  spineEdgeWidthPx: 5,
  normalArrowScale: 0.7,
  emphasizedArrowScale: 1,
  edgeOpacity: 0.9,
};
const _IS_COLORS = {
  corePath: "#1f4ea3",
  enriched: "#d73027",
  depleted: "#4575b4",
  neutral: "#64748b",
  motif: "#2563eb",
  psp: "#d97706",
  both: "#7e22ce",
};

function _isSafeRows(columns, fields) {
  if (!columns || typeof columns !== "object") return [];
  const arrays = fields.map(field => columns[field]);
  if (arrays.some(values => !Array.isArray(values))) return [];
  const length = arrays.length ? arrays[0].length : 0;
  if (arrays.some(values => values.length !== length)) return [];
  const rows = [];
  for (let i = 0; i < length; i++) {
    const row = {};
    fields.forEach(field => { row[field] = columns[field][i]; });
    rows.push(row);
  }
  return rows;
}

function _isHostId(rk) {
  return `ip-sidechains-${String(rk).replace(/[^a-zA-Z0-9]/g, "_")}`;
}

function _isSpineNodes(row) {
  return ["Ligand", "Receptor", "EM", "Target"].map((role, order) => ({
    role,
    order,
    gene: String(row[role] || ""),
    id: `path:${role}`,
  }));
}

function _isNumeric(value) {
  const number = Number(value);
  return isFinite(number) ? number : 0;
}

function _isMotifPeerForEdge(edge) {
  if (typeof ViewerPayload === "undefined") return null;
  return ViewerPayload.motifPeerRoster(
    ViewerPayload.motifPeerCohortFor(), edge.kinase, edge.owning_cluster);
}

// Convex remap of a raw strength to [0,1], anchored at [lo, hi]. gamma > 1 makes
// it accelerate: low values collapse toward 0, strong values stay high. Nothing is
// thresholded — the weakest still returns a small positive emphasis, floored to a
// faint-but-present width/opacity by the helpers below.
function _isEmphasis(value, lo, hi) {
  if (!(hi > lo)) return 0;
  const x = Math.max(0, Math.min(1, (_isNumeric(value) - lo) / (hi - lo)));
  return Math.pow(x, _IS_EMPHASIS.gamma);
}

function _isEmphasisWidth(emphasis) {
  return _IS_EMPHASIS.minWidthPx + emphasis * (_IS_EMPHASIS.maxWidthPx - _IS_EMPHASIS.minWidthPx);
}

function _isEmphasisOpacity(emphasis) {
  return _IS_EMPHASIS.minOpacity + emphasis * (_IS_EMPHASIS.maxOpacity - _IS_EMPHASIS.minOpacity);
}

function _isEmphasisNodeSize(emphasis) {
  return _IS_FOCUS.minNodeDiameterPx
    + emphasis * (_IS_FOCUS.maxNodeDiameterPx - _IS_FOCUS.minNodeDiameterPx);
}

function _isArcCenter(width, height) {
  const left = _IS_LAYOUT.horizontalMarginPx;
  const right = Math.max(left + _IS_LAYOUT.minimumSpineSpanPx,
    width - _IS_LAYOUT.horizontalMarginPx);
  return {
    x: (left + right) / 2,
    y: height + _IS_LAYOUT.spineArcCenterBelowGraphPx,
  };
}

function _isMeanAngle(angles) {
  const vector = angles.reduce((sum, angle) => ({
    x: sum.x + Math.cos(angle), y: sum.y + Math.sin(angle),
  }), { x: 0, y: 0 });
  return Math.atan2(vector.y, vector.x);
}

function _isNesDirection(signedNes) {
  if (_isNumeric(signedNes) > 0) return "enriched";
  if (_isNumeric(signedNes) < 0) return "depleted";
  return "neutral";
}

function _isLegendSample(color, lineStyle = "solid", width = 3) {
  const sample = document.createElement("span");
  sample.setAttribute("aria-hidden", "true");
  sample.style.cssText = "display:inline-block;width:26px;margin-right:6px;vertical-align:middle;"
    + `border-top:${width}px ${lineStyle} ${color};`;
  return sample;
}

function _isLegendLine(label, samples) {
  const line = document.createElement("div");
  line.style.cssText = "display:flex;align-items:center;gap:3px;min-height:20px;";
  samples.forEach(sample => line.appendChild(sample));
  line.appendChild(document.createTextNode(label));
  return line;
}

function _isNodeRelationTable(node, cy) {
  const nodeId = String(node.id());
  const nodeKind = String(node.data("kind") || "");
  const nodeLabel = String(node.data("label") || nodeId);
  const terminalRows = [];
  const chainRows = [];
  const connectedEdges = node.connectedEdges();
  const endpointLabel = endpoint => String(endpoint.data("label") || endpoint.id());

  connectedEdges.forEach(edge => {
    const kind = String(edge.data("kind") || "");
    const source = edge.source();
    const target = edge.target();
    const sourceId = String(source.id());
    const targetId = String(target.id());
    if (kind === "terminal-edge") {
      const signedNes = _isNumeric(edge.data("signed_nes"));
      if (nodeKind === "spine-node" && targetId === nodeId) {
        terminalRows.push({
          relationship: `${endpointLabel(source)} → ${nodeLabel}`,
          role: String(edge.data("role") || node.data("role") || ""),
          signedNes,
          direction: _isNesDirection(signedNes),
          evidence: String(edge.data("provenance") || "motif"),
          weight: null,
          motifPeer: edge.data("motif_peer") || null,
        });
      } else if (nodeKind === "kinase-node" && sourceId === nodeId) {
        terminalRows.push({
          relationship: `${nodeLabel} → ${endpointLabel(target)}`,
          role: String(edge.data("role") || ""),
          signedNes,
          direction: _isNesDirection(signedNes),
          evidence: String(edge.data("provenance") || "motif"),
          weight: null,
          target: targetId,
          motifPeer: edge.data("motif_peer") || null,
        });
      }
      return;
    }
    if (kind === "chain-edge" && nodeKind === "kinase-node") {
      const other = sourceId === nodeId ? target : source;
      const otherId = String(other.id());
      if (otherId === nodeId) return;
      chainRows.push({
        relationship: `${sourceId === nodeId ? nodeLabel : endpointLabel(other)} → `
          + `${sourceId === nodeId ? endpointLabel(other) : nodeLabel}`,
        role: "kinase chain",
        signedNes: null,
        direction: null,
        evidence: String(edge.data("provenance") || "motif"),
        weight: _isNumeric(edge.data("weight")),
        other: otherId,
      });
    }
  });

  terminalRows.sort((a, b) =>
    Math.abs(b.signedNes) - Math.abs(a.signedNes)
      || a.relationship.localeCompare(b.relationship));
  chainRows.sort((a, b) =>
    b.weight - a.weight || a.relationship.localeCompare(b.relationship));
  const rows = terminalRows.concat(chainRows);
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.textContent = "No relationships.";
    return empty;
  }

  let summary;
  if (nodeKind === "spine-node") {
    const enriched = terminalRows.filter(row => row.direction === "enriched").length;
    const depleted = terminalRows.filter(row => row.direction === "depleted").length;
    summary = `${terminalRows.length} kinases affecting · ${enriched} enriched · ${depleted} depleted`;
  } else {
    const targetCount = new Set(terminalRows.map(row => row.target)).size;
    const kinaseCount = new Set(chainRows.map(row => row.other)).size;
    summary = `targets ${targetCount} nodes · ${kinaseCount} kinases`;
  }

  const table = document.createElement("table");
  table.setAttribute("aria-label", `${nodeLabel} relationships`);
  table.style.cssText = "border-collapse:collapse;width:100%;font-size:11px;";
  const caption = document.createElement("caption");
  caption.textContent = summary;
  caption.style.cssText = "caption-side:top;text-align:left;font-weight:700;"
    + "color:#1e3a5f;padding-bottom:4px;";
  table.appendChild(caption);
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Relationship", "Role", "Signed NES", "Direction", "Evidence", "Weight"]
    .forEach(label => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      cell.style.cssText = "padding:2px 5px;text-align:left;white-space:nowrap;";
      headerRow.appendChild(cell);
    });
  head.appendChild(headerRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  rows.forEach(row => {
    const tableRow = document.createElement("tr");
    const values = [
      row.relationship,
      row.role,
      row.signedNes === null ? "—" : row.signedNes.toFixed(3),
      row.direction || "—",
      row.evidence,
      row.weight === null ? "—" : row.weight.toFixed(3),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      if (index === 0 && row.motifPeer) {
        const sole = Number(row.motifPeer.motif_peers_detected) === 1;
        const tip = sole
          ? "Sole plausible source here among motif-confusable candidates"
          : `${row.motifPeer.motif_peers_detected} of ${row.motifPeer.motif_peers_informative} motif-confusable candidates transcribed here`;
        cell.innerHTML = _escapeHtml(String(value))
          + ` <details class="motif-peer-details"><summary><span class="badge ${sole ? "vhi" : "lo"}" title="${_escapeHtml(tip)}">${row.motifPeer.motif_peers_detected}/${row.motifPeer.motif_peers_informative}</span></summary>`
          + `<ul class="motif-peer-roster">${(row.motifPeer.peers || []).map(peer => `<li>${_escapeHtml(String(peer.kinase || ""))} (${(Number(peer.detection_fraction || 0) * 100).toFixed(0)}%)</li>`).join("") || "<li>No motif twins</li>"}</ul></details>`;
      } else {
        cell.textContent = value;
      }
      cell.style.cssText = "padding:2px 5px;border-top:1px solid #e2e8f0;"
        + (index === 0 ? "white-space:nowrap;" : "");
      if (index === 3 && row.direction && _IS_COLORS[row.direction]) {
        cell.style.color = _IS_COLORS[row.direction];
        cell.style.fontWeight = "600";
      }
      tableRow.appendChild(cell);
    });
    body.appendChild(tableRow);
  });
  table.appendChild(body);
  return table;
}

function _isGraphForRow(shard, row) {
  const interactome = _isSafeRows(shard.interactome, [
    "source_gene", "target_gene", "provenance", "weight",
  ]).filter(edge => edge.source_gene && edge.target_gene);
  const terminalFields = [
    "source_gene", "target_gene", "role", "contrast", "provenance", "weight",
    "best_abs_nes", "signed_nes", "best_fdr", "n_sites",
  ];
  // Regenerated shards carry the inline floor-99 site list. Keep older shards
  // readable while they are being replaced; their edge caption remains the
  // only available evidence detail.
  if (shard.terminal_edges && Array.isArray(shard.terminal_edges.sites)) {
    terminalFields.push("sites");
  }
  // Pre-peer shards remain viewable; regenerated shards additionally carry the
  // kinase/owning-cluster join keys used for the motif-peer chip.
  if (shard.terminal_edges && Array.isArray(shard.terminal_edges.kinase)
      && Array.isArray(shard.terminal_edges.owning_cluster)) {
    terminalFields.unshift("kinase");
    terminalFields.splice(5, 0, "owning_cluster");
  }
  const terminal = _isSafeRows(shard.terminal_edges, terminalFields)
    .filter(edge => edge.source_gene && edge.target_gene);
  const spine = _isSpineNodes(row);
  const spineByRole = new Map(spine.map(node => [node.role, node]));
  const contrastMatchedTerminal = terminal.filter(edge => {
    const node = spineByRole.get(String(edge.role || ""));
    const owning = String(edge.owning_cluster || "");
    const expectedOwning = String(edge.role || "") === "Ligand"
      ? String(row._sender || row.sender || "")
      : String(row._receiver || row.receiver || "");
    return !!node && String(edge.target_gene).toUpperCase() === node.gene.toUpperCase()
      && String(edge.contrast || "") === String(row.contrast || "")
      && (!owning || owning === expectedOwning);
  });
  // No cutoff: every contrast-matched kinase→node edge is drawn. Weak ones are
  // suppressed by the |NES| emphasis at render time, not filtered out here.
  const terminalEdges = contrastMatchedTerminal;
  const directKinaseGenes = new Set(terminalEdges.map(edge => String(edge.source_gene)));
  // Include one-hop kinase regulators in the full view. The default first-order
  // filter hides these chain-only nodes with their kinase→kinase links.
  const chainEdges = interactome.filter(edge =>
    directKinaseGenes.has(String(edge.source_gene)) || directKinaseGenes.has(String(edge.target_gene)));
  const kinaseGenes = new Set(directKinaseGenes);
  chainEdges.forEach(edge => {
    kinaseGenes.add(String(edge.source_gene));
    kinaseGenes.add(String(edge.target_gene));
  });
  const nesMax = terminalEdges.reduce(
    (maximum, edge) => Math.max(maximum, _isNumeric(edge.best_abs_nes)), 0);
  const sitesMax = terminalEdges.reduce(
    (maximum, edge) => Math.max(maximum, _isNumeric(edge.n_sites)), 0);
  const chainMax = chainEdges.reduce(
    (maximum, edge) => Math.max(maximum, _isNumeric(edge.weight)), 0);
  return {
    spine, terminalEdges, chainEdges, kinaseGenes, directKinaseGenes,
    nesMax, sitesMax, chainMax,
  };
}

function _isTerminalSiteRows(edge) {
  if (!edge || !edge.sites) return null;
  let sites;
  try {
    sites = JSON.parse(String(edge.sites));
  } catch (err) {
    return { error: "Phosphosite detail is malformed in this shard." };
  }
  if (!Array.isArray(sites) || sites.length !== _isNumeric(edge.n_sites)) {
    return { error: "Phosphosite detail does not reconcile with the edge site count." };
  }
  return sites.slice().sort((a, b) =>
    _isNumeric(b.kl_percentile) - _isNumeric(a.kl_percentile)
      || String(a.motif || "").localeCompare(String(b.motif || "")));
}

function _isPositionedElements(graph, width, height) {
  const arcCenter = _isArcCenter(width, height);
  const spineSegments = Math.max(_IS_LAYOUT.minimumSpineSegmentCount, graph.spine.length - 1);
  const sweepRad = _IS_LAYOUT.spineSweepDeg * Math.PI / 180;
  const startAngle = -Math.PI / 2 - sweepRad / 2;
  const roleAngle = new Map(graph.spine.map(node => [node.role,
    startAngle + node.order * sweepRad / spineSegments]));
  const rolePosition = new Map(graph.spine.map(node => {
    const angle = roleAngle.get(node.role);
    return [node.role, {
      x: arcCenter.x + _IS_LAYOUT.spineArcRadiusPx * Math.cos(angle),
      y: arcCenter.y + _IS_LAYOUT.spineArcRadiusPx * Math.sin(angle),
    }];
  }));
  const targetAngles = new Map();
  graph.terminalEdges.forEach(edge => {
    const gene = String(edge.source_gene);
    if (!targetAngles.has(gene)) targetAngles.set(gene, []);
    targetAngles.get(gene).push(roleAngle.get(String(edge.role)) || startAngle);
  });
  const chainAngles = new Map();
  const addChainAngles = (gene, angles) => {
    if (!angles || !angles.length) return;
    if (!chainAngles.has(gene)) chainAngles.set(gene, []);
    chainAngles.get(gene).push(...angles);
  };
  graph.chainEdges.forEach(edge => {
    const source = String(edge.source_gene);
    const target = String(edge.target_gene);
    addChainAngles(source, targetAngles.get(target));
    addChainAngles(target, targetAngles.get(source));
  });

  const elements = [];
  graph.spine.forEach(node => {
    elements.push({ data: { id: node.id, label: node.gene, kind: "spine-node", role: node.role },
      position: rolePosition.get(node.role) });
  });
  for (let i = 0; i < graph.spine.length - 1; i++) {
    elements.push({ data: {
      id: `spine:${graph.spine[i].role}:${graph.spine[i + 1].role}`,
      source: graph.spine[i].id, target: graph.spine[i + 1].id, kind: "spine-edge",
    } });
  }

  // Per-kinase node emphasis = its strongest terminal |NES| hit. Drives node size
  // (A) and radius in the fan (B): strong kinases hug the spine node, weak fan out.
  const kinaseEmphasis = new Map();
  graph.terminalEdges.forEach(edge => {
    const gene = String(edge.source_gene);
    const emphasis = _isEmphasis(edge.best_abs_nes, _IS_EMPHASIS.nesNull, graph.nesMax);
    kinaseEmphasis.set(gene, Math.max(kinaseEmphasis.get(gene) || 0, emphasis));
  });
  const anchorAngle = new Map();
  graph.kinaseGenes.forEach(gene => {
    const angles = targetAngles.get(gene) || chainAngles.get(gene) || [];
    anchorAngle.set(gene, angles.length ? _isMeanAngle(angles) : -Math.PI / 2);
  });
  // Each pathway node owns an outward radial wedge. Multi-target kinases use the
  // circular mean of their target wedges; all other kinases stay within their
  // node's wedge, preventing adjacent fans from interleaving.
  const groups = new Map();
  [...graph.kinaseGenes].forEach(gene => {
    const key = anchorAngle.get(gene).toFixed(12);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(gene);
  });
  const orderedKeys = [...groups.keys()].sort((a, b) => Number(a) - Number(b));
  orderedKeys.forEach(key => {
    const baseAngle = Number(key);
    const genes = groups.get(key).sort((a, b) =>
      (kinaseEmphasis.get(b) || 0) - (kinaseEmphasis.get(a) || 0) || a.localeCompare(b));
    let placed = 0;
    let ring = 0;
    while (placed < genes.length) {
      const radius = _IS_LAYOUT.fanInnerRadiusPx + ring * _IS_LAYOUT.fanRingStepPx;
      const wedgeWidth = 2 * _IS_LAYOUT.halfWedgeRad * (1 - 2 * _IS_LAYOUT.fanAngleMargin);
      const capacity = Math.max(1, Math.floor(wedgeWidth * radius / _IS_LAYOUT.fanArcSpacingPx));
      const ringGenes = genes.slice(placed, placed + capacity);
      ringGenes.forEach((gene, j) => {
        const fraction = ringGenes.length === 1 ? 0.5 : j / (ringGenes.length - 1);
        const wedgeOffset = (fraction - 0.5) * 2 * _IS_LAYOUT.halfWedgeRad
          * (1 - 2 * _IS_LAYOUT.fanAngleMargin);
        const theta = baseAngle + wedgeOffset;
        const emphasis = kinaseEmphasis.get(gene) || 0;
        elements.push({ data: {
          id: `kinase:${gene}`, label: gene, kind: "kinase-node",
          emphasis, size: _isEmphasisNodeSize(emphasis),
          direct_terminal: graph.directKinaseGenes.has(gene) ? 1 : 0,
        }, position: {
          x: arcCenter.x + (_IS_LAYOUT.spineArcRadiusPx + radius) * Math.cos(theta),
          y: arcCenter.y + (_IS_LAYOUT.spineArcRadiusPx + radius) * Math.sin(theta),
        } });
      });
      placed += ringGenes.length;
      ring += 1;
    }
  });

  const seenEdges = new Set();
  graph.chainEdges.forEach((edge, index) => {
    const source = `kinase:${String(edge.source_gene)}`;
    const target = `kinase:${String(edge.target_gene)}`;
    const key = `${source}|${target}|${edge.provenance || ""}`;
    if (seenEdges.has(key)) return;
    seenEdges.add(key);
    const emphasis = _isEmphasis(edge.weight, 0, graph.chainMax);
    elements.push({ data: {
      id: `chain:${index}`, source, target, kind: "chain-edge",
      provenance: String(edge.provenance || "motif"),
      weight: _isNumeric(edge.weight), width: _isEmphasisWidth(emphasis),
      opacity: _isEmphasisOpacity(emphasis),
    } });
  });
  graph.terminalEdges.forEach((edge, index) => {
    const role = String(edge.role);
    const nesEmphasis = _isEmphasis(edge.best_abs_nes, _IS_EMPHASIS.nesNull, graph.nesMax);
    const sites = _isNumeric(edge.n_sites);
    const siteFraction = graph.sitesMax > 0
      ? Math.max(0, Math.min(1, Math.log1p(sites) / Math.log1p(graph.sitesMax)))
      : 0;
    const siteFactor = _IS_EMPHASIS.siteFloor
      + (1 - _IS_EMPHASIS.siteFloor) * siteFraction;
    const emphasis = nesEmphasis * siteFactor;
    elements.push({ data: {
      id: `terminal:${index}`, source: `kinase:${String(edge.source_gene)}`,
      target: `path:${role}`, kind: "terminal-edge", role,
      provenance: String(edge.provenance || "motif"),
      width: _isEmphasisWidth(emphasis), opacity: _isEmphasisOpacity(emphasis),
      signed_nes: _isNumeric(edge.signed_nes), nes_direction: _isNesDirection(edge.signed_nes),
      best_abs_nes: _isNumeric(edge.best_abs_nes), best_fdr: _isNumeric(edge.best_fdr),
      n_sites: sites,
      sites: edge.sites || null,
      kinase: String(edge.kinase || ""),
      owning_cluster: String(edge.owning_cluster || ""),
      motif_peer: _isMotifPeerForEdge(edge),
    } });
  });
  return elements;
}

function _isRenderCytoscape(host, graph) {
  if (typeof window.cytoscape !== "function") {
    host.innerHTML = '<div class="muted">Sidechain graph unavailable because Cytoscape did not load.</div>';
    return;
  }
  if (host._incytrSidechainCleanup) host._incytrSidechainCleanup();
  if (host._incytrSidechainCy) host._incytrSidechainCy.destroy();
  const panel = document.createElement("div");
  panel.style.cssText = "position:relative;display:flex;gap:12px;align-items:stretch;"
    + "background:#fff;box-sizing:border-box;min-width:0;";
  const graphHost = document.createElement("div");
  graphHost.style.cssText = `flex:1 1 0;min-width:0;height:${_IS_LAYOUT.graphHeightPx}px;`
    + `border:${_IS_STYLE.graphBorderWidthPx}px solid #ddd;`
    + `border-radius:${_IS_STYLE.graphCornerRadiusPx}px;background:#fff;`;
  const fullscreenButton = document.createElement("button");
  fullscreenButton.type = "button";
  fullscreenButton.textContent = "⛶ Full screen";
  fullscreenButton.setAttribute("aria-label", "View sidechain graph full screen");
  fullscreenButton.style.cssText = "position:absolute;top:10px;right:10px;z-index:2;"
    + "padding:6px 9px;border:1px solid #94a3b8;border-radius:4px;background:#fff;"
    + "color:#1e3a5f;font-size:12px;font-weight:600;cursor:pointer;box-shadow:0 1px 3px #0002;";
  const caption = document.createElement("div");
  caption.style.cssText = `margin-top:${_IS_STYLE.captionTopMarginPx}px;`
    + `font-size:${_IS_STYLE.smallTextPx}px;color:#555;`;
  caption.textContent = `${graph.kinaseGenes.size.toLocaleString()} kinase regulators; `
    + `${graph.terminalEdges.length.toLocaleString()} kinase→node, `
    + `${graph.chainEdges.length.toLocaleString()} kinase→kinase links. `
    + "Tap a node to isolate its neighborhood; hover a kinase for its label.";
  const legend = document.createElement("div");
  legend.style.cssText = `font-size:${_IS_STYLE.smallTextPx}px;color:#334155;`;
  legend.append(
    _isLegendLine("Core pathway", [_isLegendSample(_IS_COLORS.corePath, "solid", 5)]),
    _isLegendLine("Kinase → node: enriched / depleted", [
      _isLegendSample(_IS_COLORS.enriched, "dotted"),
      _isLegendSample(_IS_COLORS.depleted, "dotted"),
    ]),
    _isLegendLine("Node size: |NES| strength", []),
    _isLegendLine("Width and opacity: |NES| × #substrates (edge strength)", []),
    _isLegendLine("Chain evidence: motif / PSP / both", [
      _isLegendSample(_IS_COLORS.motif),
      _isLegendSample(_IS_COLORS.psp, "dashed"),
      _isLegendSample(_IS_COLORS.both, "dotted"),
    ]),
  );
  const edgeDetail = document.createElement("div");
  edgeDetail.style.cssText = `margin-top:${_IS_STYLE.captionTopMarginPx}px;`
    + `font-size:${_IS_STYLE.smallTextPx}px;color:#334155;min-height:1.3em;`;
  edgeDetail.textContent = "Tap an edge for its evidence.";
  const nodeRelationDetail = document.createElement("div");
  nodeRelationDetail.style.cssText = `margin-top:${_IS_STYLE.captionTopMarginPx}px;`
    + `max-height:${_IS_STYLE.relationMaxHeightPx}px;overflow:auto;`
    + `font-size:${_IS_STYLE.smallTextPx}px;color:#334155;`;
  nodeRelationDetail.textContent = "Tap a node for its relationships.";
  const siteDetail = document.createElement("div");
  siteDetail.style.cssText = `margin-top:${_IS_STYLE.captionTopMarginPx}px;`
    + `max-height:${_IS_STYLE.siteMaxHeightPx}px;overflow:auto;`
    + `font-size:${_IS_STYLE.smallTextPx}px;color:#334155;`;
  siteDetail.textContent = "Tap a terminal edge for phosphosite evidence.";
  const filterControls = document.createElement("div");
  filterControls.style.cssText = `display:grid;gap:4px;`
    + `margin-bottom:${_IS_STYLE.captionTopMarginPx}px;font-size:${_IS_STYLE.smallTextPx}px;`
    + "font-weight:600;color:#1e3a5f;";
  const makeFilterLabel = () => {
    const label = document.createElement("label");
    label.style.cssText = "display:flex;align-items:center;gap:6px;cursor:pointer;";
    return label;
  };
  const chainFilter = makeFilterLabel();
  const showChains = document.createElement("input");
  showChains.type = "checkbox";
  showChains.checked = false;
  showChains.setAttribute("aria-label", "Show kinase to kinase chains");
  chainFilter.append(showChains, document.createTextNode("Show kinase→kinase chains"));
  filterControls.append(chainFilter);
  // Dedicated side panel: it remains inside the fullscreen container while
  // giving Cytoscape its own unobstructed width.
  const infoPanel = document.createElement("aside");
  infoPanel.style.cssText = `flex:0 0 ${_IS_STYLE.infoPanelWidthPx}px;box-sizing:border-box;`
    + "padding:8px 10px;border:1px solid #cbd5e1;border-radius:5px;"
    + "background:#fffffff0;overflow:auto;";
  infoPanel.setAttribute("aria-label", "Sidechain edge evidence");
  infoPanel.append(filterControls, legend, edgeDetail, nodeRelationDetail, siteDetail);
  panel.append(graphHost, infoPanel, fullscreenButton);
  host.replaceChildren(panel, caption);
  const cy = window.cytoscape({
    container: graphHost,
    elements: _isPositionedElements(graph, graphHost.clientWidth || _IS_LAYOUT.fallbackGraphWidthPx,
      graphHost.clientHeight || _IS_LAYOUT.graphHeightPx),
    style: [
      { selector: "node", style: {
        "label": "data(label)", "font-size": _IS_STYLE.labelFontPx, "text-wrap": "wrap", "text-max-width": _IS_STYLE.labelMaxWidthPx,
        "text-valign": "center", "text-halign": "center", "color": "#1f2933",
        "background-color": "#f8fafc", "border-width": _IS_STYLE.kinaseNodeBorderWidthPx, "border-color": "#64748b",
        "width": _IS_STYLE.kinaseNodeDiameterPx, "height": _IS_STYLE.kinaseNodeDiameterPx,
      } },
      { selector: "node[kind = 'kinase-node']", style: {
        "width": "data(size)", "height": "data(size)",
      } },
      { selector: `node[kind = 'kinase-node'][emphasis < ${_IS_FOCUS.labelEmphasisMin}]`, style: {
        "label": "",
      } },
      { selector: "node[kind = 'spine-node']", style: {
        "shape": "round-rectangle", "width": _IS_STYLE.spineNodeWidthPx, "height": _IS_STYLE.spineNodeHeightPx,
        "background-color": "#e7f0ff", "border-width": _IS_STYLE.spineNodeBorderWidthPx, "border-color": "#1f4ea3",
        "font-weight": "bold",
      } },
      { selector: "edge", style: {
        "curve-style": "bezier", "target-arrow-shape": "triangle",
        "arrow-scale": _IS_STYLE.normalArrowScale, "opacity": _IS_STYLE.edgeOpacity,
      } },
      { selector: "edge[kind = 'spine-edge']", style: {
        "width": _IS_STYLE.spineEdgeWidthPx, "line-color": _IS_COLORS.corePath,
        "target-arrow-color": _IS_COLORS.corePath, "arrow-scale": _IS_STYLE.emphasizedArrowScale,
        "curve-style": "unbundled-bezier", "control-point-distances": "-38px",
      } },
      { selector: "edge[kind = 'terminal-edge']", style: {
        "width": "data(width)", "opacity": "data(opacity)",
        "line-color": _IS_COLORS.neutral, "target-arrow-color": _IS_COLORS.neutral,
        "line-style": "dotted",
      } },
      { selector: "edge[kind = 'terminal-edge'][nes_direction = 'enriched']", style: {
        "line-color": _IS_COLORS.enriched, "target-arrow-color": _IS_COLORS.enriched,
      } },
      { selector: "edge[kind = 'terminal-edge'][nes_direction = 'depleted']", style: {
        "line-color": _IS_COLORS.depleted, "target-arrow-color": _IS_COLORS.depleted,
      } },
      { selector: "edge[kind = 'chain-edge'][provenance = 'motif']", style: {
        "width": "data(width)", "opacity": "data(opacity)",
        "line-color": _IS_COLORS.motif, "target-arrow-color": _IS_COLORS.motif,
      } },
      { selector: "edge[kind = 'chain-edge'][provenance = 'psp']", style: {
        "width": "data(width)", "opacity": "data(opacity)",
        "line-color": _IS_COLORS.psp, "target-arrow-color": _IS_COLORS.psp,
        "line-style": "dashed",
      } },
      { selector: "edge[kind = 'chain-edge'][provenance = 'both']", style: {
        "width": "data(width)", "opacity": "data(opacity)",
        "line-color": _IS_COLORS.both, "target-arrow-color": _IS_COLORS.both,
        "line-style": "dotted", "arrow-scale": _IS_STYLE.emphasizedArrowScale,
      } },
      // Focus/hover (C): reveal a weak node's label on hover or when tapped; a tap
      // hard-hides everything outside the tapped node's neighborhood (not dimmed).
      { selector: "node.is-hover, node.is-focus", style: { "label": "data(label)" } },
      { selector: "node.is-focus", style: {
        "border-color": _IS_FOCUS.focusBorderColor, "border-width": _IS_FOCUS.focusBorderWidthPx,
      } },
      // A focus has only a handful of edges. Reset their opacity so the selected
      // relationship is readable instead of inheriting full-graph attenuation.
      { selector: "edge.is-focus-edge", style: { "opacity": 1 } },
      { selector: ".is-hidden, .is-chain-filtered, .is-node-filtered", style: {
        "display": "none",
      } },
    ],
    layout: { name: "preset", fit: true, padding: _IS_LAYOUT.fitPaddingPx },
    minZoom: _IS_LAYOUT.minZoom,
    maxZoom: _IS_LAYOUT.maxZoom,
  });
  const spineNodes = cy.nodes("node[kind = 'spine-node']");
  let spineCenterX = 0;
  let spineCenterY = 0;
  spineNodes.forEach(node => {
    const position = node.renderedPosition();
    spineCenterX += position.x;
    spineCenterY += position.y;
  });
  if (spineNodes.length) {
    cy.panBy({
      x: cy.width() / 2 - spineCenterX / spineNodes.length,
      y: cy.height() / 2 - spineCenterY / spineNodes.length,
    });
  }
  const resetGraphSize = () => {
    const isFullscreen = document.fullscreenElement === panel;
    graphHost.style.height = isFullscreen
      ? `${Math.max(_IS_LAYOUT.graphHeightPx, window.innerHeight - 180)}px`
      : `${_IS_LAYOUT.graphHeightPx}px`;
    panel.style.padding = isFullscreen ? "44px 12px 12px" : "0";
    panel.style.minHeight = isFullscreen ? "100vh" : "0";
    fullscreenButton.textContent = isFullscreen ? "⛶ Exit full screen" : "⛶ Full screen";
    fullscreenButton.setAttribute("aria-label", isFullscreen
      ? "Exit sidechain graph full screen" : "View sidechain graph full screen");
    cy.resize();
    cy.fit(visibleElements(cy.elements()), _IS_LAYOUT.fitPaddingPx);
  };
  const requestFullscreen = () => {
    if (document.fullscreenElement === panel) {
      document.exitFullscreen();
    } else if (panel.requestFullscreen) {
      panel.requestFullscreen();
    }
  };
  fullscreenButton.addEventListener("click", requestFullscreen);
  document.addEventListener("fullscreenchange", resetGraphSize);
  window.addEventListener("resize", resetGraphSize);
  const spine = cy.elements("node[kind = 'spine-node'], edge[kind = 'spine-edge']");
  const visibleElements = elements => elements.filter(element =>
    !element.hasClass("is-chain-filtered")
      && !element.hasClass("is-node-filtered"));
  const updateCaption = () => {
    const visibleKinaseCount = visibleElements(cy.nodes("node[kind = 'kinase-node']")).length;
    const visibleTerminalCount = visibleElements(cy.edges("edge[kind = 'terminal-edge']")).length;
    const visibleChainCount = visibleElements(cy.edges("edge[kind = 'chain-edge']")).length;
    const chainState = showChains.checked ? "Full kinase chain view." :
      "First-order kinase→gene view.";
    caption.textContent = `${visibleKinaseCount.toLocaleString()} kinase regulators; `
      + `${visibleTerminalCount.toLocaleString()} kinase→node, `
      + `${visibleChainCount.toLocaleString()} kinase→kinase links. All motif / PSP evidence. ${chainState} `
      + "Tap a node or edge to isolate its neighborhood; hover a kinase for its label.";
  };
  const applyFilters = () => {
    const chains = cy.edges("edge[kind = 'chain-edge']");
    if (showChains.checked) {
      chains.removeClass("is-chain-filtered");
    } else {
      chains.addClass("is-chain-filtered");
    }
    cy.nodes("node[kind = 'kinase-node']").forEach(node => {
      let hasVisibleEvidence = false;
      visibleElements(node.connectedEdges()).forEach(edge => {
        if (edge.data("kind") === "terminal-edge" || edge.data("kind") === "chain-edge") {
          hasVisibleEvidence = true;
        }
      });
      if (hasVisibleEvidence) {
        node.removeClass("is-node-filtered");
      } else {
        node.addClass("is-node-filtered");
      }
    });
    cy.elements().removeClass("is-hidden is-focus is-focus-edge");
    cy.fit(visibleElements(cy.elements()), _IS_LAYOUT.fitPaddingPx);
    edgeDetail.textContent = "Tap an edge for its evidence.";
    nodeRelationDetail.textContent = "Tap a node for its relationships.";
    siteDetail.textContent = "Tap a terminal edge for phosphosite evidence.";
    updateCaption();
  };
  showChains.addEventListener("change", applyFilters);
  applyFilters();
  const focus = (keep, detail) => {
    cy.elements().addClass("is-hidden").removeClass("is-focus is-focus-edge");
    keep.removeClass("is-hidden");
    keep.nodes().addClass("is-focus");
    keep.edges().addClass("is-focus-edge");
    cy.fit(visibleElements(keep), _IS_LAYOUT.fitPaddingPx);
    if (detail) edgeDetail.textContent = detail;
  };
  const renderTerminalSites = (edge, substrate, contrast) => {
    const sites = _isTerminalSiteRows({
      sites: edge.data("sites"), n_sites: edge.data("n_sites"),
    });
    if (sites === null) {
      siteDetail.textContent = "This shard has no inline phosphosite detail.";
      return;
    }
    if (!Array.isArray(sites)) {
      siteDetail.textContent = sites.error;
      return;
    }
    const wrapper = document.createElement("div");
    const heading = document.createElement("div");
    heading.style.cssText = "font-weight:700;color:#1e3a5f;padding-bottom:4px;";
    heading.textContent = `${substrate} phosphosites · ${contrast}`;
    wrapper.appendChild(heading);
    const summary = document.createElement("div");
    summary.style.cssText = "padding-bottom:5px;";
    summary.textContent = `NES ${_isNumeric(edge.data("signed_nes")).toFixed(3)} · `
      + `FDR ${_isNumeric(edge.data("best_fdr")).toPrecision(3)} · `
      + `|NES| ${_isNumeric(edge.data("best_abs_nes")).toFixed(3)}`;
    wrapper.appendChild(summary);
    const table = document.createElement("table");
    table.setAttribute("aria-label", `${substrate} floor-99 phosphosites`);
    table.style.cssText = "border-collapse:collapse;width:100%;font-size:11px;";
    const head = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["Motif", "Residue", "KL percentile"].forEach(label => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      cell.style.cssText = "padding:2px 5px;text-align:left;white-space:nowrap;";
      headerRow.appendChild(cell);
    });
    head.appendChild(headerRow);
    table.appendChild(head);
    const body = document.createElement("tbody");
    sites.forEach(site => {
      const row = document.createElement("tr");
      [String(site.motif || ""), String(site.residue_type || ""),
        _isNumeric(site.kl_percentile).toFixed(2)].forEach(value => {
        const cell = document.createElement("td");
        cell.textContent = value;
        cell.style.cssText = "padding:2px 5px;border-top:1px solid #e2e8f0;";
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
    table.appendChild(body);
    wrapper.appendChild(table);
    siteDetail.replaceChildren(wrapper);
  };
  cy.on("tap", "edge", evt => {
    const edge = evt.target;
    const source = String(edge.source().data("label") || edge.data("source"));
    const target = String(edge.target().data("label") || edge.data("target"));
    if (edge.data("kind") === "terminal-edge") {
      const signedNes = _isNumeric(edge.data("signed_nes"));
      const direction = String(edge.data("nes_direction"));
      const detail = `${source} → ${target}: NES ${signedNes.toFixed(3)} `
        + `(${direction}), FDR ${_isNumeric(edge.data("best_fdr")).toPrecision(3)}, `
        + `|NES| ${_isNumeric(edge.data("best_abs_nes")).toFixed(3)}, `
        + `sites ${_isNumeric(edge.data("n_sites"))}.`;
      focus(edge.closedNeighborhood().union(spine), detail);
      renderTerminalSites(edge, target, String(edge.data("contrast") || ""));
      return;
    }
    const detail = `${source} → ${target}: ${String(edge.data("provenance"))} `
      + `evidence, weight ${_isNumeric(edge.data("weight")).toFixed(3)}.`;
    focus(edge.closedNeighborhood().union(spine), detail);
    siteDetail.textContent = "Tap a terminal edge for phosphosite evidence.";
  });
  cy.on("mouseover", "node[kind = 'kinase-node']", evt => evt.target.addClass("is-hover"));
  cy.on("mouseout", "node[kind = 'kinase-node']", evt => evt.target.removeClass("is-hover"));
  cy.on("tap", "node", evt => {
    // Tap any node → keep only its neighborhood + the pathway spine; hard-hide the
    // rest so no faint edges remain, then zoom to what's left.
    nodeRelationDetail.replaceChildren(_isNodeRelationTable(evt.target, cy));
    siteDetail.textContent = "Tap a terminal edge for phosphosite evidence.";
    focus(evt.target.closedNeighborhood().union(spine));
  });
  cy.on("tap", evt => {
    if (evt.target === cy) {
      cy.elements().removeClass("is-hidden is-focus is-focus-edge");
      cy.fit(visibleElements(cy.elements()), _IS_LAYOUT.fitPaddingPx);
      edgeDetail.textContent = "Tap an edge for its evidence.";
      nodeRelationDetail.textContent = "Tap a node for its relationships.";
      siteDetail.textContent = "Tap a terminal edge for phosphosite evidence.";
    }
  });
  host._incytrSidechainCy = cy;
  host._incytrSidechainCleanup = () => {
    fullscreenButton.removeEventListener("click", requestFullscreen);
    document.removeEventListener("fullscreenchange", resetGraphSize);
    window.removeEventListener("resize", resetGraphSize);
    showChains.removeEventListener("change", applyFilters);
  };
}

// Called by the pathways detail-panel switcher after it has mounted the host.
async function _isRenderSidechains(rk, row) {
  const hostId = _isHostId(rk);
  const host = document.getElementById(hostId);
  if (!host || !row) return;
  if (!IncytrSidechains.hasActiveSlice()) {
    host.innerHTML = '<div class="muted">Sidechain data are unavailable for this context.</div>';
    return;
  }
  host.innerHTML = `<div class="muted" style="font-size:${_IS_STYLE.smallTextPx}px;`
    + `padding:${_IS_STYLE.loadingVerticalPaddingPx}px 0;">Loading kinase sidechains…</div>`;
  try {
    const shard = await IncytrSidechains.loadActiveShard();
    const currentHost = document.getElementById(hostId);
    if (!currentHost) return;
    if (!shard) {
      currentHost.innerHTML = '<div class="muted">Sidechain data are unavailable for this context.</div>';
      return;
    }
    _isRenderCytoscape(currentHost, _isGraphForRow(shard, row));
  } catch (err) {
    const currentHost = document.getElementById(hostId);
    if (currentHost) {
      currentHost.innerHTML = `<div class="muted">Sidechain graph load error: ${_escapeHtml(String(err && err.message ? err.message : err))}</div>`;
    }
  }
}
