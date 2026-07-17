// ---------------------------------------------------------------------------
// Incytr pathway sidechains — a row-detail Cytoscape view.
//
// The lazy shard is context-scoped and columnar. This module keeps its own
// small cache because each context has one compact shard shared by every
// pathway row; individual rows only select and walk that loaded graph.
// ---------------------------------------------------------------------------
const IncytrSidechains = (function(){
  let indexUrl = null;
  let index = null;
  let indexPromise = null;
  const shards = new Map();

  function _refs() {
    return (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.edge_slice_ref) || {};
  }

  async function _fetchJson(url) {
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
    return response.json();
  }

  async function _fetchGzipJson(url) {
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
    const url = _refs().incytr_sidechains_index;
    if (!url) return Promise.resolve(null);
    if (url !== indexUrl) {
      indexUrl = url;
      index = null;
      indexPromise = null;
      shards.clear();
    }
    if (index) return Promise.resolve(index);
    if (indexPromise) return indexPromise;
    indexPromise = _fetchJson(url).then(data => {
      if (!data || data.schema_version !== 1
          || data.slice_type !== "incytr_kinase_sidechains"
          || !data.by_context || typeof data.by_context !== "object") {
        throw new Error("sidechain index has an invalid schema");
      }
      index = data;
      return index;
    }).finally(() => { indexPromise = null; });
    return indexPromise;
  }

  function activeEntry() {
    const context = ViewerPayload.activeContext();
    return index && index.by_context ? index.by_context[context] || null : null;
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
    if (shards.has(key)) return shards.get(key);
    const promise = _fetchGzipJson(entry.url).then(data => {
      if (!data || data.schema_version !== 1
          || data.slice_type !== "incytr_kinase_sidechains"
          || data.context_id !== context) {
        throw new Error("sidechain shard has an invalid schema");
      }
      return data;
    }).catch(err => {
      shards.delete(key);
      throw err;
    });
    shards.set(key, promise);
    return promise;
  }

  return { ensureIndex, hasActiveSlice, loadActiveShard };
})();

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

function _isEdgeWidth(weight, observedMax) {
  const minimumVisibleWidth = 1.25;
  const maximumRenderedWidth = 7;
  if (!(observedMax > 0)) return minimumVisibleWidth;
  const proportion = Math.max(0, Math.min(1, _isNumeric(weight) / observedMax));
  return minimumVisibleWidth + proportion * (maximumRenderedWidth - minimumVisibleWidth);
}

function _isHash(value) {
  let hash = 0;
  const text = String(value || "");
  for (let i = 0; i < text.length; i++) hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  return Math.abs(hash);
}

function _isGraphForRow(shard, row) {
  const interactome = _isSafeRows(shard.interactome, [
    "source_gene", "target_gene", "provenance", "weight",
  ]).filter(edge => edge.source_gene && edge.target_gene);
  const terminal = _isSafeRows(shard.terminal_edges, [
    "source_gene", "target_gene", "role", "contrast", "provenance", "weight",
  ]).filter(edge => edge.source_gene && edge.target_gene);
  const spine = _isSpineNodes(row);
  const spineByRole = new Map(spine.map(node => [node.role, node]));
  const terminalEdges = terminal.filter(edge => {
    const node = spineByRole.get(String(edge.role || ""));
    return !!node && String(edge.target_gene).toUpperCase() === node.gene.toUpperCase()
      && String(edge.contrast || "") === String(row.contrast || "");
  });
  const roots = new Set(terminalEdges.map(edge => String(edge.source_gene)));
  const incoming = new Map();
  interactome.forEach((edge, index) => {
    const target = String(edge.target_gene);
    if (!incoming.has(target)) incoming.set(target, []);
    incoming.get(target).push(index);
  });

  // Reverse traversal follows source → target kinase edges upstream from the
  // terminal kinases. A visited-node set makes cycles finite while preserving
  // every edge in the reachable subgraph.
  const queue = [...roots];
  const seenNodes = new Set(queue);
  const selectedIndices = new Set();
  const distance = new Map(queue.map(gene => [gene, 0]));
  while (queue.length) {
    const target = queue.shift();
    const targetDistance = distance.get(target) || 0;
    for (const index of incoming.get(target) || []) {
      selectedIndices.add(index);
      const source = String(interactome[index].source_gene);
      const nextDistance = targetDistance + 1;
      if (!distance.has(source) || nextDistance < distance.get(source)) {
        distance.set(source, nextDistance);
      }
      if (!seenNodes.has(source)) {
        seenNodes.add(source);
        queue.push(source);
      }
    }
  }
  const chainEdges = [...selectedIndices].map(index => interactome[index]);
  const kinaseGenes = new Set(roots);
  chainEdges.forEach(edge => {
    kinaseGenes.add(String(edge.source_gene));
    kinaseGenes.add(String(edge.target_gene));
  });
  const observedMax = interactome.concat(terminalEdges).reduce(
    (maximum, edge) => Math.max(maximum, _isNumeric(edge.weight)), 0);
  return { spine, terminalEdges, chainEdges, kinaseGenes, distance, observedMax };
}

function _isPositionedElements(graph, width, height) {
  const middle = Math.max(180, height / 2);
  const left = 130;
  const right = Math.max(left + 480, width - 130);
  const gap = (right - left) / 3;
  const roleX = new Map(graph.spine.map(node => [node.role, left + node.order * gap]));
  const anchors = new Map();
  graph.terminalEdges.forEach(edge => {
    const gene = String(edge.source_gene);
    if (!anchors.has(gene)) anchors.set(gene, []);
    anchors.get(gene).push(roleX.get(String(edge.role)) || left);
  });

  const elements = [];
  graph.spine.forEach(node => {
    elements.push({ data: { id: node.id, label: node.gene, kind: "spine-node", role: node.role },
      position: { x: roleX.get(node.role), y: middle } });
  });
  for (let i = 0; i < graph.spine.length - 1; i++) {
    elements.push({ data: {
      id: `spine:${graph.spine[i].role}:${graph.spine[i + 1].role}`,
      source: graph.spine[i].id, target: graph.spine[i + 1].id, kind: "spine-edge",
    } });
  }

  const kinaseGenes = [...graph.kinaseGenes].sort((a, b) => a.localeCompare(b));
  kinaseGenes.forEach((gene, index) => {
    const terminalAnchors = anchors.get(gene) || [];
    const anchor = terminalAnchors.length
      ? terminalAnchors.reduce((sum, x) => sum + x, 0) / terminalAnchors.length
      : left + ((_isHash(gene) + index) % 4) * gap;
    const depth = graph.distance.get(gene) || 0;
    const lane = (_isHash(gene) % 9) - 4;
    const y = middle + (lane === 0 ? (index % 2 ? -1 : 1) : lane) * (38 + depth * 8);
    elements.push({ data: { id: `kinase:${gene}`, label: gene, kind: "kinase-node" },
      position: { x: anchor - depth * 54, y } });
  });

  const seenEdges = new Set();
  graph.chainEdges.forEach((edge, index) => {
    const source = `kinase:${String(edge.source_gene)}`;
    const target = `kinase:${String(edge.target_gene)}`;
    const key = `${source}|${target}|${edge.provenance || ""}`;
    if (seenEdges.has(key)) return;
    seenEdges.add(key);
    elements.push({ data: {
      id: `chain:${index}`, source, target, kind: "chain-edge",
      provenance: String(edge.provenance || "motif"),
      width: _isEdgeWidth(edge.weight, graph.observedMax),
    } });
  });
  graph.terminalEdges.forEach((edge, index) => {
    const role = String(edge.role);
    elements.push({ data: {
      id: `terminal:${index}`, source: `kinase:${String(edge.source_gene)}`,
      target: `path:${role}`, kind: "terminal-edge",
      width: _isEdgeWidth(edge.weight, graph.observedMax),
    } });
  });
  return elements;
}

function _isRenderCytoscape(host, graph) {
  if (typeof window.cytoscape !== "function") {
    host.innerHTML = '<div class="muted">Sidechain graph unavailable because Cytoscape did not load.</div>';
    return;
  }
  if (host._incytrSidechainCy) host._incytrSidechainCy.destroy();
  const graphHost = document.createElement("div");
  graphHost.style.cssText = "width:100%;height:500px;border:1px solid #ddd;border-radius:4px;background:#fff;";
  const caption = document.createElement("div");
  caption.style.cssText = "margin-top:6px;font-size:11px;color:#555;";
  caption.textContent = `${graph.terminalEdges.length.toLocaleString()} contrast-matched terminal link(s); `
    + `${graph.chainEdges.length.toLocaleString()} upstream kinase edge(s). `
    + "Chain edge color: motif (blue), PSP (orange), both (purple).";
  host.replaceChildren(graphHost, caption);
  const cy = window.cytoscape({
    container: graphHost,
    elements: _isPositionedElements(graph, graphHost.clientWidth || 900, graphHost.clientHeight || 500),
    style: [
      { selector: "node", style: {
        "label": "data(label)", "font-size": 10, "text-wrap": "wrap", "text-max-width": 96,
        "text-valign": "center", "text-halign": "center", "color": "#1f2933",
        "background-color": "#f8fafc", "border-width": 1, "border-color": "#64748b",
        "width": 34, "height": 34,
      } },
      { selector: "node[kind = 'spine-node']", style: {
        "shape": "round-rectangle", "width": 100, "height": 42,
        "background-color": "#e7f0ff", "border-width": 2, "border-color": "#1f4ea3",
        "font-weight": "bold",
      } },
      { selector: "edge", style: {
        "curve-style": "bezier", "target-arrow-shape": "triangle",
        "arrow-scale": 0.7, "opacity": 0.9,
      } },
      { selector: "edge[kind = 'spine-edge']", style: {
        "width": 5, "line-color": "#1f4ea3", "target-arrow-color": "#1f4ea3",
        "arrow-scale": 1,
      } },
      { selector: "edge[kind = 'terminal-edge']", style: {
        "width": "data(width)", "line-color": "#475569", "target-arrow-color": "#475569",
        "line-style": "dotted",
      } },
      { selector: "edge[kind = 'chain-edge'][provenance = 'motif']", style: {
        "width": "data(width)", "line-color": "#2563eb", "target-arrow-color": "#2563eb",
      } },
      { selector: "edge[kind = 'chain-edge'][provenance = 'psp']", style: {
        "width": "data(width)", "line-color": "#d97706", "target-arrow-color": "#d97706",
        "line-style": "dashed",
      } },
      { selector: "edge[kind = 'chain-edge'][provenance = 'both']", style: {
        "width": "data(width)", "line-color": "#7e22ce", "target-arrow-color": "#7e22ce",
        "line-style": "dotted", "arrow-scale": 1,
      } },
    ],
    layout: { name: "preset", fit: true, padding: 48 },
    minZoom: 0.08,
    maxZoom: 3,
  });
  host._incytrSidechainCy = cy;
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
  host.innerHTML = '<div class="muted" style="font-size:11px;padding:8px 0;">Loading kinase sidechains…</div>';
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
