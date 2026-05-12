// ---------------------------------------------------------------------------
// Incytr Heatmap tab — 22×22 sender×receiver candidate-path counts under a
// chosen significance tier (all / p05 / paper / strict) and contrast.
//
// Reads PAYLOAD.incytr_pathways:
//   senders, receivers, contrasts           (label arrays)
//   empty_deg_celltypes                     (no-data cell types, hatched)
//   heatmap_tiers[tier].counts              (Uint32, length S*R*C, sender-major)
//   heatmap_tiers[tier].gate                (gate dict, e.g. {p:0.05, pds:0.76})
//   heatmap_tiers[tier].label, .total
//   default_tier                            (initial tier selection)
//
// Click → switches to the Pathway Table tab (Phase 3) with sender, receiver,
// and contrast filters pre-set. For now we stash the requested filter on the
// store and log; the table tab handler will read it when it lands.
// ---------------------------------------------------------------------------

const _ihState = {
  tier:     null,   // initialized from PAYLOAD.incytr_pathways.default_tier
  contrast: null,   // initialized to first contrast
};

function _ihBlock() {
  return (typeof PAYLOAD !== "undefined" && PAYLOAD.incytr_pathways) || null;
}

function _ihCountAt(tier, sIdx, rIdx, cIdx) {
  const block = _ihBlock();
  if (!block) return 0;
  const counts = block.heatmap_tiers[tier] && block.heatmap_tiers[tier].counts;
  if (!counts) return 0;
  const nR = block.receivers.length, nC = block.contrasts.length;
  return counts[sIdx * nR * nC + rIdx * nC + cIdx];
}

function _ihTierGateLabel(gate) {
  if (!gate) return "";
  const parts = [];
  if (gate.p   != null) parts.push("pvalue<"   + gate.p);
  if (gate.pds != null) parts.push("|PDS|>"    + gate.pds);
  if (gate.sp  != null) parts.push("sigprob>"  + gate.sp);
  return parts.length ? parts.join(" ∧ ") : "no gate";
}

function _ihBuildControls() {
  const block = _ihBlock();
  if (!block) return;
  const tiersWrap = document.getElementById("ih-tier-radios");
  const contWrap  = document.getElementById("ih-contrast-radios");
  if (!tiersWrap || !contWrap) return;

  const tierOrder = ["all", "p05", "paper", "strict"];
  const tiersPresent = tierOrder.filter(t => block.heatmap_tiers[t]);
  if (_ihState.tier == null) {
    _ihState.tier = (block.default_tier && block.heatmap_tiers[block.default_tier])
      ? block.default_tier
      : tiersPresent[0];
  }
  tiersWrap.innerHTML = tiersPresent.map(t => {
    const meta = block.heatmap_tiers[t];
    const on = (t === _ihState.tier);
    const total = (meta.total || 0).toLocaleString();
    const title = `${_escapeHtml(meta.label || t)} — ${_ihTierGateLabel(meta.gate)} · ${total} rows`;
    return `<button class="chip${on ? " active" : ""}" data-ih-tier="${t}" title="${title}">`
      + `${_escapeHtml(t)} <span class="muted" style="margin-left:4px;">(${total})</span>`
      + `</button>`;
  }).join("");

  if (_ihState.contrast == null) _ihState.contrast = block.contrasts[0];
  contWrap.innerHTML = block.contrasts.map(c => {
    const on = (c === _ihState.contrast);
    return `<button class="chip${on ? " active" : ""}" data-ih-contrast="${c}">`
      + _escapeHtml(c) + `</button>`;
  }).join("");
}

function wireIncytrHeatmap() {
  const tiersWrap = document.getElementById("ih-tier-radios");
  const contWrap  = document.getElementById("ih-contrast-radios");
  if (tiersWrap) tiersWrap.addEventListener("click", ev => {
    const btn = ev.target.closest("[data-ih-tier]");
    if (!btn) return;
    _ihState.tier = btn.dataset.ihTier;
    _ihBuildControls();
    _ihRenderPlot();
  });
  if (contWrap) contWrap.addEventListener("click", ev => {
    const btn = ev.target.closest("[data-ih-contrast]");
    if (!btn) return;
    _ihState.contrast = btn.dataset.ihContrast;
    _ihBuildControls();
    _ihRenderPlot();
  });
}

function _ihColorscale() {
  // Sequential white → deep indigo. Zero is anchored to white so empty cells
  // and below-gate cells render as background.
  return [
    [0.00, "#ffffff"],
    [0.05, "#f0f4ff"],
    [0.20, "#c3cdf0"],
    [0.45, "#8b9add"],
    [0.70, "#5563b8"],
    [1.00, "#1f2960"],
  ];
}

function _ihRenderPlot() {
  const block = _ihBlock();
  const el = document.getElementById("ih-plot");
  const sub = document.getElementById("ih-subtitle");
  if (!block || !el) return;
  const tier = _ihState.tier;
  const tierMeta = block.heatmap_tiers[tier];
  const contrast = _ihState.contrast;
  const cIdx = block.contrasts.indexOf(contrast);
  const senders = block.senders, receivers = block.receivers;
  const nS = senders.length, nR = receivers.length;
  const empty = new Set(block.empty_deg_celltypes || []);

  // Z matrix: counts where the cell can carry data, NaN where empty-DEG. NaN
  // gates pure white via showscale: cells that are 0-but-present still render
  // as the colorscale's zero (also white) — distinguished only by tooltip and
  // by the hatched overlay below.
  const Z = [];
  const text = [];
  let maxN = 0;
  for (let r = 0; r < nR; r++) {
    const row = [];
    const tRow = [];
    for (let s = 0; s < nS; s++) {
      const isEmpty = empty.has(senders[s]) || empty.has(receivers[r]);
      if (isEmpty) {
        row.push(null);
        tRow.push("no candidate paths (empty-DEG cell type)");
      } else {
        const n = _ihCountAt(tier, s, r, cIdx);
        row.push(n);
        if (n > maxN) maxN = n;
        tRow.push(`${n.toLocaleString()} rows pass gate`);
      }
    }
    Z.push(row);
    text.push(tRow);
  }

  const traces = [{
    type: "heatmap",
    x: senders, y: receivers, z: Z,
    text, hoverinfo: "x+y+text",
    colorscale: _ihColorscale(),
    zmin: 0, zmax: Math.max(1, maxN),
    colorbar: { title: "n rows", thickness: 12, len: 0.7 },
    xgap: 1, ygap: 1,
  }];

  // Hatched overlay for empty-DEG cells — a second scatter trace with a
  // distinctive marker so the user sees "no data" instead of "0 below gate."
  const hx = [], hy = [], htxt = [];
  for (let r = 0; r < nR; r++) for (let s = 0; s < nS; s++) {
    if (empty.has(senders[s]) || empty.has(receivers[r])) {
      hx.push(senders[s]); hy.push(receivers[r]);
      htxt.push("no candidate paths (empty-DEG cell type)");
    }
  }
  if (hx.length) {
    traces.push({
      type: "scatter", mode: "markers",
      x: hx, y: hy, text: htxt, hoverinfo: "x+y+text",
      marker: { symbol: "x-thin-open", color: "#9aa0a6", size: 10, line: { width: 1.5 } },
      showlegend: false,
    });
  }

  const layout = {
    margin: { l: 110, r: 30, t: 20, b: 90 },
    height: Math.max(520, 22 * Math.max(nS, nR) + 160),
    xaxis: { title: "Sender", tickangle: -45, automargin: true,
             categoryorder: "array", categoryarray: senders },
    yaxis: { title: "Receiver", automargin: true,
             categoryorder: "array", categoryarray: receivers, autorange: "reversed" },
    plot_bgcolor: "#fafafa",
  };
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const sender = p.x, receiver = p.y;
    if (empty.has(sender) || empty.has(receiver)) return;
    // Phase 3 will pick this up and render the per-pair shard. For now we
    // stash the request on the store; the future incytr_pathways tab reads it.
    Store.dispatch({type:"SET_VIEW", key:"pendingIncytrFilter",
      value:{ sender, receiver, contrast: _ihState.contrast, tier: _ihState.tier }});
    console.info("incytr-heatmap click →", { sender, receiver, contrast: _ihState.contrast, tier: _ihState.tier });
  });

  if (sub) {
    const gate = _ihTierGateLabel(tierMeta && tierMeta.gate);
    const total = (tierMeta && tierMeta.total || 0).toLocaleString();
    sub.textContent = `Tier "${tier}" (${gate}) · contrast ${contrast} · `
      + `${total} rows pass gate across all pairs. Click a cell to drill into the table tab (Phase 3).`;
  }
}

function renderIncytrHeatmap() {
  if (!_ihBlock()) {
    const el = document.getElementById("ih-plot");
    if (el) el.innerHTML =
      '<div class="muted" style="padding:24px;">No <code>incytr_pathways</code> '
      + 'block in the payload. Rebuild with <code>pixi run viewer</code> after '
      + 'producing <code>data/incytr_factorial_outputs/receiver_cache/</code>.</div>';
    return;
  }
  _ihBuildControls();
  _ihRenderPlot();
}
