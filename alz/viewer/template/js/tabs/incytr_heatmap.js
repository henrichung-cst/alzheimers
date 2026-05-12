// ---------------------------------------------------------------------------
// Incytr Heatmap tab — 22×22 sender×receiver candidate-path counts under a
// chosen significance tier and contrast. Filter UI mirrors the Kinase tab:
// ordinal-threshold <select> for tier; dual <select>s (Disease × Timepoint)
// for contrast; Reset button + live count line.
//
// State lives in IncytrFilter (shared with the Pathway table tab) so picks
// flow across tabs. Click on a heatmap cell → seeds the table tab's
// senderIn / receiverIn / disease / timepoint filters and switches tabs.
// ---------------------------------------------------------------------------

const _IH_DISEASES = ["App", "Tau", "ApTt"];
const _IH_TIMEPOINTS = ["2mo", "4mo", "6mo"];

function _ihBlock() {
  return (typeof PAYLOAD !== "undefined" && PAYLOAD.incytr_pathways) || null;
}

function _ihContrastFromState() {
  const f = IncytrFilter.get();
  return `${f.hmDisease}_${f.hmTimepoint}`;
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
  if (gate.p   != null) parts.push("p<"   + gate.p);
  if (gate.pds != null) parts.push("|PDS|>"    + gate.pds);
  if (gate.sp  != null) parts.push("sp>"  + gate.sp);
  return parts.length ? parts.join(" ∧ ") : "no gate";
}

function _ihTierOptionLabel(tier, meta) {
  const total = (meta.total || 0).toLocaleString();
  return `${tier} — ${_ihTierGateLabel(meta.gate)} (${total})`;
}

function _ihSyncControls() {
  const block = _ihBlock();
  if (!block) return;
  const f = IncytrFilter.get();

  const tierSel = document.getElementById("ih-tier");
  if (tierSel) {
    const order = ["all", "p05", "paper", "strict"]
      .filter(t => block.heatmap_tiers[t]);
    tierSel.innerHTML = order.map(t =>
      `<option value="${_escapeHtml(t)}">${_escapeHtml(_ihTierOptionLabel(t, block.heatmap_tiers[t]))}</option>`
    ).join("");
    tierSel.value = (f.hmTier && block.heatmap_tiers[f.hmTier]) ? f.hmTier : order[0];
  }
  const dSel = document.getElementById("ih-disease");
  if (dSel) {
    dSel.innerHTML = _IH_DISEASES.map(d =>
      `<option value="${_escapeHtml(d)}">${_escapeHtml(d)}</option>`
    ).join("");
    dSel.value = f.hmDisease;
  }
  const tSel = document.getElementById("ih-timepoint");
  if (tSel) {
    tSel.innerHTML = _IH_TIMEPOINTS.map(t =>
      `<option value="${_escapeHtml(t)}">${_escapeHtml(t)}</option>`
    ).join("");
    tSel.value = f.hmTimepoint;
  }
}

function wireIncytrHeatmap() {
  const tierSel = document.getElementById("ih-tier");
  if (tierSel) tierSel.addEventListener("change", () => {
    IncytrFilter.set({ hmTier: tierSel.value });
    _ihSyncControls();
    _ihRenderPlot();
  });
  const dSel = document.getElementById("ih-disease");
  if (dSel) dSel.addEventListener("change", () => {
    IncytrFilter.set({ hmDisease: dSel.value });
    _ihRenderPlot();
  });
  const tSel = document.getElementById("ih-timepoint");
  if (tSel) tSel.addEventListener("change", () => {
    IncytrFilter.set({ hmTimepoint: tSel.value });
    _ihRenderPlot();
  });
  const resetBtn = document.getElementById("ih-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    IncytrFilter.set({ hmTier: "paper", hmDisease: "App", hmTimepoint: "2mo" });
    _ihSyncControls();
    _ihRenderPlot();
  });
}

function _ihColorscale() {
  return [
    [0.00, "#ffffff"],
    [0.05, "#f0f4ff"],
    [0.20, "#c3cdf0"],
    [0.45, "#8b9add"],
    [0.70, "#5563b8"],
    [1.00, "#1f2960"],
  ];
}

function _ihStructurallyAbsent(contrast) {
  return contrast === "ApTt_4mo";
}

function _ihRenderPlot() {
  const block = _ihBlock();
  const el = document.getElementById("ih-plot");
  const countEl = document.getElementById("ih-count");
  if (!block || !el) return;
  const f = IncytrFilter.get();
  const tier = f.hmTier;
  const tierMeta = block.heatmap_tiers[tier];
  const contrast = _ihContrastFromState();
  const absent = _ihStructurallyAbsent(contrast);
  const cIdx = block.contrasts.indexOf(contrast);
  const senders = block.senders, receivers = block.receivers;
  const nS = senders.length, nR = receivers.length;
  const empty = new Set(block.empty_deg_celltypes || []);

  const Z = [];
  const text = [];
  let maxN = 0;
  let visibleN = 0;
  for (let r = 0; r < nR; r++) {
    const row = [];
    const tRow = [];
    for (let s = 0; s < nS; s++) {
      const isEmpty = empty.has(senders[s]) || empty.has(receivers[r]);
      if (isEmpty || absent || cIdx < 0) {
        row.push(null);
        if (absent) tRow.push("ApTt × 4mo is structurally absent upstream");
        else        tRow.push("no candidate paths (empty-DEG cell type)");
      } else {
        const n = _ihCountAt(tier, s, r, cIdx);
        row.push(n);
        if (n > maxN) maxN = n;
        visibleN += n;
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
  const hx = [], hy = [], htxt = [];
  for (let r = 0; r < nR; r++) for (let s = 0; s < nS; s++) {
    if (absent || empty.has(senders[s]) || empty.has(receivers[r])) {
      hx.push(senders[s]); hy.push(receivers[r]);
      htxt.push(absent ? "structurally absent" : "no candidate paths");
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
    if (empty.has(sender) || empty.has(receiver) || absent) return;
    // Seed the table tab via shared filter state, then switch tabs.
    IncytrFilter.set({
      pair:       { sender, receiver },
      senderIn:   [sender],
      receiverIn: [receiver],
      disease:    [f.hmDisease],
      timepoint:  [f.hmTimepoint],
    });
    IncytrFilter.applyTier(tier);
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"incytrpathways"});
  });

  if (countEl) {
    const totalAtTier = (tierMeta && tierMeta.total || 0).toLocaleString();
    countEl.textContent = absent
      ? `${contrast} structurally absent · 0 rows · tier total ${totalAtTier}.`
      : `${contrast} · ${visibleN.toLocaleString()} rows at tier ${tier} (${_ihTierGateLabel(tierMeta && tierMeta.gate)}) · tier total ${totalAtTier}.`;
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
  _ihSyncControls();
  _ihRenderPlot();
}
