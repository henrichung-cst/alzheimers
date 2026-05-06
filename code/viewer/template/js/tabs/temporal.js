function _parseContrast(c) {
  const ix = c.lastIndexOf("_");
  return { geno: c.slice(0, ix), tp: c.slice(ix + 1) };
}

function _temporalKinaseMemberMask() {
  // Returns a Uint8Array over kinase ids indicating whether each kinase belongs
  // to the selected tissue scope (top_celltype_1 membership).
  const K = PAYLOAD.kinases;
  const scope = Store.state.view.temporalTissue;
  const n = K.id.length;
  const mask = new Uint8Array(n);
  if (scope === "ALL") { mask.fill(1); return mask; }
  const tissues = META.tissueCategories || {};
  if (scope.startsWith("t:")) {
    const name = scope.slice(2);
    const rs = new Set(tissues[name] || []);
    for (let i = 0; i < n; i++) mask[i] = rs.has(K.top_celltype_1[i]) ? 1 : 0;
  } else if (scope.startsWith("r:")) {
    const r = scope.slice(2);
    for (let i = 0; i < n; i++) mask[i] = (K.top_celltype_1[i] === r) ? 1 : 0;
  } else {
    mask.fill(1);
  }
  return mask;
}

function renderTemporalKinase() {
  const el = document.getElementById("temporal-plot");
  const sub = document.getElementById("tm-subtitle");
  const K = PAYLOAD.kinases;
  const fdr = Store.state.filters.fdr;
  const mask = _temporalKinaseMemberMask();
  const DG = META.diseaseGroups;
  const TPS = META.timepoints;
  const counts = {};
  for (const g of DG) counts[g] = {};
  for (const g of DG) for (const t of TPS) counts[g][t] = { up: 0, down: 0 };

  const n = K.id.length;
  let nScope = 0;
  for (let i = 0; i < n; i++) if (mask[i]) nScope++;

  for (const g of DG) {
    for (const t of TPS) {
      const c = g + "_" + t;
      const nesCol = K["NES_" + c];
      const fdrCol = K["FDR_" + c];
      if (!nesCol || !fdrCol) continue;
      let up = 0, down = 0;
      for (let i = 0; i < n; i++) {
        if (!mask[i]) continue;
        const q = fdrCol[i], nes = nesCol[i];
        if (q == null || nes == null) continue;
        if (q >= fdr) continue;
        if (nes > 0) up++;
        else if (nes < 0) down++;
      }
      counts[g][t] = { up, down };
    }
  }

  const traces = [];
  for (const g of DG) {
    const color = (META.diseaseColors || {})[g] || "#555";
    traces.push({
      type: "bar", name: g + " up",
      x: TPS, y: TPS.map(t => counts[g][t].up),
      marker: { color }, legendgroup: g,
      hovertemplate: `${g} up @ %{x}: %{y}<extra></extra>`,
    });
    traces.push({
      type: "bar", name: g + " down",
      x: TPS, y: TPS.map(t => -counts[g][t].down),
      marker: { color, opacity: 0.55 }, legendgroup: g, showlegend: true,
      hovertemplate: `${g} down @ %{x}: %{customdata}<extra></extra>`,
      customdata: TPS.map(t => counts[g][t].down),
    });
  }
  const layout = {
    barmode: "group", bargap: 0.25,
    margin: { l: 60, r: 20, t: 10, b: 40 },
    xaxis: { title: "Timepoint" },
    yaxis: { title: "Sig kinases (up − down)", zeroline: true },
    legend: { orientation: "h", y: -0.15 },
    height: 480,
    shapes: [{ type: "line", x0: -0.5, x1: TPS.length - 0.5, y0: 0, y1: 0,
               xref: "x", yref: "y", line: { color: "#000", width: 1 } }],
  };
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });
  if (sub) sub.textContent =
    `${nScope} kinases in scope · FDR < ${fdr} · diverging bars show up (+) vs down (−) counts.`;
}

// Receiver/sender/kinase-selection filter without chain-significance gating.
// Used by magnitude-based temporal metrics (mean_tpds, pct_up) so the diffuse
// phase — where the chain test reports zero passing chains but pathway burden
// is still elevated — does not get suppressed alongside the count.
function _temporalUngatedIndices() {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const rIdx = (f.receiver === "ALL") ? -1 : RECEIVERS.indexOf(f.receiver);
  const gnSet = (f.graphNodeIds && f.graphNodeIds.length)
    ? new Set(f.graphNodeIds) : null;
  const senderBit = (f.sender == null) ? 0 : (1 << f.sender);
  const senderMaskCol = BB.sender_mask;
  const kSet = (sel.kinase != null)
    ? SliceCache.kinaseBackboneSetSync(sel.kinase) : null;
  const ctIdx = (sel.celltype != null) ? sel.celltype : -1;
  const tpdsSigCol = (f.tpdsSig === "0.01") ? BB.tpds_sig_001_mask
                   : (f.tpdsSig === "0.05") ? BB.tpds_sig_005_mask
                   : (f.tpdsSig === "0.10") ? BB.tpds_sig_010_mask
                   : null;
  const out = [];
  for (let i = 0; i < n; i++) {
    if (rIdx >= 0 && BB.receiver_id[i] !== rIdx) continue;
    if (ctIdx >= 0 && BB.receiver_id[i] !== ctIdx) continue;
    if (senderBit && !(senderMaskCol[i] & senderBit)) continue;
    if (tpdsSigCol !== null && tpdsSigCol[i] === 0) continue;
    if (gnSet !== null && !gnSet.has(BB.id[i])) continue;
    if (kSet !== null && !kSet.has(BB.id[i])) continue;
    out.push(i);
  }
  return out;
}

function renderTemporalBackbone() {
  const el = document.getElementById("temporal-plot");
  const sub = document.getElementById("tm-subtitle");
  const BB = PAYLOAD.backbones;
  const metric = Store.state.view.temporalMetric;
  const DG = META.diseaseGroups;
  const TPS = META.timepoints;
  const f = Store.state.filters;
  const sel = Store.state.selection;
  const TD = PAYLOAD.tpdsDistribution || {};
  // The backbone payload is hard-gated to chains passing the chain test.
  // For count and mean_score that's appropriate (both are defined on
  // passing chains). For mean_tpds and pct_up — magnitude readouts that
  // should reflect every enumerated chain, including diffuse-phase late-
  // Tau where nothing passes — read from the build-time tpdsDistribution
  // summary. The summary aggregates per (receiver, contrast) across all
  // senders, so it answers the broad-scope question; if the user pins a
  // sender or a kinase selection, we fall back to the BB iteration with a
  // subtitle note that magnitude is now restricted to passing chains.
  const useSummary = (metric === "mean_tpds" || metric === "pct_up")
                     && f.sender == null && sel.kinase == null
                     && (!f.graphNodeIds || !f.graphNodeIds.length);

  if (useSummary) {
    const traces = [];
    const recvKeys = (f.receiver === "ALL")
      ? RECEIVERS.slice() : [f.receiver];
    for (const g of DG) {
      const color = (META.diseaseColors || {})[g] || "#555";
      const y = [];
      const cust = [];
      const totalN = [];
      for (const t of TPS) {
        const c = g + "_" + t;
        let nSum = 0, sumAbs = 0, nUp = 0, nDown = 0;
        for (const r of recvKeys) {
          const cell = TD[c + "|" + r];
          if (!cell) continue;
          nSum  += cell.n;
          sumAbs += cell.mean_abs * cell.n;
          nUp   += cell.n_up;
          nDown += cell.n_down;
        }
        if (nSum === 0) { y.push(null); cust.push([0, 0]); totalN.push(0); continue; }
        if (metric === "mean_tpds") y.push(sumAbs / nSum);
        else if (metric === "pct_up") y.push(100 * nUp / nSum);
        else y.push(null);
        cust.push([nUp, nDown]);
        totalN.push(nSum);
      }
      traces.push({
        type: "scatter", mode: "lines+markers", name: g,
        x: TPS, y, customdata: TPS.map((_, i) => [totalN[i], cust[i][0], cust[i][1]]),
        line: { color, width: 2 }, marker: { color, size: 8 },
        hovertemplate:
          "<b>" + g + "</b> %{x}<br>" +
          "value: %{y:.4f}<br>" +
          "n chains: %{customdata[0]}<br>" +
          "up / down: %{customdata[1]} / %{customdata[2]}<extra></extra>",
      });
    }
    const yTitle = (metric === "mean_tpds")
      ? "Mean |TPDS| (all enumerated chains)"
      : "% upregulated (TPDS > 0, all enumerated chains)";
    Plotly.react(el, traces, {
      margin: { l: 70, r: 20, t: 10, b: 40 },
      xaxis: { title: "Timepoint" },
      yaxis: { title: yTitle, zeroline: true },
      legend: { orientation: "h", y: -0.15 },
      height: 480,
    }, { displaylogo: false, responsive: true });
    if (sub) sub.textContent =
      `metric = ${metric} · reading per-(receiver, contrast) summary built` +
      ` from every enumerated chain · receiver=${f.receiver}.`;
    return;
  }

  // Count or mean_score, OR a sender/kinase selection is active (which the
  // summary cannot answer). Iterate the gated BB payload.
  const idx = _temporalUngatedIndices();
  const sigMaskCol = BB.significant_both_mask;
  const tpdsMin = Math.max(0, Number(Store.state.view.temporalScoreMin) || 0);

  const agg = {};
  for (const g of DG) { agg[g] = {}; for (const t of TPS)
    agg[g][t] = { countSig: 0, sumScore: 0, nFinite: 0,
                   nMagnitude: 0, sumAbsTpds: 0, nUp: 0 }; }
  for (const g of DG) {
    for (const t of TPS) {
      const c = g + "_" + t;
      const cIdx = CONTRASTS.indexOf(c);
      const tpdsCol = BB["mean_tpds_" + c];
      const obsCol = BB["observed_score_" + c];
      if (!tpdsCol || cIdx < 0) continue;
      const a = agg[g][t];
      for (let j = 0; j < idx.length; j++) {
        const i = idx[j];
        const tp = tpdsCol[i];
        if (tp == null) continue;
        if (tpdsMin > 0 && Math.abs(tp) < tpdsMin) continue;
        a.nMagnitude++;
        a.sumAbsTpds += Math.abs(tp);
        if (tp > 0) a.nUp++;
        const isSig = ((sigMaskCol[i] >> cIdx) & 1) === 1;
        if (isSig) {
          a.countSig++;
          const os = obsCol ? obsCol[i] : null;
          if (os != null) { a.sumScore += os; a.nFinite++; }
        }
      }
    }
  }

  const traces = [];
  for (const g of DG) {
    const color = (META.diseaseColors || {})[g] || "#555";
    const y = TPS.map(t => {
      const a = agg[g][t];
      if (metric === "count") return a.countSig;
      if (metric === "mean_score") return a.nFinite ? a.sumScore / a.nFinite : null;
      if (metric === "mean_tpds") return a.nMagnitude ? a.sumAbsTpds / a.nMagnitude : null;
      if (metric === "pct_up") return a.nMagnitude ? (100 * a.nUp / a.nMagnitude) : null;
      return null;
    });
    const customdata = TPS.map(t => {
      const a = agg[g][t];
      return [a.countSig, a.nMagnitude];
    });
    traces.push({
      type: "scatter", mode: "lines+markers", name: g,
      x: TPS, y, customdata,
      line: { color, width: 2 }, marker: { color, size: 8 },
      hovertemplate:
        "<b>" + g + "</b> %{x}<br>" +
        "value: %{y}<br>" +
        "passing chains: %{customdata[0]}<br>" +
        "chains with TPDS: %{customdata[1]}<extra></extra>",
    });
  }
  const yTitle = ({
    count: "Passing-chain count",
    mean_score: "Mean observed score (over passing chains)",
    mean_tpds: "Mean |TPDS| (passing chains only — selection active)",
    pct_up: "% upregulated (passing chains only — selection active)",
  })[metric] || "";
  Plotly.react(el, traces, {
    margin: { l: 70, r: 20, t: 10, b: 40 },
    xaxis: { title: "Timepoint" },
    yaxis: { title: yTitle, zeroline: true },
    legend: { orientation: "h", y: -0.15 },
    height: 480,
  }, { displaylogo: false, responsive: true });
  const restrict = (metric === "mean_tpds" || metric === "pct_up")
    ? " · magnitude restricted to passing chains because a sender/kinase selection is active"
    : "";
  if (sub) sub.textContent =
    `${idx.length.toLocaleString()} chains in current filter · metric = ${metric}` + restrict + ".";
}

function renderTemporal() {
  const el = document.getElementById("temporal-plot");
  if (!el) return;
  const level = Store.state.view.temporalLevel;
  const metricLabel = document.getElementById("tm-metric-label");
  const tissueLabel = document.getElementById("tm-tissue-label");
  if (metricLabel) metricLabel.style.display = (level === "backbone") ? "" : "none";
  if (tissueLabel) tissueLabel.style.display = (level === "kinase") ? "" : "none";
  if (level === "kinase") renderTemporalKinase();
  else renderTemporalBackbone();
}

function wireTemporalControls() {
  const levelSel = document.getElementById("tm-level");
  const metricSel = document.getElementById("tm-metric");
  const tissueSel = document.getElementById("tm-tissue");
  if (!levelSel || !metricSel || !tissueSel) return;

  // Populate tissue dropdown: All + tissue groups + per-receiver leaves.
  const opts = ['<option value="ALL">All cell types</option>'];
  const tissues = META.tissueCategories || {};
  for (const tname of Object.keys(tissues)) {
    opts.push(`<option value="t:${tname}">${tname} (tissue)</option>`);
    for (const r of tissues[tname])
      opts.push(`<option value="r:${r}">&nbsp;&nbsp;${r}</option>`);
  }
  tissueSel.innerHTML = opts.join("");

  levelSel.value = Store.state.view.temporalLevel;
  metricSel.value = Store.state.view.temporalMetric;
  tissueSel.value = Store.state.view.temporalTissue;

  levelSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"temporalLevel", value: ev.target.value}));
  metricSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"temporalMetric", value: ev.target.value}));
  tissueSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"temporalTissue", value: ev.target.value}));
  const scoreInp = document.getElementById("tm-score-min");
  if (scoreInp) {
    scoreInp.value = Store.state.view.temporalScoreMin || 0;
    scoreInp.addEventListener("change", ev =>
      Store.dispatch({type:"SET_VIEW", key:"temporalScoreMin",
                      value: Math.max(0, parseFloat(ev.target.value) || 0)}));
  }
}

// ---------------------------------------------------------------------------
// Additivity tab — merged kinase NES + backbone TPDS ApTt-additivity scatter.
// Predicted = App + Tau; observed = ApTt. y=x means perfectly additive; points
// below the diagonal = sub-additive (standing sanity check).
// ---------------------------------------------------------------------------
const _ADD_COLORS = {
  "App only":   "#d1495b",
  "Tau only":   "#2e86ab",
  "ApTt only":  "#8338ec",
  "Multi":      "#444",
};
const _ADD_CATEGORIES = ["App only", "Tau only", "ApTt only", "Multi"];
const _ADD_BACKBONE_MAX_POINTS = 20000;
