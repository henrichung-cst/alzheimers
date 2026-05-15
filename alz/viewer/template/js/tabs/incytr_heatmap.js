// ---------------------------------------------------------------------------
// Incytr Heatmap tab — sender×receiver candidate-path counts for a chosen
// contrast. Filter UI: Disease × Timepoint selects, optional pvalue gate,
// |PDS| effect-size floor, Reset button + live count line. pvalue defaults
// to off (blank) — per-animal SigProb Wald-t is unreliable in this cohort,
// so |PDS| is the recommended primary filter.
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

function _ihSnapPvalue(p) {
  // Snap user input down to the nearest precomputed pvalue threshold. Null
  // (or non-finite) opens the gate by selecting the largest threshold in the
  // grid — useful because the per-animal SigProb Wald-t is unreliable, so
  // pvalue is opt-in and |PDS| is the recommended primary filter.
  const block = _ihBlock();
  const thr = block && block.heatmap_counts && block.heatmap_counts.thresholds;
  if (!thr || !thr.length) return { value: null, index: 0, open: true };
  if (p == null || !isFinite(p)) {
    return { value: null, index: thr.length - 1, open: true };
  }
  let idx = -1;
  for (let i = 0; i < thr.length; i++) if (thr[i] <= p) idx = i;
  if (idx < 0) idx = 0;
  return { value: thr[idx], index: idx, open: false };
}

function _ihSnapAbsPds(ap) {
  // Snap up to the nearest |PDS| threshold (≥ semantics — pick the largest
  // grid step that doesn't exceed the requested value, so the on-screen
  // gate is always at least as inclusive as what the user typed). Returns
  // { value:null, index:0 } on payloads that pre-date the |PDS| axis.
  const block = _ihBlock();
  const thr = block && block.heatmap_counts && block.heatmap_counts.abs_pds_thresholds;
  if (!thr || !thr.length) return { value: null, index: 0 };
  const v = (ap == null || !isFinite(ap)) ? 0 : ap;
  let idx = 0;
  for (let i = 0; i < thr.length; i++) if (thr[i] <= v) idx = i;
  return { value: thr[idx], index: idx };
}

function _ihCountAt(sIdx, rIdx, cIdx, tIdx, apIdx) {
  const block = _ihBlock();
  if (!block) return 0;
  const hm = block.heatmap_counts;
  if (!hm || !hm.counts) return 0;
  // Back-compat: a 3D grid (no thresholds) → return the unfiltered count.
  if (!hm.shape || hm.shape.length === 3) {
    const nR = block.receivers.length, nC = block.contrasts.length;
    return hm.counts[sIdx * nR * nC + rIdx * nC + cIdx];
  }
  // Back-compat: a 4D grid (pvalue only, no |PDS| axis) → ignore apIdx.
  if (hm.shape.length === 4) {
    const [, nR, nC, nT] = hm.shape;
    return hm.counts[sIdx * nR * nC * nT + rIdx * nC * nT + cIdx * nT + tIdx];
  }
  // 5D grid (pvalue × |PDS|).
  const [, nR, nC, nT, nAP] = hm.shape;
  return hm.counts[
    sIdx * nR * nC * nT * nAP
    + rIdx * nC * nT * nAP
    + cIdx * nT * nAP
    + tIdx * nAP
    + (apIdx || 0)
  ];
}

function _ihSyncControls() {
  const block = _ihBlock();
  if (!block) return;
  const f = IncytrFilter.get();

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
  const pInput = document.getElementById("ih-pvalue");
  if (pInput) pInput.value = (f.hmPvalue == null) ? "" : f.hmPvalue;
  const apInput = document.getElementById("ih-abs-pds");
  if (apInput) apInput.value = (f.hmAbsPds == null) ? "" : f.hmAbsPds;
}

function wireIncytrHeatmap() {
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
  const pInput = document.getElementById("ih-pvalue");
  if (pInput) pInput.addEventListener("change", () => {
    if (pInput.value === "") {
      IncytrFilter.set({ hmPvalue: null });
      _ihRenderPlot();
      return;
    }
    const raw = parseFloat(pInput.value);
    if (!isFinite(raw) || raw <= 0 || raw > 1) {
      const f = IncytrFilter.get();
      pInput.value = (f.hmPvalue == null) ? "" : f.hmPvalue;
      return;
    }
    IncytrFilter.set({ hmPvalue: raw });
    _ihRenderPlot();
  });
  const apInput = document.getElementById("ih-abs-pds");
  if (apInput) apInput.addEventListener("change", () => {
    const raw = apInput.value === "" ? null : parseFloat(apInput.value);
    if (raw == null || !isFinite(raw) || raw < 0) {
      const f = IncytrFilter.get();
      apInput.value = f.hmAbsPds;
      return;
    }
    IncytrFilter.set({ hmAbsPds: raw });
    _ihRenderPlot();
  });
  const resetBtn = document.getElementById("ih-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    IncytrFilter.set({ hmDisease: "App", hmTimepoint: "2mo", hmPvalue: null, hmAbsPds: 0.01 });
    _ihSyncControls();
    _ihRenderPlot();
  });
}

const _IH_COLORSCALE = [
  [0.00, "#ffffff"],
  [0.05, "#f0f4ff"],
  [0.20, "#c3cdf0"],
  [0.45, "#8b9add"],
  [0.70, "#5563b8"],
  [1.00, "#1f2960"],
];

function _ihRenderPlot() {
  const block = _ihBlock();
  const el = document.getElementById("ih-plot");
  const countEl = document.getElementById("ih-count");
  if (!block || !el) return;
  const f = IncytrFilter.get();
  const contrast = _ihContrastFromState();
  const cIdx = block.contrasts.indexOf(contrast);
  const senders = block.senders, receivers = block.receivers;
  const nS = senders.length, nR = receivers.length;
  const empty = new Set(block.empty_deg_celltypes || []);
  const snap = _ihSnapPvalue(f.hmPvalue);
  const snapAp = _ihSnapAbsPds(f.hmAbsPds);
  const totalsByThr = (block.heatmap_counts && block.heatmap_counts.total_by_threshold) || null;
  // total_by_threshold is 1D [n_thr] on legacy payloads, 2D [n_thr][n_ap] on
  // new payloads. Handle both shapes.
  let totalAtThr = 0;
  if (totalsByThr && totalsByThr.length) {
    const row = totalsByThr[snap.index];
    totalAtThr = Array.isArray(row) ? (row[snapAp.index] || 0) : (row || 0);
  }

  const Z = [];
  const text = [];
  let maxN = 0;
  let visibleN = 0;
  for (let r = 0; r < nR; r++) {
    const row = [];
    const tRow = [];
    for (let s = 0; s < nS; s++) {
      const isEmpty = empty.has(senders[s]) || empty.has(receivers[r]);
      if (isEmpty || cIdx < 0) {
        row.push(null);
        tRow.push("no candidate paths (empty-DEG cell type)");
      } else {
        const n = _ihCountAt(s, r, cIdx, snap.index, snapAp.index);
        row.push(n);
        if (n > maxN) maxN = n;
        visibleN += n;
        const pTxt = snap.open ? "no pvalue gate" : `pvalue < ${snap.value}`;
        const apTxt = (snapAp.value != null && snapAp.value > 0)
          ? ` · |PDS| ≥ ${snapAp.value}` : "";
        tRow.push(`${n.toLocaleString()} paths · ${pTxt}${apTxt}`);
      }
    }
    Z.push(row);
    text.push(tRow);
  }

  const traces = [{
    type: "heatmap",
    x: senders, y: receivers, z: Z,
    text, hoverinfo: "x+y+text",
    colorscale: _IH_COLORSCALE,
    zmin: 0, zmax: Math.max(1, maxN),
    colorbar: { title: "n paths", thickness: 12, len: 0.7 },
    xgap: 1, ygap: 1,
  }];
  const hx = [], hy = [], htxt = [];
  for (let r = 0; r < nR; r++) for (let s = 0; s < nS; s++) {
    if (empty.has(senders[s]) || empty.has(receivers[r])) {
      hx.push(senders[s]); hy.push(receivers[r]);
      htxt.push("no candidate paths");
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
  // Plotly auto-thins category labels to every-other when the per-row pixel
  // budget drops below ~font-line-height. With 19 cell types and the prior
  // 22 px/row, ~12 px font + descender/ascender + cell-padding overflowed and
  // Plotly silently dropped every other label. Fixes:
  //   - Explicit type: "category" so tick0/dtick are unambiguous category indices
  //   - 32 px/row floor for the heatmap canvas (was 22)
  //   - Larger left/bottom margins to seat the longest cell-type names
  //   - tickfont size 11 so labels fit without breaking the row pitch
  const rowPx = 32;
  const longest = Math.max(...senders.map(s => s.length), ...receivers.map(r => r.length));
  const leftMargin  = Math.min(360, Math.max(140, longest * 6 + 30));
  const bottomMargin = Math.min(220, Math.max(100, longest * 4 + 40));
  const layout = {
    margin: { l: leftMargin, r: 30, t: 20, b: bottomMargin },
    height: Math.max(620, rowPx * Math.max(nS, nR) + leftMargin),
    // automargin: false because Plotly 2.35's automargin pass interacts
    // badly with category axes — even with tickmode:"array" + explicit
    // tickvals, it post-thins labels to nticks=10 stride=2 during the
    // remeasure. nticks: 100 also explicitly overrides the auto-nticks
    // fallback. tickvals as the category strings themselves (not numeric
    // indices) is the contract Plotly category axes honor in 2.35.
    // Lock the range to the category bounds. Plotly's autoscale otherwise
    // pads each end and the wider range triggers the auto-tick-thinning
    // (every-other label drops) regardless of tickmode/nticks. fixedrange
    // disables user zoom-out, which would re-introduce the same effect.
    xaxis: { title: "Sender", type: "category", tickangle: -45, automargin: false,
             categoryorder: "array", categoryarray: senders,
             tickmode: "array", tickvals: senders, ticktext: senders,
             range: [-0.5, nS - 0.5], fixedrange: true,
             tickfont: { size: 11 } },
    yaxis: { title: "Receiver", type: "category", automargin: false,
             categoryorder: "array", categoryarray: receivers,
             tickmode: "array", tickvals: receivers, ticktext: receivers,
             range: [nR - 0.5, -0.5], fixedrange: true,
             tickfont: { size: 11 } },
    plot_bgcolor: "#fafafa",
  };
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const sender = p.x, receiver = p.y;
    if (empty.has(sender) || empty.has(receiver)) return;
    // Seed the table tab via shared filter state, then switch tabs.
    // Propagate |PDS| into the table's sliderPds so the pathway-table view
    // starts at the same effect-size gate the user was looking at.
    IncytrFilter.set({
      pair:       { sender, receiver },
      senderIn:   [sender],
      receiverIn: [receiver],
      disease:    [f.hmDisease],
      timepoint:  [f.hmTimepoint],
      sliderP:    snap.open ? null : snap.value,
      sliderPds:  (snapAp.value != null && snapAp.value > 0) ? snapAp.value : null,
    });
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"incytrpathways"});
  });

  if (countEl) {
    const pTxt = snap.open ? "no pvalue gate" : `pvalue < ${snap.value}`;
    const apTxt = (snapAp.value != null && snapAp.value > 0)
      ? ` & |PDS| ≥ ${snapAp.value}` : "";
    countEl.textContent =
      `${contrast} · ${visibleN.toLocaleString()} paths at ${pTxt}${apTxt}`
      + ` · ${totalAtThr.toLocaleString()} across all contrasts at this threshold.`;
  }
}

function renderIncytrHeatmap() {
  if (!_ihBlock()) {
    const el = document.getElementById("ih-plot");
    if (el) el.innerHTML =
      '<div class="muted" style="padding:24px;">No <code>incytr_pathways</code> '
      + 'block in the payload. Rebuild with <code>pixi run viewer</code> after '
      + 'producing <code>outputs/reports/incytr_factorial/receiver_cache/</code>.</div>';
    return;
  }
  _ihSyncControls();
  _ihRenderPlot();
}
