function _senderOrder() {
  if (_senderOrderCache) return _senderOrderCache;
  const SENDERS = META.senderOrder || [];
  const toTissue = META.receiverToTissue || {};
  _senderOrderCache = _tissueSortedPairs(SENDERS, (s) => toTissue[s]);
  return _senderOrderCache;
}

const SENDER_GENOTYPES = ["App", "Tau", "ApTt"];
const SENDER_TIMEPOINTS = ["2mo", "4mo", "6mo"];
let _senderMatrixScaleCache = null;

function _senderMatrixGlobalScale() {
  if (_senderMatrixScaleCache) return _senderMatrixScaleCache;
  const SM = PAYLOAD.senderMatrix || {};
  let maxCount = 0;
  let maxAbsDir = 0;
  for (const key of Object.keys(SM)) {
    const cell = SM[key];
    if (!cell || cell.n === 0) continue;
    const cv = Math.log10(1 + cell.n);
    if (cv > maxCount) maxCount = cv;
    const dv = Math.abs(cell.n_up - cell.n_down);
    if (dv > maxAbsDir) maxAbsDir = dv;
  }
  if (maxCount === 0) maxCount = 1;
  if (maxAbsDir === 0) maxAbsDir = 1;
  _senderMatrixScaleCache = { maxCount, maxAbsDir };
  return _senderMatrixScaleCache;
}

// Compare-axis design: render three 22×22 matrices side-by-side. The active
// axis determines what's varied across panels (the three timepoints, or the
// three genotypes). The anchor is the dimension held fixed.
const SENDER_AXIS_PANELS = {
  timepoint: ["2mo", "4mo", "6mo"],   // fixed genotype, vary timepoint
  genotype:  ["App", "Tau", "ApTt"],  // fixed timepoint, vary genotype
};
const SENDER_ANCHOR_OPTIONS = {
  timepoint: ["App", "Tau", "ApTt"],
  genotype:  ["2mo", "4mo", "6mo"],
};

function _senderPanelContrast(axis, anchor, panelValue) {
  // Reconstruct the contrast key {genotype}_{timepoint} given which axis we
  // are varying across panels. When axis="timepoint", anchor is the genotype
  // and panelValue is the timepoint; when axis="genotype" it's reversed.
  if (axis === "timepoint") return `${anchor}_${panelValue}`;
  return `${panelValue}_${anchor}`;
}

function _setSenderAxis(nextAxis) {
  const view = Store.state.view;
  if (view.senderMatrixAxis === nextAxis) return;
  const map = view.senderMatrixLastAnchorByAxis || {};
  const restored = map[nextAxis] ||
    (nextAxis === "timepoint" ? "ApTt" : "2mo");
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixAxis", value: nextAxis});
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixAnchor", value: restored});
}

function _setSenderAnchor(nextAnchor) {
  const view = Store.state.view;
  if (view.senderMatrixAnchor === nextAnchor) return;
  const map = Object.assign({}, view.senderMatrixLastAnchorByAxis || {});
  map[view.senderMatrixAxis] = nextAnchor;
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixAnchor", value: nextAnchor});
  Store.dispatch({type:"SET_VIEW", key:"senderMatrixLastAnchorByAxis", value: map});
}

function _stepSenderAnchor(delta) {
  const view = Store.state.view;
  const opts = SENDER_ANCHOR_OPTIONS[view.senderMatrixAxis];
  const i = opts.indexOf(view.senderMatrixAnchor);
  const ni = ((i + delta) % opts.length + opts.length) % opts.length;
  _setSenderAnchor(opts[ni]);
}

function _flipSenderAxis() {
  const cur = Store.state.view.senderMatrixAxis;
  _setSenderAxis(cur === "timepoint" ? "genotype" : "timepoint");
}

function _renderSenderPanel(slotIdx, contrast, panelLabel) {
  const el = document.getElementById("sender-matrix-plot-" + slotIdx);
  if (!el) return;
  const mode = Store.state.view.senderMatrixMode;
  const SM = PAYLOAD.senderMatrix || {};
  const sRows = _senderOrder();
  const rCols = receiverOrder();
  const ctid = {};
  for (let i = 0; i < RECEIVERS.length; i++) ctid[RECEIVERS[i]] = i;

  const z = [], hover = [], cd = [];
  for (const [sname, , sid] of sRows) {
    const zrow = [], hrow = [], crow = [];
    for (const rname of rCols) {
      const rid = ctid[rname];
      const cell = SM[contrast + "|" + sid + "|" + rid];
      if (!cell || cell.n === 0) {
        zrow.push(null);
        hrow.push(`${sname} → ${rname}<br>(no backbones)`);
        crow.push({sender_id: sid, receiver: rname, n: 0});
      } else {
        let v;
        if (mode === "direction") v = cell.n_up - cell.n_down;
        else v = Math.log10(1 + cell.n);
        zrow.push(v);
        hrow.push(
          `${sname} → ${rname}<br>${contrast}<br>n=${cell.n} ` +
          `(up=${cell.n_up}, down=${cell.n_down})` +
          `<br>mean TPDS=${cell.mean_tpds}`);
        crow.push({sender_id: sid, receiver: rname, n: cell.n});
      }
    }
    z.push(zrow); hover.push(hrow); cd.push(crow);
  }

  const scale = _senderMatrixGlobalScale();
  const colorscale = (mode === "direction")
    ? [[0, DISEASE_COLORS.Tau], [0.5, "#ffffff"], [1, DISEASE_COLORS.App]]
    : "YlOrRd";
  // Show the colorbar only on the rightmost panel to save space.
  const showscale = (slotIdx === 2);
  const trace = {
    type: "heatmap",
    x: rCols, y: sRows.map(p => p[0]), z,
    text: hover, hovertemplate: "%{text}<extra></extra>",
    customdata: cd, colorscale, showscale,
    zmin: (mode === "direction") ? -scale.maxAbsDir : 0,
    zmax: (mode === "direction") ?  scale.maxAbsDir : scale.maxCount,
    zmid: (mode === "direction") ? 0 : undefined,
  };
  // Only the leftmost panel shows the y-axis sender labels to save space.
  const showY = (slotIdx === 0);
  const layout = {
    title: { text: panelLabel, font: { size: 13 } },
    margin: { l: showY ? 130 : 30, r: showscale ? 60 : 8, t: 30, b: 110 },
    xaxis: { tickangle:-45, automargin:true, tickfont:{size:9} },
    yaxis: { automargin:true, autorange:"reversed",
             dtick:1, tickfont:{size:9}, showticklabels: showY },
    height: 560,
  };
  Plotly.react(el, [trace], layout, {displaylogo:false, responsive:true});

  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const d = ev.points[0].customdata;
    if (!d || d.n === 0) return;
    Store.dispatch({type:"SET_FILTER", key:"sender", value: d.sender_id});
    Store.dispatch({type:"SET_FILTER", key:"receiver", value: d.receiver});
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"pathway"});
  });
}

function renderSenderMatrix() {
  const view = Store.state.view;
  const axis = view.senderMatrixAxis;
  const anchor = view.senderMatrixAnchor;
  const panels = SENDER_AXIS_PANELS[axis];
  for (let i = 0; i < 3; i++) {
    const c = _senderPanelContrast(axis, anchor, panels[i]);
    _renderSenderPanel(i, c, c);
  }
  const sub = document.getElementById("sm-subtitle");
  if (sub) {
    sub.textContent = (axis === "timepoint")
      ? `${anchor} at ${panels.join(", ")} — color scale pinned across all nine contrasts.`
      : `${panels.join(", ")} at ${anchor} — color scale pinned across all nine contrasts.`;
  }
}

function _populateSenderAnchorSelect() {
  const sel = document.getElementById("sm-anchor");
  if (!sel) return;
  const view = Store.state.view;
  const opts = SENDER_ANCHOR_OPTIONS[view.senderMatrixAxis];
  sel.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join("");
  sel.value = view.senderMatrixAnchor;
  const lab = document.getElementById("sm-anchor-label");
  if (lab) {
    lab.firstChild.nodeValue =
      (view.senderMatrixAxis === "timepoint") ? "Genotype: " : "Timepoint: ";
  }
}

function wireSenderMatrix() {
  const modeSel = document.getElementById("sm-mode");
  if (modeSel) {
    modeSel.value = Store.state.view.senderMatrixMode;
    modeSel.addEventListener("change", ev => {
      Store.dispatch({type:"SET_VIEW", key:"senderMatrixMode",
                      value: ev.target.value});
    });
  }
  const axisSel = document.getElementById("sm-axis");
  if (axisSel) {
    axisSel.value = Store.state.view.senderMatrixAxis;
    axisSel.addEventListener("change", ev => _setSenderAxis(ev.target.value));
  }
  const anchorSel = document.getElementById("sm-anchor");
  _populateSenderAnchorSelect();
  if (anchorSel) {
    anchorSel.addEventListener("change", ev => _setSenderAnchor(ev.target.value));
  }
}

function wireSenderMatrixKeyboard() {
  document.addEventListener("keydown", ev => {
    if (Store.state.view.activeTab !== "senders") return;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    let handled = false;
    if (ev.key === "ArrowLeft")       { _stepSenderAnchor(-1); handled = true; }
    else if (ev.key === "ArrowRight") { _stepSenderAnchor(+1); handled = true; }
    else if (ev.key === "ArrowUp" ||
             ev.key === "ArrowDown")  { _flipSenderAxis(); handled = true; }
    if (handled) ev.preventDefault();
  });
}

// ---------------------------------------------------------------------------
// Temporal Dynamics tab — merged kinase Direction-over-Time + pathway Temporal.
// Level toggle picks which entity is aggregated; both render to #temporal-plot.
// ---------------------------------------------------------------------------
