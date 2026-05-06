function _tissueSortedPairs(names, tissueOf) {
  const pairs = names.map((n, i) => [n, tissueOf(n, i) || "zzz", i]);
  pairs.sort((a, b) => {
    if (a[1] !== b[1]) return a[1] < b[1] ? -1 : 1;
    return a[0] < b[0] ? -1 : (a[0] > b[0] ? 1 : 0);
  });
  return pairs;
}

function receiverOrder() {
  return _tissueSortedPairs(RECEIVERS, (_, i) => TISSUE_CAT[i]).map(p => p[0]);
}

function renderOverview() {
  const el = document.getElementById("overview-plot");
  if (!el) return;
  const f = Store.state.filters;
  const mode = Store.state.view.overviewMode;  // 'count' | 'direction'
  const rows = receiverOrder();
  const cols = CONTRASTS;

  // Build z matrix + hover + customdata.
  const z = [], hover = [], cd = [];
  for (const r of rows) {
    const zrow = [], hrow = [], crow = [];
    for (const c of cols) {
      const cell = PAYLOAD.overview[c + "|" + r];
      if (!cell || cell.n === 0) {
        zrow.push(null); hrow.push(`${r} | ${c}<br>(no sig backbones)`);
        crow.push({receiver:r, contrast:c, n:0});
      } else {
        let v;
        if (mode === "direction") v = cell.n_up - cell.n_down;
        else v = Math.log10(1 + cell.n);
        zrow.push(v);
        hrow.push(
          `${r} | ${c}<br>n=${cell.n} (up=${cell.n_up}, down=${cell.n_down})` +
          `<br>mean TPDS=${cell.mean_tpds}`);
        crow.push({receiver:r, contrast:c, n:cell.n});
      }
    }
    z.push(zrow); hover.push(hrow); cd.push(crow);
  }

  // Contrast filter: dim non-selected columns by blanking cells.
  if (f.contrast !== "ALL") {
    const keep = cols.indexOf(f.contrast);
    for (let i = 0; i < z.length; i++)
      for (let j = 0; j < z[i].length; j++)
        if (j !== keep) z[i][j] = null;
  }

  const colorscale = (mode === "direction")
    ? [[0, DISEASE_COLORS.Tau], [0.5, "#ffffff"], [1, DISEASE_COLORS.App]]
    : "YlOrRd";
  const trace = {
    type:"heatmap", x:cols, y:rows, z, text:hover,
    hovertemplate:"%{text}<extra></extra>", customdata:cd,
    colorscale, showscale:true,
    zmid: (mode === "direction") ? 0 : undefined,
  };
  const layout = {
    margin:{l:130, r:20, t:10, b:90},
    xaxis:{tickangle:-30, automargin:true},
    yaxis:{automargin:true, autorange:"reversed"},
    height:560,
  };
  Plotly.react(el, [trace], layout, {displaylogo:false, responsive:true});

  // Plotly.react preserves the DOM node, so detach prior listeners first.
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const d = ev.points[0].customdata;
    if (!d || d.n === 0) return;
    Store.dispatch({type:"SET_SELECTION", key:"backbone", value:null});
    Store.dispatch({type:"SET_FILTER", key:"receiver", value:d.receiver});
  });
}

// ---------------------------------------------------------------------------
// Sender × Receiver tab
// ---------------------------------------------------------------------------
let _senderOrderCache = null;
