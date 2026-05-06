
async function renderKinaseBackbones(kinase_id) {
  const container = document.getElementById("ke-detail-backbones");
  if (!container) return;
  _ensureKinaseIndexes();
  if (!_presentKinaseSet.has(kinase_id)) {
    container.innerHTML = '<div class="muted">No significant edges for this kinase.</div>';
    container.classList.remove("muted");
    return;
  }
  let rows;
  try {
    rows = await SliceCache.loadKinase(kinase_id);
  } catch (e) {
    if (Store.state.selection.kinase !== kinase_id) return;
    container.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    return;
  }
  if (Store.state.selection.kinase !== kinase_id) return;

  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const filtered = (cIdx >= 0)
    ? rows.filter(r => r.contrast_id === cIdx)
    : rows;

  const BB = PAYLOAD.backbones;
  const bbIdxById = _backboneIdxById;
  const K = PAYLOAD.kinases;
  const ki = _kinaseIdxById.get(kinase_id);
  const kinaseName = ki != null ? K.name[ki] : "";

  // Group edge rows by backbone, keeping per-contrast support+concordance.
  const groups = new Map();
  for (const r of filtered) {
    let g = groups.get(r.backbone_id);
    if (!g) {
      g = { backbone_id: r.backbone_id, byContrast: new Array(CONTRASTS.length).fill(null), maxAbs: 0 };
      groups.set(r.backbone_id, g);
    }
    g.byContrast[r.contrast_id] = {
      support: r.support_contribution,
      concordance: r.concordance,
    };
    const m = Math.abs(r.support_contribution);
    if (m > g.maxAbs) g.maxAbs = m;
  }

  // For each group, determine per-contrast role (imputed step vs enrichment-only)
  // by checking whether this kinase appears in BB.imputed_nodes_union_<contrast>.
  const imputedColsCache = CONTRASTS.map(c => BB["imputed_nodes_union_" + c]);
  const grouped = Array.from(groups.values());
  for (const g of grouped) {
    const bi = bbIdxById.get(g.backbone_id);
    let nImp = 0, nEnr = 0;
    for (let ci = 0; ci < CONTRASTS.length; ci++) {
      const cell = g.byContrast[ci];
      if (!cell) continue;
      const raw = (bi != null && imputedColsCache[ci]) ? (imputedColsCache[ci][bi] || "") : "";
      const isImputed = raw && kinaseName &&
        raw.split(";").some(s => s.trim() === kinaseName);
      cell.role = isImputed ? "imp" : "enr";
      if (isImputed) nImp++; else nEnr++;
    }
    g.role = (nImp > 0 && nEnr > 0) ? "mixed" : (nImp > 0 ? "imp" : "enr");
    g.nContrasts = nImp + nEnr;
  }

  grouped.sort((a, b) => b.maxAbs - a.maxAbs);
  const TOP = 100;
  const shown = grouped.slice(0, TOP);

  const roleLabel = { imp: "imputed-step", enr: "enrichment", mixed: "mixed" };
  const roleClass = { imp: "imp", enr: "expr", mixed: "mix" };

  const supCell = (cell) => {
    if (!cell) return '<td class="bb-sup"></td>';
    const v = cell.support;
    const m = Math.abs(v);
    const dir = cell.concordance > 0 ? "↑" : (cell.concordance < 0 ? "↓" : "·");
    const color = v > 0 ? "var(--up-red, #c53030)" : (v < 0 ? "var(--down-blue, #2b6cb0)" : "#999");
    const mark = cell.role === "imp" ? "★" : "";
    const title = `${v.toFixed(3)} (${cell.role === "imp" ? "imputed step" : "enrichment"})`;
    return `<td class="bb-sup" title="${title}" style="color:${color};white-space:nowrap;padding:2px 4px;font-size:11px;text-align:center;">` +
      `<span style="font-weight:600;">${dir}${m.toFixed(2)}</span>` +
      (mark ? `<span style="margin-left:2px;">${mark}</span>` : "") +
      `</td>`;
  };

  const shortContrast = (c) => c.replace(/_(\d+)mo$/, "·$1").replace(/^ApTt/, "AT");
  const headContrasts = CONTRASTS.map(c =>
    `<th title="Display label: ${shortContrast(c)}\nRaw column: support_contribution_${c}\nDefinition: Kinase support contribution for ${c}." style="padding:2px 4px;font-size:11px;text-align:center;white-space:nowrap;">${shortContrast(c)}</th>`
  ).join("");
  const parts = [
    `<div class="muted" style="margin-bottom:4px;">Showing top ${shown.length} of ${grouped.length} backbones` +
    (cIdx >= 0 ? ` (contrast ${f.contrast})` : "") +
    ` · ★ = kinase imputed as a pathway step; otherwise support is from enrichment of substrates.</div>`,
    '<div style="overflow-x:auto;max-width:100%;">',
    '<table class="data-table" style="font-size:11px;"><thead><tr>',
    '<th title="Display label: Receiver\nRaw column: receiver\nDefinition: Receiver cell type for the pathway backbone.">Receiver</th>',
    '<th title="Display label: Receptor\nRaw column: Receptor\nDefinition: Receptor gene in the pathway backbone.">Receptor</th>',
    '<th title="Display label: EM\nRaw column: EM\nDefinition: Extracellular-matrix or intermediate effector molecule in the pathway backbone.">EM</th>',
    '<th title="Display label: Target\nRaw column: Target\nDefinition: Target gene in the pathway backbone.">Target</th>',
    headContrasts,
    '<th title="Display label: Role\nRaw column: pathway_evidence_backbone\nDefinition: Whether this kinase is an imputed pathway step or substrate-enrichment support.">Role</th>',
    '</tr></thead><tbody>',
  ];
  for (const g of shown) {
    const bi = bbIdxById.get(g.backbone_id);
    const rcv = bi != null ? RECEIVERS[BB.receiver_id[bi]] : "?";
    const rcp = bi != null ? BB.Receptor[bi] : "?";
    const em  = bi != null ? BB.EM[bi] : "?";
    const tgt = bi != null ? BB.Target[bi] : "?";
    const supCells = g.byContrast.map(supCell).join("");
    const cls = roleClass[g.role] || "lo";
    parts.push(
      `<tr><td>${rcv}</td><td>${rcp}</td><td>${em}</td><td>${tgt}</td>` +
      supCells +
      `<td><span class="badge ${cls}">${roleLabel[g.role]}</span></td></tr>`
    );
  }
  parts.push("</tbody></table></div>");
  container.innerHTML = parts.join("");
}
