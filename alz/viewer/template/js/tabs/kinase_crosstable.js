// ---------------------------------------------------------------------------
// Kinase Crosstable — cross-dataset agreement view (master/detail).
//
// MASTER (left): a slim, agreement-grouped list of kinases — one row per kinase
// with the mouse 3×3 NES glyph, the direction badge, the human per-donor strip,
// and mouse/human location tiers at the selected cluster. The per-sample NES
// detail lives in the DETAIL panel, not as wide columns.
//
// DETAIL (right, #kx-detail): a cross-dataset comparison of the selected kinase:
// Song mouse, Mukesh human AD (+ CTRL visual reference), and 5xFAD cortex /
// hippocampus age tiles under a verdict header, with two sub-tabs:
//   Activity      — _renderKinaseNesPlot (mouse), _khRenderNESAcrossDonors
//                   (human), and local 5xFAD age/status summaries.
//   Specificity   — _kxRenderSpecAligned: ONE cluster-aligned reference table,
//                   Song (primary mouse) + WMB / SEA-AD / HBCA cross-check pills.
//
// Indexes are joined client-side from existing PAYLOAD keys (see _kxBuildIndexes).
// The cross-dataset join key is the kinase abbreviation + residue (species-neutral;
// no ortholog mapping). Selection is LOCAL to this tab (does not drive the other
// tabs' selection). Agreement is recomputed live against the FDR slider.
// ---------------------------------------------------------------------------

let _KX_ROWS = null;
let _KX_HUMAN_BY_NAME = null;
let _KX_HUMAN_PERDONOR = null;    // Map<kid, Map<donor, {nes, fdr}>>
let _KX_HUMAN_SPEC_BY_NAME = null;
let _KX_WMB_BY_KIN_CLUSTER = null; // Map<`kid|cluster`, wmb_specificity>
let _KX_WMB_MAX_BY_KID = null;     // Map<kid, {frac, cluster}> — peak WMB specificity (for "Any")
let _KX_SONG_BY_KID = null;        // Map<kid, {tau, topCluster, topShare}> — Song location specificity (per kinase)
let _KX_SEAAD_MAX_BY_NAME = null;  // Map<name, {score, cluster}> — peak SEA-AD specificity (for "Any")
let _KX_DECOMP_BY_KIN_CTX = null;
let _KX_F5_BY_KEY = null;          // Map<`kinase|residue|tissue`, {rows: Map<age,row>}>
let _KX_F5_ATTR_BY_KEY = null;     // Map<`kinase|tissue`, compact attribution summary>
let _KX_F5_AGREE_BY_KEY = null;    // Map<`kinase|tissue|track|age`, bulk-vs-decomp agreement>
let _KX_F5_CELLTYPES = [];
let _KX_CLUSTERS = null;
let _KX_AD_DONORS = [];
let _KX_CTRL_DONORS = [];
let _KX_INITIALIZED = false;
let _KX_F5_ATTR_READY = false;     // 5xFAD attribution-summary sidecar (P2) loaded + indexed
let _kxVisible = [];

// Song location specificity (the favored mouse signal), shown as fold over the
// even-split baseline (1/31 ≈ 0.032, meta.song_uniform) — the same vocabulary as
// the WMB cross-check and the detail pane: ≥10× / ≥5× / ≥2× / ≥1×. The fold is
// the kinase's peak cell-type share (song_top_share); τ — the Yanai concentration
// index over the 31 spine clusters — rides in the tooltip. See
// docs/foundation/concordance.md §6.
const _KX_SONG_UNIFORM = (PAYLOAD && PAYLOAD.meta && PAYLOAD.meta.song_uniform) || (1 / 31);

// Uniform baseline for the WMB cross-check tier. WMB specificity is a share
// normalized over the retained WMB classes that carry atlas cells (~9), so the
// honest "even split" is 1/N_retained, emitted canonically as meta.wmb_uniform.
let _KX_WMB_UNIFORM = (PAYLOAD && PAYLOAD.meta && PAYLOAD.meta.wmb_uniform) || (1 / 9);

function _kxState() {
  if (!Store.state.view.crosstable) {
    Store.state.view.crosstable = {
      cluster: "",
      residueTrack: "",
      search: "",
      sortKey: "agree_score",
      sortDir: -1,
      mSpecMin: 0,            // minimum mouse Song location tier (0=any, 2/5/10 × even-split)
      hSpecMin: 0,            // minimum human SEA-AD expr tier (0=any,1,2,5,10 ×)
      allSamples: false,      // median + agreement over ALL measured units, not just sig
      agreeCat: "",           // agreement category to show ("" = any), via the toolbar dropdown
      compareScope: "three_way",
      compareState: "",
      f5Celltype: "",
      f5Confidence: "",
      f5SnrnaMin: 0,
      f5WmbMin: 0,
      selectedKey: null,      // `name|residue` of the kinase shown in the detail panel
      detailTab: "activity",  // "activity" | "specificity"
    };
  }
  return Store.state.view.crosstable;
}

const _KX_F5_AGES = [3, 6, 9, 12];
const _KX_F5_TISSUES = ["cortex", "hippocampus"];
const _KX_STATUS_META = {
  sig_up: {cls: "badge vhi", label: "sig up", short: "up", dir: 1},
  sig_down: {cls: "badge hi", label: "sig down", short: "down", dir: -1},
  mixed_sig: {cls: "badge mix", label: "mixed_sig", short: "mixed", dir: 0},
  nonsig_measured: {cls: "badge lo", label: "measured", short: "n/s", dir: 0},
  missing: {cls: "badge lo", label: "missing", short: "n/a", dir: null},
};
const _KX_COMPARE_LABELS = {
  three_way: "3-way",
  song_mukesh: "Song vs Mukesh",
  song_f5: "Song vs 5xFAD",
  mukesh_f5: "Mukesh vs 5xFAD",
  f5_tissue: "5xFAD tissue split",
};
const _KX_STATE_FILTER_LABELS = {
  sig_agree: "significant agree",
  agree: "agree",
  sig_disagree: "significant disagree",
  disagree: "disagree",
};
const _KX_STATE_FILTER_VALUES = new Set(Object.keys(_KX_STATE_FILTER_LABELS));

function _kxNormResidue(v) {
  const s = String(v || "").toUpperCase();
  if (s === "Y" || s === "PY") return "Y";
  return "ST";
}

function _kxF5TrackForResidue(residue) {
  return _kxNormResidue(residue) === "Y" ? "y" : "st";
}

function _kxF5Key(name, residue, tissue) {
  return `${name || ""}|${_kxNormResidue(residue)}|${tissue || ""}`;
}

function _kxF5AttrKey(name, tissue) {
  return `${name || ""}|${tissue || ""}`;
}

function _kxF5Num(v) {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function _kxStatusDir(status) {
  const m = _KX_STATUS_META[status || "missing"];
  return m ? m.dir : null;
}

function _kxStatusIsSig(status) {
  return status === "sig_up" || status === "sig_down";
}

function _kxDirectionStatus(nes, measured, sig) {
  if (!measured) return "missing";
  const n = _kxF5Num(nes);
  if (!sig) return "nonsig_measured";
  if (n == null || n === 0) return "nonsig_measured";
  return n > 0 ? "sig_up" : "sig_down";
}

function _kxBuildIndexes() {
  if (_KX_INITIALIZED) return;
  const K = ViewerPayload.kinases();
  const famMap = (META && META.familyMap) || {};

  _KX_ROWS = [];
  for (let i = 0; i < K.id.length; i++) {
    const nesByC = {}; const fdrByC = {};
    for (const c of CONTRASTS) {
      nesByC[c] = K["NES_" + c] ? K["NES_" + c][i] : null;
      fdrByC[c] = K["FDR_" + c] ? K["FDR_" + c][i] : null;
    }
    _KX_ROWS.push({
      kid: K.id[i],
      name: K.name[i],
      gene: K.gene_symbol[i] || "",
      family: famMap[K.name[i]] || "",
      residue: (K.residue_type && K.residue_type[i]) || "ST",
      peak_NES: K.peak_NES[i],
      trajectory: K.trajectory ? (K.trajectory[i] || "") : "",
      n_sig_contrasts: K.n_sig_contrasts ? (K.n_sig_contrasts[i] || 0) : 0,
      n_celltype_candidates: K.n_celltype_candidates ? (K.n_celltype_candidates[i] || 0) : 0,
      _nes: nesByC, _fdr: fdrByC,
    });
  }

  _KX_DECOMP_BY_KIN_CTX = new Map();
  const D = PAYLOAD.decomposition_index || {kinase_id: []};
  for (let k = 0; k < D.kinase_id.length; k++) {
    const ctName = CONTRASTS[D.contrast_id[k]];
    if (!ctName) continue;
    const key = `${D.kinase_id[k]}|${ctName}|${D.cell_type[k]}`;
    _KX_DECOMP_BY_KIN_CTX.set(key, {nes: D.decomp_nes[k], fdr: D.decomp_fdr[k]});
  }
  const decompClusters = new Set();
  for (let k = 0; k < D.cell_type.length; k++) decompClusters.add(D.cell_type[k]);

  _KX_WMB_BY_KIN_CLUSTER = new Map();
  _KX_WMB_MAX_BY_KID = new Map();
  _KX_SONG_BY_KID = new Map();
  const A = PAYLOAD.attribution_index;
  if (A && A.kinase_id) {
    for (let i = 0; i < A.kinase_id.length; i++) {
      const kid = A.kinase_id[i], ct = A.cell_type[i];
      const v = A.wmb_specificity ? A.wmb_specificity[i] : null;
      if (v != null && isFinite(v)) {
        const key = `${kid}|${ct}`;
        // attribution_index can carry multiple rows per (kinase, cluster) across
        // contrasts/confidence strata; keep the max specificity seen (it's a
        // cluster-level property, not contrast-level — picks the populated row).
        const prev = _KX_WMB_BY_KIN_CLUSTER.get(key);
        if (prev == null || v > prev) _KX_WMB_BY_KIN_CLUSTER.set(key, v);
        // Peak specificity across clusters, for the "Any cell type" pivot.
        const mx = _KX_WMB_MAX_BY_KID.get(kid);
        if (!mx || v > mx.frac) _KX_WMB_MAX_BY_KID.set(kid, {frac: v, cluster: ct});
      }
      // Song location specificity is per-kinase (tau + the cell type it
      // concentrates in), constant across the rows for a kid — stamp once.
      if (!_KX_SONG_BY_KID.has(kid) && A.song_tau) {
        const t = A.song_tau[i];
        if (t != null && isFinite(t)) {
          _KX_SONG_BY_KID.set(kid, {
            tau: t,
            topCluster: (A.song_top_cluster && A.song_top_cluster[i]) || "",
            topShare: (A.song_top_share && isFinite(A.song_top_share[i])) ? A.song_top_share[i] : null,
          });
        }
      }
    }
  }

  _KX_HUMAN_BY_NAME = new Map();
  _KX_HUMAN_PERDONOR = new Map();
  _KX_HUMAN_SPEC_BY_NAME = new Map();
  _KX_SEAAD_MAX_BY_NAME = new Map();
  const specClusters = new Set();
  const H = PAYLOAD.human;
  _KX_AD_DONORS = [];
  _KX_CTRL_DONORS = [];
  if (H) {
    _KX_AD_DONORS = Array.isArray(H.donors) ? H.donors.slice() : [];
    _KX_CTRL_DONORS = Array.isArray(H.ctrl_donors) ? H.ctrl_donors.slice() : [];
  }
  if (H && H.kinases) {
    const HK = H.kinases;
    for (let i = 0; i < HK.id.length; i++) {
      const name = HK.name[i];
      const residue = HK.residue_type[i] || "ST";
      const key = `${name}|${residue}`;
      _KX_HUMAN_BY_NAME.set(key, {
        kid: HK.id[i],
        median_nes: HK.median_nes_sig_only ? HK.median_nes_sig_only[i] : null,
        n_sig: HK.n_donors_sig ? HK.n_donors_sig[i] : null,
        n_up: HK.n_donors_up ? HK.n_donors_up[i] : null,
        n_down: HK.n_donors_down ? HK.n_donors_down[i] : null,
        n_tested: HK.n_donors_tested ? HK.n_donors_tested[i] : null,
        sea_ad_lfc: HK.sea_ad_lfc ? HK.sea_ad_lfc[i] : null,
        sea_ad_n: HK.sea_ad_n_supertypes ? HK.sea_ad_n_supertypes[i] : null,
      });
    }
  }
  if (H && H.perdonor_index) {
    const PD = H.perdonor_index;
    for (let i = 0; i < PD.kinase_id.length; i++) {
      const kid = PD.kinase_id[i];
      let m = _KX_HUMAN_PERDONOR.get(kid);
      if (!m) { m = new Map(); _KX_HUMAN_PERDONOR.set(kid, m); }
      m.set(PD.donor[i], {nes: PD.NES[i], fdr: PD.FDR[i]});
    }
  }
  if (H && H.celltype_specificity) {
    const CS = H.celltype_specificity;
    const seaad = CS.seaad_mtg && CS.seaad_mtg.ranked_by_kinase;
    const hbca = CS.allen_hbca && CS.allen_hbca.ranked_by_kinase;
    const allNames = new Set([
      ...(seaad ? Object.keys(seaad) : []),
      ...(hbca ? Object.keys(hbca) : []),
    ]);
    for (const name of allNames) {
      const entry = {seaad: new Map(), hbca: new Map()};
      if (seaad && seaad[name]) {
        for (const r of seaad[name]) {
          entry.seaad.set(r.celltype, {score: r.score, lfc: r.sea_ad_lfc, expr: r.mean_log2_expression});
          specClusters.add(r.celltype);
          if (r.score != null && isFinite(r.score)) {
            const mx = _KX_SEAAD_MAX_BY_NAME.get(name);
            if (!mx || r.score > mx.score) _KX_SEAAD_MAX_BY_NAME.set(name, {score: r.score, cluster: r.celltype});
          }
        }
      }
      if (hbca && hbca[name]) {
        for (const r of hbca[name]) {
          entry.hbca.set(r.celltype, {score: r.score, expr: r.mean_log2_expression});
          specClusters.add(r.celltype);
        }
      }
      _KX_HUMAN_SPEC_BY_NAME.set(name, entry);
    }
  }

  _KX_F5_BY_KEY = new Map();
  _KX_F5_AGREE_BY_KEY = new Map();
  // _KX_F5_ATTR_BY_KEY / _KX_F5_CELLTYPES are built by _kxBuildF5AttrIndex() once the
  // 5xFAD attribution-summary sidecar (P2) loads — see _kxEnsureF5AttrData().
  const F5 = PAYLOAD.supporting_5xfad;
  if (F5) {
    for (const row of (F5.rows || [])) {
      if (row.analysis_track === "raw_phospho") continue;
      if (row.analysis_track && row.analysis_track !== "stoichiometry") continue;
      const tissue = row.tissue || "";
      if (!_KX_F5_TISSUES.includes(tissue)) continue;
      const residue = _kxNormResidue(row.residue_type || row.track);
      const key = _kxF5Key(row.kinase, residue, tissue);
      let rec = _KX_F5_BY_KEY.get(key);
      if (!rec) {
        rec = {
          kinase: row.kinase || "",
          gene: row.gene_symbol || row.kinase || "",
          tissue,
          residue,
          track: row.track || _kxF5TrackForResidue(residue),
          assay: row.assay || "",
          rows: new Map(),
        };
        _KX_F5_BY_KEY.set(key, rec);
      }
      rec.rows.set(Number(row.age_months), row);
    }
    for (const row of (F5.celltype_agreement_index || [])) {
      const tissue = row.tissue || "";
      if (!_KX_F5_TISSUES.includes(tissue)) continue;
      const key = `${row.kinase || ""}|${tissue}|${row.track || _kxF5TrackForResidue(row.residue_type)}|${Number(row.age_months)}`;
      _KX_F5_AGREE_BY_KEY.set(key, row);
    }
  }

  const clusters = Array.from(decompClusters);
  for (const c of specClusters) if (!decompClusters.has(c)) clusters.push(c);
  clusters.sort();
  _KX_CLUSTERS = clusters;

  for (const r of _KX_ROWS) r._human = _KX_HUMAN_BY_NAME.get(`${r.name}|${r.residue}`) || null;

  // Union: append human-only kinases (measured in Mukesh but not in the mouse
  // MEA) so the agreement view can categorize them. Mouse columns render dashes;
  // these rows are flagged _humanOnly so the detail shows a mouse-side placeholder.
  const seen = new Set(_KX_ROWS.map(r => `${r.name}|${r.residue}`));
  for (const [key, human] of _KX_HUMAN_BY_NAME) {
    if (seen.has(key)) continue;
    const bar = key.lastIndexOf("|");
    const name = key.slice(0, bar);
    const residue = key.slice(bar + 1);
    _KX_ROWS.push({
      kid: human.kid, name, gene: "", family: famMap[name] || "", residue,
      peak_NES: null, trajectory: "", n_sig_contrasts: 0, n_celltype_candidates: 0,
      _nes: {}, _fdr: {}, _human: human, _humanOnly: true,
    });
    seen.add(key);
  }

  if (_KX_F5_BY_KEY) {
    for (const rec of _KX_F5_BY_KEY.values()) {
      const key = `${rec.kinase}|${rec.residue}`;
      if (seen.has(key)) continue;
      _KX_ROWS.push({
        kid: null, name: rec.kinase, gene: rec.gene || "", family: famMap[rec.kinase] || "", residue: rec.residue,
        peak_NES: null, trajectory: "", n_sig_contrasts: 0, n_celltype_candidates: 0,
        _nes: {}, _fdr: {}, _human: null, _humanOnly: true, _f5Only: true,
      });
      seen.add(key);
    }
  }

  // Default pivot is "" = Any cell type (peak specificity across clusters).
  _KX_INITIALIZED = true;
}

// Build the 5xFAD attribution-summary index from the loaded sidecar rows (P2). The raw
// list lives in kinase_fivexfad.js (_f5AttributionSummaryRows); kept identical to the
// former inline build that ran inside _kxBuildIndexes.
function _kxBuildF5AttrIndex() {
  _KX_F5_ATTR_BY_KEY = new Map();
  const celltypes = new Set();
  const rows = (typeof _f5AttributionSummaryRows === "function") ? _f5AttributionSummaryRows() : [];
  for (const row of (rows || [])) {
    const tissue = row.tissue || "";
    if (!_KX_F5_TISSUES.includes(tissue)) continue;
    const key = _kxF5AttrKey(row.kinase, tissue);
    let rec = _KX_F5_ATTR_BY_KEY.get(key);
    if (!rec) {
      rec = {
        kinase: row.kinase || "",
        tissue,
        celltypes: [],
        best_confidence_tier: "none",
        top_fivexfad_fold_over_uniform: null,
      };
      _KX_F5_ATTR_BY_KEY.set(key, rec);
    }
    if (_kxConfRank(row.best_confidence_tier) > _kxConfRank(rec.best_confidence_tier)) {
      rec.best_confidence_tier = row.best_confidence_tier || "none";
    }
    const topFold = _kxF5Num(row.top_fivexfad_fold_over_uniform);
    if (topFold != null && (rec.top_fivexfad_fold_over_uniform == null || topFold > rec.top_fivexfad_fold_over_uniform)) {
      rec.top_fivexfad_fold_over_uniform = topFold;
    }
    for (const ct of (row.celltypes || [])) {
      if (!ct.cell_type) continue;
      celltypes.add(ct.cell_type);
      rec.celltypes.push({...ct, age_months: row.age_months});
    }
  }
  _KX_F5_CELLTYPES = Array.from(celltypes).sort();
}

// Ensure the 5xFAD attribution-summary sidecar is loaded, then build its index. On first
// call it kicks the (shared, promise-cached) fetch and re-renders when it resolves; the
// table renders immediately with mouse+human columns and the 5xFAD columns fill in on the
// follow-up render. Idempotent once ready.
function _kxEnsureF5AttrData() {
  if (_KX_F5_ATTR_READY) return;
  const block = (typeof _f5Block === "function") ? _f5Block() : null;
  if (!block || !block.celltype_attribution_summary_shard || typeof _f5EnsureShardData !== "function") {
    _kxBuildF5AttrIndex();
    _KX_F5_ATTR_READY = true;
    return;
  }
  _f5EnsureShardData().then(() => {
    _kxBuildF5AttrIndex();
    _KX_F5_ATTR_READY = true;
    _kxSyncControls();
    _kxRenderTable();
  });
}

// ---------- specificity tier badges ----------

// Log2 tier reuses _khSpecTierRank/_khSpecTierBadge from kinase_human.js (same
// thresholds, same CSS classes); WMB tier stays local because its uniform
// baseline depends on the active levy_t5 spine size.
function _kxLog2TierBadge(score, atCluster) {
  if (score == null || !isFinite(score)) return `<td class="muted">–</td>`;
  const at = atCluster ? ` @ ${atCluster}` : "";
  const rank = _khSpecTierRank(score);
  if (!rank) return `<td class="muted" title="log2 ${score.toFixed(3)}${_escapeHtml(at)}">&lt;1×</td>`;
  return `<td style="text-align:center;padding:2px 4px;" title="log2 ${score.toFixed(3)}${_escapeHtml(at)}">${_khSpecTierBadge(rank)}</td>`;
}
// Primary mouse specificity: Song location specificity (per kinase), as fold over
// even-split (1/31). The fold is the kinase's peak cell-type share; τ rides in the
// tooltip. `pinned` is the active cluster pivot (or "") — when a cluster is pinned
// we flag whether it IS the kinase's top cell type, but the tier is per-kinase (it
// does not vary by cluster the way the WMB share does).
function _kxSongTierBadge(song, pinned) {
  if (!song || song.topShare == null || !isFinite(song.topShare)) return `<td class="muted">–</td>`;
  const u = _KX_SONG_UNIFORM;
  const share = song.topShare;
  const top = song.topCluster || "?";
  const pct = `${(share * 100).toFixed(0)}% in ${top}`;
  const tauTxt = (song.tau != null && isFinite(song.tau)) ? ` · τ ${song.tau.toFixed(2)}` : "";
  let cls, label;
  if (share >= 10 * u) { cls = "badge vhi"; label = "≥10×"; }
  else if (share >= 5 * u) { cls = "badge hi";  label = "≥5×"; }
  else if (share >= 2 * u) { cls = "badge mid"; label = "≥2×"; }
  else if (share >= u)     { cls = "badge lo";  label = "≥1×"; }
  else { return `<td class="muted" title="below 1× even-split (share ${share.toFixed(3)} vs 1/31 ≈ ${u.toFixed(3)}) · ${_escapeHtml(pct)}${tauTxt}">&lt;1×</td>`; }
  let mark = "";
  if (pinned) mark = (pinned === top)
    ? ` <span title="this pinned cell type IS where it concentrates">★</span>`
    : ` <span class="muted" title="concentrates in ${_escapeHtml(top)}, not the pinned ${_escapeHtml(pinned)}">·</span>`;
  const tip = `Song location specificity ${(share / u).toFixed(1)}× even-split (share ${share.toFixed(3)} vs 1/31 ≈ ${u.toFixed(3)}) · ${pct}${tauTxt}`;
  return `<td style="text-align:center;padding:2px 4px;"><span class="${cls}" title="${_escapeHtml(tip)}">${label}</span>${mark}</td>`;
}
function _kxWmbTierBadge(frac, atCluster) {
  if (frac == null || !isFinite(frac)) return `<td class="muted">–</td>`;
  const u = _KX_WMB_UNIFORM;
  const at = atCluster ? ` @ ${atCluster}` : "";
  let cls, label;
  if (frac >= 10 * u) { cls = "badge vhi"; label = "≥10×"; }
  else if (frac >= 5 * u) { cls = "badge hi"; label = "≥5×"; }
  else if (frac >= 2 * u) { cls = "badge mid"; label = "≥2×"; }
  else if (frac >= u)     { cls = "badge lo"; label = "≥1×"; }
  else                    { return `<td class="muted" title="${frac.toFixed(4)} vs uniform ${u.toFixed(4)}${_escapeHtml(at)}">&lt;1×</td>`; }
  return `<td style="text-align:center;padding:2px 4px;"><span class="${cls}" title="WMB specificity ${frac.toFixed(4)} vs uniform ${u.toFixed(4)}${_escapeHtml(at)}">${label}</span></td>`;
}

// ---------- cell-type specificity corroboration (detail sub-tab) ----------

// Fold-over-the-average-cell-type → tier. WMB carries a fold-over-uniform
// directly; the human references (SEA-AD MTG, HBCA) carry a log2 ratio whose
// fold is 2^score (cell-type mean / brain-wide mean). Song uses its share with
// _msTier (fold over the 1/31 even-split). All four collapse to one vocabulary:
// ≥10× / ≥5× / ≥2× / ≥1× more concentrated than the atlas's average cell type.
function _kxFoldTier(fold) {
  if (fold == null || !isFinite(fold)) return 0;
  if (fold >= 10) return 10;
  if (fold >= 5) return 5;
  if (fold >= 2) return 2;
  if (fold >= 1) return 1;
  return 0;
}
// One pill renderer for every reference column so the design language is
// identical across Song / WMB / SEA-AD / HBCA (badge vhi/hi/mid/lo; muted <1×).
function _kxRefPill(tier, tip) {
  if (!tier) return `<td class="muted" title="${_escapeHtml(tip)}">&lt;1×</td>`;
  const cls = tier >= 10 ? "vhi" : tier >= 5 ? "hi" : tier >= 2 ? "mid" : "lo";
  return `<td style="text-align:center;padding:2px 4px;"><span class="badge ${cls}" title="${_escapeHtml(tip)}">≥${tier}×</span></td>`;
}

// Cluster-aligned reference-corroboration table for one kinase: where does it
// localize, and do the independent atlases agree with Song's call? Rows = the
// union of Levy-T5 clusters ranked by any reference, columns = the four fold
// pills (Song primary; WMB / SEA-AD / HBCA cross-checks). Song share + τ + WMB
// fold per cluster come from kinase_celltype_evidence (by kid); SEA-AD + HBCA
// scores from _KX_HUMAN_SPEC_BY_NAME (by name). Human-only kinases have no kid →
// Song/WMB dashed, human pills populated.
function _kxRenderSpecAligned(hostEl, row) {
  if (typeof _ensureKinaseIndexes === "function") _ensureKinaseIndexes();
  const EV = PAYLOAD.kinase_celltype_evidence || {cell_type: []};
  const byCluster = new Map();
  const at = (c) => { let o = byCluster.get(c); if (!o) { o = {cluster: c}; byCluster.set(c, o); } return o; };

  const kid = row._humanOnly ? null : row.kid;
  if (kid != null && typeof _evidenceByKinase !== "undefined" && _evidenceByKinase) {
    for (const k of (_evidenceByKinase.get(kid) || [])) {
      const c = EV.cell_type[k];
      if (!c) continue;
      const o = at(c);
      const sShare = EV.song_specificity ? EV.song_specificity[k] : null;
      o.song = (sShare != null && isFinite(sShare)) ? sShare : null;
      o.songTau = EV.song_tau ? EV.song_tau[k] : null;
      o.wmbFold = EV.wmb_fold ? EV.wmb_fold[k] : null;
    }
  }
  const spec = _KX_HUMAN_SPEC_BY_NAME.get(row.name);
  if (spec) {
    for (const [c, e] of spec.seaad) { const o = at(c); o.seaad = e.score; o.seaadLfc = e.lfc; }
    for (const [c, e] of spec.hbca) { const o = at(c); o.hbca = e.score; }
  }

  const rows = Array.from(byCluster.values());
  if (!rows.length) {
    hostEl.innerHTML = `<div class="kx-detail-placeholder muted">No cell-type specificity references for this kinase.</div>`;
    return;
  }
  for (const o of rows) {
    o._t = {
      song: o.song != null ? _msTier(o.song) : 0,
      wmb: _kxFoldTier(o.wmbFold),
      seaad: o.seaad != null ? _kxFoldTier(Math.pow(2, o.seaad)) : 0,
      hbca: o.hbca != null ? _kxFoldTier(Math.pow(2, o.hbca)) : 0,
    };
  }
  // Rank by Song first when the kinase has Song data; the other references are
  // cross-checks. Human-only kinases have no Song side, so they fall back to the
  // strongest available human/cross-reference tier.
  rows.sort((a, b) => {
    const hasSongA = a.song != null && isFinite(a.song);
    const hasSongB = b.song != null && isFinite(b.song);
    if (hasSongA !== hasSongB) return hasSongB - hasSongA;
    if (hasSongA && hasSongB) {
      if (a._t.song !== b._t.song) return b._t.song - a._t.song;
      if ((a.song || 0) !== (b.song || 0)) return (b.song || 0) - (a.song || 0);
    }
    const ma = Math.max(a._t.wmb, a._t.seaad, a._t.hbca);
    const mb = Math.max(b._t.wmb, b._t.seaad, b._t.hbca);
    return mb !== ma ? mb - ma : (a.cluster || "").localeCompare(b.cluster || "");
  });

  const u = _KX_SONG_UNIFORM;
  const body = rows.map(o => {
    const t = o._t;
    const songTip = o.song != null
      ? `Song ${(o.song / u).toFixed(1)}× even-split (share ${o.song.toFixed(3)} vs 1/31 ≈ ${u.toFixed(3)})${(o.songTau != null && isFinite(o.songTau)) ? ` · τ ${o.songTau.toFixed(2)}` : ""}`
      : "Not measured in mouse (Song)";
    const wmbTip = (o.wmbFold != null && isFinite(o.wmbFold)) ? `WMB ${o.wmbFold.toFixed(2)}× uniform` : "No WMB cross-check at this cluster";
    const seaTip = o.seaad != null
      ? `SEA-AD expr ${Math.pow(2, o.seaad).toFixed(1)}× brain mean (log2 ${o.seaad.toFixed(2)}). Location evidence only; SEA-AD LFC is shown in the NES tab.`
      : "No SEA-AD cross-check at this cluster";
    const hbcaTip = o.hbca != null ? `HBCA ${Math.pow(2, o.hbca).toFixed(1)}× brain mean (log2 ${o.hbca.toFixed(2)})` : "No HBCA cross-check at this cluster";
    const songCell = kid == null ? `<td class="muted" title="Not measured in mouse (Song)">–</td>` : _kxRefPill(t.song, songTip);
    const wmbCell = kid == null ? `<td class="muted" title="Not measured in mouse (WMB)">–</td>` : _kxRefPill(t.wmb, wmbTip);
    return `<tr><td>${_escapeHtml(o.cluster)}</td>${songCell}${wmbCell}${_kxRefPill(t.seaad, seaTip)}${_kxRefPill(t.hbca, hbcaTip)}</tr>`;
  }).join("");

  const head = `<thead><tr>` +
    `<th title="Levy-T5 cluster — the shared axis: WMB classes, SEA-AD supertypes, and HBCA superclusters are all rolled up to this nomenclature before ranking.">Cell type</th>` +
    `<th title="Song location specificity — the primary mouse call. This kinase's share of expression in this cluster across the levy_t5 pseudobulk, as fold over the even-split baseline (1/31 ≈ 0.032). τ in tooltip.">Song</th>` +
    `<th title="WMB (Allen Whole Mouse Brain) atlas cross-check — fold over the retained-class uniform (1/9 ≈ 0.111).">WMB</th>` +
    `<th title="SEA-AD MTG human expression reference — 2^log2(cell-type mean / brain-wide mean). Location evidence only; SEA-AD LFC is shown in the NES tab.">SEA-AD expr</th>` +
    `<th title="Allen HBCA human atlas cross-check — 2^log2(cell-type mean / brain-wide mean).">HBCA</th>` +
    `</tr></thead>`;
  hostEl.innerHTML =
    `<p class="kx-spec-note muted">Cell-type localization of <b>${_escapeHtml(row.gene || row.name)}</b> across reference atlases, on the shared Levy-T5 axis. ` +
    `<b>Song</b> is the primary mouse call; <b>WMB</b> (mouse), <b>SEA-AD expr</b> and <b>HBCA</b> (human) are independent cross-checks. ` +
    `Pills are fold over each atlas's average cell type: ≥10× / ≥5× / ≥2× / ≥1×.</p>` +
    `<div class="kx-spec-table-wrap"><table class="data-table kx-spec-table">${head}<tbody>${body}</tbody></table></div>`;
}

// ---------- cross-dataset agreement model ----------

// Labels + badge styling for the mouse/human direction categories.
const _KX_AGREE_META = {
  "concordant-up":   {cls: "badge vhi", glyph: "↑ same",     label: "Same direction — up",     tip: "Significant in BOTH datasets, both up in disease."},
  "concordant-down": {cls: "badge hi",  glyph: "↓ same",     label: "Same direction — down",   tip: "Significant in BOTH datasets, both down in disease."},
  "discordant":      {cls: "badge mix", glyph: "opposite",   label: "Opposite direction",      tip: "Significant in both datasets but opposite direction."},
  "mouse-only":      {cls: "badge mid", glyph: "M only",    label: "Mouse-only",        tip: "Significant in mouse (Song) only."},
  "human-only":      {cls: "badge imp", glyph: "H only",    label: "Human-only",        tip: "Significant in human (Mukesh) only."},
  "neither":         {cls: "badge lo",  glyph: "–",         label: "Neither",           tip: "Not significant in either dataset at this FDR."},
};

function _kxMedian(arr) {
  if (!arr.length) return null;
  const a = arr.slice().sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function _kxConfRank(conf) {
  const rank = {none: 0, low: 1, moderate: 2, high: 3, very_high: 4};
  return rank[conf || "none"] || 0;
}

function _kxConfPass(conf, threshold) {
  if (!threshold) return true;
  return _kxConfRank(conf || "none") >= _kxConfRank(threshold);
}

function _kxMeasuredDirection(nes) {
  const n = _kxF5Num(nes);
  if (n == null || n === 0) return 0;
  return n > 0 ? 1 : -1;
}

function _kxSourceFromValues(values, fdrGate, allSamples) {
  const measured = values
    .filter(v => v && _kxF5Num(v.nes) != null)
    .map(v => ({nes: _kxF5Num(v.nes), fdr: _kxF5Num(v.fdr)}));
  if (!measured.length) {
    return {status: "missing", sig: false, measured: false, nes: null, measuredNes: null, dir: null, measuredDir: null, nSig: 0, nMeasured: 0};
  }
  const sigVals = measured.filter(v => v.fdr != null && v.fdr < fdrGate);
  const measuredNes = _kxMedian(measured.map(v => v.nes));
  const sigNes = sigVals.length ? _kxMedian(sigVals.map(v => v.nes)) : null;
  const callNes = allSamples ? measuredNes : sigNes;
  const status = _kxDirectionStatus(callNes, true, allSamples || sigVals.length > 0);
  return {
    status,
    sig: _kxStatusIsSig(status),
    measured: true,
    nes: callNes,
    measuredNes,
    dir: _kxStatusDir(status),
    measuredDir: _kxMeasuredDirection(measuredNes),
    nSig: sigVals.length,
    nMeasured: measured.length,
  };
}

function _kxF5Rec(row, tissue) {
  return _KX_F5_BY_KEY ? (_KX_F5_BY_KEY.get(_kxF5Key(row.name, row.residue, tissue)) || null) : null;
}

function _kxF5AttrSummary(row, tissue) {
  return _KX_F5_ATTR_BY_KEY ? (_KX_F5_ATTR_BY_KEY.get(_kxF5AttrKey(row.name, tissue)) || null) : null;
}

function _kxF5AttrRows(row, tissue) {
  const s = _kxF5AttrSummary(row, tissue);
  return s && Array.isArray(s.celltypes) ? s.celltypes : [];
}

function _kxF5WmbTier(attrRow) {
  const spec = _kxF5Num(attrRow && attrRow.wmb_specificity);
  if (typeof _wmbTier === "function") return _wmbTier(spec);
  return _kxFoldTier(spec);
}

function _kxF5NativeTier(attrRow) {
  const fold = _kxF5Num(attrRow && attrRow.fivexfad_fold_over_uniform);
  return _kxFoldTier(fold);
}

function _kxF5ScopedAttrRows(row, tissue, state) {
  const s = state || _kxState();
  return _kxF5AttrRows(row, tissue).filter(r => {
    if (s.f5Celltype && r.cell_type !== s.f5Celltype) return false;
    if (!_kxConfPass(r.confidence_tier || "none", s.f5Confidence)) return false;
    if (s.f5SnrnaMin && _kxF5NativeTier(r) < Number(s.f5SnrnaMin)) return false;
    if (s.f5WmbMin && _kxF5WmbTier(r) < Number(s.f5WmbMin)) return false;
    return true;
  });
}

function _kxF5TissueSource(row, tissue, fdrGate, allSamples) {
  const rec = _kxF5Rec(row, tissue);
  if (!rec) {
    return {status: "missing", sig: false, measured: false, nes: null, measuredNes: null, dir: null, measuredDir: null, nSig: 0, nMeasured: 0, _values: []};
  }
  const values = _KX_F5_AGES.map(age => {
    const r = rec.rows.get(age);
    return r ? {nes: r.NES, fdr: r.FDR} : null;
  }).filter(Boolean);
  const source = _kxSourceFromValues(values, fdrGate, allSamples);
  source._values = values
    .map(v => ({nes: _kxF5Num(v.nes), fdr: _kxF5Num(v.fdr)}))
    .filter(v => v.nes != null);
  if (!allSamples) {
    const dirs = new Set(values
      .filter(v => _kxF5Num(v.fdr) != null && _kxF5Num(v.fdr) < fdrGate)
      .map(v => _kxMeasuredDirection(v.nes))
      .filter(Boolean));
    if (dirs.size > 1) {
      source.status = "mixed_sig";
      source.sig = true;
      source.dir = 0;
    }
  }
  return source;
}

function _kxAggregateF5(cortex, hippocampus, fdrGate, allSamples) {
  const parts = [cortex, hippocampus];
  const values = parts.flatMap(p => p._values || []);
  const sigValues = values.filter(v => v.fdr != null && v.fdr < fdrGate);
  const displayValues = allSamples ? values : sigValues;
  const displayMedian = displayValues.length ? _kxMedian(displayValues.map(v => v.nes)) : null;
  const measuredMedian = values.length ? _kxMedian(values.map(v => v.nes)) : null;
  if (parts.some(p => p.status === "mixed_sig")) {
    return {status: "mixed_sig", sig: true, measured: true, nes: displayMedian, measuredNes: measuredMedian, dir: 0, measuredDir: 0,
      nSig: (cortex.nSig || 0) + (hippocampus.nSig || 0), nMeasured: (cortex.nMeasured || 0) + (hippocampus.nMeasured || 0)};
  }
  const sigDirs = parts.filter(p => _kxStatusIsSig(p.status)).map(p => p.dir);
  if (sigDirs.includes(1) && sigDirs.includes(-1)) {
    return {status: "mixed_sig", sig: true, measured: true, nes: displayMedian, measuredNes: measuredMedian, dir: 0, measuredDir: 0,
      nSig: (cortex.nSig || 0) + (hippocampus.nSig || 0), nMeasured: (cortex.nMeasured || 0) + (hippocampus.nMeasured || 0)};
  }
  if (sigDirs.includes(1) || sigDirs.includes(-1)) {
    const dir = sigDirs.includes(1) ? 1 : -1;
    return {status: dir > 0 ? "sig_up" : "sig_down", sig: true, measured: true, nes: displayMedian, measuredNes: measuredMedian,
      dir, measuredDir: dir, nSig: (cortex.nSig || 0) + (hippocampus.nSig || 0), nMeasured: (cortex.nMeasured || 0) + (hippocampus.nMeasured || 0)};
  }
  const measured = parts.filter(p => p.measured);
  if (!measured.length) {
    return {status: "missing", sig: false, measured: false, nes: null, measuredNes: null, dir: null, measuredDir: null, nSig: 0, nMeasured: 0};
  }
  const dirs = new Set(measured.map(p => p.measuredDir).filter(Boolean));
  return {status: "nonsig_measured", sig: false, measured: true, nes: null, measuredNes: measuredMedian,
    dir: 0, measuredDir: dirs.size === 1 ? Array.from(dirs)[0] : 0, nSig: 0, nMeasured: (cortex.nMeasured || 0) + (hippocampus.nMeasured || 0)};
}

function _kxComparePair(a, b) {
  if (!a.measured && !b.measured) return "missing";
  if (!a.measured || !b.measured) return "missing_one";
  if (a.status === "mixed_sig" || b.status === "mixed_sig") return "mixed_sig";
  if (a.sig && b.sig) return a.dir === b.dir ? "sig_same" : "sig_opposite";
  if (a.sig || b.sig) return "source_only";
  if (a.measuredDir && b.measuredDir && a.measuredDir === b.measuredDir) return "directional_same";
  if (a.measuredDir && b.measuredDir && a.measuredDir !== b.measuredDir) return "directional_opposite";
  return "measured";
}

function _kxThreeWayState(r) {
  const sources = [r._songSource, r._mukeshSource, r._f5Source];
  if (r._f5Source.status === "mixed_sig") return "mixed_sig";
  const measured = sources.filter(s => s && s.measured);
  if (measured.length < 3) return "missing_one";
  const sig = sources.filter(s => s.sig);
  if (sig.length === 3) {
    const dirs = new Set(sig.map(s => s.dir));
    return dirs.size === 1 ? "sig_same" : "sig_opposite";
  }
  if (sig.length >= 2 && new Set(sig.map(s => s.dir)).size > 1) return "sig_opposite";
  const dirs = new Set(measured.map(s => s.measuredDir).filter(Boolean));
  if (dirs.size === 1) return "directional_same";
  if (dirs.size > 1) return "directional_opposite";
  return "measured";
}

function _kxTissueState(r) {
  const c = r._f5Tissues && r._f5Tissues.cortex;
  const h = r._f5Tissues && r._f5Tissues.hippocampus;
  if (!c || !h || (!c.measured && !h.measured)) return "missing";
  if (!c.measured || !h.measured) return "missing_one";
  if (c.status === "mixed_sig" || h.status === "mixed_sig" || r._f5Source.status === "mixed_sig") return "mixed_sig";
  if (c.sig && h.sig) return c.dir === h.dir ? "sig_same" : "mixed_sig";
  if (c.sig || h.sig) return c.sig ? "cortex_only" : "hippocampus_only";
  if (c.measuredDir && h.measuredDir && c.measuredDir === h.measuredDir) return "directional_same";
  if (c.measuredDir && h.measuredDir && c.measuredDir !== h.measuredDir) return "directional_opposite";
  return "measured";
}

function _kxComparisonState(r, scope) {
  if (scope === "song_mukesh") return _kxComparePair(r._songSource, r._mukeshSource);
  if (scope === "song_f5") return _kxComparePair(r._songSource, r._f5Source);
  if (scope === "mukesh_f5") return _kxComparePair(r._mukeshSource, r._f5Source);
  if (scope === "f5_tissue") return _kxTissueState(r);
  return _kxThreeWayState(r);
}

function _kxStatePassesComparison(r, scope, state) {
  if (!state) return true;
  if (!_KX_STATE_FILTER_VALUES.has(state)) return true;
  const cat = _kxComparisonState(r, scope);
  if (state === "sig_agree") return cat === "sig_same";
  if (state === "agree") return cat === "sig_same" || cat === "directional_same";
  if (state === "sig_disagree") return cat === "sig_opposite" || cat === "mixed_sig";
  if (state === "disagree") return cat === "sig_opposite" || cat === "directional_opposite" || cat === "mixed_sig";
  return cat === state;
}

function _kxComparisonMeta(scope, state) {
  const scopeLabel = _KX_COMPARE_LABELS[scope] || _KX_COMPARE_LABELS.three_way;
  const map = {
    sig_same: {cls: "badge vhi", glyph: "sig same", label: "Significant same direction", tip: `${scopeLabel}: all required significant sources point the same direction.`},
    directional_same: {cls: "badge hi", glyph: "dir same", label: "Directional agreement", tip: `${scopeLabel}: measured sources point the same direction, allowing nonsignificant evidence.`},
    sig_opposite: {cls: "badge mix", glyph: "opposite", label: "Opposite significant direction", tip: `${scopeLabel}: significant sources oppose each other.`},
    directional_opposite: {cls: "badge mix", glyph: "dir opp", label: "Directional opposition", tip: `${scopeLabel}: measured sources point opposite directions without all sides significant.`},
    source_only: {cls: "badge mid", glyph: "one side", label: "Source-only", tip: `${scopeLabel}: only one side is significant at the active FDR.`},
    cortex_only: {cls: "badge mid", glyph: "ctx only", label: "Cortex-only", tip: "5xFAD cortex is significant and hippocampus is not."},
    hippocampus_only: {cls: "badge mid", glyph: "hip only", label: "Hippocampus-only", tip: "5xFAD hippocampus is significant and cortex is not."},
    mixed_sig: {cls: "badge mix", glyph: "mixed_sig", label: "5xFAD mixed_sig", tip: "5xFAD cortex and hippocampus contain significant opposing evidence."},
    missing_one: {cls: "badge imp", glyph: "missing", label: "Missing one side", tip: `${scopeLabel}: at least one required source is missing.`},
    missing: {cls: "badge lo", glyph: "n/a", label: "Missing", tip: `${scopeLabel}: no comparable evidence is measured.`},
    measured: {cls: "badge lo", glyph: "measured", label: "Measured", tip: `${scopeLabel}: evidence is measured but does not meet a stronger agreement category.`},
  };
  return map[state] || map.measured;
}

// Stamp _mouseSig / _humanSig / _agreeCategory / _agreeScore on each row, live
// against the FDR gate. BOTH summary NES are the median over the dataset's units
// (mouse contrasts / human AD donors) that are significant at the gate — the same
// statistic on both sides, so the magnitudes and direction signs are comparable
// (peak vs median would bias the mouse magnitude upward). Significance and sign
// are both derived from the SAME live threshold.
// allSamples = include EVERY measured unit in the median (and treat "measured"
// as the inclusion criterion for the category), instead of only units FDR-
// significant at the gate. Lets the user see the all-sample direction/agreement.
function _kxComputeAgreement(rows, fdrGate, allSamples) {
  for (const r of rows) {
    const songVals = [];
    for (const c of (CONTRASTS || [])) songVals.push({nes: r._nes[c], fdr: r._fdr[c]});
    const song = _kxSourceFromValues(songVals, fdrGate, allSamples);

    const humanVals = [];
    const pd = r._human ? _KX_HUMAN_PERDONOR.get(r._human.kid) : null;
    if (pd) for (const d of _KX_AD_DONORS) {
      const e = pd.get(d);
      if (e) humanVals.push({nes: e.nes, fdr: e.fdr});
    }
    const mukesh = _kxSourceFromValues(humanVals, fdrGate, allSamples);

    const cortex = _kxF5TissueSource(r, "cortex", fdrGate, allSamples);
    const hippocampus = _kxF5TissueSource(r, "hippocampus", fdrGate, allSamples);
    const f5 = _kxAggregateF5(cortex, hippocampus, fdrGate, allSamples);

    r._songSource = song;
    r._mukeshSource = mukesh;
    r._f5Tissues = {cortex, hippocampus};
    r._f5Source = f5;
    r._mouseSig = song.sig; r._humanSig = mukesh.sig;
    r._mNes = song.nes; r._hNes = mukesh.nes; r._f5Nes = f5.nes;

    const cat = _kxComparisonState(r, _kxState().compareScope || "three_way");
    r._agreeCategory = cat;
    // Internal sort rank only: categories first, then available evidence breadth.
    const rank = {sig_same: 6, directional_same: 5, source_only: 4, cortex_only: 4, hippocampus_only: 4, measured: 3, missing_one: 2, missing: 1, directional_opposite: -2, mixed_sig: -3, sig_opposite: -4};
    r._agreeScore = (rank[cat] || 0) * 100 + (song.nSig || 0) + (mukesh.nSig || 0) + (f5.nSig || 0);
  }
}

// Median-of-significant NES driving the Agree call. "–" when nothing is
// significant on that side at the live gate (so the row's category is *-only or
// neither, not same/opposite direction). Colored by sign, same red/blue as glyphs.
function _kxMedNesCell(val) {
  if (val == null || !isFinite(val)) return `<td class="muted kx-nes-num">–</td>`;
  const col = val >= 0 ? "#c53030" : "#2b6cb0";
  return `<td class="kx-nes-num" style="color:${col};" title="median NES driving the Agree call">${val >= 0 ? "+" : ""}${val.toFixed(2)}</td>`;
}

function _kxAgreeCategoryCell(cat) {
  const s = _kxState();
  const m = _kxComparisonMeta(s.compareScope || "three_way", cat);
  return `<td style="text-align:center;padding:2px 4px;"><span class="${m.cls}" title="${_escapeHtml(m.tip)}">${m.glyph}</span></td>`;
}

// Mouse NES-profile glyph: adapt the crosstable row's contrast-keyed _nes/_fdr
// dicts to the array form (indexed by CONTRASTS) that _renderNesProfile expects,
// reusing the kinase-explorer glyph verbatim. maxAbs=3 matches the score scale.
function _kxMakeNesProfileRow(r) {
  const nesArr = [], fdrArr = [];
  for (let ci = 0; ci < (CONTRASTS || []).length; ci++) {
    const c = CONTRASTS[ci];
    nesArr[ci] = (r._nes && r._nes[c] != null) ? r._nes[c] : null;
    fdrArr[ci] = (r._fdr && r._fdr[c] != null) ? r._fdr[c] : null;
  }
  return {id: r.kid, _nes: nesArr, _fdr: fdrArr};
}
function _kxMouseGlyphCell(r, fdrGate) {
  if (r._humanOnly) return `<td class="kx-mglyph muted" title="Not measured in mouse">–</td>`;
  return `<td class="kx-mglyph">${_renderNesProfile(_kxMakeNesProfileRow(r), fdrGate, 3)}</td>`;
}

// Human per-donor NES strip — same .npc markup + color encoding as
// _khRenderProfile, kept local because that function dereferences the human
// tab's mutable globals (_KH/_KHState), which are null until the human tab
// renders. AD donors first, then a muted CTRL group (for reference — controls
// are scored against the CTRL mean so they bias toward zero by design).
function _kxDonorCells(pd, donorList, fdrGate) {
  const cells = [];
  for (const d of donorList) {
    const e = pd.get(d);
    const nes = e ? e.nes : null;
    const fdrV = e ? e.fdr : null;
    const sig = fdrV != null && isFinite(fdrV) && fdrV < fdrGate;
    let bg = "#fff";
    if (nes != null && isFinite(nes)) {
      const a = Math.min(1, Math.abs(nes) / 3);
      const rgb = nes >= 0 ? [197, 48, 48] : [43, 108, 176];
      bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
    }
    const tip = nes == null ? `${d}: n/a`
      : `${d}: NES ${nes.toFixed(2)}${fdrV != null ? `, FDR ${fdrV.toExponential(1)}` : ""}${sig ? " (sig)" : ""}`;
    cells.push(`<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_escapeHtml(tip)}"></div>`);
  }
  return cells;
}
function _kxHumanGlyphCell(r, fdrGate) {
  const pd = r._human ? _KX_HUMAN_PERDONOR.get(r._human.kid) : null;
  if (!pd || !_KX_AD_DONORS.length) return `<td class="kx-hglyph muted" title="Not measured in human">–</td>`;
  const adCells = _kxDonorCells(pd, _KX_AD_DONORS, fdrGate);
  const adBlock = `<div class="nes-profile-cell" style="grid-template-columns:repeat(${adCells.length || 1},1fr);">${adCells.join("")}</div>`;
  const ctrlCells = _KX_CTRL_DONORS.length ? _kxDonorCells(pd, _KX_CTRL_DONORS, fdrGate) : [];
  const ctrlBlock = ctrlCells.length
    ? `<span class="nes-profile-spacer" aria-hidden="true"></span><div class="nes-profile-cell kh-ctrl-group" style="grid-template-columns:repeat(${ctrlCells.length},1fr);" title="CTRL donors scored against the same CTRL mean — reference only; they bias toward zero by design.">${ctrlCells.join("")}</div>`
    : "";
  return `<td class="kx-hglyph"><div class="nes-profile-wrap">${adBlock}${ctrlBlock}</div></td>`;
}

function _kxF5Tile(row, tissue, age, fdrGate) {
  const rec = _kxF5Rec(row, tissue);
  const entry = rec ? rec.rows.get(age) : null;
  if (!entry) return `<div class="npc" title="${_escapeHtml(`5xFAD ${tissue} ${age}mo: not measured`)}"></div>`;
  const nes = _kxF5Num(entry.NES);
  const fdr = _kxF5Num(entry.FDR);
  const sig = fdr != null && fdr < fdrGate;
  let bg = "#fff";
  if (nes != null) {
    const a = Math.min(1, Math.abs(nes) / 3);
    const rgb = nes >= 0 ? [197, 48, 48] : [43, 108, 176];
    bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
  }
  const title = `5xFAD ${tissue} ${age}mo TG vs WT: NES ${nes == null ? "n/a" : nes.toFixed(2)}, FDR ${fdr == null ? "n/a" : fdr.toExponential(1)}${sig ? " (sig)" : ""}`;
  return `<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_escapeHtml(title)}"></div>`;
}

function _kxF5GlyphCell(row, tissue, fdrGate) {
  const rec = _kxF5Rec(row, tissue);
  if (!rec) return `<td class="kx-f5glyph muted" title="5xFAD ${_escapeHtml(tissue)} not measured">–</td>`;
  const labels = _KX_F5_AGES.map(age => `<span>${age}</span>`).join("");
  const cells = _KX_F5_AGES.map(age => _kxF5Tile(row, tissue, age, fdrGate)).join("");
  const source = row._f5Tissues && row._f5Tissues[tissue] ? row._f5Tissues[tissue] : null;
  const meta = _KX_STATUS_META[source ? source.status : "missing"] || _KX_STATUS_META.missing;
  return `<td class="kx-f5glyph" title="${_escapeHtml(`5xFAD ${tissue}: ${meta.label}`)}">` +
    `<div class="nes-profile-wrap kx-f5-wrap">` +
    `<span class="kx-f5-tissue">${_escapeHtml(tissue === "hippocampus" ? "Hip" : "Ctx")}</span>` +
    `<div class="nes-profile-age-stack">` +
    `<div class="nes-profile-age-labels">${labels}</div>` +
    `<div class="nes-profile-cell" style="grid-template-columns:repeat(4,1fr);">${cells}</div>` +
    `</div></div></td>`;
}

// ---------- NES-tab LFC direction support ----------

function _kxFinite(v) {
  return v != null && isFinite(Number(v));
}

function _kxSigned(v, digits) {
  if (!_kxFinite(v)) return "–";
  const n = Number(v);
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
}

function _kxDirectionSupport(nes, lfc) {
  const n = _kxFinite(nes) ? Number(nes) : null;
  const l = _kxFinite(lfc) ? Number(lfc) : null;
  if (n == null || l == null || n === 0 || Math.abs(l) < 0.1) {
    return {
      cls: "badge lo",
      label: "no clear change",
      tip: "Needs a signed NES and |LFC| ≥ 0.1 to call direction support.",
    };
  }
  const same = (n > 0) === (l > 0);
  return same
    ? {cls: "badge hi", label: "same direction", tip: "LFC sign matches the NES activity direction."}
    : {cls: "badge mix", label: "opposite direction", tip: "LFC sign is opposite the NES activity direction."};
}

function _kxBestSongLfcEvidence(row, cluster, useTopCluster = true) {
  if (row._humanOnly || row.kid == null) return null;
  const AI = PAYLOAD.attribution_index || {};
  if (!AI.kinase_id) return null;
  const song = _KX_SONG_BY_KID.get(row.kid) || null;
  const preferredCell = cluster || (useTopCluster && song && song.topCluster) || "";
  const candidates = [];
  for (let j = 0; j < AI.kinase_id.length; j++) {
    if (AI.kinase_id[j] !== row.kid) continue;
    const cell = AI.cell_type ? AI.cell_type[j] : "";
    if (preferredCell && cell !== preferredCell) continue;
    const lfc = AI.song_lfc ? AI.song_lfc[j] : null;
    if (!_kxFinite(lfc)) continue;
    const contrastId = AI.contrast_id ? AI.contrast_id[j] : null;
    const contrast = CONTRASTS && contrastId != null ? CONTRASTS[contrastId] : "";
    const nes = (AI.nes && _kxFinite(AI.nes[j])) ? AI.nes[j]
      : (contrast && row._nes ? row._nes[contrast] : null);
    candidates.push({
      source: "Song LFC",
      scope: `${cell || "cell type n/a"}${contrast ? ` · ${contrast}` : ""}`,
      lfc: Number(lfc),
      nes: _kxFinite(nes) ? Number(nes) : null,
      fdr: AI.fdr ? AI.fdr[j] : null,
      confidence: AI.confidence_tier ? AI.confidence_tier[j] : "",
    });
  }
  if (!candidates.length && preferredCell) {
    // If the top/pinned cell type has no LFC row, fall back to the strongest
    // available Song LFC for the kinase so the NES tab still surfaces the evidence.
    return _kxBestSongLfcEvidence(row, "", false);
  }
  candidates.sort((a, b) => Math.abs(b.lfc) - Math.abs(a.lfc));
  return candidates[0] || null;
}

function _kxRenderDirectionSupport(hostEl, row) {
  if (!hostEl) return;
  const s = _kxState();
  const rows = [];
  const song = _kxBestSongLfcEvidence(row, s.cluster);
  if (song) rows.push(song);
  const human = row._human || null;
  if (human && _kxFinite(human.sea_ad_lfc)) {
    rows.push({
      source: "SEA-AD LFC",
      scope: `${human.sea_ad_n || "n/a"} human MTG supertypes`,
      lfc: Number(human.sea_ad_lfc),
      nes: row._hNes,
      fdr: null,
      confidence: "",
    });
  }
  if (row._f5Source && row._f5Source.measured && row._f5Source.status !== "mixed_sig") {
    const f5Attrs = _KX_F5_TISSUES.flatMap(t => _kxF5ScopedAttrRows(row, t, s)
      .map(a => ({...a, _tissue: t})))
      .filter(a => _kxF5Num(a.fivexfad_lfc) != null)
      .sort((a, b) => Math.abs(_kxF5Num(b.fivexfad_lfc) || 0) - Math.abs(_kxF5Num(a.fivexfad_lfc) || 0));
    const best = f5Attrs[0];
    if (best) {
      rows.push({
        source: "5xFAD snRNA LFC",
        scope: `${best._tissue} · ${best.cell_type || "cell type n/a"}`,
        lfc: Number(best.fivexfad_lfc),
        nes: row._f5Source.nes || row._f5Source.measuredNes,
        fdr: null,
        confidence: best.confidence_tier || "",
      });
    }
  }

  if (!rows.length) {
    hostEl.innerHTML = `<div class="kx-direction-support muted">No Song LFC or SEA-AD LFC direction evidence for this kinase.</div>`;
    return;
  }
  const body = rows.map(r => {
    const support = _kxDirectionSupport(r.nes, r.lfc);
    const fdrTxt = _kxFinite(r.fdr) ? Number(r.fdr).toFixed(3) : "–";
    const tip = `${support.tip} NES ${_kxSigned(r.nes, 2)} · LFC ${_kxSigned(r.lfc, 3)}`;
    return `<tr>` +
      `<td>${_escapeHtml(r.source)}</td>` +
      `<td class="muted">${_escapeHtml(r.scope)}</td>` +
      `<td class="kx-nes-num">${_kxSigned(r.nes, 2)}</td>` +
      `<td class="kx-nes-num">${_kxSigned(r.lfc, 3)}</td>` +
      `<td class="kx-nes-num">${_escapeHtml(fdrTxt)}</td>` +
      `<td><span class="${support.cls}" title="${_escapeHtml(tip)}">${_escapeHtml(support.label)}</span></td>` +
      `</tr>`;
  }).join("");
  hostEl.innerHTML =
    `<section class="kx-direction-support">` +
    `<div class="kx-detail-col-head">Direction support</div>` +
    `<p class="muted">LFC evidence is interpreted with NES activity, not as cell-type specificity.</p>` +
    `<table class="data-table kx-direction-table"><thead><tr>` +
    `<th title="LFC evidence source.">Source</th>` +
    `<th title="Cell type / contrast or human summary scope used for this direction check.">Scope</th>` +
    `<th title="NES activity direction used as the reference sign.">NES</th>` +
    `<th title="Disease-vs-control transcript log2 fold change.">LFC</th>` +
    `<th title="FDR for the matching Song attribution row, when available.">FDR</th>` +
    `<th title="Whether the LFC sign supports the NES activity direction.">Support</th>` +
    `</tr></thead><tbody>${body}</tbody></table>` +
    `</section>`;
}

function _kxStatusBadge(status, title) {
  const m = _KX_STATUS_META[status || "missing"] || _KX_STATUS_META.missing;
  return `<span class="${m.cls}" title="${_escapeHtml(title || m.label)}">${_escapeHtml(m.label)}</span>`;
}

function _kxF5AgreementFor(row, tissue, age) {
  const track = _kxF5TrackForResidue(row.residue);
  return _KX_F5_AGREE_BY_KEY ? (_KX_F5_AGREE_BY_KEY.get(`${row.name}|${tissue}|${track}|${age}`) || null) : null;
}

function _kxRenderF5AgeTable(row, tissue, fdrGate) {
  const rec = _kxF5Rec(row, tissue);
  if (!rec) return `<div class="kx-detail-placeholder muted">5xFAD ${_escapeHtml(tissue)} not measured.</div>`;
  const body = _KX_F5_AGES.map(age => {
    const r = rec.rows.get(age);
    const agr = _kxF5AgreementFor(row, tissue, age);
    if (!r) {
      return `<tr><td>${age}mo</td><td colspan="5" class="muted">not measured</td></tr>`;
    }
    const nes = _kxF5Num(r.NES);
    const fdr = _kxF5Num(r.FDR);
    const sig = fdr != null && fdr < fdrGate;
    const col = nes == null ? "" : (nes >= 0 ? "color:#c53030;" : "color:#2b6cb0;");
    return `<tr>` +
      `<td>${age}mo</td>` +
      `<td class="kx-nes-num" style="${col}">${nes == null ? "–" : `${nes >= 0 ? "+" : ""}${nes.toFixed(2)}`}</td>` +
      `<td class="kx-nes-num">${fdr == null ? "–" : fdr.toExponential(1)}</td>` +
      `<td>${sig ? `<span class="badge hi">sig</span>` : `<span class="muted">n/s</span>`}</td>` +
      `<td>${agr ? _escapeHtml(agr.agreement_state || "–") : `<span class="muted">–</span>`}</td>` +
      `<td class="muted">${agr && agr.top_cell_type ? _escapeHtml(agr.top_cell_type) : "–"}</td>` +
      `</tr>`;
  }).join("");
  return `<table class="data-table kx-f5-detail-table"><thead><tr>` +
    `<th>Age</th><th>NES</th><th>FDR</th><th>Call</th><th>Bulk/decomp</th><th>Top cell</th>` +
    `</tr></thead><tbody>${body}</tbody></table>`;
}

function _kxRenderF5AttrMini(row, tissue) {
  const rows = _kxF5ScopedAttrRows(row, tissue, _kxState())
    .slice()
    .sort((a, b) => (_kxConfRank(b.confidence_tier) - _kxConfRank(a.confidence_tier)) ||
      ((_kxF5Num(b.fivexfad_fold_over_uniform) || -1) - (_kxF5Num(a.fivexfad_fold_over_uniform) || -1)))
    .slice(0, 4);
  if (!rows.length) return `<div class="muted">No matching 5xFAD attribution rows.</div>`;
  const body = rows.map(r => {
    const fold = _kxF5Num(r.fivexfad_fold_over_uniform);
    const lfc = _kxF5Num(r.fivexfad_lfc);
    return `<tr>` +
      `<td>${_escapeHtml(r.cell_type || "")}${r.age_months ? ` <span class="muted">${_escapeHtml(String(r.age_months))}mo</span>` : ""}</td>` +
      `<td>${_escapeHtml((r.confidence_tier || "none").replace("_", " "))}</td>` +
      `<td class="kx-nes-num">${fold == null ? "–" : fold.toFixed(1) + "x"}</td>` +
      `<td class="kx-nes-num">${lfc == null ? "–" : `${lfc >= 0 ? "+" : ""}${lfc.toFixed(2)}`}</td>` +
      `</tr>`;
  }).join("");
  return `<table class="data-table kx-f5-attr-table"><thead><tr>` +
    `<th>Cell type</th><th>Conf</th><th>snRNA</th><th>LFC</th>` +
    `</tr></thead><tbody>${body}</tbody></table>`;
}

function _kxRenderF5Detail(hostEl, row, fdrGate) {
  if (!hostEl) return;
  const f5 = row._f5Source || _kxAggregateF5(
    _kxF5TissueSource(row, "cortex", fdrGate, _kxState().allSamples),
    _kxF5TissueSource(row, "hippocampus", fdrGate, _kxState().allSamples),
    fdrGate,
    _kxState().allSamples);
  const cortex = row._f5Tissues ? row._f5Tissues.cortex : null;
  const hip = row._f5Tissues ? row._f5Tissues.hippocampus : null;
  const mixedNote = f5.status === "mixed_sig"
    ? `<div class="notice show">5xFAD aggregate is <strong>mixed_sig</strong>: cortex and hippocampus contain significant opposing evidence.</div>`
    : "";
  hostEl.innerHTML = `<section class="kx-f5-detail">` +
    `<div class="kx-detail-col-head">5xFAD · cortex and hippocampus</div>` +
    `<p class="muted">Aggregate ${_kxStatusBadge(f5.status, "Aggregate 5xFAD cortex+hippocampus status")} · ` +
    `cortex ${_kxStatusBadge(cortex ? cortex.status : "missing")} · hippocampus ${_kxStatusBadge(hip ? hip.status : "missing")}</p>` +
    mixedNote +
    `<h5>Cortex</h5>${_kxRenderF5AgeTable(row, "cortex", fdrGate)}${_kxRenderF5AttrMini(row, "cortex")}` +
    `<h5>Hippocampus</h5>${_kxRenderF5AgeTable(row, "hippocampus", fdrGate)}${_kxRenderF5AttrMini(row, "hippocampus")}` +
    `</section>`;
}

// ---------- slim master header + row builders ----------

function _kxBuildHeader(s) {
  const TH = (key, label, title, extra) => {
    const sortAttr = key ? ` data-sortkey="${key}"` : "";
    const cls = extra ? ` class="${extra}"` : "";
    return `<th${sortAttr}${cls} title="${_escapeHtml(title)}">${label}</th>`;
  };
  const cells = [];
  cells.push(TH("name", "Kinase", "Kinase identifier from MEA. Click to sort."));
  cells.push(TH("gene", "Gene", "Gene symbol. Click to sort."));
  cells.push(TH("residue", "Res", "Phospho-residue track (ST or Y). Click to sort."));
  cells.push(TH("family", "Family", "Kinase family. Click to sort."));
  cells.push(TH(null, "Mouse", "Mouse (Song) bulk MEA NES across 3 diseases × 3 timepoints (red=up, blue=down; outlined=FDR<header).", "kx-mglyph"));
  cells.push(TH(null, "Human", "Human (Mukesh) per-donor MEA NES: AD donors, then a muted CTRL reference group (red=up, blue=down; outlined=FDR<header).", "kx-hglyph"));
  cells.push(TH(null, "5xFAD cortex", "5xFAD cortex TG-vs-WT kinase MEA across 3/6/9/12 months. Red=up, blue=down, outline=FDR<header.", "kx-f5glyph"));
  cells.push(TH(null, "5xFAD hip", "5xFAD hippocampus TG-vs-WT kinase MEA across 3/6/9/12 months. Red=up, blue=down, outline=FDR<header.", "kx-f5glyph"));
  cells.push(TH("m_med", "M med", "Mouse median NES over the contrasts feeding the direction call (FDR-significant, or all when 'All samples' is on). – = none. Sort by magnitude.", "kx-nes-num"));
  cells.push(TH("h_med", "H med", "Human median NES over the AD donors feeding the direction call (FDR-significant, or all when 'All samples' is on). – = none. Sort by magnitude.", "kx-nes-num"));
  cells.push(TH("f5_med", "5xFAD med", "5xFAD median NES over cortex and hippocampus age units feeding the direction call (FDR-significant, or all when 'All samples' is on). – = none. Sort by magnitude.", "kx-nes-num"));
  cells.push(TH("agree_score", "Crossplay", "Categorical comparison under the active scope: Song, Mukesh AD, and/or 5xFAD aggregate cortex+hippocampus. CTRL donors are visual reference only.", "kx-agree-col"));
  const specScope = s.cluster ? `at ${s.cluster}` : "— peak across all clusters (hover for the cluster)";
  cells.push(TH("m_spec", "Song", `Song location evidence, as fold over the even-split baseline (1/31 ≈ 0.032): ≥10× / ≥5× / ≥2× / ≥1×. The fold is the kinase's peak cell-type share; the concentration index τ and the cell type it concentrates in are in the tooltip. One value per kinase. ★ = the pinned cluster IS that cell type.`, "kx-spec"));
  cells.push(TH("wmb", "WMB", `WMB atlas cross-check: independent mouse-brain location tier ${specScope} (× uniform 1/${(1 / _KX_WMB_UNIFORM).toFixed(0)}). Confirms the Song call against an outside atlas.`, "kx-spec"));
  cells.push(TH("h_spec", "SEA-AD expr", `Human SEA-AD MTG expression location tier ${specScope}. This is expression enrichment, not SEA-AD LFC.`, "kx-spec"));
  return `<thead><tr>${cells.join("")}</tr></thead>`;
}

// Resolve a row's mouse (WMB) + human (SEA-AD) specificity for the active pivot:
// "" = Any cell type → peak across clusters (with the argmax cluster for the
// tooltip); a named cluster → the value at that cluster. Shared by the Song /
// SEA-AD expr columns AND their minimum-tier filters so they agree.
function _kxResolveSpec(r, cluster) {
  const song = _KX_SONG_BY_KID.get(r.kid) || null;  // per-kinase, pivot-independent
  if (cluster) {
    const w = _KX_WMB_BY_KIN_CLUSTER.get(`${r.kid}|${cluster}`);
    const spec = _KX_HUMAN_SPEC_BY_NAME.get(r.name);
    const seaadCt = spec && spec.seaad.get(cluster);
    return {song, wmb: (w == null ? null : w), wmbAt: null, seaad: seaadCt ? seaadCt.score : null, seaadAt: null};
  }
  const wmx = _KX_WMB_MAX_BY_KID.get(r.kid);
  const smx = _KX_SEAAD_MAX_BY_NAME.get(r.name);
  return {
    song,
    wmb: wmx ? wmx.frac : null, wmbAt: wmx ? wmx.cluster : null,
    seaad: smx ? smx.score : null, seaadAt: smx ? smx.cluster : null,
  };
}

function _kxBuildRow(r, s, fdrGate) {
  const sp = _kxResolveSpec(r, s.cluster);
  const key = `${r.name}|${r.residue}`;

  const tds = [];
  tds.push(`<td>${_escapeHtml(r.name)}</td>`);
  tds.push(`<td>${_escapeHtml(r.gene)}</td>`);
  tds.push(`<td class="muted">${_escapeHtml(r.residue)}</td>`);
  tds.push(`<td class="muted">${_escapeHtml(r.family)}</td>`);
  tds.push(_kxMouseGlyphCell(r, fdrGate));
  tds.push(_kxHumanGlyphCell(r, fdrGate));
  tds.push(_kxF5GlyphCell(r, "cortex", fdrGate));
  tds.push(_kxF5GlyphCell(r, "hippocampus", fdrGate));
  tds.push(_kxMedNesCell(r._mNes));
  tds.push(_kxMedNesCell(r._hNes));
  tds.push(_kxMedNesCell(r._f5Nes));
  tds.push(_kxAgreeCategoryCell(r._agreeCategory));
  tds.push(r._humanOnly ? `<td class="muted">–</td>` : _kxSongTierBadge(sp.song, s.cluster));
  tds.push(r._humanOnly ? `<td class="muted">–</td>` : _kxWmbTierBadge(sp.wmb, sp.wmbAt));
  tds.push(_kxLog2TierBadge(sp.seaad, sp.seaadAt));
  const sel = s.selectedKey === key ? " selected" : "";
  return `<tr class="kx-data-row${sel}" data-key="${_escapeHtml(key)}">${tds.join("")}</tr>`;
}

const _KX_STRING_KEYS = new Set(["name", "gene", "residue", "family"]);

function _kxSortRows(rows, s) {
  const cluster = s.cluster;
  if (s.sortKey === "agree_score") {
    // Signed sort: same-direction (high +) first, opposite-direction (negative) last.
    rows.sort((a, b) => s.sortDir * ((b._agreeScore || 0) - (a._agreeScore || 0)));
    return;
  }
  if (_KX_STRING_KEYS.has(s.sortKey)) {
    const k = s.sortKey;
    rows.sort((a, b) => s.sortDir * (a[k] || "").localeCompare(b[k] || ""));
    return;
  }
  rows.sort((a, b) => {
    let av, bv;
    if (s.sortKey === "m_spec") {
      // Song peak cell-type share (the fold driver) — per kinase, pivot-independent.
      const sa = _KX_SONG_BY_KID.get(a.kid), sb = _KX_SONG_BY_KID.get(b.kid);
      const ta = sa ? sa.topShare : null, tb = sb ? sb.topShare : null;
      const ax = (ta == null || !isFinite(ta)) ? -Infinity : ta;
      const bx = (tb == null || !isFinite(tb)) ? -Infinity : tb;
      return s.sortDir * (bx - ax);
    } else if (s.sortKey === "wmb") {
      if (cluster) {
        av = _KX_WMB_BY_KIN_CLUSTER.get(`${a.kid}|${cluster}`);
        bv = _KX_WMB_BY_KIN_CLUSTER.get(`${b.kid}|${cluster}`);
      } else {
        const ma = _KX_WMB_MAX_BY_KID.get(a.kid), mb = _KX_WMB_MAX_BY_KID.get(b.kid);
        av = ma ? ma.frac : null; bv = mb ? mb.frac : null;
      }
    } else if (s.sortKey === "h_spec") {
      if (cluster) {
        const ea = _KX_HUMAN_SPEC_BY_NAME.get(a.name);
        const eb = _KX_HUMAN_SPEC_BY_NAME.get(b.name);
        const ca = ea && ea.seaad.get(cluster);
        const cb = eb && eb.seaad.get(cluster);
        av = ca ? ca.score : null; bv = cb ? cb.score : null;
      } else {
        const ma = _KX_SEAAD_MAX_BY_NAME.get(a.name), mb = _KX_SEAAD_MAX_BY_NAME.get(b.name);
        av = ma ? ma.score : null; bv = mb ? mb.score : null;
      }
    } else if (s.sortKey === "m_med") {
      av = a._mNes; bv = b._mNes;
    } else if (s.sortKey === "h_med") {
      av = a._hNes; bv = b._hNes;
    } else if (s.sortKey === "f5_med") {
      av = a._f5Nes; bv = b._f5Nes;
    } else {
      av = a.peak_NES; bv = b.peak_NES;
    }
    const aMissing = av == null || !isFinite(av);
    const bMissing = bv == null || !isFinite(bv);
    if (aMissing && bMissing) return (a.name || "").localeCompare(b.name || "");
    if (aMissing) return 1;
    if (bMissing) return -1;
    const ax = Math.abs(av);
    const bx = Math.abs(bv);
    return s.sortDir < 0 ? (bx - ax) : (ax - bx);
  });
}

// ---------- master render ----------

function _kxFilteredRows(s) {
  const searchQ = (s.search || "").toLowerCase().trim();
  const trackQ = (s.residueTrack || "").trim();
  const mMin = +(s.mSpecMin || 0);
  const hMin = +(s.hSpecMin || 0);
  const hLog2Min = hMin > 0 ? Math.log2(hMin) : 0;  // SEA-AD tiers are log2-based
  return _KX_ROWS.filter(r => {
    if (trackQ && r.residue !== trackQ) return false;
    if (searchQ) {
      const hay = (r.name + " " + r.gene).toLowerCase();
      if (!hay.includes(searchQ)) return false;
    }
    if (mMin > 0 || hMin > 0) {
      const sp = _kxResolveSpec(r, s.cluster);
      // mMin is a fold-over-even-split threshold (2 / 5 / 10 ×), on the Song peak share.
      if (mMin > 0 && (!sp.song || sp.song.topShare == null || sp.song.topShare < mMin * _KX_SONG_UNIFORM)) return false;
      if (hMin > 0 && (sp.seaad == null || sp.seaad < hLog2Min)) return false;
    }
    if (s.f5Celltype || s.f5Confidence || s.f5SnrnaMin || s.f5WmbMin) {
      const hasAttr = _KX_F5_TISSUES.some(tissue => _kxF5ScopedAttrRows(r, tissue, s).length > 0);
      if (!hasAttr) return false;
    }
    return true;
  });
}

function _kxRenderTable() {
  _kxBuildIndexes();
  _kxEnsureF5AttrData();
  const s = _kxState();
  const fdrGate = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const wrap = document.getElementById("kx-table-wrap");
  const countEl = document.getElementById("kx-count");
  if (!wrap) return;

  const rows = _kxFilteredRows(s);
  _kxComputeAgreement(rows, fdrGate, s.allSamples);

  const head = _kxBuildHeader(s);

  // Comparison-state filter lives here because source statuses are recomputed
  // live against the active FDR threshold.
  const shown = rows.filter(r => _kxStatePassesComparison(r, s.compareScope || "three_way", s.compareState || ""));
  _kxSortRows(shown, s);
  // Stamp resolved spec onto each row for export, then store the visible set.
  for (const r of shown) {
    const sp = _kxResolveSpec(r, s.cluster);
    r._exportSongFold = (sp.song && sp.song.topShare != null) ? sp.song.topShare / _KX_SONG_UNIFORM : null;
    r._exportWmb = sp.wmb;
    r._exportSeaad = sp.seaad;
  }
  _kxVisible = shown.slice();
  const bodyParts = shown.map(r => _kxBuildRow(r, s, fdrGate));
  wrap.innerHTML = `<div class="ke-table-wrap" style="overflow:auto;max-height:75vh;"><table class="data-table" id="kx-table">${head}<tbody>${bodyParts.join("")}</tbody></table></div>`;

  if (countEl) {
    const n = shown.length;
    let sigAgree = 0, agree = 0, sigDisagree = 0, disagree = 0;
    for (const r of rows) {
      const scope = s.compareScope || "three_way";
      if (_kxStatePassesComparison(r, scope, "sig_agree")) sigAgree++;
      if (_kxStatePassesComparison(r, scope, "agree")) agree++;
      if (_kxStatePassesComparison(r, scope, "sig_disagree")) sigDisagree++;
      if (_kxStatePassesComparison(r, scope, "disagree")) disagree++;
    }
    const filterLabel = s.compareState ? ` · state=${_KX_STATE_FILTER_LABELS[s.compareState] || s.compareState}` : "";
    countEl.textContent = `${n.toLocaleString()} kinase${n === 1 ? "" : "s"} · ${_KX_COMPARE_LABELS[s.compareScope] || "3-way"}${filterLabel} · sig agree ${sigAgree} · agree ${agree} · sig disagree ${sigDisagree} · disagree ${disagree} · cluster=${s.cluster || "any"} · ${s.allSamples ? "all samples" : `fdr<${fdrGate}`}`;
  }

  wrap.querySelectorAll("th[data-sortkey]").forEach(th => {
    th.style.cursor = "pointer";
    const k = th.dataset.sortkey;
    if (s.sortKey === k) th.innerHTML += ` <span class="muted" style="font-size:10px;">${s.sortDir > 0 ? "▲" : "▼"}</span>`;
    th.addEventListener("click", () => {
      if (s.sortKey === k) s.sortDir = -s.sortDir;
      else { s.sortKey = k; s.sortDir = _KX_STRING_KEYS.has(k) ? 1 : -1; }
      _kxRenderTable();
    });
  });
  wrap.querySelectorAll("tr.kx-data-row").forEach(tr => {
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      s.selectedKey = tr.dataset.key;
      // Local selection only — highlight in place, repaint the detail panel.
      wrap.querySelectorAll("tr.kx-data-row.selected").forEach(x => x.classList.remove("selected"));
      tr.classList.add("selected");
      _kxRenderDetail();
    });
  });

  _kxRenderDetail();
}

// ---------- cross-dataset detail panel ----------

function _kxNotMeasured(species) {
  const lbl = species === "mouse" ? "mouse (Song)" : "human (Mukesh)";
  return `<div class="kx-detail-placeholder muted">Not measured in ${lbl}.</div>`;
}

// _khRenderNESAcrossDonors / _khRenderAttribution read _KHState.auditDonor; pick
// this human row's first AD donor with a finite NES so the panels populate.
function _kxPrimeHumanDonor(hrow) {
  if (typeof _KHState === "undefined" || !_KH || !Array.isArray(_KH.donors)) return;
  let pick = null;
  if (hrow && hrow._nes) {
    for (let di = 0; di < _KH.donors.length; di++) {
      const v = hrow._nes[di];
      if (v != null && isFinite(v)) { pick = _KH.donors[di]; break; }
    }
  }
  _KHState.auditDonor = pick || _KH.donors[0] || null;
}

function _kxRenderDetail() {
  const host = document.getElementById("kx-detail");
  if (!host) return;
  const s = _kxState();
  const row = s.selectedKey ? _KX_ROWS.find(r => `${r.name}|${r.residue}` === s.selectedKey) : null;
  if (!row) {
    host.innerHTML = `<div class="kx-detail-idle muted">Select a kinase to compare mouse and human activity.</div>`;
    return;
  }
  const fdrGate = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  _kxComputeAgreement([row], fdrGate, s.allSamples);
  const meta = _kxComparisonMeta(s.compareScope || "three_way", row._agreeCategory);
  const mouseKid = row._humanOnly ? null : row.kid;
  const humanKid = row._human ? row._human.kid : null;

  // Verdict-header significance counts (live).
  let mSig = 0, mTot = 0;
  for (const c of (CONTRASTS || [])) {
    const f = row._fdr[c];
    if (f != null && isFinite(f)) { mTot++; if (f < fdrGate) mSig++; }
  }
  let hSig = 0, hTot = 0;
  const pd = humanKid != null ? _KX_HUMAN_PERDONOR.get(humanKid) : null;
  if (pd) for (const d of _KX_AD_DONORS) {
    const e = pd.get(d);
    if (e && e.nes != null && isFinite(e.nes)) { hTot++; if (e.fdr != null && e.fdr < fdrGate) hSig++; }
  }
  const mNesTxt = row._mNes != null ? `${row._mNes >= 0 ? "+" : ""}${row._mNes.toFixed(2)}` : "n/a";
  const hNesTxt = row._hNes != null ? `${row._hNes >= 0 ? "+" : ""}${row._hNes.toFixed(2)}` : "n/a";
  const f5NesTxt = row._f5Nes != null ? `${row._f5Nes >= 0 ? "+" : ""}${row._f5Nes.toFixed(2)}` : "n/a";
  const allS = s.allSamples;
  const nesLbl = allS ? "median NES (all samples)" : "median NES";
  const mCntTxt = allS ? `${mTot} contrast${mTot === 1 ? "" : "s"}` : `${mSig}/${mTot || "–"} contrasts sig`;
  const hCntTxt = allS ? `${hTot} donor${hTot === 1 ? "" : "s"}` : `${hSig}/${hTot || "–"} donors sig`;
  const f5Cnt = row._f5Source || {nSig: 0, nMeasured: 0};
  const f5CntTxt = allS
    ? `${f5Cnt.nMeasured || 0} tissue-age unit${(f5Cnt.nMeasured || 0) === 1 ? "" : "s"}`
    : `${f5Cnt.nSig || 0}/${f5Cnt.nMeasured || "–"} tissue-age units sig`;
  const tab = s.detailTab || "activity";

  host.innerHTML = `
    <div class="kx-detail-header">
      <div class="kx-detail-title">
        <b>${_escapeHtml(row.name)}</b>
        <span class="muted">${_escapeHtml(row.gene || "")}${row.gene ? " · " : ""}${_escapeHtml(row.residue)}</span>
        <span class="${meta.cls}" title="${_escapeHtml(meta.tip)}">${meta.glyph} ${_escapeHtml(meta.label)}</span>
      </div>
      <div class="kx-detail-verdict muted">
        Mouse (Song) ${nesLbl} <b>${mNesTxt}</b> (${mCntTxt}) ·
        Human (Mukesh AD) ${nesLbl} <b>${hNesTxt}</b> (${hCntTxt}) ·
        5xFAD ${nesLbl} <b>${f5NesTxt}</b> (${f5CntTxt}) ·
        5xFAD aggregate ${_kxStatusBadge(row._f5Source ? row._f5Source.status : "missing")}
      </div>
    </div>
    <nav class="kx-detail-tabs">
      <button data-kxd-tab="activity" class="${tab === "activity" ? "active" : ""}">NES Activity</button>
      <button data-kxd-tab="specificity" class="${tab === "specificity" ? "active" : ""}">Cell-type Specificity</button>
    </nav>
    ${tab === "specificity"
      ? `<div id="kx-detail-spec" class="kx-detail-spec"></div>`
      : `<div id="kx-detail-direction"></div><div class="kx-detail-grid">
      <div class="kx-detail-col"><div class="kx-detail-col-head">Mouse · Song</div><div id="kx-detail-m"></div></div>
      <div class="kx-detail-col"><div class="kx-detail-col-head">Human · Mukesh</div><div id="kx-detail-h"></div></div>
      <div class="kx-detail-col"><div id="kx-detail-f5"></div></div>
    </div>`}`;

  host.querySelectorAll("button[data-kxd-tab]").forEach(b => b.addEventListener("click", () => {
    s.detailTab = b.dataset.kxdTab;
    _kxRenderDetail();
  }));

  if (typeof _ensureKinaseIndexes === "function") _ensureKinaseIndexes();

  if (tab === "activity") {
    _kxRenderDirectionSupport(document.getElementById("kx-detail-direction"), row);
    const mEl = document.getElementById("kx-detail-m");
    const hEl = document.getElementById("kx-detail-h");
    const f5El = document.getElementById("kx-detail-f5");
    const humanReady = (typeof _KH_HAS !== "undefined") && _KH_HAS && _KH;
    const hrow = (humanReady && humanKid != null && typeof _khAllRows === "function")
      ? _khAllRows().find(x => x.id === humanKid) : null;
    if (mouseKid != null && typeof _renderKinaseNesPlot === "function") _renderKinaseNesPlot("kx-detail-m", mouseKid);
    else mEl.innerHTML = _kxNotMeasured("mouse");
    if (hrow && typeof _khRenderNESAcrossDonors === "function") {
      _kxPrimeHumanDonor(hrow);
      _khRenderNESAcrossDonors("kx-detail-h", hrow, _KHState.auditDonor);
    } else hEl.innerHTML = _kxNotMeasured("human");
    _kxRenderF5Detail(f5El, row, fdrGate);
  } else {
    // Cell-type Specificity: one cluster-aligned reference-corroboration table
    // (Song primary + WMB / SEA-AD / HBCA cross-checks), not two verbatim tables.
    const specEl = document.getElementById("kx-detail-spec");
    if (specEl) _kxRenderSpecAligned(specEl, row);
  }
}

// ---------- controls + wiring ----------

function _kxSyncControls() {
  const s = _kxState();
  const clSel = document.getElementById("kx-cluster");
  if (clSel) {
    clSel.innerHTML = `<option value="">Any cell type</option>` + (_KX_CLUSTERS || []).map(c =>
      `<option value="${_escapeHtml(c)}">${_escapeHtml(c)}</option>`
    ).join("");
    clSel.value = (s.cluster && (_KX_CLUSTERS || []).includes(s.cluster)) ? s.cluster : "";
  }
  const tSel = document.getElementById("kx-track");
  if (tSel) tSel.value = s.residueTrack || "";
  const sInput = document.getElementById("kx-search");
  if (sInput) sInput.value = s.search || "";
  const mSel = document.getElementById("kx-mspec");
  if (mSel) mSel.value = String(s.mSpecMin || 0);
  const hSel = document.getElementById("kx-hspec");
  if (hSel) hSel.value = String(s.hSpecMin || 0);
  const aChk = document.getElementById("kx-allsamples");
  if (aChk) aChk.checked = !!s.allSamples;
  const agSel = document.getElementById("kx-agree");
  if (agSel) agSel.value = s.agreeCat || "";
  const scopeSel = document.getElementById("kx-compare-scope");
  if (scopeSel) scopeSel.value = s.compareScope || "three_way";
  const stateSel = document.getElementById("kx-compare-state");
  if (stateSel) {
    if (s.compareState && !_KX_STATE_FILTER_VALUES.has(s.compareState)) s.compareState = "";
    stateSel.value = s.compareState || "";
  }
  const f5Cell = document.getElementById("kx-f5-celltype");
  if (f5Cell) {
    const cur = s.f5Celltype || "";
    f5Cell.innerHTML = `<option value="">Any</option>` + (_KX_F5_CELLTYPES || []).map(c =>
      `<option value="${_escapeHtml(c)}">${_escapeHtml(c)}</option>`
    ).join("");
    f5Cell.value = (_KX_F5_CELLTYPES || []).includes(cur) ? cur : "";
    if (f5Cell.value !== cur) s.f5Celltype = "";
  }
  const f5Conf = document.getElementById("kx-f5-confidence");
  if (f5Conf) f5Conf.value = s.f5Confidence || "";
  const f5Snrna = document.getElementById("kx-f5-snrna");
  if (f5Snrna) f5Snrna.value = String(s.f5SnrnaMin || 0);
  const f5Wmb = document.getElementById("kx-f5-wmb");
  if (f5Wmb) f5Wmb.value = String(s.f5WmbMin || 0);
}

function wireKinaseCrosstable() {
  // Index build is deferred to first render (_kxRenderTable / renderKinaseCrosstable both
  // call _kxBuildIndexes, guarded by _KX_INITIALIZED). Wiring only attaches DOM listeners,
  // mirroring the lazy 5xFAD tab — keeps ~170k row iterations off the boot path (audit P3).
  const clSel = document.getElementById("kx-cluster");
  if (clSel) clSel.addEventListener("change", () => { _kxState().cluster = clSel.value; _kxRenderTable(); });
  const tSel = document.getElementById("kx-track");
  if (tSel) tSel.addEventListener("change", () => { _kxState().residueTrack = tSel.value; _kxRenderTable(); });
  const sInput = document.getElementById("kx-search");
  if (sInput) sInput.addEventListener("input", () => { _kxState().search = sInput.value; _kxRenderTable(); });
  const mSel = document.getElementById("kx-mspec");
  if (mSel) mSel.addEventListener("change", () => { _kxState().mSpecMin = +mSel.value; _kxRenderTable(); });
  const hSel = document.getElementById("kx-hspec");
  if (hSel) hSel.addEventListener("change", () => { _kxState().hSpecMin = +hSel.value; _kxRenderTable(); });
  const aChk = document.getElementById("kx-allsamples");
  if (aChk) aChk.addEventListener("change", () => { _kxState().allSamples = aChk.checked; _kxRenderTable(); });
  const agSel = document.getElementById("kx-agree");
  if (agSel) agSel.addEventListener("change", () => { _kxState().agreeCat = agSel.value; _kxRenderTable(); });
  const scopeSel = document.getElementById("kx-compare-scope");
  if (scopeSel) scopeSel.addEventListener("change", () => {
    const s = _kxState();
    s.compareScope = scopeSel.value || "three_way";
    s.compareState = "";
    _kxSyncControls();
    _kxRenderTable();
  });
  const stateSel = document.getElementById("kx-compare-state");
  if (stateSel) stateSel.addEventListener("change", () => { _kxState().compareState = stateSel.value; _kxRenderTable(); });
  const f5Cell = document.getElementById("kx-f5-celltype");
  if (f5Cell) f5Cell.addEventListener("change", () => { _kxState().f5Celltype = f5Cell.value; _kxRenderTable(); });
  const f5Conf = document.getElementById("kx-f5-confidence");
  if (f5Conf) f5Conf.addEventListener("change", () => { _kxState().f5Confidence = f5Conf.value; _kxRenderTable(); });
  const f5Snrna = document.getElementById("kx-f5-snrna");
  if (f5Snrna) f5Snrna.addEventListener("change", () => { _kxState().f5SnrnaMin = +f5Snrna.value; _kxRenderTable(); });
  const f5Wmb = document.getElementById("kx-f5-wmb");
  if (f5Wmb) f5Wmb.addEventListener("change", () => { _kxState().f5WmbMin = +f5Wmb.value; _kxRenderTable(); });
  const reset = document.getElementById("kx-reset");
  if (reset) reset.addEventListener("click", () => {
    const s = _kxState();
    s.cluster = "";   // Any cell type
    s.residueTrack = ""; s.search = "";
    s.mSpecMin = 0; s.hSpecMin = 0; s.allSamples = false;
    s.sortKey = "agree_score"; s.sortDir = -1;
    s.agreeCat = ""; s.compareScope = "three_way"; s.compareState = "";
    s.f5Celltype = ""; s.f5Confidence = ""; s.f5SnrnaMin = 0; s.f5WmbMin = 0;
    s.selectedKey = null; s.detailTab = "activity";
    _kxSyncControls();
    _kxRenderTable();
  });

  const exportBtn = document.getElementById("kx-export");
  if (exportBtn) exportBtn.addEventListener("click", exportCrosstableCsv);
}

function exportCrosstableCsv() {
  const stamp = new Date().toISOString().slice(0, 10);
  const headers = ["Kinase","Gene","Residue","Family","Mouse_med_NES","Human_med_NES","5xFAD_med_NES","Crossplay","Song_fold","WMB_tier","SEAAD_log2"];
  const keys    = ["name","gene","residue","family","_mNes","_hNes","_f5Nes","_agreeCategory","_exportSongFold","_exportWmb","_exportSeaad"];
  csvDownload(csvSerialize(headers, keys, _kxVisible), `crosstable_${stamp}.csv`);
}

function renderKinaseCrosstable() {
  _kxBuildIndexes();
  _kxEnsureF5AttrData();
  _kxSyncControls();
  _kxRenderTable();
}
