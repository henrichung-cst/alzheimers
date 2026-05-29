// ---------------------------------------------------------------------------
// SliceCache — lazy loader for per-entity edge parquets.
// Backbone-bucket slices back the backbone-selection highlight in the kinase
// explorer; decomp-OLS shards back the Attribution drawer's per-cell evidence.
// LRU-capped to avoid unbounded memory.
// ---------------------------------------------------------------------------
const SliceCache = (function(){
  // PAYLOAD is async-loaded now; read it lazily on first method call.
  let ESR = null;
  let BUCKET_SIZE = 256;
  let dPresent = null, sPresent = null, iPresent = null, hPresent = null;
  let iPresentByDonor = null;       // T-cell: {donor1: Set, donor2: Set}
  function _ensureInit() {
    if (ESR !== null) return;
    ESR = (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.edge_slice_ref) || {};
    BUCKET_SIZE = ESR.bucket_size || 256;
    dPresent = new Set((ESR.present_decomp_ols_kinase_ids || []).map(Number));
    sPresent = new Set(
      (ESR.present_song_concordance_genes || []).map(g => String(g).toUpperCase())
    );
    const si = (PAYLOAD && PAYLOAD.incytr_pathways && PAYLOAD.incytr_pathways.slice_index) || {};
    iPresent = new Set((si.present || []).map(([s, r]) => s + "||" + r));
    iPresentByDonor = {};
    if (si.by_donor) {
      for (const d of Object.keys(si.by_donor)) {
        iPresentByDonor[d] = new Set(
          (si.by_donor[d].present || []).map(([s, r]) => s + "||" + r)
        );
      }
    }
    hPresent = new Set((ESR.present_human_perdonor_kinase_ids || []).map(Number));
  }
  const MAX = 16;                          // LRU cap (per side)
  const bCache = new Map();                // bucket_id -> rows[]

  function _lruTouch(cache, key, value){
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    while (cache.size > MAX) cache.delete(cache.keys().next().value);
  }

  async function _fetchParquet(url){
    let resp;
    try {
      resp = await fetch(url);
    } catch (e) {
      if (window.location.protocol === "file:") {
        throw new Error(
          "Browser blocked local sidecar fetches under file://. " +
          "Serve outputs/reports/unified_viewer over HTTP and open that URL."
        );
      }
      throw e;
    }
    if (!resp.ok) throw new Error(`fetch ${url} → ${resp.status}`);
    const buf = await resp.arrayBuffer();
    if (typeof hyparquet === "undefined") {
      throw new Error("parquet reader not loaded (hyparquet missing)");
    }
    return await hyparquet.parquetReadObjects({
      file: buf, compressors: hyparquet.compressors,
    });
  }

  async function loadBackboneBucket(backbone_id){
    _ensureInit();
    const bkt = Math.floor(backbone_id / BUCKET_SIZE);
    if (bCache.has(bkt)) {
      const v = bCache.get(bkt); _lruTouch(bCache, bkt, v); return v;
    }
    const pad = String(bkt).padStart(3, "0");
    const url = `${ESR.backbone_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(bCache, bkt, rows);
    return rows;
  }

  async function backboneEdges(backbone_id){
    const rows = await loadBackboneBucket(backbone_id);
    return rows.filter(r => r.backbone_id === backbone_id);
  }

  // Decomp-OLS shards: per-kinase substrate-site OLS for every (contrast, cell_type).
  const dCache = new Map();              // kinase_id -> rows[]
  async function loadDecompOls(kinase_id){
    _ensureInit();
    if (!dPresent.has(Number(kinase_id))) return [];
    if (dCache.has(kinase_id)) {
      const v = dCache.get(kinase_id); _lruTouch(dCache, kinase_id, v); return v;
    }
    if (!ESR.decomp_ols_url) return [];
    const pad = String(kinase_id).padStart(3, "0");
    const url = `${ESR.decomp_ols_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(dCache, kinase_id, rows);
    return rows;
  }

  // Song concordance shards: one parquet per uppercased gene symbol.
  const sCache = new Map();              // GENE_UPPER -> rows[]
  async function loadSongConcordance(geneSymbol){
    _ensureInit();
    const g = String(geneSymbol || "").toUpperCase();
    if (!g || !sPresent.has(g)) return [];
    if (sCache.has(g)) {
      const v = sCache.get(g); _lruTouch(sCache, g, v); return v;
    }
    if (!ESR.song_concordance_url) return [];
    const url = `${ESR.song_concordance_url}${encodeURIComponent(g)}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(sCache, g, rows);
    return rows;
  }

  // Incytr pathway shards: one parquet per (sender, receiver) pair under
  // edge_slices/incytr_pathways/<sanitized_sender>__<sanitized_receiver>.parquet.
  // Sanitize rule matches alz/integration/load.R:sanitize_celltype — replace
  // "/" with "-" and " " with "_". Sender raw, receiver display name (the
  // payload-side senders/receivers arrays already carry canonical display).
  const iCache = new Map();              // "sender||receiver" -> rows[]
  function _incytrSanitize(name) {
    return String(name).replace(/\//g, "-").replace(/ /g, "_").replace(/\./g, "");
  }
  async function loadIncytrShard(sender, receiver) {
    _ensureInit();
    const skey = sender + "||" + receiver;
    // T-cell: per-donor shards. Mouse: flat shards.
    const donor = (Store.state.selection && Store.state.selection.donor) || null;
    const perDonorSet = donor && iPresentByDonor ? iPresentByDonor[donor] : null;
    if (perDonorSet) {
      if (!perDonorSet.has(skey)) return [];
    } else if (!iPresent.has(skey)) {
      return [];
    }
    const lkey = donor ? donor + "||" + skey : skey;
    if (iCache.has(lkey)) {
      const v = iCache.get(lkey); _lruTouch(iCache, lkey, v); return v;
    }
    const base = ESR.incytr_pathways_url || "edge_slices/incytr_pathways/";
    const prefix = donor && perDonorSet ? `${donor}__` : "";
    const url = `${base}${prefix}${_incytrSanitize(sender)}__${_incytrSanitize(receiver)}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(iCache, lkey, rows);
    return rows;
  }

  // Human per-donor substrate shards: one parquet per kinase id with
  // (donor, leading_substrates, substrate_motifs). Returns a Map keyed by
  // donor for O(1) lookup; empty Map when the kinase has no leading-edge
  // hits in any donor (kinase not in present_human_perdonor_kinase_ids).
  const hCache = new Map();              // kinase_id -> Map<donor, {leading,motifs}>
  const hInflight = new Map();           // kinase_id -> Promise<Map>
  async function loadHumanPerdonorSubstrate(kinase_id) {
    _ensureInit();
    const kid = Number(kinase_id);
    if (!hPresent.has(kid)) return new Map();
    if (hCache.has(kid)) {
      const v = hCache.get(kid); _lruTouch(hCache, kid, v); return v;
    }
    if (hInflight.has(kid)) return hInflight.get(kid);
    if (!ESR.human_perdonor_url) return new Map();
    const pad = String(kid).padStart(3, "0");
    const url = `${ESR.human_perdonor_url}${pad}.parquet`;
    // Coalesce concurrent fetches — the Trace and Running Enrichment sub-tabs
    // both render asynchronously from the same kinase selection and would
    // otherwise race on the same shard.
    const p = _fetchParquet(url).then(rows => {
      const byDonor = new Map();
      for (const r of rows) {
        byDonor.set(String(r.donor), {
          leading: r.leading_substrates || "",
          motifs: r.substrate_motifs || "",
          klPercentiles: r.substrate_kl_percentiles || "",
        });
      }
      _lruTouch(hCache, kid, byDonor);
      hInflight.delete(kid);
      return byDonor;
    }).catch(err => {
      hInflight.delete(kid);
      throw err;
    });
    hInflight.set(kid, p);
    return p;
  }

  return { loadBackboneBucket, backboneEdges, loadDecompOls, loadIncytrShard,
           loadSongConcordance, loadHumanPerdonorSubstrate,
           get backboneCacheSize(){ return bCache.size; },
           get decompOlsCacheSize(){ return dCache.size; },
           get incytrCacheSize(){ return iCache.size; },
           get songConcordanceCacheSize(){ return sCache.size; },
           get humanPerdonorCacheSize(){ return hCache.size; } };
})();
window.SliceCache = SliceCache;

// ---------------------------------------------------------------------------
// Header wiring
// ---------------------------------------------------------------------------
