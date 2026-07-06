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
  let iPresentByContext = null;     // context id -> Set("sender||receiver")
  function _ensureInit() {
    if (ESR !== null) return;
    ESR = (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.edge_slice_ref) || {};
    BUCKET_SIZE = ESR.bucket_size || 256;
    dPresent = new Set((ESR.present_decomp_ols_kinase_ids || []).map(Number));
    sPresent = new Set(
      (ESR.present_song_concordance_genes || []).map(g => String(g).toUpperCase())
    );
    const si = ViewerPayload.incytrSliceIndex();
    iPresent = new Set((si.present || []).map(([s, r]) => s + "||" + r));
    iPresentByContext = {};
    for (const ctx of ViewerPayload.contexts()) {
      const idx = ViewerPayload.incytrSliceIndex(ctx.id);
      iPresentByContext[ctx.id] = new Set(
        (idx.present || []).map(([s, r]) => s + "||" + r)
      );
    }
    hPresent = new Set((ESR.present_human_perdonor_kinase_ids || []).map(Number));
  }
  const MAX = 16;                          // LRU cap (per side)
  const bCache = new Map();                // bucket_id -> rows[]
  const bInflight = new Map();             // bucket_id -> Promise<rows[]>

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
          "Serve the viewer output directory over HTTP and open that URL."
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
    if (bInflight.has(bkt)) return bInflight.get(bkt);
    const pad = String(bkt).padStart(3, "0");
    const url = `${ESR.backbone_url}${pad}.parquet`;
    const p = _fetchParquet(url).then(rows => {
      _lruTouch(bCache, bkt, rows);
      bInflight.delete(bkt);
      return rows;
    }).catch(err => {
      bInflight.delete(bkt);
      throw err;
    });
    bInflight.set(bkt, p);
    return p;
  }

  async function backboneEdges(backbone_id){
    const rows = await loadBackboneBucket(backbone_id);
    return rows.filter(r => r.backbone_id === backbone_id);
  }

  // Decomp-OLS shards: per-kinase substrate-site OLS for every (contrast, cell_type).
  const dCache = new Map();              // kinase_id -> rows[]
  const dInflight = new Map();           // kinase_id -> Promise<rows[]>
  async function loadDecompOls(kinase_id){
    _ensureInit();
    const kid = Number(kinase_id);
    if (!dPresent.has(kid)) return [];
    if (dCache.has(kid)) {
      const v = dCache.get(kid); _lruTouch(dCache, kid, v); return v;
    }
    if (dInflight.has(kid)) return dInflight.get(kid);
    if (!ESR.decomp_ols_url) return [];
    const pad = String(kid).padStart(3, "0");
    const url = `${ESR.decomp_ols_url}${pad}.parquet`;
    const p = _fetchParquet(url).then(rows => {
      _lruTouch(dCache, kid, rows);
      dInflight.delete(kid);
      return rows;
    }).catch(err => {
      dInflight.delete(kid);
      throw err;
    });
    dInflight.set(kid, p);
    return p;
  }

  // Song concordance shards: one parquet per uppercased gene symbol.
  const sCache = new Map();              // GENE_UPPER -> rows[]
  const sInflight = new Map();           // GENE_UPPER -> Promise<rows[]>
  async function loadSongConcordance(geneSymbol){
    _ensureInit();
    const g = String(geneSymbol || "").toUpperCase();
    if (!g || !sPresent.has(g)) return [];
    if (sCache.has(g)) {
      const v = sCache.get(g); _lruTouch(sCache, g, v); return v;
    }
    if (sInflight.has(g)) return sInflight.get(g);
    if (!ESR.song_concordance_url) return [];
    const url = `${ESR.song_concordance_url}${encodeURIComponent(g)}.parquet`;
    const p = _fetchParquet(url).then(rows => {
      _lruTouch(sCache, g, rows);
      sInflight.delete(g);
      return rows;
    }).catch(err => {
      sInflight.delete(g);
      throw err;
    });
    sInflight.set(g, p);
    return p;
  }

  // Incytr pathway shards: one parquet per (sender, receiver) pair under
  // edge_slices/incytr_pathways/<sanitized_sender>__<sanitized_receiver>.parquet.
  // Sanitize rule matches alz/integration/load.R:sanitize_celltype — replace
  // "/" with "-" and " " with "_". Sender raw, receiver display name (the
  // payload-side senders/receivers arrays already carry canonical display).
  const iCache = new Map();              // "context||sender||receiver" -> rows[]
  const iInflight = new Map();           // "context||sender||receiver" -> Promise<rows[]>
  async function loadIncytrShard(sender, receiver) {
    _ensureInit();
    const skey = sender + "||" + receiver;
    // Prefer context-scoped shard indexes. Legacy flat indexes remain supported
    // for older payloads that predate viewer schema v2.
    const context = ViewerPayload.activeContext();
    const contextSet = context && iPresentByContext ? iPresentByContext[context] : null;
    if (contextSet) {
      if (!contextSet.has(skey)) return [];
    } else if (!iPresent.has(skey)) {
      return [];
    }
    const lkey = context ? context + "||" + skey : skey;
    if (iCache.has(lkey)) {
      const v = iCache.get(lkey); _lruTouch(iCache, lkey, v); return v;
    }
    if (iInflight.has(lkey)) return iInflight.get(lkey);
    const ctxIdx = context ? ViewerPayload.incytrSliceIndex(context) : null;
    const base = (ctxIdx && ctxIdx.base_url) || ESR.incytr_pathways_url || "edge_slices/incytr_pathways/";
    const fname = ViewerPayload.incytrShardFilename(sender, receiver, context);
    const url = `${base}${fname}`;
    const p = _fetchParquet(url).then(rows => {
      _lruTouch(iCache, lkey, rows);
      iInflight.delete(lkey);
      return rows;
    }).catch(err => {
      iInflight.delete(lkey);
      throw err;
    });
    iInflight.set(lkey, p);
    return p;
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

  // Substrate-pair shards: one parquet per kinase (Early AD Kinases tab),
  // holding that kinase's shared human↔5xFAD motif pairs across all 8 contexts.
  // Filename sanitizes non-alphanumeric runs to '_' (matches the Python builder).
  const pbCache = new Map();             // KINASE -> rows[]
  const pbInflight = new Map();          // KINASE -> Promise<rows[]>
  let pbPresent = null;
  function _sanitizeKinase(name) {
    return String(name || "").replace(/[^A-Za-z0-9]+/g, "_");
  }
  async function loadSubstratePairs(kinase) {
    _ensureInit();
    const k = String(kinase || "");
    if (pbPresent === null) {
      pbPresent = new Set((ESR.present_substrate_pairs_kinases || []).map(String));
    }
    if (!k || !pbPresent.has(k)) return [];
    if (pbCache.has(k)) {
      const v = pbCache.get(k); _lruTouch(pbCache, k, v); return v;
    }
    if (pbInflight.has(k)) return pbInflight.get(k);
    if (!ESR.substrate_pairs_url) return [];
    const url = `${ESR.substrate_pairs_url}${encodeURIComponent(_sanitizeKinase(k))}.parquet`;
    const p = _fetchParquet(url).then(rows => {
      _lruTouch(pbCache, k, rows);
      pbInflight.delete(k);
      return rows;
    }).catch(err => {
      pbInflight.delete(k);
      throw err;
    });
    pbInflight.set(k, p);
    return p;
  }

  return { loadBackboneBucket, backboneEdges, loadDecompOls, loadIncytrShard,
           loadSongConcordance, loadHumanPerdonorSubstrate, loadSubstratePairs,
           get backboneCacheSize(){ return bCache.size; },
           get decompOlsCacheSize(){ return dCache.size; },
           get incytrCacheSize(){ return iCache.size; },
           get songConcordanceCacheSize(){ return sCache.size; },
           get humanPerdonorCacheSize(){ return hCache.size; },
           get substratePairsCacheSize(){ return pbCache.size; } };
})();
window.SliceCache = SliceCache;

// ---------------------------------------------------------------------------
// Header wiring
// ---------------------------------------------------------------------------
