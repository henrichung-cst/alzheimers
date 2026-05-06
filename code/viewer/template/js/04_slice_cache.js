window.getFilteredIndices = getFilteredIndices;

// ---------------------------------------------------------------------------
// SliceCache — lazy loader for per-entity edge parquets (Unit E).
// Kinase slices and backbone-bucket slices are fetched on demand via the
// URLs in PAYLOAD.edge_slice_ref. LRU-capped to avoid unbounded memory.
// Parquet decoding uses hyparquet (CDN-loaded) when available; falls back
// to reporting an error message on the selected entity's side panel.
// ---------------------------------------------------------------------------
const SliceCache = (function(){
  const ESR = PAYLOAD.edge_slice_ref || {};
  const BUCKET_SIZE = ESR.bucket_size || 256;
  const MAX = 16;                          // LRU cap (per side)
  const kCache = new Map();                // kinase_id -> {backbone_id, contrast_id, support_contribution, concordance}
  const bCache = new Map();                // bucket_id -> same shape + kinase_id

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

  // Persistent (non-LRU) sets of backbone_ids per kinase, for sync filter use.
  const kBbSets = new Map();
  function _populateBbSet(kinase_id, rows){
    const s = new Set();
    for (const r of rows) s.add(r.backbone_id);
    kBbSets.set(kinase_id, s);
  }
  function kinaseBackboneSetSync(kinase_id){
    return kBbSets.has(kinase_id) ? kBbSets.get(kinase_id) : null;
  }

  async function loadKinase(kinase_id){
    if (kCache.has(kinase_id)) {
      const v = kCache.get(kinase_id); _lruTouch(kCache, kinase_id, v); return v;
    }
    const pad = String(kinase_id).padStart(3, "0");
    const url = `${ESR.kinase_url}${pad}.parquet`;
    const rows = await _fetchParquet(url);
    _lruTouch(kCache, kinase_id, rows);
    if (!kBbSets.has(kinase_id)) _populateBbSet(kinase_id, rows);
    return rows;
  }

  async function loadBackboneBucket(backbone_id){
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

  // per_backbone_summary.parquet — fetched once, indexed by backbone_id.
  // ~64,592 rows × 7 cols ≈ ~3 MB; small enough to keep wholesale in memory.
  let _bbSummaryAll = null;
  let _bbSummaryIdx = null;
  let _bbSummaryPromise = null;
  async function _loadBackboneSummary(){
    if (_bbSummaryAll) return _bbSummaryAll;
    if (_bbSummaryPromise) return _bbSummaryPromise;
    const url = ESR.backbone_summary_url;
    if (!url) throw new Error("backbone_summary_url missing in edge_slice_ref");
    _bbSummaryPromise = (async () => {
      const rows = await _fetchParquet(url);
      _bbSummaryAll = rows;
      const idx = new Map();
      for (let i = 0; i < rows.length; i++) {
        const bid = rows[i].backbone_id;
        let arr = idx.get(bid);
        if (!arr) { arr = []; idx.set(bid, arr); }
        arr.push(i);
      }
      _bbSummaryIdx = idx;
      _bbSummaryPromise = null;
      return rows;
    })();
    return _bbSummaryPromise;
  }
  async function backboneSummary(backbone_id){
    const rows = await _loadBackboneSummary();
    const arr = _bbSummaryIdx.get(backbone_id) || [];
    return arr.map(i => rows[i]);
  }

  // Decomp-OLS shards: per-kinase substrate-site OLS for every (contrast, wmb_class).
  // Backs the Attribution drawer's per-cell evidence section.
  const dCache = new Map();              // kinase_id -> rows[]
  const dPresent = new Set((ESR.present_decomp_ols_kinase_ids || []).map(Number));
  async function loadDecompOls(kinase_id){
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

  return { loadKinase, loadBackboneBucket, backboneEdges, backboneSummary,
           kinaseBackboneSetSync, loadDecompOls,
           get kinaseCacheSize(){ return kCache.size; },
           get backboneCacheSize(){ return bCache.size; },
           get decompOlsCacheSize(){ return dCache.size; } };
})();
window.SliceCache = SliceCache;

// ---------------------------------------------------------------------------
// Header wiring
// ---------------------------------------------------------------------------
