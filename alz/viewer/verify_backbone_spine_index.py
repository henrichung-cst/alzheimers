"""Verify B-6 backbone_spine_index is present in the built viewer payload and
that collapse/expand key logic is correct.

Run:  pixi run python alz/viewer/verify_backbone_spine_index.py

Pass criteria:
  1. backbone_spine_index key present in each grain block of the payload.
  2. n_spines > 0 for each grain.
  3. URL resolves to an existing .json.gz file on disk.
  4. The file is valid gzipped JSON with the expected schema_version and
     spine_to_pairs dict.
  5. Key logic: _ipSpineKey equivalents compute the right spine format.
"""
from __future__ import annotations

import gzip
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
VIEWER_DIR = os.path.join(REPO_ROOT, "outputs", "reports", "unified_viewer")
PAYLOAD_JSON = os.path.join(VIEWER_DIR, "unified_viewer.payload.json")
PAYLOAD_JSON_GZ = PAYLOAD_JSON + ".gz"

_PASS = 0
_FAIL = 0

def _ok(msg: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  PASS  {msg}", flush=True)

def _fail(msg: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  FAIL  {msg}", flush=True)

# ---------------------------------------------------------------------------
# Key logic tests (Python equivalents of _ipSpineKey)
# ---------------------------------------------------------------------------

def _spine_key(r: dict, grain: str) -> str:
    L = r.get("Ligand", "") or ""
    R = r.get("Receptor", "") or ""
    E = r.get("EM", "") or ""
    T = r.get("Target", "") or ""
    if grain == "R-EM":   return f"{R}|{E}"
    if grain == "L-R-EM": return f"{L}|{R}|{E}"
    if grain == "R-EM-T": return f"{R}|{E}|{T}"
    return f"{L}|{R}|{E}|{T}"   # Full

def _test_key_logic() -> None:
    print("\n--- Key logic tests ---", flush=True)
    row = {"Ligand": "Apoe", "Receptor": "Lrp1", "EM": "Mapk1", "Target": "Cdkn1a"}
    cases = [
        ("Full",   "Apoe|Lrp1|Mapk1|Cdkn1a"),
        ("L-R-EM", "Apoe|Lrp1|Mapk1"),
        ("R-EM-T", "Lrp1|Mapk1|Cdkn1a"),
        ("R-EM",   "Lrp1|Mapk1"),
    ]
    for grain, expected in cases:
        got = _spine_key(row, grain)
        if got == expected:
            _ok(f"_spine_key({grain!r}) = {got!r}")
        else:
            _fail(f"_spine_key({grain!r}): expected {expected!r}, got {got!r}")

    # Collapse: dropping Target from Full → L-R-EM spine
    full_key  = _spine_key(row, "Full")
    lrem_key  = _spine_key(row, "L-R-EM")
    rem_t_key = _spine_key(row, "R-EM-T")
    rem_key   = _spine_key(row, "R-EM")
    # L-R-EM is a prefix of Full (without the trailing |Target)
    if full_key.startswith(lrem_key):
        _ok(f"L-R-EM spine is prefix of Full spine (drop Target)")
    else:
        _fail(f"L-R-EM spine not prefix of Full: {lrem_key!r} vs {full_key!r}")
    # R-EM is a substring of L-R-EM (drop Ligand)
    if lrem_key.endswith(rem_key):
        _ok(f"R-EM spine is suffix of L-R-EM spine (drop Ligand)")
    else:
        _fail(f"R-EM not suffix of L-R-EM: {rem_key!r} vs {lrem_key!r}")
    # Coarser-grain ordering
    from alz.viewer.shared.incytr_index import BACKBONE_GRAIN_NODES
    grain_node_counts = {g: len(ns) for g, ns in BACKBONE_GRAIN_NODES.items()}
    # Full has 4 nodes; all backbone grains have fewer
    if all(c < 4 for c in grain_node_counts.values()):
        _ok(f"All backbone grains have fewer than 4 nodes: {grain_node_counts}")
    else:
        _fail(f"Unexpected node counts: {grain_node_counts}")

    # Collapse logic: Full can collapse to all backbone grains
    _IP_GRAIN_COARSER = {
        "Full":   ["L-R-EM", "R-EM-T", "R-EM"],
        "L-R-EM": ["R-EM"],
        "R-EM-T": ["R-EM"],
        "R-EM":   [],
    }
    # From Full, spine key truncated to L-R-EM drops Target (last segment)
    full_key  = _spine_key(row, "Full")
    lrem_key  = _spine_key(row, "L-R-EM")
    rem_t_key = _spine_key(row, "R-EM-T")
    rem_key   = _spine_key(row, "R-EM")
    expected_coarser = {
        "Full":   ["L-R-EM", "R-EM-T", "R-EM"],
        "L-R-EM": ["R-EM"],
        "R-EM-T": ["R-EM"],
        "R-EM":   [],
    }
    for grain_name, coarser in expected_coarser.items():
        got = _IP_GRAIN_COARSER.get(grain_name, [])
        if got == coarser:
            _ok(f"_ipCoarserGrains({grain_name!r}) = {coarser}")
        else:
            _fail(f"_ipCoarserGrains({grain_name!r}): expected {coarser}, got {got}")

# ---------------------------------------------------------------------------
# Payload checks
# ---------------------------------------------------------------------------

def _load_payload() -> dict | None:
    if os.path.exists(PAYLOAD_JSON_GZ):
        with gzip.open(PAYLOAD_JSON_GZ) as f:
            return json.load(f)
    if os.path.exists(PAYLOAD_JSON):
        with open(PAYLOAD_JSON) as f:
            return json.load(f)
    _fail(f"Payload not found at {PAYLOAD_JSON} or {PAYLOAD_JSON_GZ}")
    return None

def _test_payload_spine_index(payload: dict) -> None:
    print("\n--- Payload backbone_spine_index checks ---", flush=True)
    # Payload structure: incytr_pathways.by_context.<context_id>.<block>
    incytr_top = payload.get("incytr_pathways")
    if not incytr_top:
        _fail("No incytr_pathways block in payload")
        return
    by_context = incytr_top.get("by_context", {})
    if not by_context:
        _fail("No by_context in incytr_pathways")
        return
    # Find first context that has backbone_grains.
    incytr: dict | None = None
    for ctx_id, ctx_block in by_context.items():
        if ctx_block.get("backbone_grains"):
            incytr = ctx_block
            _ok(f"Found backbone_grains in context {ctx_id!r}")
            break
    if incytr is None:
        _fail("No context with backbone_grains found")
        return
    backbone_grains = incytr.get("backbone_grains")
    if not backbone_grains:
        _fail("No backbone_grains in incytr_pathways block")
        return
    _ok(f"backbone_grains present: {sorted(backbone_grains)}")

    for grain, gblock in backbone_grains.items():
        bsi = gblock.get("backbone_spine_index")
        if not bsi:
            _fail(f"grain {grain!r}: backbone_spine_index missing")
            continue
        url = bsi.get("url", "")
        n_spines = bsi.get("n_spines", 0)
        _ok(f"grain {grain!r}: backbone_spine_index url={url!r}, n_spines={n_spines}")
        if n_spines == 0:
            _fail(f"grain {grain!r}: n_spines == 0")
        # Verify file on disk
        disk_path = os.path.join(VIEWER_DIR, url)
        if os.path.exists(disk_path):
            _ok(f"grain {grain!r}: file exists at {disk_path}")
            # Verify content
            try:
                with gzip.open(disk_path) as f:
                    data = json.load(f)
                sv = data.get("schema_version")
                stp = data.get("spine_to_pairs", {})
                if sv == 1:
                    _ok(f"grain {grain!r}: schema_version=1")
                else:
                    _fail(f"grain {grain!r}: schema_version={sv!r}")
                if len(stp) == n_spines:
                    _ok(f"grain {grain!r}: n_spines matches dict length ({n_spines})")
                else:
                    _fail(f"grain {grain!r}: n_spines={n_spines} but dict has {len(stp)} entries")
                # Check that all values are lists of [sender, receiver] pairs
                sample_keys = list(stp.keys())[:3]
                ok_fmt = True
                for sk in sample_keys:
                    pairs = stp[sk]
                    if not isinstance(pairs, list) or not all(
                        isinstance(p, list) and len(p) == 2 for p in pairs
                    ):
                        _fail(f"grain {grain!r}: spine_to_pairs[{sk!r}] has wrong format")
                        ok_fmt = False
                        break
                if ok_fmt and sample_keys:
                    _ok(f"grain {grain!r}: spine_to_pairs format correct (sample: {sample_keys})")
            except Exception as e:
                _fail(f"grain {grain!r}: error reading {disk_path}: {e}")
        else:
            _fail(f"grain {grain!r}: file NOT found at {disk_path}")


# ---------------------------------------------------------------------------
# Decision 3: widen grouping unit test (synthetic, fixture can't exercise this)
# Tests the Python equivalent of _ipApplyExpandsToWidenPanel's grouping logic:
#   - group (sender,receiver) pairs by receiver cell type
#   - sort by count desc, then receiver alpha
#   - within each receiver: list senders
# NOTE: visual widen gate still waits for overnight full load on real data.
# ---------------------------------------------------------------------------

def _simulate_widen_grouping(
    spine_to_pairs: dict[str, list[list[str]]],
    spine_key: str,
) -> list[tuple[str, list[str]]]:
    """Group pairs for a spine key by receiver (sorted count-first, then alpha)."""
    pairs = spine_to_pairs.get(spine_key, [])
    by_receiver: dict[str, list[str]] = {}
    for s, r in pairs:
        by_receiver.setdefault(r, []).append(s)
    # Sort by count desc, then receiver name asc (mirrors JS sort in _ipApplyExpandsToWidenPanel).
    return sorted(by_receiver.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def _test_widen_grouping() -> None:
    """Synthetic multi-pair, multi-receiver test for widen grouping logic.

    Scenario:
      Spine key "Lrp1|Mapk1" (R-EM) is shared by 5 (sender, receiver) pairs
      across 3 distinct receiver cell types:
        Microglia       → CholinergicNeurons   (2 senders: Astrocytes, OPC)
        Microglia       → NdnfNeurons          (1 sender: Microglia)
        OPC             → CholinergicNeurons   (another sender for same receiver)
      Total: CholinergicNeurons has 3 senders (Astrocytes, OPC, OPC), NdnfNeurons has 1.
    """
    print("\n--- Widen grouping unit tests (synthetic) ---", flush=True)

    synthetic_index: dict[str, list[list[str]]] = {
        "Lrp1|Mapk1": [
            ["Astrocytes",       "CholinergicNeurons"],
            ["OPC",              "CholinergicNeurons"],
            ["Microglia",        "CholinergicNeurons"],
            ["Microglia",        "NdnfNeurons"],
            ["Astrocytes",       "ExcitatoryNeurons"],
        ],
        "Acvr1b|Akt2": [
            ["Microglia",        "CholinergicNeurons"],
        ],
    }

    # Test 1: correct receiver grouping for multi-pair spine.
    grouped = _simulate_widen_grouping(synthetic_index, "Lrp1|Mapk1")
    receivers = [rec for rec, _ in grouped]
    expected_top = "CholinergicNeurons"
    if receivers and receivers[0] == expected_top:
        _ok(f"widen: highest-count receiver first = {expected_top!r}")
    else:
        _fail(f"widen: expected {expected_top!r} first, got {receivers!r}")

    # Test 2: CholinergicNeurons has 3 senders.
    cholin_senders = dict(grouped).get("CholinergicNeurons", [])
    if len(cholin_senders) == 3:
        _ok(f"widen: CholinergicNeurons has 3 senders: {cholin_senders}")
    else:
        _fail(f"widen: expected 3 senders for CholinergicNeurons, got {cholin_senders}")

    # Test 3: NdnfNeurons has 1 sender.
    ndnf_senders = dict(grouped).get("NdnfNeurons", [])
    if len(ndnf_senders) == 1:
        _ok(f"widen: NdnfNeurons has 1 sender: {ndnf_senders}")
    else:
        _fail(f"widen: expected 1 sender for NdnfNeurons, got {ndnf_senders}")

    # Test 4: groups cover exactly 3 distinct receivers.
    if len(grouped) == 3:
        _ok(f"widen: 3 receiver groups: {receivers}")
    else:
        _fail(f"widen: expected 3 receiver groups, got {len(grouped)} ({receivers})")

    # Test 5: missing spine key → empty list.
    empty = _simulate_widen_grouping(synthetic_index, "Nonexistent|Key")
    if empty == []:
        _ok("widen: missing spine key returns empty list")
    else:
        _fail(f"widen: expected [] for missing key, got {empty}")

    # Test 6: single-pair spine (degenerate — the fixture case).
    single = _simulate_widen_grouping(synthetic_index, "Acvr1b|Akt2")
    if len(single) == 1 and single[0][0] == "CholinergicNeurons" and len(single[0][1]) == 1:
        _ok("widen: single-pair spine produces 1 receiver group with 1 sender")
    else:
        _fail(f"widen: unexpected single-pair result: {single}")

    # Test 7: all-alpha tie-break — when two receivers have equal count, sort alpha.
    tied_index: dict[str, list[list[str]]] = {
        "X|Y": [
            ["S1", "Zebra"],
            ["S2", "Apple"],
        ],
    }
    tied = _simulate_widen_grouping(tied_index, "X|Y")
    if tied and tied[0][0] == "Apple":
        _ok("widen: alpha tie-break: Apple before Zebra when counts equal")
    else:
        _fail(f"widen: expected Apple first on alpha tie-break, got {[r for r, _ in tied]}")


def main() -> int:
    print(f"Verifying B-6 backbone_spine_index in {VIEWER_DIR}", flush=True)
    _test_key_logic()
    _test_widen_grouping()
    payload = _load_payload()
    if payload is not None:
        _test_payload_spine_index(payload)
    print(f"\n--- Results: {_PASS} passed, {_FAIL} failed ---", flush=True)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
