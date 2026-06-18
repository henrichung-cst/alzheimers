from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from alz.bulk_mea import enrich


class MeaEnrichPerformanceHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        enrich._percentile_cache.clear()

    def tearDown(self) -> None:
        enrich._percentile_cache.clear()

    def test_motif_cache_key_depends_on_order_and_kinase_type(self) -> None:
        motifs = pd.Series(["AAAAAAAsAAAAAAA", "BBBBBBBtBBBBBBB"])

        key_a = enrich._motif_cache_key(motifs, kin_type="ser_thr")
        key_b = enrich._motif_cache_key(motifs.iloc[::-1], kin_type="ser_thr")
        key_c = enrich._motif_cache_key(motifs, kin_type="tyrosine")

        self.assertEqual(key_a[0], "ser_thr")
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_percentile_cache_get_moves_hit_to_recent_position(self) -> None:
        old_max = enrich._PERCENTILE_CACHE_MAX_ENTRIES
        enrich._PERCENTILE_CACHE_MAX_ENTRIES = 2
        try:
            first = ("ser_thr", "percentile_rank", "first")
            second = ("ser_thr", "percentile_rank", "second")
            third = ("ser_thr", "percentile_rank", "third")

            enrich._store_cached_percentiles(first, pd.DataFrame({"K1": [1.0]}))
            enrich._store_cached_percentiles(second, pd.DataFrame({"K1": [2.0]}))
            self.assertIsNotNone(enrich._get_cached_percentiles(first))
            enrich._store_cached_percentiles(third, pd.DataFrame({"K1": [3.0]}))

            self.assertIn(first, enrich._percentile_cache)
            self.assertIn(third, enrich._percentile_cache)
            self.assertNotIn(second, enrich._percentile_cache)
        finally:
            enrich._PERCENTILE_CACHE_MAX_ENTRIES = old_max


if __name__ == "__main__":
    unittest.main()
