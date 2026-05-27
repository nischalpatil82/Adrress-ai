"""
Smoke tests for the fuzzy_engine.v2 stack.

These do NOT hit the network; the geocoder is replaced with NullGeocoder.
Run with:
    python -m pytest tests/test_v2_smoke.py -q
or:
    python tests/test_v2_smoke.py
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestNormalize(unittest.TestCase):
    def test_basic(self) -> None:
        from fuzzy_engine.v2.normalize import parse, normalize_text

        s = "RPNC Systems, 3rd Floor, Bannerghatta Road, Bengaluru-560029"
        norm = normalize_text(s)
        self.assertNotIn(",", norm)
        self.assertEqual(norm.lower(), norm)

        p = parse(s)
        self.assertEqual(p.pincode, "560029")
        self.assertEqual(p.city, "bangalore")  # alias collapsed
        self.assertEqual(p.road_anchor, "bannerghatta")

    def test_pincode_repair(self) -> None:
        from fuzzy_engine.v2.normalize import repair_pincode
        self.assertEqual(repair_pincode("56029"), "056029")
        self.assertEqual(repair_pincode("5600291"), "560029")
        self.assertIsNone(repair_pincode("abc"))


class TestPincodeIndex(unittest.TestCase):
    def test_missing_csv_no_crash(self) -> None:
        from fuzzy_engine.v2.verify import PincodeIndex
        idx = PincodeIndex(Path("does_not_exist.csv"), enable_live_fallback=False)
        self.assertFalse(idx.loaded)
        self.assertIsNone(idx.lookup("560029"))


class TestVerifierOffline(unittest.TestCase):
    def test_null_geocoder_path(self) -> None:
        from fuzzy_engine.v2.verify import AddressVerifier, NullGeocoder, PincodeIndex
        pincodes = PincodeIndex(Path("does_not_exist.csv"), enable_live_fallback=False)
        v = AddressVerifier(provider=NullGeocoder(), validator=None, pincodes=pincodes)
        out = v.verify("anything", expected_pincode="560029")
        self.assertFalse(out.geocoded)
        self.assertIn("geocode_miss", out.notes)


class TestPrefixTrie(unittest.TestCase):
    def test_build_and_query(self) -> None:
        from fuzzy_engine.v2.retrieval import PrefixTrie
        items = [
            (1, "Koramangala 5th Block Bangalore"),
            (2, "Koramangala 4th Block Bangalore"),
            (3, "Indiranagar 100ft Road Bangalore"),
        ]
        trie = PrefixTrie()
        trie.build(items)
        hits = trie.search("koramang", k=5)
        addrs = {h.address for h in hits}
        self.assertEqual(len(hits), 2)
        self.assertTrue(any("4th" in a for a in addrs))
        self.assertTrue(any("5th" in a for a in addrs))


class TestRerankerFeaturesShape(unittest.TestCase):
    def test_featurize_length(self) -> None:
        from fuzzy_engine.v2.reranker import FEATURE_NAMES, Reranker
        from fuzzy_engine.v2.retrieval import Candidate
        c = Candidate(addr_id=1, address="Mg Road Bangalore",
                      scores={"bm25": 0.8, "faiss": 0.7})
        feats = Reranker._featurize("mg road bangalore", c)
        self.assertEqual(len(feats), len(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
