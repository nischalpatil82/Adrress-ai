"""
fuzzy_engine.v2
===============
Next-generation address correction stack.

Layers:
    L1 normalize.py   - tokenize, transliterate, structured parse
    L2 speller.py     - typo correction (T5 + LM + dictionaries)
    L3 retrieval.py   - prefix trie + BM25 + FAISS dense retrieval
    L4 reranker.py    - LightGBM rerank with calibrated probabilities
    L5 verify.py      - Google Geocoding + India Post pincode validation

Public entry point:
    from fuzzy_engine.v2 import AddressPipeline
    pipe = AddressPipeline.from_config()
    pipe.correct("rpnc systems 3rd flour berrergata rood bengaluru")
    pipe.autocomplete("koramang", k=5)
"""

from fuzzy_engine.v2.orchestrator import AddressPipeline

__all__ = ["AddressPipeline"]
__version__ = "2.0.0"
