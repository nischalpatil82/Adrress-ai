"""
fuzzy_engine -- Enterprise-grade RapidFuzz Address Correction Engine.

Usage (CSV):
    from fuzzy_engine import AddressCorrector
    corrector = AddressCorrector("data/realistic_addresses.csv")

Usage (MySQL Database):
    from fuzzy_engine import AddressCorrector
    corrector = AddressCorrector.from_database()

Then:
    result = corrector.correct("prestig apertmnebt 12 main road banashankri bangalor")
"""

from fuzzy_engine.corrector import AddressCorrector

__all__ = ["AddressCorrector"]
__version__ = "2.0.0"
