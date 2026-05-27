"""
fuzzy_engine.v2.warm_geocode_cache
==================================
Bulk-geocodes every address in models/addresses.npy and stores the result
in the v2 SQLite geocode cache. Subsequent calls to AddressVerifier hit the
cache instead of the network, making /v2/correct nearly instant in production.

Usage:
    # Default: respects V2_GEOCODER (auto -> nominatim if no key)
    python -m fuzzy_engine.v2.warm_geocode_cache

    # Force a specific provider for the warm-up
    python -m fuzzy_engine.v2.warm_geocode_cache --provider locationiq

    # Resume after interruption (skips rows already cached)
    python -m fuzzy_engine.v2.warm_geocode_cache --resume

    # Limit the run to first N rows (for a smoke test)
    python -m fuzzy_engine.v2.warm_geocode_cache --limit 200

Throughput notes:
    - Public Nominatim: 1 req/s -> ~20 hours for 73k rows.
    - LocationIQ free : 2 req/s -> ~10 hours for 73k rows.
    - LocationIQ paid : 60 req/s -> ~20 minutes.
The script is fully resumable; safe to Ctrl+C any time and re-run with
--resume to pick up where you left off.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import numpy as np

from fuzzy_engine.v2.config import ADDRESSES_PATH
from fuzzy_engine.v2.verify import (
    AddressVerifier,
    GeocodeCache,
    GoogleGeocoder,
    LocationIQGeocoder,
    NominatimGeocoder,
    NullGeocoder,
    OpenCageGeocoder,
    auto_select_provider,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2.warm_cache")


PROVIDERS = {
    "auto": auto_select_provider,
    "google": lambda: GoogleGeocoder(),
    "locationiq": lambda: LocationIQGeocoder(),
    "opencage": lambda: OpenCageGeocoder(),
    "nominatim": lambda: NominatimGeocoder(),
    "null": lambda: NullGeocoder(),
}


_stop = False


def _install_signal_handler() -> None:
    def handler(signum, frame):  # noqa: ARG001
        global _stop
        log.warning("Interrupt received; finishing the current request and exiting.")
        _stop = True
    signal.signal(signal.SIGINT, handler)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=list(PROVIDERS), default="auto")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process the first N addresses (0 = all)")
    p.add_argument("--resume", action="store_true",
                   help="Skip addresses already in the cache")
    p.add_argument("--addresses", default=str(ADDRESSES_PATH),
                   help="Path to addresses.npy")
    p.add_argument("--report-every", type=int, default=50,
                   help="Log progress every N addresses")
    args = p.parse_args()

    addrs_path = Path(args.addresses)
    if not addrs_path.exists():
        log.error("addresses.npy not found at %s", addrs_path)
        return 2

    log.info("Loading %s ...", addrs_path)
    addresses = list(np.load(addrs_path, allow_pickle=True))
    if args.limit > 0:
        addresses = addresses[: args.limit]
    log.info("Loaded %d addresses.", len(addresses))

    provider = PROVIDERS[args.provider]()
    log.info("Using provider: %s", provider.name)
    if provider.name == "null":
        log.error("Null provider selected; nothing to do. "
                  "Set V2_GEOCODER or pass --provider.")
        return 3

    verifier = AddressVerifier(provider=provider)
    cache: GeocodeCache = verifier.cache

    _install_signal_handler()

    # ---------------------------------------------------------------
    n_total = len(addresses)
    n_cached = n_geocoded = n_failed = n_skipped = 0
    t_start = time.time()

    for i, addr in enumerate(addresses, 1):
        if _stop:
            break

        # Skip if cached and resume mode
        if args.resume:
            cached = cache.get(provider.name, addr)
            if cached is not None:
                n_skipped += 1
                if i % args.report_every == 0:
                    _log_progress(i, n_total, t_start, n_geocoded,
                                  n_failed, n_skipped)
                continue

        try:
            geo = provider.geocode(addr)
            if geo is None:
                n_failed += 1
                # Cache the miss so we don't re-hit it (TTL still applies)
                cache.put(provider.name, addr, None)
            else:
                n_geocoded += 1
                cache.put(provider.name, addr, geo)
        except Exception as exc:  # noqa: BLE001
            log.warning("query #%d failed: %s", i, exc)
            n_failed += 1

        if i % args.report_every == 0:
            _log_progress(i, n_total, t_start, n_geocoded,
                          n_failed, n_skipped)

    log.info("---- DONE ----")
    log.info("processed=%d  geocoded=%d  miss/failed=%d  skipped(cached)=%d  elapsed=%.1fs",
             i, n_geocoded, n_failed, n_skipped, time.time() - t_start)
    return 0


def _log_progress(i: int, n: int, t_start: float,
                  ok: int, fail: int, skip: int) -> None:
    pct = 100.0 * i / max(n, 1)
    elapsed = time.time() - t_start
    rate = i / elapsed if elapsed > 0 else 0.0
    eta_s = (n - i) / rate if rate > 0 else 0.0
    log.info("[%5d/%5d  %5.1f%%]  ok=%d  fail=%d  skip=%d  rate=%.2f/s  eta=%s",
             i, n, pct, ok, fail, skip, rate, _fmt_eta(eta_s))


def _fmt_eta(s: float) -> str:
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


if __name__ == "__main__":
    sys.exit(main())
