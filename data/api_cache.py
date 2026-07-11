"""Shared in-memory TTL cache and rate limiter for market data APIs."""

import time
from typing import Any, Dict, Optional


class TTLCache:
    """Simple in-memory cache with per-key TTL."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry:
            return None
        if time.time() - entry["fetched_at"] >= entry["ttl"]:
            del self._store[key]
            return None
        return entry["data"]

    def set(self, key: str, data: Any, ttl_seconds: int):
        self._store[key] = {"data": data, "fetched_at": time.time(), "ttl": ttl_seconds}

    def clear(self):
        self._store.clear()


class RateLimiter:
    """Token-bucket style rate limiter per provider."""

    def __init__(self):
        self._last_call: Dict[str, float] = {}
        self._min_interval: Dict[str, float] = {
            "finnhub": 1.0,       # 60/min
            "alpha_vantage": 3600.0,  # ~25/day, cache aggressively
            "fmp": 30.0,          # ~250/day
        }

    def can_call(self, provider: str) -> bool:
        min_interval = self._min_interval.get(provider, 60.0)
        last = self._last_call.get(provider, 0)
        return (time.time() - last) >= min_interval

    def record_call(self, provider: str):
        self._last_call[provider] = time.time()


# Global instances
api_cache = TTLCache()
rate_limiter = RateLimiter()

# Cache TTLs (seconds)
FINNHUB_CACHE_TTL = 900      # 15 min
ALPHA_VANTAGE_CACHE_TTL = 14400  # 4 hours
FMP_CACHE_TTL = 1800         # 30 min
