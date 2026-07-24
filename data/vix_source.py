"""VIX (market volatility index) fetcher — used as a market-wide risk-off gauge."""

import time
import yfinance as yf
from utils.logger import logger

_vix_cache = {"value": None, "fetched_at": 0}
CACHE_TTL_SECONDS = 900  # 15 minutes — VIX doesn't need per-second freshness


def get_current_vix() -> float | None:
    """Returns the latest VIX close, or None if the fetch fails."""
    now = time.time()
    if _vix_cache["value"] is not None and (now - _vix_cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return _vix_cache["value"]

    try:
        data = yf.Ticker("^VIX").history(period="1d", interval="15m")
        if data.empty:
            return _vix_cache["value"]
        latest = float(data["Close"].iloc[-1])
        _vix_cache["value"] = latest
        _vix_cache["fetched_at"] = now
        return latest
    except Exception as e:
        logger.warning(f"VIX fetch failed (non-critical): {e}")
        return _vix_cache["value"]


def is_high_volatility_regime(threshold: float = 25.0) -> bool:
    """True if VIX is currently above the given threshold (default 25 = elevated risk)."""
    vix = get_current_vix()
    if vix is None:
        return False  # fail safe — don't assume high volatility if we can't check
    return vix > threshold