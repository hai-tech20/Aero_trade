import requests
import time
import logging
from config.settings import APP_NAME

logger = logging.getLogger(APP_NAME)

_binance_cache = {"data": None, "fetched_at": 0}
CACHE_TTL_SECONDS = 300  # 5 minutes


def get_binance_btc_klines(interval: str = "1h", limit: int = 50) -> list | None:
    """Fetch Binance BTCUSDT klines (public endpoint, no key needed)."""
    now = time.time()
    if _binance_cache["data"] and (now - _binance_cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return _binance_cache["data"]

    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        _binance_cache["data"] = data
        _binance_cache["fetched_at"] = now
        logger.debug(f"[BINANCE] Fetched {len(data)} BTCUSDT klines")
        return data
    except requests.RequestException as e:
        logger.warning(
            f"Binance klines fetch failed (non-critical, using broker price only): {e}"
        )
        return None


def get_binance_btc_price() -> float | None:
    """Get latest Binance BTCUSDT close price from klines."""
    klines = get_binance_btc_klines()
    if not klines or len(klines) < 1:
        return None
    try:
        return float(klines[-1][4])  # kline index 4 is close price
    except (IndexError, ValueError):
        return None


def check_btc_divergence(broker_price: float, divergence_threshold_pct: float = 1.5) -> dict:
    """Check divergence between broker BTCUSD price and Binance BTCUSDT price.

    Returns:
        dict: { "diverges": bool, "pct_divergence": float, "binance_price": float | None, "msg": str }
    """
    binance_price = get_binance_btc_price()
    if binance_price is None:
        return {
            "diverges": False,
            "pct_divergence": 0.0,
            "binance_price": None,
            "msg": "Binance price unavailable — using broker price only",
        }

    pct_divergence = abs((broker_price - binance_price) / ((broker_price + binance_price) / 2)) * 100
    diverges = pct_divergence > divergence_threshold_pct
    msg = ""
    if diverges:
        msg = f"BTCUSD: broker price ${broker_price:.2f} vs Binance ${binance_price:.2f} — {pct_divergence:.2f}% divergence, proceeding with caution"
    else:
        msg = f"BTCUSD: broker price ${broker_price:.2f} vs Binance ${binance_price:.2f} — {pct_divergence:.2f}% divergence, within threshold"

    return {
        "diverges": diverges,
        "pct_divergence": pct_divergence,
        "binance_price": binance_price,
        "msg": msg,
    }
