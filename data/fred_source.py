"""FRED (Federal Reserve Economic Data) fetcher — macro indicators like interest rates."""

import os
import time
import requests
from utils.logger import logger

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

_cache: dict = {}
CACHE_TTL_SECONDS = 3600 * 6  # 6 hours — macro data doesn't change intraday


def get_series_latest(series_id: str) -> float | None:
    """
    Fetch the latest value for a FRED series.
    Common series IDs: 'FEDFUNDS' (Fed funds rate), 'DGS10' (10-year treasury yield).
    """
    now = time.time()
    cached = _cache.get(series_id)
    if cached and (now - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["value"]

    try:
        resp = requests.get(BASE_URL, params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        for obs in observations:
            if obs.get("value") not in (".", None):
                value = float(obs["value"])
                _cache[series_id] = {"value": value, "fetched_at": now}
                return value
        return None
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id} (non-critical): {e}")
        return cached["value"] if cached else None


def is_rates_rising(series_id: str = "FEDFUNDS", lookback: int = 3) -> bool | None:
    """
    Checks whether the last `lookback` observations for a series show an upward trend.
    Returns None if data isn't available (fail-safe — don't assume a trend you can't confirm).
    """
    try:
        resp = requests.get(BASE_URL, params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": lookback,
        }, timeout=10)
        resp.raise_for_status()
        values = [float(o["value"]) for o in resp.json().get("observations", []) if o.get("value") not in (".", None)]
        if len(values) < 2:
            return None
        return all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    except Exception as e:
        logger.warning(f"FRED trend check failed for {series_id}: {e}")
        return None