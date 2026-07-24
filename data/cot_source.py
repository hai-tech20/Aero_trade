"""COT (Commitment of Traders) fetcher — institutional positioning, updated weekly."""

import time
from utils.logger import logger

_cache: dict = {"data": None, "fetched_at": 0}
CACHE_TTL_SECONDS = 3600 * 24  # 24 hours — COT data only updates once a week anyway


def get_latest_cot_positioning(currency_code: str) -> dict | None:
    """
    Returns net institutional positioning for a currency from the latest weekly COT report,
    or None if unavailable. currency_code should match CFTC naming (e.g. 'EURO FX').
    """
    now = time.time()
    if _cache["data"] is not None and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
        df = _cache["data"]
    else:
        try:
            import cot_reports as cot
            df = cot.cot_all(cot_report_type="legacy_fut")
            _cache["data"] = df
            _cache["fetched_at"] = now
        except Exception as e:
            logger.warning(f"COT report fetch failed (non-critical): {e}")
            return None

    try:
        matches = df[df["Market and Exchange Names"].str.contains(currency_code, case=False, na=False)]
        if matches.empty:
            return None
        latest = matches.sort_values("As of Date in Form YYYY-MM-DD").iloc[-1]
        long_pos = float(latest.get("Noncommercial Positions-Long (All)", 0))
        short_pos = float(latest.get("Noncommercial Positions-Short (All)", 0))
        net = long_pos - short_pos
        return {
            "currency": currency_code,
            "net_position": net,
            "bias": "bullish" if net > 0 else ("bearish" if net < 0 else "neutral"),
            "report_date": str(latest.get("As of Date in Form YYYY-MM-DD", "")),
        }
    except Exception as e:
        logger.warning(f"COT parsing failed for {currency_code}: {e}")
        return None