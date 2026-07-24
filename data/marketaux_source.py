"""Marketaux news sentiment fetcher — higher free quota (100/day) than Alpha Vantage (25/day)."""

import os
import time
import requests
from utils.logger import logger

MARKETAUX_TOKEN = os.getenv("MARKETAUX_API_TOKEN", "")
BASE_URL = "https://api.marketaux.com/v1/news/all"

_cache: dict = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes


def get_symbol_sentiment(symbol: str) -> dict:
    """
    Returns {'bullish': float, 'bearish': float, 'confidence': float, 'article_count': int}
    Falls back to neutral if the API fails or the daily quota is exhausted.
    """
    now = time.time()
    cached = _cache.get(symbol)
    if cached and (now - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["data"]

    neutral = {"bullish": 0.5, "bearish": 0.5, "confidence": 0.0, "article_count": 0}
    try:
        resp = requests.get(BASE_URL, params={
            "symbols": symbol,
            "filter_entities": "true",
            "language": "en",
            "api_token": MARKETAUX_TOKEN,
        }, timeout=10)
        if resp.status_code == 429:
            logger.warning(f"Marketaux rate limit hit for {symbol} — using neutral sentiment")
            return neutral
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            result = neutral
        else:
            scores = []
            for article in data:
                for entity in article.get("entities", []):
                    if entity.get("symbol", "").upper() == symbol.upper() and "sentiment_score" in entity:
                        scores.append(entity["sentiment_score"])
            if not scores:
                result = neutral
            else:
                avg = sum(scores) / len(scores)
                bullish = max(0.0, min(1.0, 0.5 + avg / 2))
                result = {
                    "bullish": bullish,
                    "bearish": 1.0 - bullish,
                    "confidence": min(1.0, len(scores) / 10),
                    "article_count": len(data),
                }

        _cache[symbol] = {"data": result, "fetched_at": now}
        return result
    except Exception as e:
        logger.warning(f"Marketaux fetch failed for {symbol} (non-critical): {e}")
        return cached["data"] if cached else neutral