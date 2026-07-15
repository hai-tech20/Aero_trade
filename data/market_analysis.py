"""
Real market analysis via Finnhub, Alpha Vantage, and FMP.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from utils.logger import logger
from data.api_cache import (
    api_cache, rate_limiter,
    FINNHUB_CACHE_TTL, ALPHA_VANTAGE_CACHE_TTL, FMP_CACHE_TTL,
)
from core.symbols import get_news_identifiers, get_currencies_from_symbol, normalize_symbol

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FMP_KEY = os.getenv("FMP_API_KEY", "")

NEUTRAL_SENTIMENT = {"bullish": 0.5, "bearish": 0.5, "confidence": 0.0}


class MarketAnalysis:
    """Fetch real-time market news, sentiment, and economic calendar."""

    def __init__(self):
        self._high_impact_events: List[Dict] = []
        self._events_fetched_at: float = 0

    def _neutral(self, reason: str = "") -> Dict:
        result = dict(NEUTRAL_SENTIMENT)
        if reason:
            logger.warning(f"[MARKET ANALYSIS] Using neutral sentiment: {reason}")
        return result

    def get_symbol_news_sentiment(self, symbol: str) -> Dict:
        """Finnhub news sentiment for a symbol."""
        cache_key = f"finnhub:{normalize_symbol(symbol)}"
        cached = api_cache.get(cache_key)
        if cached is not None:
            return cached

        if not FINNHUB_KEY:
            return self._neutral("FINNHUB_API_KEY not set")

        if not rate_limiter.can_call("finnhub"):
            return self._neutral("Finnhub rate limit")

        identifiers = get_news_identifiers(symbol)
        sentiments = []

        for ident in identifiers[:2]:
            try:
                rate_limiter.record_call("finnhub")
                resp = requests.get(
                    "https://finnhub.io/api/v1/news-sentiment",
                    params={"symbol": ident, "token": FINNHUB_KEY},
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning(f"Finnhub error {resp.status_code} for {ident}")
                    continue
                data = resp.json()
                bullish = data.get("sentiment", {}).get("bullishPercent", 50) / 100.0
                bearish = data.get("sentiment", {}).get("bearishPercent", 50) / 100.0
                articles = data.get("buzz", {}).get("articlesInLastWeek", 0)
                conf = min(articles / 50, 1.0)
                sentiments.append({"bullish": bullish, "bearish": bearish, "confidence": conf})
            except Exception as e:
                logger.error(f"Finnhub request failed for {ident}: {e}")

        if not sentiments:
            result = self._neutral("Finnhub returned no data")
        else:
            result = {
                "bullish": sum(s["bullish"] for s in sentiments) / len(sentiments),
                "bearish": sum(s["bearish"] for s in sentiments) / len(sentiments),
                "confidence": max(s["confidence"] for s in sentiments),
            }

        api_cache.set(cache_key, result, FINNHUB_CACHE_TTL)
        return result

    def get_alpha_vantage_sentiment(self, symbol: str) -> Dict:
        """Alpha Vantage NEWS_SENTIMENT — called infrequently."""
        cache_key = f"av:{normalize_symbol(symbol)}"
        cached = api_cache.get(cache_key)
        if cached is not None:
            return cached

        if not ALPHA_VANTAGE_KEY:
            return self._neutral("ALPHA_VANTAGE_API_KEY not set")

        if not rate_limiter.can_call("alpha_vantage"):
            return self._neutral("Alpha Vantage rate limit")

        tickers = get_news_identifiers(symbol)
        ticker = tickers[0] if tickers else symbol

        try:
            rate_limiter.record_call("alpha_vantage")
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "apikey": ALPHA_VANTAGE_KEY,
                    "limit": 10,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                return self._neutral(f"Alpha Vantage HTTP {resp.status_code}")

            data = resp.json()
            if "Note" in data or "Information" in data:
                return self._neutral("Alpha Vantage rate limited")

            feed = data.get("feed", [])
            if not feed:
                result = self._neutral("Alpha Vantage no feed data")
            else:
                scores = []
                for item in feed:
                    for ts in item.get("ticker_sentiment", []):
                        if ts.get("ticker", "").upper() in ticker.upper() or True:
                            score = float(ts.get("ticker_sentiment_score", 0))
                            scores.append(score)
                if scores:
                    avg = sum(scores) / len(scores)
                    bullish = max(0, min(1, 0.5 + avg))
                    bearish = 1.0 - bullish
                    result = {"bullish": bullish, "bearish": bearish, "confidence": min(len(scores) / 10, 1.0)}
                else:
                    result = self._neutral("Alpha Vantage no ticker scores")

            api_cache.set(cache_key, result, ALPHA_VANTAGE_CACHE_TTL)
            return result
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return self._neutral(str(e))

    def get_economic_calendar(self) -> List[Dict]:
        """FMP economic calendar for high-impact events."""
        cache_key = "fmp:calendar"
        cached = api_cache.get(cache_key)
        if cached is not None:
            return cached

        if not FMP_KEY:
            logger.warning("[MARKET ANALYSIS] FMP_API_KEY not set — calendar empty")
            return []

        if not rate_limiter.can_call("fmp"):
            return cached if cached else []

        try:
            rate_limiter.record_call("fmp")
            today = datetime.now().strftime("%Y-%m-%d")
            end = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            resp = requests.get(
                "https://financialmodelingprep.com/api/v3/economic_calendar",
                params={"from": today, "to": end, "apikey": FMP_KEY},
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"FMP calendar error {resp.status_code}")
                return []

            events = resp.json()
            if not isinstance(events, list):
                return []

            parsed = []
            for ev in events:
                impact = str(ev.get("impact", "")).lower()
                if impact in ("high", "3", "medium"):
                    parsed.append({
                        "event": ev.get("event", ""),
                        "currency": ev.get("currency", ""),
                        "impact": impact,
                        "time": ev.get("date", ""),
                        "country": ev.get("country", ""),
                    })

            api_cache.set(cache_key, parsed, FMP_CACHE_TTL)
            self._high_impact_events = parsed
            return parsed
        except Exception as e:
            logger.error(f"FMP calendar error: {e}")
            return []

    def is_news_blackout(self, symbol: str, window_minutes: int = 15) -> tuple:
        """Check if high-impact event affects symbol currencies within window."""
        events = self.get_economic_calendar()
        currencies = get_currencies_from_symbol(symbol)
        now = datetime.now()

        for ev in events:
            if ev.get("impact") not in ("high", "3"):
                continue
            ev_currency = ev.get("currency", "").upper()
            if ev_currency not in currencies:
                continue
            try:
                ev_time = datetime.fromisoformat(ev["time"].replace("Z", "+00:00").replace("+00:00", ""))
            except (ValueError, TypeError):
                continue
            delta = abs((ev_time - now).total_seconds()) / 60
            if delta <= window_minutes:
                return True, f"High-impact event: {ev.get('event')} ({ev_currency}) in {delta:.0f}min"
        return False, ""

    async def get_forex_news(self, symbol: str = "EURUSD") -> List[Dict]:
        """Get recent news headlines via Finnhub company-news / general."""
        cache_key = f"finnhub:news:{normalize_symbol(symbol)}"
        cached = api_cache.get(cache_key)
        if cached is not None:
            return cached

        if not FINNHUB_KEY:
            return []

        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": "forex", "token": FINNHUB_KEY},
                timeout=10,
            )
            if resp.status_code != 200:
                return []
            items = resp.json()[:10]
            news = [{
                "title": n.get("headline", ""),
                "summary": n.get("summary", ""),
                "impact": "medium",
                "timestamp": datetime.fromtimestamp(n.get("datetime", 0)).isoformat(),
                "source": n.get("source", "Finnhub"),
            } for n in items]
            api_cache.set(cache_key, news, FINNHUB_CACHE_TTL)
            return news
        except Exception as e:
            logger.error(f"Finnhub news error: {e}")
            return []

    async def get_market_sentiment(self, symbol: str) -> Dict:
        """Combined sentiment from Finnhub (primary) + Alpha Vantage (secondary)."""
        finnhub = self.get_symbol_news_sentiment(symbol)
        av = self.get_alpha_vantage_sentiment(symbol)

        # Weight Finnhub higher since it's refreshed more often
        bullish = finnhub["bullish"] * 0.7 + av["bullish"] * 0.3
        bearish = finnhub["bearish"] * 0.7 + av["bearish"] * 0.3
        confidence = max(finnhub["confidence"], av["confidence"] * 0.5)

        return {
            "symbol": symbol,
            "bullish": bullish,
            "bearish": bearish,
            "neutral": max(0, 1.0 - bullish - bearish),
            "confidence": confidence,
            "timestamp": datetime.now().timestamp(),
        }

    async def analyze_market_conditions(self, symbol: str) -> Dict:
        """Comprehensive market analysis — single entry point for SignalFusion + risk gating."""
        try:
            news = await self.get_forex_news(symbol)
            sentiment = await self.get_market_sentiment(symbol)
            events = self.get_economic_calendar()
            blackout, blackout_reason = self.is_news_blackout(symbol)

            # If sentiment is neutral (no API keys), use a simple price-based sentiment
            if sentiment.get("confidence", 0.0) <= 0.1:
                from data.market_data import MarketDataFeed
                feed = MarketDataFeed()
                try:
                    df = await feed.get_latest_data(symbol, "5min", 50)
                    if df is not None and len(df) >= 20:
                        close = df["close"].values
                        recent_change = (close[-1] - close[-20]) / close[-20]
                        if recent_change > 0.001:
                            sentiment["bullish"] = 0.6
                            sentiment["bearish"] = 0.4
                            sentiment["confidence"] = 0.3
                        elif recent_change < -0.001:
                            sentiment["bullish"] = 0.4
                            sentiment["bearish"] = 0.6
                            sentiment["confidence"] = 0.3
                except Exception:
                    pass

            bias = "neutral"
            if sentiment.get("bullish", 0.5) > 0.5:
                bias = "bullish"
            elif sentiment.get("bearish", 0.5) > 0.5:
                bias = "bearish"

            return {
                "symbol": symbol,
                "bias": bias,
                "sentiment": sentiment,
                "news_count": len(news),
                "upcoming_events": len(events),
                "news_blackout": blackout,
                "blackout_reason": blackout_reason,
                "analysis": f"Market shows {bias} bias with {len(news)} recent news items",
                "timestamp": datetime.now().timestamp(),
            }
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
            return {
                "symbol": symbol,
                "bias": "neutral",
                "sentiment": self._neutral(str(e)),
                "news_count": 0,
                "upcoming_events": 0,
                "news_blackout": False,
                "blackout_reason": "",
                "analysis": "Analysis unavailable",
                "timestamp": datetime.now().timestamp(),
            }


market_analysis = MarketAnalysis()
