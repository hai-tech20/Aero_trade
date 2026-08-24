"""
ADVANCED DATA SOURCE INTEGRATION — AERO TRADE MASTERY EDITION
============================================================
Aggregates macro & sentiment features to enrich model input:
  - FRED: VIX, DXY proxy (DTWEXBGS), T10Y2Y yield curve, FFR, inflation
  - Alpha Vantage: FX, market data, news + sentiment when available
  - yfinance supplement: VIX, SPY, QQQ, DXY, GLD, BTC proxies for correlation
  - Binance public API: crypto market structure & volume
  - Economic calendar + News volume proxies

Designed to be non-blocking: graceful fallback when API keys are missing.
"""

from __future__ import annotations
import os
import time
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from utils.logger import TradingLogger
logger = TradingLogger("DataSource")

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _env_key(name: str) -> str:
    return os.environ.get(name, "") or ""


def _read_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 12) -> Optional[Any]:
    try:
        req = Request(url, headers=headers or {"User-Agent": "AeroTrade/2.0"})
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except Exception as ex:
        logger.debug(f"HTTP fetch failed for {url[:60]}: {ex}")
        return None


def _csv_series_to_df(txt: str) -> pd.DataFrame:
    from io import StringIO
    lines = [l for l in txt.splitlines() if l and not l.startswith("#")]
    try:
        return pd.read_csv(StringIO("\n".join(lines)))
    except Exception:
        return pd.DataFrame()


# ========================================================
# FRED data (free API, https://fred.stlouisfed.org/docs/api/fred/)
# ========================================================
class FredFetcher:
    """VIX, yield curve, fed funds, USD index proxies."""

    SERIES = {
        "VIX": "VIXCLS",
        "DXY": "DTWEXBGS",
        "YIELD10Y2Y": "T10Y2Y",
        "EFFR": "EFFR",
        "CPI": "CPIAUCSL",
        "GDP": "GDP",
        "UNRATE": "UNRATE",
        "FEDFUNDS": "FEDFUNDS",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _env_key("FRED_API_KEY") or ""
        self._cache_file = CACHE_DIR / "fred_cache.json"
        self._cache: Dict[str, Dict[str, float]] = self._load()

    def _load(self) -> Dict[str, Dict[str, float]]:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text())
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        try:
            self._cache_file.write_text(json.dumps(self._cache, default=str))
        except Exception:
            pass

    def fetch_latest(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        today = datetime.utcnow().date().isoformat()
        # If cached today, use cache
        if self._cache.get("_date") == today and len(self._cache) > 2:
            for k, v in self._cache.items():
                if k != "_date":
                    out[k] = float(v) if isinstance(v, (int, float)) else 0.0
            return out
        if not self.api_key:
            # Try free proxy: yfinance fallback VIX only via yf
            return self._yf_proxy_macro()
        base = "https://api.stlouisfed.org/fred/series/observations"
        for name, sid in self.SERIES.items():
            url = f"{base}?series_id={sid}&api_key={self.api_key}&file_type=json&sort_order=desc&limit=1"
            data = _read_json(url)
            if data and "observations" in data and len(data["observations"]):
                try:
                    val = float(data["observations"][0]["value"])
                    out[name.lower()] = val
                    self._cache[name.lower()] = val
                except Exception:
                    pass
            time.sleep(0.25)
        self._cache["_date"] = today
        self._save()
        if not out:
            return self._yf_proxy_macro()
        return out

    def _yf_proxy_macro(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            import yfinance as yf
            proxies = {
                "vix": "^VIX",
                "spx": "^GSPC",
                "dxy": "DX-Y.NYB",
                "tnx": "^TNX",
                "gld": "GLD",
                "btc": "BTC-USD",
            }
            for k, t in proxies.items():
                try:
                    h = yf.download(t, period="5d", progress=False, auto_adjust=True)
                    if not h.empty and "Close" in h.columns:
                        out[k] = float(h["Close"].iloc[-1])
                        if len(h) >= 2:
                            out[f"{k}_chg"] = float(h["Close"].iloc[-1] / h["Close"].iloc[-2] - 1)
                        else:
                            out[f"{k}_chg"] = 0.0
                except Exception:
                    continue
        except Exception:
            pass
        return out


# ========================================================
# Alpha Vantage — FX weekly + news sentiment (free key 5req/min)
# ========================================================
class AlphaVantageFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _env_key("ALPHA_VANTAGE_API_KEY") or ""
        self._cache_file = CACHE_DIR / "alpha_cache.json"
        self._cache: Dict[str, Any] = {}
        try:
            if self._cache_file.exists():
                self._cache = json.loads(self._cache_file.read_text())
        except Exception:
            pass

    def save(self):
        try:
            self._cache_file.write_text(json.dumps(self._cache, default=str))
        except Exception:
            pass

    def fetch_news_sentiment(self, tickers: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Return aggregate sentiment features (0..1 bullish score, volume counts, avg relevance).
        """
        out = {"news_bullish_pct": 0.5, "news_volume_z": 0.0, "news_avg_relevance": 0.0}
        if not self.api_key:
            return out
        today = datetime.utcnow().date().isoformat()
        if self._cache.get("news_date") == today:
            return {**out, **self._cache.get("news_data", {})}
        topics = ",".join([
            "top_news", "earnings", "ipo", "mergers_and_acquisitions", "economy_monetary",
            "economy_macro", "energy_transportation", "finance", "life_sciences",
            "manufacturing", "real_estate", "retail_wholesale", "technology"
        ])
        url = (
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&"
            f"apikey={self.api_key}&limit=200&topics={topics}"
        )
        data = _read_json(url)
        if not data or "feed" not in data:
            return out
        feed = data["feed"]
        scores = []
        rels = []
        bullish = 0
        for f in feed:
            try:
                s = float(f.get("overall_sentiment_score", 0.0))
                lab = str(f.get("overall_sentiment_label", "Neutral")).lower()
                rel = float(f.get("avg_relevance_score_overall", 0.0))
                scores.append(s)
                rels.append(rel)
                if "bullish" in lab or s > 0.15:
                    bullish += 1
            except Exception:
                continue
        if scores:
            out["news_bullish_pct"] = float(bullish / max(1, len(scores)))
            out["news_volume_z"] = float(np.clip(len(scores) / 200.0, 0, 1))
            out["news_avg_relevance"] = float(np.mean(rels))
            out["news_sentiment_score_mean"] = float(np.mean(scores))
            out["news_sentiment_score_std"] = float(np.std(scores))
        self._cache["news_date"] = today
        self._cache["news_data"] = out
        self.save()
        return out


# ========================================================
# Binance public — crypto data, order-book proxies, funding rates
# ========================================================
class BinancePublic:
    BASE = "https://api.binance.com"

    def fetch_futures_funding(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        out = {"funding_rate": 0.0, "open_interest_chg": 0.0, "funding_bullish": 0.5}
        try:
            data = _read_json(f"{self.BASE}/fapi/v1/fundingRate?symbol={symbol}&limit=10")
            if data and isinstance(data, list) and len(data):
                rates = [float(r["fundingRate"]) for r in data if "fundingRate" in r]
                if rates:
                    out["funding_rate"] = float(rates[-1])
                    out["funding_bullish"] = float(rates[-1] > 0.0001) - float(rates[-1] < -0.0001)
                    if len(rates) >= 2:
                        out["funding_rate_chg"] = float(rates[-1] - rates[0])
        except Exception:
            pass
        return out

    def fetch_spot_klines(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 50) -> pd.DataFrame:
        try:
            data = _read_json(f"{self.BASE}/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}")
            if data:
                df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume",
                                                 "ct", "qv", "nt", "tbbav", "tbqav", "ignore"])
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                return df
        except Exception:
            pass
        return pd.DataFrame()


# ========================================================
# Composed: single call gives everything the ML needs as feature dictionary
# ========================================================
class MacroContext:
    """Aggregates all available external macro context into a small dict of numerical features."""

    def __init__(self, fred_key: Optional[str] = None, alpha_key: Optional[str] = None, force_refresh_hours: int = 6):
        self.fred = FredFetcher(api_key=fred_key)
        self.alpha = AlphaVantageFetcher(api_key=alpha_key)
        self.binance = BinancePublic()
        self.force_refresh = timedelta(hours=force_refresh_hours)
        self._ts: Optional[datetime] = None
        self._last: Dict[str, float] = {}

    def collect(self) -> Dict[str, float]:
        now = datetime.utcnow()
        if self._ts and (now - self._ts) < self.force_refresh and self._last:
            return dict(self._last)
        out: Dict[str, float] = {}
        try:
            out.update(self.fred.fetch_latest())
        except Exception:
            pass
        try:
            out.update(self.alpha.fetch_news_sentiment())
        except Exception:
            pass
        try:
            out.update(self.binance.fetch_futures_funding("BTCUSDT"))
            out.update({f"bn_{k}": v for k, v in self.binance.fetch_futures_funding("ETHUSDT").items()})
        except Exception:
            pass
        # Normalize all numeric
        for k, v in list(out.items()):
            try:
                out[k] = float(np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6))
            except Exception:
                del out[k]
        self._last = out
        self._ts = now
        return out

    # Returns feature vector aligned with feature_engine expectations
    def as_feature_dict(self) -> Dict[str, float]:
        ctx = self.collect()
        feat = {}
        # Normalize to z-style or 0..1
        for k, v in ctx.items():
            if v is None:
                feat[k] = 0.0
                continue
            feat[k] = float(v)
        # BTC correlation flag
        feat["macro_btc_chg"] = float(ctx.get("btc_chg", 0.0))
        feat["macro_vix_level"] = float(np.clip(ctx.get("vix", 20) / 40.0, 0, 1))
        feat["macro_high_vol_flag"] = float(ctx.get("vix", 20) > 25)
        feat["macro_yield_curve_inv"] = float(ctx.get("yield10y2y", 0) < 0)
        return feat
