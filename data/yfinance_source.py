"""Supplementary historical data source using yfinance.

IMPORTANT: yfinance data is used ONLY for:
(a) initial model pretraining when broker history is too shallow, and
(b) stress-testing/backtesting.

Never use yfinance data for live trading decisions or position sizing.
Broker live feeds are authoritative for trading.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
import yfinance as yf

from utils.logger import logger


YFINANCE_TICKER_MAP: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}


def get_yfinance_ticker(symbol: str) -> str:
    base = symbol.upper()
    if base in YFINANCE_TICKER_MAP:
        return YFINANCE_TICKER_MAP[base]
    if base.endswith("W"):
        base = base[:-1]
    if base in YFINANCE_TICKER_MAP:
        return YFINANCE_TICKER_MAP[base]
    if len(base) == 6 and base.isalpha():
        return f"{base}=X"
    return symbol


def fetch_yfinance_history(symbol: str, years: int = 10) -> Optional[pd.DataFrame]:
    try:
        ticker = get_yfinance_ticker(symbol)
        logger.info(f"[YFINANCE] Fetching {ticker} ({years}y)")
        period = f"{years}y"
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            logger.warning(f"[YFINANCE] No data for {ticker}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
        })
        df.index = pd.to_datetime(df.index)
        df.index.name = "time"
        logger.success(f"[YFINANCE] Fetched {len(df)} days for {ticker}")
        return df
    except Exception as e:
        logger.error(f"[YFINANCE] Failed for {symbol}: {e}")
        return None


def fetch_crash_window(symbol: str, event_date: str, window_days: int = 10) -> Optional[pd.DataFrame]:
    try:
        dt = datetime.strptime(event_date, "%Y-%m-%d")
        start = dt - timedelta(days=window_days)
        end = dt + timedelta(days=window_days)
        ticker = get_yfinance_ticker(symbol)
        logger.info(f"[YFINANCE] Fetching crash window {start.date()} to {end.date()} for {ticker}")
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            logger.warning(f"[YFINANCE] No crash-window data for {ticker}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume", "Adj Close": "adj_close",
        })
        df.index = pd.to_datetime(df.index)
        df.index.name = "time"
        logger.success(f"[YFINANCE] Crash window fetched: {len(df)} days for {ticker}")
        return df
    except Exception as e:
        logger.error(f"[YFINANCE] Crash window failed for {symbol}: {e}")
        return None


def save_supplementary_history(symbol: str, df: pd.DataFrame, timeframe: str = "1d") -> Optional[str]:
    try:
        from config.settings import DATA_DIR
        out_path = DATA_DIR / f"{symbol}_{timeframe}_yfinance.parquet"
        df.to_parquet(out_path, compression="gzip")
        logger.info(f"[YFINANCE] Saved supplementary history to {out_path}")
        return str(out_path)
    except Exception as e:
        logger.error(f"[YFINANCE] Failed to save: {e}")
        return None


def load_supplementary_history(symbol: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
    try:
        from config.settings import DATA_DIR
        path = DATA_DIR / f"{symbol}_{timeframe}_yfinance.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        logger.info(f"[YFINANCE] Loaded supplementary history: {len(df)} bars")
        return df
    except Exception as e:
        logger.error(f"[YFINANCE] Failed to load supplementary history: {e}")
        return None
