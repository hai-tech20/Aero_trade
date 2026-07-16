
"""Download real historical data from Yahoo Finance (fallback if MT5 not connected)."""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from utils.logger import logger
from config.settings import DATA_DIR

def download_yfinance_history(symbol: str = "EURUSD=X",
                              timeframe: str = "5m",
                              period: str = "60d"):
    """Download 5-minute data from Yahoo Finance and save as parquet."""
    out_path = DATA_DIR / "EURUSD_5min_history.parquet"
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=timeframe)
        if len(df) == 0:
            logger.error("No data downloaded from Yahoo Finance")
            return None

        # Rename columns to match expected format
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "time"
        df.to_parquet(out_path, compression="gzip")
        logger.success(f"Saved {len(df)} bars from Yahoo Finance to {out_path}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        logger.error(f"Yahoo Finance download failed: {e}")
        return None

if __name__ == "__main__":
    download_yfinance_history()
