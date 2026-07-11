"""Bulk historical data download from MT5."""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from core.broker.mt5_bridge import MT5Bridge
from config.settings import DATA_DIR, TRADING_CONFIG, DATA_CONFIG
from utils.logger import logger


def download_full_history(
    bridge: MT5Bridge,
    symbol: str,
    timeframe: str,
    years: int = None,
) -> pd.DataFrame:
    years = years or DATA_CONFIG.get("history_years", 5)
    all_chunks = []
    chunk_size = 5000
    cutoff = datetime.now() - timedelta(days=365 * years)
    before_dt = datetime.now()

    while True:
        df_chunk = bridge.get_historical_data_before(symbol, timeframe, before_dt, chunk_size)
        if df_chunk is None or len(df_chunk) == 0:
            break
        all_chunks.append(df_chunk)
        oldest_time = df_chunk.index.min()
        if oldest_time <= cutoff:
            break
        before_dt = oldest_time.to_pydatetime() if hasattr(oldest_time, "to_pydatetime") else oldest_time

    if not all_chunks:
        return pd.DataFrame()

    full_df = pd.concat(all_chunks).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    full_df = full_df[full_df.index >= cutoff]

    out_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
    full_df.to_parquet(out_path, compression="gzip")
    logger.info(f"Saved {len(full_df)} bars to {out_path}")
    return full_df


def download_all_for_bridge(bridge: MT5Bridge, pairs: list = None, timeframes: list = None):
    pairs = pairs or TRADING_CONFIG.get("pairs", ["EURUSD"])
    timeframes = timeframes or TRADING_CONFIG.get("timeframes", ["5min"])
    for symbol in pairs:
        for tf in timeframes:
            try:
                download_full_history(bridge, symbol, tf)
            except Exception as e:
                logger.error(f"Failed to download {symbol} {tf}: {e}")
