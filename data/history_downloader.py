"""Bulk historical data download from MT5."""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, cast

from core.broker.mt5_bridge import MT5Bridge
from config.settings import DATA_DIR, TRADING_CONFIG, DATA_CONFIG
from utils.logger import logger


def download_full_history(
    bridge: MT5Bridge,
    symbol: str,
    timeframe: str,
    years: Optional[int] = None,
    chunk_size: int = 1500,
    chunk_delay_seconds: float = 0.75,
) -> pd.DataFrame:
    resolved_years = int(years if years is not None else DATA_CONFIG.get("history_years", 5))
    all_chunks: List[pd.DataFrame] = []
    cutoff = datetime.now() - timedelta(days=365 * resolved_years)
    before_dt = datetime.now()
    failures = []

    while True:
        df_chunk = bridge.get_historical_data_before(symbol, timeframe, before_dt, chunk_size)
        if df_chunk is None or len(df_chunk) == 0:
            failures.append({
                "before_dt": str(before_dt),
                "last_error": getattr(bridge, "_get_last_error_details")(),
            })
            break
        all_chunks.append(df_chunk)
        oldest_time = cast(datetime, df_chunk.index.to_list()[0] if len(df_chunk.index) else before_dt)
        if oldest_time <= cutoff:
            break
        before_dt = oldest_time
        import time
        time.sleep(chunk_delay_seconds)

    if not all_chunks:
        logger.error(f"No history downloaded for {symbol} {timeframe}; failures={failures}")
        return pd.DataFrame()

    full_df = cast(pd.DataFrame, pd.concat(all_chunks).sort_index())
    full_df = cast(pd.DataFrame, full_df[~full_df.index.duplicated(keep="first")])
    full_df = cast(pd.DataFrame, full_df[full_df.index >= cutoff])

    out_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
    full_df.to_parquet(out_path, compression="gzip")
    logger.info(f"Saved {len(full_df)} bars to {out_path}")
    if failures:
        logger.warning(f"History download for {symbol} {timeframe} completed with trailing failures: {failures}")
    return full_df


def download_all_for_bridge(bridge: MT5Bridge, pairs: Optional[list[str]] = None, timeframes: Optional[list[str]] = None):
    pairs = pairs or cast(list[str], TRADING_CONFIG.get("pairs", ["EURUSD"]))
    timeframes = timeframes or cast(list[str], TRADING_CONFIG.get("timeframes", ["5min"]))
    for symbol in pairs:
        for tf in timeframes:
            try:
                download_full_history(bridge, symbol, tf)
            except Exception as e:
                logger.error(f"Failed to download {symbol} {tf}: {e}")
