"""Bulk historical data download from MT5."""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, cast

from core.broker.mt5_bridge import MT5Bridge
from config.settings import DATA_DIR, TRADING_CONFIG, DATA_CONFIG
from utils.logger import logger


def _timeframe_delta(timeframe: str) -> timedelta:
    return {
        "1min": timedelta(minutes=1),
        "5min": timedelta(minutes=5),
        "15min": timedelta(minutes=15),
        "30min": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
    }.get(timeframe, timedelta(minutes=1))


def download_full_history(
    bridge: MT5Bridge,
    symbol: str,
    timeframe: str,
    years: Optional[int] = None,
    chunk_size: int = 1500,
    chunk_delay_seconds: float = 0.75,
) -> pd.DataFrame:
    if not bridge.ensure_symbol_selected(symbol):
        logger.error(f"Cannot download history because symbol is not selected/visible: {symbol}")
        return pd.DataFrame()
    resolved_years = int(years if years is not None else DATA_CONFIG.get("history_years", 5))
    all_chunks: List[pd.DataFrame] = []
    cutoff = datetime.now() - timedelta(days=365 * resolved_years)
    before_dt = datetime.now()
    failures = []
    previous_oldest: Optional[datetime] = None
    previous_chunk: Optional[pd.DataFrame] = None
    page_step = _timeframe_delta(timeframe)
    debug_trace = os.getenv("AERO_TRACE_HISTORY", "").lower() in {"1", "true", "yes"}
    iteration = 0

    while True:
        iteration += 1
        requested_before = before_dt
        df_chunk = bridge.get_historical_data_before(symbol, timeframe, requested_before, chunk_size)
        if df_chunk is None or len(df_chunk) == 0:
            failures.append({
                "before_dt": str(requested_before),
                "last_error": getattr(bridge, "_get_last_error_details")(),
                "iteration": iteration,
            })
            if debug_trace:
                logger.info(
                    f"History page {symbol} {timeframe} iter={iteration} before={requested_before} -> empty/none | last_error={getattr(bridge, '_get_last_error_details')()['raw']}"
                )
            break

        oldest_time = cast(datetime, df_chunk.index.min().to_pydatetime() if len(df_chunk.index) else requested_before)
        newest_time = cast(datetime, df_chunk.index.max().to_pydatetime() if len(df_chunk.index) else requested_before)
        overlap = False
        duplicate_oldest = previous_oldest is not None and oldest_time >= previous_oldest
        if previous_chunk is not None and len(previous_chunk.index) > 0:
            overlap = bool(previous_chunk.index.intersection(df_chunk.index).size)

        if debug_trace:
            logger.info(
                f"History page {symbol} {timeframe} iter={iteration} before={requested_before} rows={len(df_chunk)} oldest={oldest_time} newest={newest_time} overlap={overlap} duplicate_oldest={duplicate_oldest}"
            )

        if duplicate_oldest:
            failures.append({
                "before_dt": str(requested_before),
                "last_error": {"code": None, "message": "stalled_duplicate_oldest_page", "raw": None},
                "iteration": iteration,
                "oldest_time": str(oldest_time),
            })
            logger.warning(
                f"Detected stalled history paging for {symbol} {timeframe}: oldest_time={oldest_time} previous_oldest={previous_oldest}"
            )
            break

        all_chunks.append(df_chunk)
        previous_oldest = oldest_time
        previous_chunk = df_chunk
        if oldest_time <= cutoff:
            break
        before_dt = oldest_time - page_step
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


def download_all_for_bridge(
    bridge: MT5Bridge,
    pairs: Optional[list[str]] = None,
    timeframes: Optional[list[str]] = None,
    years: Optional[int] = None,
):
    pairs = pairs or cast(list[str], TRADING_CONFIG.get("pairs", ["EURUSD"]))
    timeframes = timeframes or cast(list[str], TRADING_CONFIG.get("timeframes", ["5min"]))
    for symbol in pairs:
        for tf in timeframes:
            try:
                download_full_history(bridge, symbol, tf, years=years)
            except Exception as e:
                logger.error(f"Failed to download {symbol} {tf}: {e}")
