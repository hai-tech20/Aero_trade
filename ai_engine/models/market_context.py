"""
CROSS-PAIR MARKET CONTEXT FEATURES — AERO TRADE MASTERY EDITION
================================================================
Every symbol's model currently only sees its own price history. Forex pairs
move together — USD strength affects EURUSD/GBPUSD/USDJPY simultaneously,
AUD/NZD vs JPY/CHF is a classic risk-on/risk-off gauge, and gold often leads
broader risk sentiment. This module builds those cross-market signals from
pairs we already have full history for, so no new data source is required.

Two mirrored entry points, which MUST stay in exact key order agreement or
training and live inference will silently misalign feature slots:
  - compute_market_context_series() : batch, time-indexed, for training
  - compute_market_context_snapshot(): single point-in-time, for live inference

Both return values keyed by MARKET_CONTEXT_KEYS in the same order, appended
into the feature vector as ext_<key> by feature_engine.py's existing
extended_context mechanism.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

from utils.logger import TradingLogger

logger = TradingLogger("MarketContext")

# USD is the quote currency in these (pair rising = USD weakening -> flip sign)
_USD_QUOTE = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
# USD is the base currency in these (pair rising = USD strengthening)
_USD_BASE = ["USDJPY", "USDCHF", "USDCAD"]
_RISK_PAIRS = ["AUDJPY", "NZDJPY"]
_GOLD = "XAUUSD"

_WINDOWS = (5, 20, 60)

MARKET_CONTEXT_KEYS = (
    [f"usd_strength_{w}" for w in _WINDOWS]
    + [f"risk_sentiment_{w}" for w in _WINDOWS]
    + [f"gold_mom_{w}" for w in _WINDOWS]
)


def _load_closes(data_dir, symbol: str, timeframe: str) -> Optional[pd.Series]:
    path = data_dir / f"{symbol}_{timeframe}_history.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "time" in df.columns:
            idx = pd.to_datetime(df["time"])
        else:
            idx = pd.to_datetime(df.index)
        s = pd.Series(df["close"].astype(float).values, index=idx)
        return s[~s.index.duplicated(keep="last")].sort_index()
    except Exception as ex:
        logger.warning(f"market_context: failed to load {symbol}/{timeframe}: {ex}")
        return None


def compute_market_context_series(data_dir, timeframe: str) -> pd.DataFrame:
    """
    Batch/training path. Loads the reference basket once for a timeframe and
    returns a time-indexed DataFrame with MARKET_CONTEXT_KEYS as columns.
    Caller reindexes/aligns this to each symbol's own dataframe (ffill for
    any timestamp gaps) before passing into compute_all_features_matrix.
    """
    series: Dict[str, pd.Series] = {}
    for sym in _USD_QUOTE + _USD_BASE + _RISK_PAIRS + [_GOLD]:
        s = _load_closes(data_dir, sym, timeframe)
        if s is not None:
            series[sym] = s

    if not series:
        logger.warning(f"market_context: no reference basket data available for {timeframe}")
        return pd.DataFrame(columns=list(MARKET_CONTEXT_KEYS))

    # Union index across the whole basket, forward-filled — reference pairs
    # occasionally have small gaps that shouldn't zero out the whole context.
    all_idx = sorted(set().union(*[s.index for s in series.values()]))
    aligned = pd.DataFrame(index=pd.DatetimeIndex(all_idx))
    for sym, s in series.items():
        aligned[sym] = s.reindex(aligned.index).ffill()

    returns = {}
    for sym in series:
        returns[sym] = aligned[sym].pct_change().fillna(0.0)

    out = pd.DataFrame(index=aligned.index)
    quote_present = [s for s in _USD_QUOTE if s in returns]
    base_present = [s for s in _USD_BASE if s in returns]
    risk_present = [s for s in _RISK_PAIRS if s in returns]

    usd_strength_1bar = pd.Series(0.0, index=aligned.index)
    n_usd = 0
    for sym in quote_present:
        usd_strength_1bar = usd_strength_1bar - returns[sym]
        n_usd += 1
    for sym in base_present:
        usd_strength_1bar = usd_strength_1bar + returns[sym]
        n_usd += 1
    if n_usd > 0:
        usd_strength_1bar = usd_strength_1bar / n_usd

    risk_1bar = pd.Series(0.0, index=aligned.index)
    if risk_present:
        for sym in risk_present:
            risk_1bar = risk_1bar + returns[sym]
        risk_1bar = risk_1bar / len(risk_present)

    gold_1bar = returns.get(_GOLD, pd.Series(0.0, index=aligned.index))

    for w in _WINDOWS:
        out[f"usd_strength_{w}"] = usd_strength_1bar.rolling(w, min_periods=1).sum()
        out[f"risk_sentiment_{w}"] = risk_1bar.rolling(w, min_periods=1).sum()
        out[f"gold_mom_{w}"] = gold_1bar.rolling(w, min_periods=1).sum()

    return out[list(MARKET_CONTEXT_KEYS)].astype(float)


def compute_market_context_snapshot(recent_closes: Dict[str, pd.Series]) -> Dict[str, float]:
    """
    Live inference path. `recent_closes` maps reference-basket symbol -> a
    Series of its most recent closes (needs >= max(_WINDOWS)+1 bars for the
    longest window to be meaningful; shorter series degrade gracefully).
    Returns a flat dict with the exact same MARKET_CONTEXT_KEYS as
    compute_market_context_series's columns, for a single current snapshot.
    """
    result = {k: 0.0 for k in MARKET_CONTEXT_KEYS}
    try:
        returns: Dict[str, np.ndarray] = {}
        for sym, s in recent_closes.items():
            if s is None or len(s) < 2:
                continue
            arr = s.astype(float).to_numpy()
            returns[sym] = np.diff(arr) / np.maximum(arr[:-1], 1e-9)

        if not returns:
            return result

        max_len = max(len(r) for r in returns.values())

        def _basket_avg(symbols, sign_map):
            total = np.zeros(max_len)
            count = 0
            for sym in symbols:
                if sym not in returns:
                    continue
                r = returns[sym]
                padded = np.zeros(max_len)
                padded[-len(r):] = r
                total += sign_map[sym] * padded
                count += 1
            return total / count if count > 0 else total

        usd_sign = {s: -1.0 for s in _USD_QUOTE}
        usd_sign.update({s: 1.0 for s in _USD_BASE})
        usd_1bar = _basket_avg([s for s in (_USD_QUOTE + _USD_BASE) if s in returns], usd_sign)

        risk_sign = {s: 1.0 for s in _RISK_PAIRS}
        risk_1bar = _basket_avg([s for s in _RISK_PAIRS if s in returns], risk_sign)

        gold_1bar = returns.get(_GOLD, np.zeros(max_len))
        if len(gold_1bar) < max_len:
            padded = np.zeros(max_len)
            padded[-len(gold_1bar):] = gold_1bar
            gold_1bar = padded

        for w in _WINDOWS:
            result[f"usd_strength_{w}"] = float(np.sum(usd_1bar[-w:]))
            result[f"risk_sentiment_{w}"] = float(np.sum(risk_1bar[-w:]))
            result[f"gold_mom_{w}"] = float(np.sum(gold_1bar[-w:]))
    except Exception as ex:
        logger.warning(f"market_context snapshot failed: {ex}")
    return result


def reference_basket_symbols() -> list:
    return list(dict.fromkeys(_USD_QUOTE + _USD_BASE + _RISK_PAIRS + [_GOLD]))
