"""
Macro Bias Engine -- 1H/4H top-down structure and bias detection.
Uses the same fractal swing-detection approach as sniper_logic.SniperLogic._find_swings,
kept standalone so it can run independently of the 1min/5min sniper instance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from utils.logger import logger


def find_swings(df: pd.DataFrame, left_right: int = 3) -> Tuple[List[int], List[int]]:
    """Find swing highs/lows indices using a small fractal window."""
    if df is None or len(df) < (left_right * 2 + 3):
        return [], []
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    sh, sl = [], []
    for i in range(left_right, len(df) - left_right):
        h = highs[i]
        l = lows[i]
        if h == np.max(highs[i - left_right: i + left_right + 1]) and h > highs[i - 1] and h > highs[i + 1]:
            sh.append(i)
        if l == np.min(lows[i - left_right: i + left_right + 1]) and l < lows[i - 1] and l < lows[i + 1]:
            sl.append(i)
    return sh, sl


def classify_structure(df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int]) -> str:
    """
    'bullish' if the last 2 swing highs are ascending AND last 2 swing lows are ascending.
    'bearish' for the mirror case. 'ranging' otherwise.
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "ranging"
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    last_two_highs = [highs[i] for i in swing_highs[-2:]]
    last_two_lows = [lows[i] for i in swing_lows[-2:]]
    ascending_highs = last_two_highs[-1] > last_two_highs[-2]
    ascending_lows = last_two_lows[-1] > last_two_lows[-2]
    descending_highs = last_two_highs[-1] < last_two_highs[-2]
    descending_lows = last_two_lows[-1] < last_two_lows[-2]
    if ascending_highs and ascending_lows:
        return "bullish"
    if descending_highs and descending_lows:
        return "bearish"
    return "ranging"


def get_key_zones(df: pd.DataFrame, swing_highs: List[int], swing_lows: List[int],
                   tolerance_pct: float = 0.0015) -> List[Dict]:
    """Cluster nearby swing points into support/resistance zones."""
    zones: List[Dict] = []

    def _cluster(indices: List[int], price_col: str, zone_type: str):
        if not indices:
            return
        prices = sorted(float(df[price_col].iloc[i]) for i in indices)
        cluster = [prices[0]]
        for p in prices[1:]:
            if abs(p - cluster[-1]) / cluster[-1] <= tolerance_pct:
                cluster.append(p)
            else:
                zones.append({"price": float(np.mean(cluster)), "type": zone_type, "touches": len(cluster)})
                cluster = [p]
        zones.append({"price": float(np.mean(cluster)), "type": zone_type, "touches": len(cluster)})

    _cluster(swing_highs, "high", "resistance")
    _cluster(swing_lows, "low", "support")
    return zones


def get_macro_bias(symbol: str, df_1h: pd.DataFrame = None, df_4h: pd.DataFrame = None) -> Dict:
    """
    Combine 1H and 4H structure into a single macro bias.
    If 1H and 4H disagree, returns 'neutral' -- the HTF layer should not force a bias
    when the two timeframes conflict.
    """
    result = {"bias": "neutral", "key_zones": [], "confidence": 0.0, "htf_1h": "unknown", "htf_4h": "unknown"}
    try:
        bias_1h = "unknown"
        bias_4h = "unknown"
        zones: List[Dict] = []

        if df_1h is not None and len(df_1h) >= 30:
            sh1, sl1 = find_swings(df_1h, left_right=3)
            bias_1h = classify_structure(df_1h, sh1, sl1)
            zones += get_key_zones(df_1h, sh1, sl1)

        if df_4h is not None and len(df_4h) >= 30:
            sh4, sl4 = find_swings(df_4h, left_right=3)
            bias_4h = classify_structure(df_4h, sh4, sl4)
            zones += get_key_zones(df_4h, sh4, sl4)

        result["htf_1h"] = bias_1h
        result["htf_4h"] = bias_4h
        result["key_zones"] = zones

        if bias_1h == bias_4h and bias_1h in ("bullish", "bearish"):
            result["bias"] = bias_1h
            result["confidence"] = 0.7
        elif bias_4h in ("bullish", "bearish") and bias_1h == "ranging":
            result["bias"] = bias_4h
            result["confidence"] = 0.4
        else:
            result["bias"] = "neutral"
            result["confidence"] = 0.0

        return result
    except Exception as e:
        logger.error(f"Error computing macro bias for {symbol}: {e}")
        return result