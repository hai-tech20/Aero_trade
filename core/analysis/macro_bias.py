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


def _zone_proximity(current_price: float, zones: List[Dict], direction: str, tolerance_pct: float = 0.0025) -> Dict:
    """
    Is price currently sitting at a validated HTF zone that supports `direction`?
    A long wants to be near a 'support' zone (buying the dip into structure), a
    short wants to be near a 'resistance' zone -- entering AT a zone is a
    materially different (better) entry than the same directional bias taken
    mid-range with no nearby structure.
    """
    if not zones or current_price <= 0 or direction not in ("bullish", "bearish"):
        return {"at_key_zone": False, "zone_quality": 0, "zone_distance_pct": None}
    wanted_type = "support" if direction == "bullish" else "resistance"
    candidates = [z for z in zones if z["type"] == wanted_type]
    if not candidates:
        return {"at_key_zone": False, "zone_quality": 0, "zone_distance_pct": None}
    nearest = min(candidates, key=lambda z: abs(z["price"] - current_price) / current_price)
    dist_pct = abs(nearest["price"] - current_price) / current_price
    at_zone = dist_pct <= tolerance_pct
    return {
        "at_key_zone": bool(at_zone),
        "zone_quality": int(nearest["touches"]) if at_zone else 0,
        "zone_distance_pct": float(dist_pct),
    }


def get_macro_bias(
    symbol: str,
    df_1h: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
    df_1d: pd.DataFrame = None,
    df_30min: pd.DataFrame = None,
    current_price: float = None,
) -> Dict:
    """
    Top-down multi-timeframe confluence: 30min/1H are the "setup" tier, 4H/1D
    are the "trend" tier. Returns a continuous confluence_score (0..1) rather
    than only a binary bias -- a trade taken when 3 timeframes agree AND price
    sits at a validated HTF zone is a genuinely different quality of entry
    than one timeframe alone flipping bullish, and callers should be able to
    demand more from the LTF trigger when confluence is weak instead of just
    gating pass/fail. df_1d and df_30min are optional so this stays backward
    compatible for callers that only have 1H/4H (e.g. before 30min history
    exists for a symbol).
    """
    result = {
        "bias": "neutral", "key_zones": [], "confidence": 0.0,
        "htf_1h": "unknown", "htf_4h": "unknown", "htf_1d": "unknown", "htf_30min": "unknown",
        "confluence_score": 0.0, "agreeing_timeframes": 0, "total_timeframes": 0,
        "at_key_zone": False, "zone_quality": 0, "zone_distance_pct": None,
    }
    try:
        # Weight higher timeframes more heavily -- a daily trend flip means more
        # than a 30min one for "am I trading with or against the real tide".
        tf_weights = {"30min": 1.0, "1h": 1.5, "4h": 2.0, "1d": 2.5}
        frames = {"30min": df_30min, "1h": df_1h, "4h": df_4h, "1d": df_1d}
        structures: Dict[str, str] = {}
        zones: List[Dict] = []

        for tf, df in frames.items():
            if df is not None and len(df) >= 30:
                sh, sl = find_swings(df, left_right=3)
                structures[tf] = classify_structure(df, sh, sl)
                zones += get_key_zones(df, sh, sl)
            else:
                structures[tf] = "unknown"

        result["htf_1h"] = structures["1h"]
        result["htf_4h"] = structures["4h"]
        result["htf_1d"] = structures["1d"]
        result["htf_30min"] = structures["30min"]
        result["key_zones"] = zones

        available = {tf: s for tf, s in structures.items() if s != "unknown"}
        result["total_timeframes"] = len(available)

        if available:
            bull_w = sum(tf_weights[tf] for tf, s in available.items() if s == "bullish")
            bear_w = sum(tf_weights[tf] for tf, s in available.items() if s == "bearish")
            total_w = sum(tf_weights[tf] for tf in available)
            result["agreeing_timeframes"] = sum(1 for s in available.values() if s in ("bullish", "bearish"))

            if bull_w > bear_w and bull_w > 0:
                result["bias"] = "bullish"
                result["confluence_score"] = round(bull_w / total_w, 3)
            elif bear_w > bull_w and bear_w > 0:
                result["bias"] = "bearish"
                result["confluence_score"] = round(bear_w / total_w, 3)
            else:
                result["bias"] = "neutral"
                result["confluence_score"] = 0.0

            # confidence stays for backward compatibility with existing callers
            result["confidence"] = result["confluence_score"] if result["bias"] != "neutral" else 0.0

        if current_price and result["bias"] in ("bullish", "bearish"):
            zp = _zone_proximity(current_price, zones, result["bias"])
            result.update(zp)
            if zp["at_key_zone"]:
                # Being AT a validated zone in the direction of confluence is a
                # genuine entry-quality signal, not just "trend agrees" -- boost
                # the score, capped at 1.0.
                result["confluence_score"] = round(min(1.0, result["confluence_score"] + 0.15), 3)

        return result
    except Exception as e:
        logger.error(f"Error computing macro bias for {symbol}: {e}")
        return result