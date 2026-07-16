from __future__ import annotations


def make_model_key(symbol: str, timeframe: str, family: str = "rf") -> str:
    return f"{family}_{symbol}_{timeframe}"


def make_scaler_key(symbol: str, timeframe: str) -> str:
    return f"scaler_{symbol}_{timeframe}"


def base_model_key_for_timeframe(timeframe: str) -> str:
    return "gbm_15min" if timeframe == "15min" else f"rf_{timeframe}"
