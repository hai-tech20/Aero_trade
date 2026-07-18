"""Per-symbol volatility/statistics profile."""

from typing import Dict
import pandas as pd
from core.symbols import symbol_asset_class, get_pip_size


def compute_volatility_profile(df: pd.DataFrame, is_crypto: bool = False) -> Dict:
    if df is None or len(df) < 20:
        return {}
    try:
        close_col = "close" if "close" in df.columns else "Close"
        high_col = "high" if "high" in df.columns else "High"
        low_col = "low" if "low" in df.columns else "Low"
        returns = df[close_col].pct_change().dropna()
        days_factor = 365 if is_crypto else 252
        intraday_range = ((df[high_col] - df[low_col]) / df[low_col]).dropna()
        profile = {
            "avg_intraday_range_pct": float(intraday_range.mean() * 100) if len(intraday_range) > 0 else 0.0,
            "annualized_volatility_pct": float(returns.std() * (days_factor ** 0.5) * 100) if len(returns) > 0 else 0.0,
            "worst_single_day_pct": float(returns.min() * 100) if len(returns) > 0 else 0.0,
            "best_single_day_pct": float(returns.max() * 100) if len(returns) > 0 else 0.0,
        }
        return profile
    except Exception as e:
        return {"error": str(e)}


def validate_stop_loss_distance(symbol: str, stop_loss_pips: float, profile: Dict) -> Dict:
    if not profile or "error" in profile:
        return {"valid": True, "reason": "No profile available"}
    avg_range = profile.get("avg_intraday_range_pct", 0)
    from core.symbols import symbol_asset_class, get_pip_size
    cls = symbol_asset_class(symbol)
    pip = get_pip_size(symbol)
    if cls == "crypto":
        current_price = 50000.0
        sl_pct = (stop_loss_pips * pip) / current_price * 100
    elif cls == "metal":
        current_price = 2000.0
        sl_pct = (stop_loss_pips * pip) / current_price * 100
    else:
        current_price = 1.1000
        sl_pct = (stop_loss_pips * pip) / current_price * 100
    if avg_range > 0 and sl_pct < avg_range * 0.3:
        return {
            "valid": False,
            "reason": f"Stop loss {sl_pct:.2f}% is too tight vs avg intraday range {avg_range:.2f}% — likely to be hit by noise",
        }
    return {"valid": True, "reason": "Stop loss distance is reasonable"}


def get_symbol_profile(symbol: str) -> Dict:
    from config.settings import DATA_DIR
    import os
    paths = [
        DATA_DIR / f"{symbol}_1d_yfinance.parquet",
        DATA_DIR / f"{symbol}_1min_history.parquet",
        DATA_DIR / f"{symbol}_5min_history.parquet",
    ]
    for path in paths:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                cls = symbol_asset_class(symbol)
                return compute_volatility_profile(df, is_crypto=(cls == "crypto"))
            except Exception:
                continue
    return {}
