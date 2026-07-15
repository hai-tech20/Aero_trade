"""
Trading Bot Configuration — AERO TRADE
"""

import os
from pathlib import Path
from typing import Dict, List, Any

APP_NAME = "AERO TRADE"

BASE_DIR = Path(__file__).parent.parent

LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ai_engine" / "models" / "saved"
CACHE_DIR = BASE_DIR / ".cache"
ASSETS_DIR = BASE_DIR / "assets" / "branding"

for d in (LOG_DIR, DATA_DIR, MODEL_DIR, CACHE_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRADING_CONFIG: Dict[str, Any] = {
    "demo_mode": True,
    "pairs": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"],
    "timeframes": ["1min", "5min", "15min"],
    "risk_per_trade": 0.02,
    "max_daily_loss": 0.05,
    "max_positions": 10,
    "max_position_size": 0.15,
    "default_lot_size": 0.01,
    "broker": "",
    "account_currency": "USD",
    "scalping_enabled": True,
    "immediate_execution": True,
    "aggressive_mode": False,
    "news_blackout_minutes": 15,
    "max_symbols": 50,
}

AI_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.30,
    "update_interval": 5,
}

SIGNAL_FUSION_CONFIG: Dict[str, Any] = {
    "weights": {
        "ai": 0.25,
        "technical": 0.25,
        "price_action": 0.30,
        "market_structure": 0.10,
        # Sentiment: only Alpha Vantage is used for directional signal;
        # Finnhub only provides article count (news volume flag, not sentiment)
        "sentiment": 0.10,
    },
    "min_confidence": {
        "ai": 0.30,
        "technical": 0.30,
        "price_action": 0.20,
        # Sentiment confidence only from Alpha Vantage
        "sentiment": 0.10,
        "combined": 0.35,
    },
    "min_sources": 2,
    "time_window_seconds": 300,
    "decay_factor": 0.95,
    "require_confirmation": True,
}

RISK_CONFIG: Dict[str, Any] = {
    "stop_loss_pips": 100,
    "take_profit_pips": 0,
    "variable_tp": True,
    "tp_multiplier": 1.5,
    "max_drawdown_pct": 10.0,
    "max_daily_trades": 200,
    "max_consecutive_losses": 5,
    "max_position_pct": 5.0,
    "max_daily_loss_pct": 5.0,
    "scalping_sl_pips": 50,
    "scalping_tp_pips": 30,
    "crypto_sl_dollars": 200,
    "crypto_tp_dollars": 150,
    "metal_sl_pips": 30,
    "metal_tp_pips": 20,
}

BROKER_CONFIG: Dict[str, Any] = {
    "name": "",
    "use_mt5": True,
    "leverage": 100,
    "verify_timeout_seconds": 15,
}

DATA_CONFIG: Dict[str, Any] = {
    "history_days": 365,
    "history_years": 5,
    "update_interval": 60,
}

LOGGING_CONFIG: Dict[str, Any] = {
    "log_file": LOG_DIR / "aero_trade.log",
}

DEV_CONFIG: Dict[str, Any] = {
    "debug": True,
    "paper_trading": True,
}


def load_env_vars():
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            pass
    os.environ.setdefault("TRADING_MODE", "DEMO")
    os.environ.setdefault("LOG_LEVEL", "INFO")


load_env_vars()


def validate_settings() -> bool:
    errors = []
    if TRADING_CONFIG["risk_per_trade"] > 0.20:
        errors.append("Risk per trade must be ≤ 20%")
    weights_sum = sum(SIGNAL_FUSION_CONFIG["weights"].values())
    if abs(weights_sum - 1.0) > 0.001:
        errors.append(f"Signal weights must sum to 1.0 (got {weights_sum})")
    for p in (LOG_DIR, DATA_DIR, MODEL_DIR):
        if not p.exists():
            try:
                p.mkdir(parents=True)
            except Exception as e:
                errors.append(f"Cannot create directory {p}: {e}")
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}")
        return False
    return True


def is_demo_mode() -> bool:
    return TRADING_CONFIG["demo_mode"]


def get_model_path(name: str) -> Path:
    return MODEL_DIR / f"{name}.joblib"


def get_data_path(symbol: str, timeframe: str) -> Path:
    return DATA_DIR / f"{symbol}_{timeframe}_history.parquet"


if __name__ == "__main__":
    print(f"✓ {APP_NAME} settings loaded")
    print(f"Base dir: {BASE_DIR}")
    validate_settings()
