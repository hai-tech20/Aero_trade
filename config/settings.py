"""
Trading Bot Configuration — AERO TRADE
"""

import os
from pathlib import Path
from typing import Any, Dict

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
    "pairs": [
        "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD",
        "EURCAD", "GBPCHF", "GBPJPY", "USDCAD",
        "AUDUSD", "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCHF",
        "GBPAUD", "GBPCAD", "GBPNZD", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
        "AUDCAD", "AUDNZD", "AUDCHF", "NZDCAD", "NZDCHF", "CADCHF",
        "USDZAR", "USDMXN", "USDSGD", "USDNOK", "USDSEK",
    ],
    "timeframes": ["1min", "5min", "15min", "30min", "1h", "4h", "1d"],
    # Every model uses a fixed 60-bar horizon for its training target
    # regardless of timeframe — on 4h/1d that's a ~10-day to ~3-month horizon,
    # not an intraday move. Live trading should only pick from timeframes
    # short enough that the resulting signal is actually intraday-relevant;
    # 4h/1d models still train and can be inspected, just not traded live.
    # 1min excluded from live selection: its models backtest at similar
    # accuracy to longer timeframes, but on 1-minute bars the absolute price
    # move is small enough that spread eats a much larger share of the edge
    # — the same statistical accuracy is measurably less profitable net of
    # real trading cost. Every trade overnight on 2026-08-24 came from a
    # 1min model and the session lost money; 5min+ timeframes give the same
    # underlying edge more room to clear transaction costs.
    # 30min/1h further excluded: with the flat 60-bar horizon above, their
    # real target move is ~30h/~60h out — genuinely multi-day swing setups,
    # not intraday, despite the "30min"/"1h" label. Only 5min (5h horizon)
    # and 15min (15h horizon) stay within a trading day.
    "live_intraday_timeframes": ["5min", "15min"],
    "risk_per_trade": 0.015,
    "max_daily_loss": 0.035,
    "max_positions": 8,
    "max_position_size": 0.08,
    "default_lot_size": 0.01,
    "broker": "",
    "account_currency": "USD",
    "scalping_enabled": True,
    "immediate_execution": True,
    "aggressive_mode": False,
    "news_blackout_minutes": 25,
    "max_symbols": 50,
}

AI_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "min_confidence": 0.60,
    "update_interval": 5,
    "min_model_accuracy": 0.60,
    "promotion_absolute_floor": 0.63,
    "promotion_baseline_margin": 0.02,
    "use_ultra_ensemble": True,
    "minimum_active_submodels": 3,
    "ensemble_agreement_threshold": 0.70,
    # This machine has 8 cores / 8GB RAM — all-core parallelism (-1) caused
    # a hard crash after 21 combos of a long training run (memory allocation
    # failures cascading through multiple sub-models, then an uncaught
    # MemoryError killed the whole process). Each parallel tree-building
    # worker holds its own memory; more workers = more simultaneous peak
    # usage. 4 trades some wall-clock speed for enough headroom that the
    # live bot server (also running concurrently) doesn't get starved.
    "ensemble_n_jobs": 4,
    "ensemble_fast_mode": True,
    "max_train_rows_ltf": 25000,
    "max_train_rows_mtf": 35000,
    # Lowered from 45000 — this was the single biggest per-combo memory
    # driver for 30min/1h training (larger row count -> larger feature
    # matrix + larger trees), and 30min/1h combos were the ones that
    # actually crashed the machine (feature matrix at 45k rows x 239 feats
    # alone is ~43MB in float64 before any model even starts training, on
    # top of 7 sub-models each holding their own copies/trees). 22000 keeps
    # training on real, ample data while meaningfully cutting peak memory
    # so this can coexist with the live bot server on an 8GB machine.
    "max_train_rows_htf": 22000,
}

SIGNAL_FUSION_CONFIG: Dict[str, Any] = {
    "weights": {
        "ai": 0.38,
        "technical": 0.14,
        "price_action": 0.30,
        "market_structure": 0.08,
        "sentiment": 0.10,
    },
    "min_confidence": {
        "ai": 0.55,
        "technical": 0.40,
        "price_action": 0.45,
        "sentiment": 0.20,
        "combined": 0.65,
    },
    "min_sources": 3,
    "time_window_seconds": 300,
    "decay_factor": 0.95,
    "require_confirmation": True,
    "require_price_action_confirmation": True,
    "require_ai_above_floor": True,
    "require_macro_bias_alignment": True,
    "sniper_required_conditions": 1,
    "final_combined_confidence_floor": 0.70,
    # When sniper finds no structural signal at all, a trade can still fire on
    # AI + technical + macro alone if their combined confidence clears the
    # normal floor by this much extra — compensates for skipping structural
    # confirmation instead of requiring it unconditionally.
    "favourable_no_sniper_confidence_bonus": 0.12,
}

RISK_CONFIG: Dict[str, Any] = {
    "stop_loss_pips": 50,
    "take_profit_pips": 125,
    "variable_tp": True,
    "tp_multiplier": 2.5,
    "min_rr_ratio": 2.0,
    "target_rr_ratio": 2.5,
    "aggressive_rr_ratio": 3.2,
    "max_drawdown_pct": 45.0,
    "max_daily_trades": 40,
    "max_consecutive_losses": 3,
    "max_position_pct": 2.0,
    "max_daily_loss_pct": 3.5,
    "scalping_sl_pips": 40,
    "scalping_tp_pips": 100,
    "crypto_sl_dollars": 150,
    "crypto_tp_dollars": 450,
    "metal_sl_pips": 25,
    "metal_tp_pips": 75,
    "allow_early_close": True,
    "early_close_min_confidence": 0.78,
    "early_close_confirm_cycles": 3,
    "floating_loss_size_reduction_pct": 35.0,
    "floating_loss_threshold_pct": 1.5,
    "use_atr_sl_tp": True,
    "atr_sl_multiplier": 1.2,
    "atr_tp_multiplier": 3.0,
    "break_even_enabled": True,
    "break_even_trigger_r": 0.8,
    "break_even_pad_r": 0.1,
    "trailing_stop_enabled": True,
    "trailing_trigger_r": 1.5,
    "trailing_distance_atr": 1.0,
    "partial_tp_enabled": True,
    "partial_tp_1_rr": 1.0,
    "partial_tp_1_close_pct": 0.35,
    "partial_tp_2_rr": 2.0,
    "partial_tp_2_close_pct": 0.35,
    "adx_entry_threshold": 20.0,
    "adx_block_entry_below": 18.0,
}

BROKER_CONFIG: Dict[str, Any] = {
    "name": "",
    "use_mt5": True,
    "leverage": 100,
    "verify_timeout_seconds": 15,
}

DATA_CONFIG: Dict[str, Any] = {
    "history_days": 365,
    "history_years": 15,
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
