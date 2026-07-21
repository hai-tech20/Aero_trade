"""Risk Management System — AERO TRADE"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
from datetime import datetime

from utils.logger import logger
from config.settings import RISK_CONFIG, TRADING_CONFIG
from core.symbols import get_tick_value_from_broker


class RiskManager:
    """Advanced risk management with dynamic sizing and news gating."""

    def __init__(self):
        self.trade_history = []
        self.daily_stats = {
            "date": datetime.now().date(),
            "trades": 0, "wins": 0, "losses": 0,
            "pnl": 0.0, "max_drawdown": 0.0,
        }
        self.trading_halted = False
        self.halt_reason = ""
        self.volatility_profiles: Dict[str, Dict] = {}
        logger.info("Risk Manager initialized")

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        if df is None or len(df) < period + 1:
            return 0.0
        try:
            high = (df["high"] if "high" in df.columns else df["High"]).to_numpy(dtype=float)
            low = (df["low"] if "low" in df.columns else df["Low"]).to_numpy(dtype=float)
            close = (df["close"] if "close" in df.columns else df["Close"]).to_numpy(dtype=float)
            plus_dm = np.maximum(high[1:] - high[:-1], 0)
            minus_dm = np.maximum(low[:-1] - low[1:], 0)
            tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
            tr_series = pd.Series(tr)
            plus_dm_series = pd.Series(plus_dm)
            minus_dm_series = pd.Series(minus_dm)
            atr_values = tr_series.rolling(window=period).mean().tolist()
            atr = float(atr_values[-1]) if len(tr) >= period else float(np.mean(tr))
            plus_di_series = 100 * plus_dm_series.rolling(window=period).mean() / (atr + 1e-9)
            minus_di_series = 100 * minus_dm_series.rolling(window=period).mean() / (atr + 1e-9)
            dx_series = pd.Series(100 * np.abs(plus_di_series - minus_di_series) / (plus_di_series + minus_di_series + 1e-9))
            adx_values = dx_series.rolling(window=period).mean().tolist()
            adx = float(adx_values[-1]) if len(dx_series) >= period else float(np.mean(dx_series.to_numpy(dtype=float)))
            return adx
        except Exception:
            return 0.0

    def check_trend_filter(self, symbol: str, df: pd.DataFrame, adx_threshold: float = 20.0) -> Tuple[bool, str]:
        adx = self._compute_adx(df)
        if adx < adx_threshold:
            return False, f"ADX {adx:.1f} below threshold {adx_threshold} — trend-following entry suppressed (choppy market)"
        return True, f"ADX {adx:.1f} — trend-following entry allowed"

    def update_volatility_profile(self, symbol: str, df: pd.DataFrame):
        try:
            from data.volatility_profiler import compute_volatility_profile, symbol_asset_class
            cls = symbol_asset_class(symbol)
            self.volatility_profiles[symbol] = compute_volatility_profile(df, is_crypto=(cls == "crypto"))
        except Exception as e:
            logger.error(f"Failed to update volatility profile for {symbol}: {e}")

    def get_volatility_profile(self, symbol: str) -> Dict:
        return self.volatility_profiles.get(symbol, {})

    def calculate_dynamic_position_size(
        self,
        account_balance: float,
        stop_loss_pips: float,
        fused_confidence: float,
        sentiment_agrees: bool,
        symbol: str,
        symbol_info: Dict[str, Any] | None = None,
    ) -> float:
        if account_balance < 100:
            base_risk_pct = 0.02
        elif account_balance < 1000:
            base_risk_pct = 0.03
        elif account_balance < 10_000:
            base_risk_pct = 0.02
        else:
            base_risk_pct = 0.01

        confidence_multiplier = 0.75 + (fused_confidence * 0.5)
        if not sentiment_agrees:
            confidence_multiplier *= 0.5

        risk_amount = account_balance * base_risk_pct * confidence_multiplier
        pip_value = get_tick_value_from_broker(symbol_info or {}, default=10.0)

        position_size = risk_amount / max(stop_loss_pips, 1) / pip_value

        # Round/clamp to the broker's REAL per-symbol volume rules, not a hardcoded 0.01 —
        # different symbols require different minimums and step sizes (e.g. 0.1 instead of
        # 0.01), and sending a size that isn't an exact multiple of the real step gets
        # rejected as "Invalid volume" even if it looked fine by a generic assumption.
        info = symbol_info or {}
        vol_min = float(info.get("volume_min") or 0.01)
        vol_max = float(info.get("volume_max") or 100.0)
        vol_step = float(info.get("volume_step") or 0.01)

        if vol_step > 0:
            position_size = round(position_size / vol_step) * vol_step
        position_size = max(vol_min, min(position_size, vol_max))
        decimals = len(str(vol_step).split(".")[-1]) if "." in str(vol_step) else 2
        return round(position_size, decimals)

    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float,
        stop_loss_pips: float,
        symbol: str,
        confidence: float = 1.0,
        use_compounding: bool = False,
        win_streak: int = 0,
        symbol_info: Dict[str, Any] | None = None,
        is_demo: bool = True,
    ) -> float:
        aggressive = TRADING_CONFIG.get("aggressive_mode", False) and is_demo
        if not aggressive:
            use_compounding = False

        try:
            base_risk = account_balance * risk_per_trade
            if use_compounding and win_streak > 0:
                compound_multiplier = min(1.0 + (win_streak * 0.5), 3.0)
                risk_amount = base_risk * compound_multiplier
                if confidence > 0.8:
                    risk_amount *= 1.5
            else:
                risk_amount = base_risk

            risk_amount *= min(confidence, 1.0)
            pip_value = get_tick_value_from_broker(symbol_info or {}, default=10.0)

            if stop_loss_pips > 0:
                position_size = risk_amount / (stop_loss_pips * pip_value)
            else:
                position_size = risk_amount / (20 * pip_value)

            max_pct = RISK_CONFIG.get("max_position_pct", 5.0)
            max_lots = (account_balance * max_pct / 100) / 1000
            position_size = min(position_size, max_lots)
            position_size = max(round(position_size / 0.01) * 0.01, 0.01)
            return position_size
        except Exception as e:
            logger.error(f"Position size error: {e}")
            return 0.01

    def validate_trade(
        self,
        symbol: str,
        position_size: float,
        account_balance: float,
        open_positions: List[Dict[str, Any]],
        market_conditions: Dict[str, Any] | None = None,
        is_demo: bool = True,
        live_trading_enabled: bool = False,
    ) -> Tuple[bool, str]:
        if self.trading_halted:
            return False, self.halt_reason or "Trading halted"

        if not is_demo and not live_trading_enabled:
            return False, "Live trading not enabled for this account"

        reasons = []

        # News blackout gating
        if market_conditions and market_conditions.get("news_blackout"):
            reasons.append(market_conditions.get("blackout_reason", "High-impact news event"))

        if self.daily_stats["trades"] >= RISK_CONFIG["max_daily_trades"]:
            reasons.append("Daily trade limit reached")

        if self.daily_stats["pnl"] <= -RISK_CONFIG["max_daily_loss_pct"] / 100 * account_balance:
            reasons.append("Daily loss limit reached")
            if not is_demo:
                self.trading_halted = True
                self.halt_reason = "Daily loss limit breached on live account"

        recent = self.trade_history[-RISK_CONFIG["max_consecutive_losses"]:]
        if len(recent) >= RISK_CONFIG["max_consecutive_losses"]:
            if all(t.get("pnl", 0) < 0 for t in recent):
                reasons.append("Consecutive losses limit")

        if self.daily_stats["max_drawdown"] >= RISK_CONFIG["max_drawdown_pct"]:
            reasons.append("Max drawdown reached")
            if not is_demo:
                self.trading_halted = True
                self.halt_reason = "Max drawdown breached on live account"

        if reasons:
            return False, ", ".join(reasons)
        return True, "Trade allowed"

    def record_trade(self, trade: Dict):
        self.trade_history.append(trade)
        today = datetime.now().date()
        if self.daily_stats["date"] != today:
            self.daily_stats = {
                "date": today, "trades": 0, "wins": 0, "losses": 0,
                "pnl": 0.0, "max_drawdown": 0.0,
            }
        self.daily_stats["trades"] += 1
        self.daily_stats["pnl"] += trade.get("pnl", 0)
        if trade.get("pnl", 0) > 0:
            self.daily_stats["wins"] += 1
        else:
            self.daily_stats["losses"] += 1

    def should_stop_trading(self, account_balance: float, initial_balance: float, is_demo: bool = True) -> Tuple[bool, str]:
        if initial_balance <= 0:
            return False, ""
        drawdown_pct = (initial_balance - account_balance) / initial_balance * 100
        if drawdown_pct >= RISK_CONFIG["max_drawdown_pct"]:
            if not is_demo:
                self.trading_halted = True
                self.halt_reason = f"Max drawdown: {drawdown_pct:.1f}%"
            return True, f"Max drawdown reached: {drawdown_pct:.1f}%"
        if self.daily_stats["pnl"] <= -RISK_CONFIG["max_daily_loss_pct"] / 100 * account_balance:
            if not is_demo:
                self.trading_halted = True
                self.halt_reason = "Daily loss limit reached"
            return True, "Daily loss limit reached"
        return False, ""

    def get_risk_metrics(self) -> Dict:
        if not self.trade_history:
            return {}
        pnls = [t.get("pnl", 0) for t in self.trade_history]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "total_trades": len(self.trade_history),
            "win_rate": wins / len(pnls) * 100 if pnls else 0,
            "total_pnl": sum(pnls),
            "daily_stats": self.daily_stats,
            "trading_halted": self.trading_halted,
        }

    def reset_halt(self):
        self.trading_halted = False
        self.halt_reason = ""


risk_manager = RiskManager()
