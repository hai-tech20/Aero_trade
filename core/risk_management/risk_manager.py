"""Risk Management System — AERO TRADE MASTERY EDITION"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from utils.logger import logger
from config.settings import RISK_CONFIG, TRADING_CONFIG
from core.symbols import get_tick_value_from_broker


class RiskManager:
    """Advanced risk management with dynamic sizing, news gating, break-even, trailing, partial TP."""

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
        # Per-position exit management state
        self._exit_state: Dict[str, Dict[str, Any]] = {}
        logger.info("Risk Manager initialized")

    # ========================================================================
    # ADVANCED EXIT MANAGEMENT — Break-Even, Trailing Stop, Partial Take-Profit
    # ========================================================================

    def _atr_from_df(self, df: Optional[pd.DataFrame], period: int = 14) -> float:
        """ATR helper; returns 0 if unavailable."""
        if df is None or len(df) < period + 2:
            return 0.0
        try:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values
            tr = []
            for i in range(1, len(df)):
                tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
            return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))
        except Exception:
            return 0.0

    def _pips_to_price(self, symbol: str, pips: float, current_price: float, symbol_info: Optional[Dict] = None) -> float:
        """Convert pips/dollars to absolute price movement."""
        try:
            if symbol_info:
                tick = float(symbol_info.get("point") or symbol_info.get("tick_size") or 0.00001)
                tick_value = float(symbol_info.get("tick_value") or 10.0)
                if tick > 0 and tick_value > 0:
                    per_pip = (tick * 10) * (10.0 / tick_value)
                    return pips * per_pip * max(current_price, 1.0) * 0.0001
            # Fallback: approximate
            if symbol.upper().startswith("XAU") or symbol.upper() == "GOLD":
                return pips * 0.1
            if "JPY" in symbol.upper():
                return pips * 0.01
            if "BTC" in symbol.upper() or symbol.upper() == "BTCUSD":
                return pips
            return pips * 0.0001
        except Exception:
            return 0.0

    def _ensure_position_state(self, position_id: str, entry_price: float, direction: str,
                               sl_price: float, tp_price: float,
                               symbol: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if position_id in self._exit_state:
            return self._exit_state[position_id]
        atr = self._atr_from_df(df, 14)
        cls = symbol.upper()
        if cls.startswith("XAU") or cls == "GOLD":
            tick_r = 0.1
        elif "JPY" in cls:
            tick_r = 0.01
        elif "BTC" in cls:
            tick_r = 5.0
        else:
            tick_r = 0.0001
        sl_dist = abs(entry_price - sl_price) or atr * RISK_CONFIG.get("atr_sl_multiplier", 1.2) or (tick_r * 50)
        tp_dist = abs(entry_price - tp_price) or atr * RISK_CONFIG.get("atr_tp_multiplier", 3.0) or (tick_r * 125)
        state = {
            "entry_price": float(entry_price),
            "direction": direction,  # "BUY" or "SELL"
            "initial_sl": float(sl_price),
            "initial_tp": float(tp_price),
            "sl_distance_r": float(sl_dist),  # distance in price units = 1R
            "tp_distance_r": float(tp_dist),
            "symbol": symbol,
            "is_buy": direction.upper() == "BUY",
            "break_even_applied": False,
            "trailing_applied": False,
            "partial_tp_1_closed": False,
            "partial_tp_2_closed": False,
            "current_sl": float(sl_price),
            "current_tp": float(tp_price),
            "atr": float(atr),
            "opened_at": datetime.utcnow().isoformat(),
            "highest_observed": float(entry_price),
            "lowest_observed": float(entry_price),
        }
        self._exit_state[position_id] = state
        return state

    def evaluate_position_exit(self,
                                position_id: str,
                                current_price: float,
                                entry_price: float,
                                direction: str,
                                sl_price: float,
                                tp_price: float,
                                symbol: str,
                                df: Optional[pd.DataFrame] = None,
                                symbol_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Core exit-manager for an open position.

        Evaluates the state and returns:
          {
            "action": "HOLD" | "MOVE_SL" | "PARTIAL_CLOSE_1" | "PARTIAL_CLOSE_2",
            "new_sl": float,
            "new_tp": float,
            "close_fraction": float (0..1),
            "reason": str,
            "break_even": bool,
            "trailing": bool,
            "rr_state": str,
          }
        """
        s = self._ensure_position_state(position_id, entry_price, direction, sl_price, tp_price, symbol, df)
        entry = s["entry_price"]
        one_r = max(s["sl_distance_r"], 1e-9)
        is_buy = s["is_buy"]

        # P&L in R-multiples (positive = in profit)
        if is_buy:
            move_r = (current_price - entry) / one_r
        else:
            move_r = (entry - current_price) / one_r

        # Update extrema
        s["highest_observed"] = max(s["highest_observed"], current_price)
        s["lowest_observed"] = min(s["lowest_observed"], current_price)

        atr = self._atr_from_df(df, 14) or s["atr"] or one_r
        s["atr"] = atr

        result: Dict[str, Any] = {
            "action": "HOLD",
            "new_sl": s["current_sl"],
            "new_tp": s["current_tp"],
            "close_fraction": 0.0,
            "reason": "",
            "break_even": s["break_even_applied"],
            "trailing": s["trailing_applied"],
            "rr_state": f"{move_r:.2f}R",
        }

        cfg = RISK_CONFIG
        be_trigger = float(cfg.get("break_even_trigger_r", 0.8))
        be_pad = float(cfg.get("break_even_pad_r", 0.1))
        trail_trigger = float(cfg.get("trailing_trigger_r", 1.5))
        trail_dist = float(cfg.get("trailing_distance_atr", 1.0))
        ptp1_rr = float(cfg.get("partial_tp_1_rr", 1.0))
        ptp1_pct = float(cfg.get("partial_tp_1_close_pct", 0.35))
        ptp2_rr = float(cfg.get("partial_tp_2_rr", 2.0))
        ptp2_pct = float(cfg.get("partial_tp_2_close_pct", 0.35))

        # -----------------------------------------------------------
        # 1) Partial Take-Profit #1 (at +1R): close 35%, trail the rest
        # -----------------------------------------------------------
        if cfg.get("partial_tp_enabled", True) and not s["partial_tp_1_closed"] and move_r >= ptp1_rr:
            s["partial_tp_1_closed"] = True
            result["action"] = "PARTIAL_CLOSE_1"
            result["close_fraction"] = ptp1_pct
            result["reason"] = f"Partial TP #1 hit at {move_r:.2f}R — close {int(ptp1_pct*100)}%"
            # Move SL to entry after first TP (free trade on remaining)
            if not s["break_even_applied"]:
                s["break_even_applied"] = True
                if is_buy:
                    s["current_sl"] = max(s["current_sl"], entry + be_pad * one_r)
                else:
                    s["current_sl"] = min(s["current_sl"], entry - be_pad * one_r)
                result["new_sl"] = s["current_sl"]
                result["break_even"] = True
            return result

        # -----------------------------------------------------------
        # 2) Partial Take-Profit #2 (at +2R): close another 35%
        # -----------------------------------------------------------
        if cfg.get("partial_tp_enabled", True) and not s["partial_tp_2_closed"] and move_r >= ptp2_rr:
            s["partial_tp_2_closed"] = True
            result["action"] = "PARTIAL_CLOSE_2"
            result["close_fraction"] = ptp2_pct
            result["reason"] = f"Partial TP #2 hit at {move_r:.2f}R — close {int(ptp2_pct*100)}%"
            # After +2R, tighten trailing aggressively
            s["trailing_applied"] = True
            return result

        # -----------------------------------------------------------
        # 3) Break-Even (at +0.8R): move SL to entry + small pad
        # -----------------------------------------------------------
        if cfg.get("break_even_enabled", True) and not s["break_even_applied"] and move_r >= be_trigger:
            s["break_even_applied"] = True
            if is_buy:
                new_sl = entry + be_pad * one_r
                s["current_sl"] = max(s["current_sl"], new_sl)
            else:
                new_sl = entry - be_pad * one_r
                s["current_sl"] = min(s["current_sl"], new_sl)
            result["action"] = "MOVE_SL"
            result["new_sl"] = s["current_sl"]
            result["reason"] = f"Break-Even moved SL (trade at {move_r:.2f}R)"
            result["break_even"] = True
            return result

        # -----------------------------------------------------------
        # 4) Trailing Stop (at +1.5R): trail by 1 ATR
        # -----------------------------------------------------------
        if cfg.get("trailing_stop_enabled", True) and move_r >= trail_trigger:
            s["trailing_applied"] = True
            trail_price_distance = max(one_r * 0.8, atr * trail_dist)
            if is_buy:
                # Trail from highest observed
                candidate_sl = s["highest_observed"] - trail_price_distance
                if candidate_sl > s["current_sl"]:
                    s["current_sl"] = candidate_sl
                    result["action"] = "MOVE_SL"
                    result["new_sl"] = s["current_sl"]
                    result["reason"] = f"Trailing SL updated @ {move_r:.2f}R (hi={s['highest_observed']:.5f})"
            else:
                candidate_sl = s["lowest_observed"] + trail_price_distance
                if candidate_sl < s["current_sl"]:
                    s["current_sl"] = candidate_sl
                    result["action"] = "MOVE_SL"
                    result["new_sl"] = s["current_sl"]
                    result["reason"] = f"Trailing SL updated @ {move_r:.2f}R (lo={s['lowest_observed']:.5f})"
            result["trailing"] = True
            if result["action"] == "HOLD":
                result["reason"] = f"Trailing active, no SL update needed ({move_r:.2f}R)"
            return result

        result["reason"] = f"In position — {move_r:.2f}R (waiting for triggers)"
        return result

    def clear_position_exit_state(self, position_id: str) -> None:
        self._exit_state.pop(position_id, None)

    def compute_atr_sl_tp_prices(self,
                                  symbol: str,
                                  direction: str,
                                  entry_price: float,
                                  df: Optional[pd.DataFrame] = None,
                                  symbol_info: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Return (sl_price, tp_price) using ATR multipliers from RISK_CONFIG.
        GUARANTEES tp_distance >= sl_distance (target 2.5R minimum).
        """
        atr = self._atr_from_df(df, 14)
        sl_mult = float(RISK_CONFIG.get("atr_sl_multiplier", 1.2))
        tp_mult = float(RISK_CONFIG.get("atr_tp_multiplier", 3.0))
        target_rr = float(RISK_CONFIG.get("target_rr_ratio", 2.5))
        is_buy = direction.upper() == "BUY"

        cls = symbol.upper()
        if atr <= 0:
            # Default per-asset-class
            if cls.startswith("XAU") or cls == "GOLD":
                atr = 3.0
            elif "BTC" in cls:
                atr = 80.0
            elif "JPY" in cls:
                atr = 0.20
            else:
                atr = 0.0010

        sl_dist = atr * sl_mult
        tp_dist = max(atr * tp_mult, sl_dist * target_rr)

        # Safety floor: SL must never be tighter than a sane multiple of the
        # actual current spread, regardless of what ATR computed to. A stop
        # placed inside (or barely outside) the spread can be hit on the very
        # next tick with zero adverse price movement — this is what happened
        # to the first live trade, where a 1min-ATR-derived SL came out to
        # ~1.4 pips on a 5min-timeframe signal and got clipped in 39 seconds.
        info = symbol_info or {}
        spread_points = float(info.get("spread") or 0)
        point_size = float(info.get("point") or 0)
        if spread_points > 0 and point_size > 0:
            min_sl_dist = spread_points * point_size * 5.0
            if sl_dist < min_sl_dist:
                sl_dist = min_sl_dist
                tp_dist = max(tp_dist, sl_dist * target_rr)

        # Ensure tp > sl in absolute terms (2.5R minimum regardless of multiplier)
        if tp_dist < sl_dist * target_rr:
            tp_dist = sl_dist * target_rr

        if is_buy:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
        return float(sl_price), float(tp_price)

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
        floating_pnl: float = 0.0,
        floating_pnl_pct: float = 0.0,
    ) -> float:
        # Smoothly decreasing risk % as balance grows — standard practice:
        # tiny accounts need more headroom just to clear broker minimum lot
        # sizes, while large accounts should risk a smaller % since the same
        # % represents a much bigger absolute amount. This still scales
        # UP in absolute dollar terms as balance grows (risk_amount =
        # balance x pct), it just doesn't grow linearly forever — a
        # deliberate, standard risk-management shape, not a cap on growth.
        # The reward:risk ratio (2.5R minimum) is a separate, unaffected
        # protection enforced elsewhere.
        if account_balance < 50:
            base_risk_pct = 0.07
        elif account_balance < 1000:
            base_risk_pct = 0.045
        elif account_balance < 10_000:
            base_risk_pct = 0.03
        else:
            base_risk_pct = 0.015

        confidence_multiplier = 0.75 + (fused_confidence * 0.5)
        if not sentiment_agrees:
            confidence_multiplier *= 0.5

        # Reduce size when sitting on meaningful unrealized losses
        loss_threshold = RISK_CONFIG.get("floating_loss_threshold_pct", 2.0)
        reduction_pct = RISK_CONFIG.get("floating_loss_size_reduction_pct", 25.0) / 100.0
        if floating_pnl_pct < -loss_threshold:
            confidence_multiplier *= (1.0 - reduction_pct)
            logger.info(
                f"reducing position size by {reduction_pct*100:.0f}% — "
                f"currently sitting on unrealized losses ({floating_pnl_pct:.1f}% floating, ${floating_pnl:.2f})"
            )

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

    def get_max_positions_for_balance(self, account_balance: float) -> int:
        """
        How many concurrent positions this account can reasonably hold —
        was previously a single fixed config value (8) regardless of
        balance, meaning a $10 account and a $10,000 account got the same
        ceiling. A tiny account holding 8 simultaneous positions spreads
        already-thin margin dangerously; a large account is unnecessarily
        capped. Falls back to the static config value if balance is unknown.
        """
        if account_balance <= 0:
            return int(TRADING_CONFIG.get("max_positions", 8))
        if account_balance < 50:
            return 2
        if account_balance < 1000:
            return 4
        if account_balance < 10_000:
            return 8
        return 12

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
        aggressive = TRADING_CONFIG.get("aggressive_mode", False)
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

        reasons = []

        # News blackout gating
        if market_conditions and market_conditions.get("news_blackout"):
            reasons.append(market_conditions.get("blackout_reason", "High-impact news event"))

        if self.daily_stats["trades"] >= RISK_CONFIG["max_daily_trades"]:
            reasons.append("Daily trade limit reached")

        if self.daily_stats["pnl"] <= -RISK_CONFIG["max_daily_loss_pct"] / 100 * account_balance:
            reasons.append("Daily loss limit reached")

        recent = self.trade_history[-RISK_CONFIG["max_consecutive_losses"]:]
        if len(recent) >= RISK_CONFIG["max_consecutive_losses"]:
            if all(t.get("pnl", 0) < 0 for t in recent):
                reasons.append("Consecutive losses limit")

        if self.daily_stats["max_drawdown"] >= RISK_CONFIG["max_drawdown_pct"]:
            reasons.append("Max drawdown reached")

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
            return True, f"Max drawdown reached: {drawdown_pct:.1f}%"
        if self.daily_stats["pnl"] <= -RISK_CONFIG["max_daily_loss_pct"] / 100 * account_balance:
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
