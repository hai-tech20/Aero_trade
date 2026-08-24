"""
AERO TRADE — Main Trading Engine
"""

import asyncio
import json

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).parent))

from ai_engine.models.predictor import HybridAIPredictor
from ai_engine.models.model_keys import make_model_key
from core.analysis.macro_bias import get_macro_bias
from ai_engine.models.market_context import compute_market_context_snapshot, reference_basket_symbols
from ai_engine.training.model_status import get_best_timeframe_for_symbol
from core.sessions import is_favorable_session
from decision_maker.signal_fusion import SignalFusion, Signal, SignalSource, FusedSignal
from core.broker.mt5_broker_client import MT5BrokerClient
from core.broker.credential_store import get_account_meta
from core.strategies.sniper_logic import sniper_logic
from core.risk_management.risk_manager import risk_manager
from core.ledger.trade_ledger import (
    init_ledger, record_open_trade, resolve_orphan_trade,
    build_self_analysis,
)
from core.ledger.close_attribution import classify_close, TRAINABLE_CLOSED_BY
from core.symbols import symbol_asset_class, price_distance_to_pips, get_tick_value_from_broker
from data.market_data import MarketDataFeed
from data.market_analysis import market_analysis
from data.binance_data import check_btc_divergence
from data.vix_source import is_high_volatility_regime
from data.fred_source import is_rates_rising
from data.cot_source import get_latest_cot_positioning
from utils.logger import TradingLogger
from config.settings import (
    APP_NAME, TRADING_CONFIG, RISK_CONFIG, AI_CONFIG, SIGNAL_FUSION_CONFIG,
)


class AeroTradeBot:
    """AERO TRADE trading bot orchestrator."""

    def __init__(self, account_id: str = "default", config: Dict[str, Any] | None = None):
        self.account_id = account_id
        self.config = config or {}
        self.logger = TradingLogger("AeroTrade")
        self.running = False
        self.ai_predictor = HybridAIPredictor()
        self.signal_fusion = SignalFusion(SIGNAL_FUSION_CONFIG)
        self.market_data = MarketDataFeed()
        self.broker = MT5BrokerClient(account_id)
        self.active_positions: List[Dict] = []
        self.account_info = {}
        self.available_symbols: List[str] = []
        self.mt5_bridge = None
        self.win_streak = 0
        self.initial_balance = 0.0
        self.is_demo = True
        self.live_trading_enabled = False
        self._log_callbacks: List = []
        self.activity_log: List[Dict] = []
        self._last_slot_log_time: float = 0.0
        self._last_slot_count: int = -1
        self._last_self_analysis_time: float = 0.0
        self._open_orders_cache: List[Dict] = []
        self._floating_pnl: float = 0.0
        self._floating_pnl_pct: float = 0.0
        self._open_win_count: int = 0
        self._open_loss_count: int = 0
        self._opposing_signal_counts: Dict[str, int] = {}
        self._market_context_snapshot: Dict[str, float] = {}

    def on_log(self, callback):
        self._log_callbacks.append(callback)

    def _emit_log(self, level: str, msg: str):
        for cb in self._log_callbacks:
            try:
                cb(level, msg)
            except Exception:
                pass

    def _log_activity(self, symbol: str, message: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "message": message,
        }
        self.activity_log.append(entry)
        if len(self.activity_log) > 5000:
            self.activity_log = self.activity_log[-5000:]

    def get_activity(self, limit: int = 50) -> List[Dict]:
        return self.activity_log[-limit:]

    async def initialize(self, creds: Dict[str, Any] | None = None):
        self.logger.info(f"Initializing {APP_NAME}...")
        init_ledger()
        await self._initialize_ai()
        await self.ai_predictor.start_retraining_scheduler()

        if creds:
            connected = await self.broker.connect_with_credentials(creds)
        else:
            connected = await self.broker.connect(demo=True)

        if not connected:
            raise RuntimeError(
                f"Could not establish a real MT5 connection for account {self.account_id} — "
                "refusing to start the bot with no live broker connection."
            )

        try:
            meta = get_account_meta(self.account_id)
            self.is_demo = meta.get("is_demo", True)
            self.live_trading_enabled = meta.get("live_trading_enabled", False)
        except KeyError:
            self.is_demo = True

        self.account_info = await self.broker.get_account_info()
        self.initial_balance = self.account_info.get("balance", 0)

        configured_pairs = [str(pair) for pair in TRADING_CONFIG.get("pairs", ["EURUSD"])]
        if self.broker.use_mt5 and self.broker.mt5_bridge:
            self.mt5_bridge = self.broker.mt5_bridge
            await asyncio.sleep(1)
            # get_all_symbols() returns None on a transient query failure —
            # this only runs once at startup, so a single bad timeout here
            # (e.g. under memory pressure) used to permanently strand the
            # whole bot session with zero tradeable symbols. Retry a few
            # times before falling back to the configured pair list as-is.
            all_symbols = None
            for attempt in range(3):
                all_symbols = self.mt5_bridge.get_all_symbols()
                if all_symbols is not None:
                    break
                self.logger.warning(f"get_all_symbols failed (attempt {attempt + 1}/3), retrying...")
                await asyncio.sleep(2)
            if all_symbols is None:
                self.logger.warning("get_all_symbols failed after retries — falling back to configured pairs unfiltered")
                self.available_symbols = configured_pairs
            else:
                available = [symbol for symbol in configured_pairs if symbol in all_symbols]
                missing = [symbol for symbol in configured_pairs if symbol not in all_symbols]
                self.available_symbols = available
                if missing:
                    self.logger.warning(f"Configured symbols not available from broker: {', '.join(missing)}")
        else:
            self.available_symbols = configured_pairs

        await self.market_data.connect()

        # Reconcile active_positions from the broker's live state. This list
        # is in-memory only — every restart starts it empty. A position
        # opened by a previous instance and closed after a restart was
        # invisible to monitor_positions() entirely (not even reaching the
        # "orphan" fallback, since its loop only iterates self.active_positions),
        # so the close was never logged, never recorded in the trade ledger,
        # and never fed into online learning. Given how often this process
        # gets restarted (memory pressure, fixes, retraining), this silently
        # dropped a meaningful share of real trade outcomes. confidence/
        # timeframe/features aren't recoverable from the broker, so these
        # reconciled entries use safe defaults — good enough for correct
        # close detection, P&L, and SL/TP-based attribution going forward.
        try:
            broker_positions = await self.broker.get_open_orders()
            if broker_positions:
                known_ids = {str(p.get("order_id")) for p in self.active_positions}
                restored = 0
                for bp in broker_positions:
                    oid = str(bp.get("order_id"))
                    if oid in known_ids:
                        continue
                    self.active_positions.append({
                        "order_id": oid,
                        "symbol": bp.get("symbol", ""),
                        "direction": bp.get("type", ""),
                        "entry_price": float(bp.get("entry_price", 0) or 0),
                        "size": float(bp.get("volume", 0) or 0),
                        "confidence": 0.0,
                        "features_json": "{}",
                        "timeframe": "5min",
                        "stop_loss": float(bp.get("stop_loss", 0) or 0),
                        "take_profit": float(bp.get("take_profit", 0) or 0),
                    })
                    restored += 1
                if restored:
                    self.logger.info(f"Reconciled {restored} pre-existing broker position(s) into active_positions")
        except Exception as e:
            self.logger.warning(f"Position reconciliation at startup failed: {e}")

        self.logger.info(f"{APP_NAME} initialized — {len(self.available_symbols)} symbols")

    async def _initialize_ai(self):
        model_path = Path("ai_engine/models/saved/")
        if model_path.exists() and any(model_path.glob("*.joblib")):
            self.logger.info("Pre-trained models loaded")
        else:
            self.logger.warning("No pre-trained models — training in background")
            asyncio.create_task(self._train_initial_models())

    async def _train_initial_models(self):
        self.logger.info("Background model training started")

    async def run(self):
        self.running = True
        self.logger.info("Trading loop started")
        sleep_interval = 1 if TRADING_CONFIG.get("scalping_enabled") else 5
        try:
            while self.running:
                await self.trading_iteration()
                await asyncio.sleep(sleep_interval)
        except Exception as e:
            self.logger.error(f"Main loop error: {e}")
        finally:
            await self.shutdown()

    def get_floating_stats(self) -> Dict[str, Any]:
        return {
            "floating_pnl": self._floating_pnl,
            "floating_pnl_pct": self._floating_pnl_pct,
            "open_win_count": self._open_win_count,
            "open_loss_count": self._open_loss_count,
            "open_positions_count": len(self._open_orders_cache),
        }

    def _compute_floating_stats(self, open_orders: List[Dict]):
        balance = max(self.account_info.get("balance", 1), 1)
        self._floating_pnl = sum(float(o.get("profit", 0) or o.get("pnl_usd", 0)) for o in open_orders)
        self._floating_pnl_pct = (self._floating_pnl / balance) * 100
        self._open_win_count = sum(1 for o in open_orders if float(o.get("profit", 0) or 0) > 0)
        self._open_loss_count = sum(1 for o in open_orders if float(o.get("profit", 0) or 0) < 0)
        self._open_orders_cache = open_orders

    def _log_slot_summary(self, open_count: int):
        import time
        max_pos = risk_manager.get_max_positions_for_balance(self.account_info.get("balance", 0))
        now = time.time()
        available = max_pos - open_count
        if open_count != self._last_slot_count or (now - self._last_slot_log_time) >= 60:
            self._log_activity("-", f"Position slots: {open_count}/{max_pos} used — {available} available")
            self._last_slot_count = open_count
            self._last_slot_log_time = now

    async def _maybe_log_self_analysis(self):
        import time
        now = time.time()
        if now - self._last_self_analysis_time < 14400:  # every 4 hours
            return
        try:
            report = build_self_analysis(self.account_id, days=30)
            overall = report.get("overall", {})
            self._log_activity(
                "-",
                f"Self-analysis: {overall.get('total_trades', 0)} trades, "
                f"win rate {overall.get('win_rate', 0)*100:.1f}%, "
                f"P&L ${overall.get('pnl_usd', 0):.2f}, "
                f"flags: {len(report.get('flags', []))}",
            )
            for flag in report.get("flags", [])[:3]:
                self._log_activity("-", f"Analysis flag: {flag}")
            self._last_self_analysis_time = now
        except Exception as e:
            self.logger.warning(f"Self-analysis log failed: {e}")

    async def trading_iteration(self):
        try:
            # Refresh account_info FIRST, before the drawdown check that
            # depends on it. The previous order (check, then refresh later
            # in this same function) meant that if the refresh ever failed
            # or returned incomplete data even once, account_info stayed
            # stuck at whatever it last was — and since the drawdown check
            # returns early on "stop", the refresh that would fix it was
            # never reached again. That is a real self-perpetuating deadlock
            # this session hit live: a single bad read got read as balance=0,
            # computed as 100% drawdown, and aborted every single cycle
            # forever after, with no way to recover without a restart.
            try:
                fresh_info = await self.broker.get_account_info()
                if fresh_info and fresh_info.get("balance") is not None:
                    self.account_info = fresh_info
            except Exception as e:
                self.logger.warning(f"account_info refresh failed, keeping last known value: {e}")

            current_balance = self.account_info.get("balance", 0)
            # A missing/zero balance reading means "no valid data yet", not
            # "the account is empty" — treating it as 100% drawdown was the
            # exact bug above. Only evaluate the drawdown check against a
            # balance that looks real.
            if current_balance > 0:
                stop, reason = risk_manager.should_stop_trading(
                    current_balance, self.initial_balance, self.is_demo,
                )
                if stop:
                    self.logger.warning(f"Trading stopped: {reason}")
                    self._log_activity("-", f"cycle aborted — {reason}")
                    return

            # Always monitor first — never skip symbol analysis due to position cap
            await self.monitor_positions()

            open_orders = await self.broker.get_open_orders()
            if open_orders is None:
                self.logger.warning("trading_iteration: could not confirm broker position state — skipping stats update this cycle")
                open_orders = []
            self._compute_floating_stats(open_orders)
            broker_count = len(open_orders)
            self._log_slot_summary(broker_count)

            if self.broker.use_mt5 and self.broker.mt5_bridge and not self.broker.mt5_bridge.connected:
                self.logger.warning("MT5 connection lost — attempting reconnect")
                try:
                    from core.broker.credential_store import load_broker_credentials
                    creds = load_broker_credentials(self.account_id)
                    await self.broker.connect_with_credentials(creds)
                except Exception as e:
                    self.logger.error(f"MT5 reconnect failed: {e}")

            # Market context (USD strength / risk sentiment / gold momentum) is
            # market-wide, not per-symbol — compute it ONCE per cycle here and
            # reuse across every symbol below, instead of re-fetching the same
            # ~10 reference pairs 35 times per cycle. Must match what training
            # used (same ai_engine.models.market_context helpers) or this is
            # the exact train/serve mismatch class already fixed once this
            # session (feature_engine.py winsorization).
            if self.mt5_bridge:
                try:
                    recent_closes = {}
                    for ref_sym in reference_basket_symbols():
                        ref_df = self.mt5_bridge.get_historical_data(ref_sym, "1min", count=65)
                        if ref_df is not None and len(ref_df) >= 2 and "close" in ref_df.columns:
                            recent_closes[ref_sym] = ref_df["close"]
                    self._market_context_snapshot = compute_market_context_snapshot(recent_closes)
                except Exception as e:
                    self.logger.warning(f"market context snapshot failed: {e}")
                    self._market_context_snapshot = {}

            symbols_analyzed = 0
            for symbol in self.available_symbols[:TRADING_CONFIG.get("max_symbols", 50)]:
                try:
                    await self._process_symbol(symbol)
                    symbols_analyzed += 1
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    self._log_activity(symbol, f"error — {e}")

            await self.monitor_positions()
            await self._maybe_log_self_analysis()
            self._log_activity(
                "-",
                f"cycle complete — {symbols_analyzed} symbols analyzed — "
                f"floating P&L ${self._floating_pnl:.2f} ({self._floating_pnl_pct:.1f}%) — "
                f"next cycle in {1 if TRADING_CONFIG.get('scalping_enabled') else 5}s",
            )
        except Exception as e:
            self.logger.error(f"Iteration error: {e}")
            self._log_activity("-", f"cycle error — {e}")

    def _has_active_trading_model(self, symbol: str, timeframe: str = "1min") -> bool:
        best = get_best_timeframe_for_symbol(symbol)
        if best is not None:
            return True
        ultra_key = HybridAIPredictor._ultra_key(symbol, timeframe)
        return ultra_key in self.ai_predictor.ultra_ensembles

    async def _process_symbol(self, symbol: str):
        best = get_best_timeframe_for_symbol(symbol)
        if best is None:
            self._log_activity(symbol, "analyzed — no active promoted model for symbol (checked 1min/5min/15min) — trading disabled")
            return
        active_timeframe = best["timeframe"]

        if not is_favorable_session(symbol):
            self._log_activity(symbol, f"analyzed — outside favorable trading session for {symbol} — skipped")
            return

        df_1min: pd.DataFrame | None = None
        df_5min: pd.DataFrame | None = None
        df_15min: pd.DataFrame | None = None
        if self.mt5_bridge:
            df_1min = self.mt5_bridge.get_historical_data(symbol, "1min", count=200)
            df_5min = self.mt5_bridge.get_historical_data(symbol, "5min", count=200)
            df_15min = self.mt5_bridge.get_historical_data(symbol, "15min", count=200)
        if df_1min is None or len(df_1min) < 50:
            df_1min = await self.market_data.get_latest_data(symbol, "1min", lookback=200)
        if df_5min is None or len(df_5min) < 20:
            df_5min = await self.market_data.get_latest_data(symbol, "5min", lookback=200)
        if df_15min is None or len(df_15min) < 20:
            df_15min = await self.market_data.get_latest_data(symbol, "15min", lookback=200)
        if df_1min is None or len(df_1min) < 50:
            self._log_activity(symbol, "analyzed — no data available — skipped")
            return

        self._log_activity(symbol, f"analyzed — using {active_timeframe} model (accuracy {best['accuracy']:.4f}, best available)")

        # Market analysis + sentiment
        market_bias = await market_analysis.analyze_market_conditions(symbol)
        sentiment = market_bias.get("sentiment", {})
        await self.signal_fusion.add_signal(Signal(
            source=SignalSource.SENTIMENT,
            direction=1 if market_bias.get("bias") == "bullish" else (-1 if market_bias.get("bias") == "bearish" else 0),
            confidence=sentiment.get("confidence", 0),
            strength=sentiment.get("bullish", 0.5) - sentiment.get("bearish", 0.5),
            timestamp=datetime.now().timestamp(),
            metadata={"news_count": market_bias.get("news_count", 0), "symbol": symbol},
        ))

        # Sniper signal
        sniper_logic.update_order_blocks(symbol, df_1min)
        if df_5min is None:
            self._log_activity(symbol, "analyzed — missing 5min context — skipped")
            return
        df_1h = None
        df_4h = None
        df_1d = None
        df_30min = None
        if self.mt5_bridge:
            df_1h = self.mt5_bridge.get_historical_data(symbol, "1h", count=200)
            df_4h = self.mt5_bridge.get_historical_data(symbol, "4h", count=200)
            df_1d = self.mt5_bridge.get_historical_data(symbol, "1d", count=200)
            df_30min = self.mt5_bridge.get_historical_data(symbol, "30min", count=200)
        current_price_for_macro = float(df_1min["close"].iloc[-1]) if df_1min is not None and len(df_1min) else None
        macro = get_macro_bias(symbol, df_1h, df_4h, df_1d=df_1d, df_30min=df_30min, current_price=current_price_for_macro)
        sniper_signal = sniper_logic.get_sniper_signal(symbol, df_1min, df_5min, macro_bias=macro["bias"])
        sniper_found = sniper_signal["direction"] is not None

        if sniper_found:
            direction_str = sniper_signal["direction"]
            sniper_dir = 1 if direction_str == "BUY" else -1
            await self.signal_fusion.add_signal(Signal(
                source=SignalSource.PRICE_ACTION,
                direction=sniper_dir,
                confidence=sniper_signal.get("confidence", 0),
                strength=sniper_dir * sniper_signal.get("confidence", 0),
                timestamp=datetime.now().timestamp(),
                metadata={"reason": sniper_signal.get("reason", "")},
            ))
        else:
            # No structural (sniper) signal — don't hard-stop here. Still let
            # AI + technical + macro speak for themselves below; they just
            # have to clear a materially higher combined-confidence bar
            # (favourable_no_sniper_confidence_bonus) since there's no
            # liquidity-sweep/CHoCH/order-block confirmation backing them up.
            self._log_activity(symbol, f"analyzed — no price-action structure ({sniper_signal.get('reason', '')}) — evaluating on AI+technical alone")

        # AI signal
        try:
            df_map = {"1min": df_1min, "5min": df_5min, "15min": df_15min}
            df_for_ai = df_map.get(active_timeframe, df_1min)
            ai_pred = await self.ai_predictor.predict(
                df_for_ai, timeframe=active_timeframe, symbol=symbol,
                extended_context=self._market_context_snapshot,
            )
            if ai_pred and ai_pred.confidence > 0 and ai_pred.direction != 0:
                await self.signal_fusion.add_signal(Signal(
                    source=SignalSource.AI,
                    direction=ai_pred.direction,
                    confidence=ai_pred.confidence,
                    strength=ai_pred.direction * ai_pred.confidence,
                    timestamp=datetime.now().timestamp(),
                    metadata={"model": ai_pred.model_used},
                ))
            else:
                self._log_activity(symbol, "analyzed — no active AI prediction for symbol — trading disabled")
                return
        except Exception as e:
            self.logger.warning(f"AI prediction failed: {e}")
            self._log_activity(symbol, "analyzed — AI prediction failed — trading disabled")
            return

        # Technical signal
        tech = await self.get_technical_signal(df_1min, market_bias)
        await self.signal_fusion.add_signal(Signal(
            source=SignalSource.TECHNICAL,
            direction=tech["direction"],
            confidence=tech["confidence"],
            strength=tech["direction"] * tech["confidence"],
            timestamp=datetime.now().timestamp(),
            metadata=tech.get("indicators", {}),
        ))

        confluence_score = float(macro.get("confluence_score", 0.0))
        fused = await self.signal_fusion.fuse_signals(symbol, macro_bias=macro.get("bias"), confluence_score=confluence_score)

        # Multi-timeframe confluence modulates how much we demand from the
        # LTF trigger: a single timeframe's opinion shouldn't carry the same
        # weight as 3-4 timeframes agreeing + price sitting at a validated
        # HTF zone. Strong confluence (score->1.0) eases the floor slightly;
        # weak/absent confluence (score->0.0) raises it — capped at ±0.05 so
        # this shifts entry *quality* requirements without gutting the base
        # safety floor either direction.
        base_min_conf = float(SIGNAL_FUSION_CONFIG.get("final_combined_confidence_floor", 0.70))
        confluence_adjustment = (0.5 - confluence_score) * 0.10
        min_conf = max(0.60, min(0.85, base_min_conf + confluence_adjustment))
        if not sniper_found:
            fallback_bonus = float(SIGNAL_FUSION_CONFIG.get("favourable_no_sniper_confidence_bonus", 0.12))
            min_conf = min(0.92, min_conf + fallback_bonus)

        if fused.action == "HOLD" or fused.confidence < min_conf:
            self._log_activity(
                symbol,
                f"analyzed — fused confidence {fused.confidence:.2f} below {min_conf:.2f} floor "
                f"(confluence={confluence_score:.2f}, {macro.get('agreeing_timeframes', 0)}/{macro.get('total_timeframes', 0)} TFs, "
                f"at_key_zone={macro.get('at_key_zone', False)}, sniper={'yes' if sniper_found else 'no'}) or HOLD ({fused.details.get('reason', '')}) — no trade"
            )
            return

        if any(p.get("symbol") == symbol for p in self.active_positions):
            existing = next(p for p in self.active_positions if p.get("symbol") == symbol)
            await self._maybe_early_close(symbol, existing, fused, market_bias)
            self._log_activity(symbol, f"analyzed — fused confidence {fused.confidence:.2f} — signal: {fused.action} — already in position — skipped new entry")
            return

        sentiment_agrees = (
            (fused.action == "BUY" and market_bias.get("bias") != "bearish") or
            (fused.action == "SELL" and market_bias.get("bias") != "bullish") or
            market_bias.get("bias") == "neutral"
        )

        # ADX/trend-strength filter — HARD BLOCK (non-trending = no entries)
        try:
            adx_threshold = float(RISK_CONFIG.get("adx_block_entry_below", 18.0))
            adx_ok, adx_msg = risk_manager.check_trend_filter(symbol, df_1min, adx_threshold=adx_threshold)
            if not adx_ok:
                self._log_activity(symbol, f"analyzed — HARD ADX BLOCK {adx_msg} — entry blocked (ADX<{adx_threshold:.0f})")
                return
        except Exception as e:
            self.logger.warning(f"ADX check failed for {symbol}: {e}")
            self._log_activity(symbol, "analyzed — ADX check errored, treating as HARD BLOCK for safety")
            return

        entry_kind = "sniper" if sniper_found else "favourable (AI+technical, no structure)"
        self._log_activity(symbol, f"analyzed — fused signal {fused.action} @ {fused.confidence:.2f} — entry: {entry_kind}")
        tf_df_map = {"1min": df_1min, "5min": df_5min, "15min": df_15min, "30min": df_30min, "1h": df_1h}
        atr_df = tf_df_map.get(active_timeframe)
        if atr_df is None or len(atr_df) < 20:
            atr_df = df_1min
        await self.execute_trade(
            fused, symbol, df_1min, market_bias, sniper_signal, sentiment_agrees,
            active_timeframe=active_timeframe, atr_df=atr_df,
        )

    async def _maybe_early_close(
        self,
        symbol: str,
        position: Dict,
        fused: FusedSignal,
        market_bias: Dict,
    ):
        """Conservative early-close: opt-in, requires sustained opposing signal."""
        if not RISK_CONFIG.get("allow_early_close", False):
            return

        pos_dir = position.get("direction", "")
        opposing = (
            (pos_dir == "BUY" and fused.action == "SELL") or
            (pos_dir == "SELL" and fused.action == "BUY")
        )
        min_conf = RISK_CONFIG.get("early_close_min_confidence", 0.65)
        confirm_cycles = RISK_CONFIG.get("early_close_confirm_cycles", 2)

        if opposing and fused.confidence >= min_conf:
            key = f"{symbol}:{position.get('order_id')}"
            self._opposing_signal_counts[key] = self._opposing_signal_counts.get(key, 0) + 1
        else:
            key = f"{symbol}:{position.get('order_id')}"
            self._opposing_signal_counts[key] = 0
            return

        if self._opposing_signal_counts[key] < confirm_cycles:
            self._log_activity(
                symbol,
                f"early-close watch — opposing {fused.action} @ {fused.confidence:.2f} "
                f"({self._opposing_signal_counts[key]}/{confirm_cycles} cycles)",
            )
            return

        reason = (
            f"fused signal reversed to {fused.action} @ {fused.confidence:.2f} confidence, "
            f"closing early after {confirm_cycles} consecutive opposing cycles"
        )
        self._log_activity(symbol, f"EARLY CLOSE triggered — {reason}")
        order_id = position.get("order_id")
        result = await self.broker.close_order(order_id)
        if result.get("success"):
            position["bot_initiated_close"] = True
            position["close_reason"] = reason
            self._opposing_signal_counts.pop(key, None)
            self.logger.info(f"Early close executed for {symbol} #{order_id}")
        else:
            self._log_activity(symbol, f"early close FAILED — {result.get('error')}")

    async def get_technical_signal(self, market_data_df: pd.DataFrame, market_analysis_result: Dict[str, Any] | None = None) -> Dict[str, Any]:
        import numpy as np
        if len(market_data_df) < 20:
            return {"direction": 0, "confidence": 0.0, "indicators": {}}
        close = market_data_df["close"].to_numpy(dtype=float)
        ma_fast = np.mean(close[-5:])
        ma_slow = np.mean(close[-20:])
        ma_signal = 1 if ma_fast > ma_slow else -1
        delta = np.diff(close[-14:])
        gains = delta[delta > 0].sum()
        losses = abs(delta[delta < 0].sum())
        rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
        rsi_signal = -1 if rsi > 70 else (1 if rsi < 30 else 0)
        signals = [s for s in [ma_signal, rsi_signal] if s != 0]
        if not signals:
            return {"direction": 0, "confidence": 0.0, "indicators": {}}
        direction = 1 if sum(signals) > 0 else -1
        confidence = min(abs(sum(signals)) / len(signals), 0.8)
        return {"direction": direction, "confidence": confidence, "indicators": {"ma": ma_signal, "rsi": rsi_signal}}

    async def execute_trade(
        self,
        decision: FusedSignal,
        symbol: str,
        market_data_df: pd.DataFrame,
        market_analysis_result: Dict[str, Any] | None = None,
        sniper_signal: Dict[str, Any] | None = None,
        sentiment_agrees: bool = True,
        active_timeframe: str = "5min",
        atr_df: pd.DataFrame | None = None,
    ):
        min_conf = AI_CONFIG.get("min_confidence", 0.55)
        if decision.confidence < min_conf:
            return

        # Real-time slot check immediately before placement (mid-cycle freeing)
        open_orders = await self.broker.get_open_orders()
        if open_orders is None:
            # Can't confirm current slot usage — safer to skip this cycle's
            # entry than risk opening a position on top of an undercounted
            # slot.
            self._log_activity(symbol, "slot check — could not confirm broker position state — skipping order")
            return
        max_pos = risk_manager.get_max_positions_for_balance(self.account_info.get("balance", 0))
        if len(open_orders) >= max_pos:
            self._log_activity(
                symbol,
                f"slot check — {len(open_orders)}/{max_pos} positions open — skipping order",
            )
            return

        balance = self.account_info.get("balance", 0)
        symbol_info = self.mt5_bridge.get_symbol_info(symbol) if self.mt5_bridge else {}

        order_type = "BUY" if decision.action == "BUY" else "SELL"
        current_price = market_data_df["close"].iloc[-1]

        # ============= ATR-based SL/TP with 2.5R minimum GUARANTEE =============
        # Primary: use risk_manager ATR-based pricing (respects atr_sl_mult/atr_tp_mult + min_rr_ratio).
        # ATR must come from the SAME timeframe the signal was actually generated
        # on — always using 1min data here (regardless of active_timeframe) made
        # SL/TP distances for 5min/15min/30min/1h signals absurdly tight relative
        # to the move those signals actually predict, since 1min ATR is much
        # smaller than higher-timeframe ATR. A too-tight SL gets clipped by
        # ordinary spread/noise almost immediately, before the real move has any
        # chance to play out — exactly what happened to the first live trade.
        atr_source_df = atr_df if atr_df is not None and len(atr_df) >= 20 else market_data_df
        atr_sl_price, atr_tp_price = risk_manager.compute_atr_sl_tp_prices(
            symbol, order_type, current_price,
            df=atr_source_df, symbol_info=symbol_info,
        )
        cls = symbol_asset_class(symbol)

        # Fallback: sniper-provided SL/TP ONLY if they satisfy minimum R:R 2.5
        sniper_sl_valid = False
        sniper_tp_valid = False
        sl_price = atr_sl_price
        tp_price = atr_tp_price
        if sniper_signal and sniper_signal.get("stop_loss") and sniper_signal.get("take_profit"):
            s_sl = float(sniper_signal["stop_loss"])
            s_tp = float(sniper_signal["take_profit"])
            if order_type == "BUY":
                sl_dist = max(current_price - s_sl, 1e-9)
                tp_dist = max(s_tp - current_price, 1e-9)
            else:
                sl_dist = max(s_sl - current_price, 1e-9)
                tp_dist = max(current_price - s_tp, 1e-9)
            target_min_rr = float(RISK_CONFIG.get("min_rr_ratio", 2.0))
            if sl_dist > 0 and tp_dist >= sl_dist * target_min_rr:
                sl_price = s_sl
                tp_price = s_tp
                sniper_sl_valid = True
                sniper_tp_valid = True
            else:
                self._log_activity(
                    symbol,
                    f"sniper SL/TP failed R:R test (sl_dist={sl_dist:.6f} tp_dist={tp_dist:.6f} "
                    f"need tp>={sl_dist*target_min_rr:.6f}) — falling back to ATR levels",
                )
        sl_pips = price_distance_to_pips(symbol, abs(current_price - sl_price))
        tp_pips = price_distance_to_pips(symbol, abs(current_price - tp_price))
        # Final belt-and-suspenders invariant: tp_dist_pips >= sl_dist_pips * target_rr_ratio
        rr_actual = (tp_pips / sl_pips) if sl_pips > 0 else 0.0
        rr_min = float(RISK_CONFIG.get("target_rr_ratio", 2.5))
        if rr_actual < rr_min and sl_pips > 0:
            sl_distance = abs(current_price - sl_price)
            if order_type == "BUY":
                tp_price = current_price + sl_distance * rr_min
            else:
                tp_price = current_price - sl_distance * rr_min
            tp_pips = price_distance_to_pips(symbol, abs(current_price - tp_price))
            self._log_activity(
                symbol,
                f"TP distance BOOSTED to meet {rr_min:.1f}R minimum — actual was {rr_actual:.2f}R. "
                f"SL={sl_price:.6f} TP={tp_price:.6f}",
            )
        
        # Binance BTCUSD divergence check
        has_divergence = False
        if symbol.upper() in ("BTCUSD", "BTCUSDT", "XBTUSD"):
            divergence = check_btc_divergence(current_price)
            self._log_activity(symbol, divergence["msg"])
            self.logger.info(divergence["msg"])
            has_divergence = divergence["diverges"]

        # News volume spike caution flag (reduce position if >50 articles)
        has_high_news_volume = False
        if market_analysis_result and market_analysis_result.get("sentiment", {}).get("article_count", 0) > 50:
            has_high_news_volume = True
            self._log_activity(symbol, f"High news volume ({market_analysis_result['sentiment']['article_count']} articles) — proceeding with caution")

        position_size = risk_manager.calculate_dynamic_position_size(
            balance, sl_pips, decision.confidence, sentiment_agrees, symbol, symbol_info,
            floating_pnl=self._floating_pnl,
            floating_pnl_pct=self._floating_pnl_pct,
        )
        if self._floating_pnl_pct < -RISK_CONFIG.get("floating_loss_threshold_pct", 2.0):
            self._log_activity(
                symbol,
                f"reducing position size by {RISK_CONFIG.get('floating_loss_size_reduction_pct', 25):.0f}% — "
                f"currently sitting on unrealized losses ({self._floating_pnl_pct:.1f}% floating)",
            )
        
        # Slightly reduce position size if there's divergence or high news volume
        if has_divergence:
            position_size = position_size * 0.7  # 30% reduction
            self._log_activity(symbol, "reducing position size by 30% due to price divergence")
        if has_high_news_volume:
            position_size = position_size * 0.8  # 20% reduction for high news volume
            self._log_activity(symbol, "reducing position size by 20% due to high news volume")

        # VIX regime check — reduce size during elevated market-wide volatility
        try:
            if is_high_volatility_regime(threshold=25.0):
                position_size = position_size * 0.7
                self._log_activity(symbol, "reducing position size by 30% due to high VIX (elevated market volatility)")
        except Exception as e:
            self.logger.warning(f"VIX check failed for {symbol}: {e}")

        # FRED rate-trend check — bias against long AUD/NZD when US rates are climbing
        try:
            base_currency = symbol[:3].upper()
            if base_currency in ("AUD", "NZD") and decision.action == "BUY":
                if is_rates_rising("FEDFUNDS"):
                    position_size = position_size * 0.7
                    self._log_activity(symbol, "reducing position size by 30% — rising US rates typically pressure commodity currencies")
        except Exception as e:
            self.logger.warning(f"FRED rate check failed for {symbol}: {e}")

        # COT institutional positioning — adjust size based on agreement with fused signal
        try:
            cot_currency_map = {
                "EUR": "EURO FX", "GBP": "BRITISH POUND", "JPY": "JAPANESE YEN",
                "AUD": "AUSTRALIAN DOLLAR", "NZD": "NEW ZEALAND DOLLAR",
                "CAD": "CANADIAN DOLLAR", "CHF": "SWISS FRANC",
            }
            base_currency = symbol[:3].upper()
            cot_name = cot_currency_map.get(base_currency)
            if cot_name:
                cot_data = get_latest_cot_positioning(cot_name)
                if cot_data:
                    cot_bullish = cot_data["bias"] == "bullish"
                    signal_bullish = decision.action == "BUY"
                    if cot_bullish == signal_bullish:
                        position_size = position_size * 1.1
                        self._log_activity(symbol, f"COT positioning agrees ({cot_data['bias']}) — slightly increasing size")
                    else:
                        position_size = position_size * 0.85
                        self._log_activity(symbol, f"COT positioning disagrees ({cot_data['bias']}) — reducing size")
        except Exception as e:
            self.logger.warning(f"COT check failed for {symbol}: {e}")

        # Re-align to the broker's real volume step after ALL reductions above — multiplying
        # by any of these factors can produce a size that's no longer a valid step multiple,
        # which was causing "Invalid volume" rejections even after the initial sizing was fixed.
        vol_step = float(symbol_info.get("volume_step") or 0.01)
        if vol_step > 0:
            position_size = round(position_size / vol_step) * vol_step
            decimals = len(str(vol_step).split(".")[-1]) if "." in str(vol_step) else 2
            position_size = round(position_size, decimals)

        volume_min = symbol_info.get("volume_min", 0.01)

        if position_size < volume_min:
            # Kept in sync with risk_manager.calculate_dynamic_position_size's
            # micro-account tier — a 2% ceiling here while sizing targets 5%
            # for tiny balances would contradict itself and skip trades that
            # sizing had already decided were acceptable.
            max_safe_risk_pct = 0.07 if balance < 50 else 0.03
            risk_at_min = volume_min * sl_pips * get_tick_value_from_broker(symbol_info or {}, default=10.0)
            if risk_at_min > balance * max_safe_risk_pct:
                self._log_activity(symbol, f"risk check FAILED — calculated size {position_size:.4f} lots below broker minimum {volume_min} — skipping because min size would risk ${risk_at_min:.2f} ({risk_at_min/balance*100:.1f}% of ${balance:.2f})")
                return
            position_size = volume_min

        allowed, reason = risk_manager.validate_trade(
            symbol, position_size, balance, self.active_positions,
            market_analysis_result or {}, self.is_demo, self.live_trading_enabled,
        )
        if not allowed:
            self._log_activity(symbol, f"risk check FAILED — {reason}")
            self.logger.debug(f"Trade blocked: {reason}")
            return
        if position_size <= 0:
            self._log_activity(symbol, "risk check FAILED — position size <= 0 — trade skipped")
            return

        # Log planned R:R before placement (ATR-based 2.5R minimum guaranteed above)
        rr_for_log = (tp_pips / sl_pips) if sl_pips > 0 else 0.0
        self._log_activity(
            symbol,
            f"signal: {decision.action} — confidence {decision.confidence:.2f} — size {position_size:.2f} lots "
            f"— SL={sl_price:.6f} TP={tp_price:.6f} (~{rr_for_log:.2f}R) — placing order",
        )
        self.logger.info(
            f"[{APP_NAME}] {order_type} {symbol} | Size: {position_size:.2f} | Conf: {decision.confidence:.2f} | SL={sl_price:.6f} TP={tp_price:.6f} (R={rr_for_log:.2f})"
        )

        order_result = await self.broker.place_order(
            symbol=symbol,
            order_type=order_type,
            volume=position_size,
            sl_price=float(sl_price),
            tp_price=float(tp_price),
            comment=f"AeroTrade-{decision.action}-R{rr_for_log:.1f}",
        )

        if order_result.get("success"):
            order_id = str(order_result.get("order_id", order_result.get("ticket", "")))
            fill_price = float(order_result.get("price") or current_price)
            features_json = "{}"
            try:
                feats = await self.ai_predictor.prepare_features(market_data_df)
                if feats.size > 0:
                    features_json = json.dumps(feats.flatten().tolist()[:50])
            except Exception:
                pass

            record_open_trade(
                order_id, self.account_id, symbol, order_type,
                fill_price, position_size,
                decision.confidence, features_json, sentiment_agrees,
                decision.details.get("model", "fusion"),
                timeframe=active_timeframe,
            )
            pos_dict = {
                "order_id": order_id,
                "symbol": symbol,
                "direction": order_type,
                "entry_price": fill_price,
                "size": position_size,
                "confidence": decision.confidence,
                "features_json": features_json,
                "timeframe": active_timeframe,
                "stop_loss": float(sl_price),
                "take_profit": float(tp_price),
            }
            self.active_positions.append(pos_dict)
            # Initialize exit state machine for partial-TP / BE / trailing
            try:
                risk_manager._ensure_position_state(
                    order_id, entry=fill_price, direction=order_type,
                    sl=float(sl_price), tp=float(tp_price),
                    symbol=symbol, df=market_data_df,
                )
            except Exception as sm_err:
                self.logger.warning(f"Exit state-machine init skipped for {order_id}: {sm_err}")
            self._log_activity(
                symbol,
                f"order placed — {order_type} {position_size:.2f} lots #{order_id} @ {fill_price:.6f} — SL={sl_price:.6f} TP={tp_price:.6f}",
            )
            self.logger.success(f"Order placed: {order_type} {symbol} #{order_id} @ {fill_price:.6f}")
        else:
            self._log_activity(symbol, f"order FAILED — {order_result.get('error')}")
            self.logger.error(f"Order failed: {order_result.get('error')}")

    async def monitor_positions(self):
        open_orders = await self.broker.get_open_orders()
        if open_orders is None:
            # A failed/timed-out query is NOT the same as "broker confirms
            # zero positions" — treating it that way caused a real open
            # position to be logged as closed (a fake breakeven trade) while
            # it was still genuinely live at the broker, and dropped it out
            # of active management (trailing stop / partial TP) as a result.
            # Skip reconciliation entirely this cycle; try again next cycle.
            self.logger.warning("monitor_positions: could not confirm broker position state this cycle — skipping close reconciliation")
            return
        broker_ids = {str(o.get("order_id")) for o in open_orders}
        local_ids = [str(p.get("order_id")) for p in self.active_positions]

        self.logger.debug(
            f"monitor_positions check: local={len(self.active_positions)} "
            f"broker={len(open_orders)} local_ids={local_ids} "
            f"broker_ids={sorted(broker_ids)}"
        )

        # Update floating profit on still-open positions
        broker_map = {str(o.get("order_id")): o for o in open_orders}
        for pos in self.active_positions:
            oid = str(pos.get("order_id"))
            if oid in broker_map:
                pos["floating_pnl"] = float(broker_map[oid].get("profit", 0) or 0)
                bsl = float(broker_map[oid].get("stop_loss") or 0)
                btp = float(broker_map[oid].get("take_profit") or 0)
                if bsl and bsl > 0:
                    pos["stop_loss"] = bsl
                if btp and btp > 0:
                    pos["take_profit"] = btp

        # ========== Exit state-machine: Break-Even / Trailing / Partial-TP ==========
        be_enabled = RISK_CONFIG.get("break_even_enabled", True)
        ts_enabled = RISK_CONFIG.get("trailing_stop_enabled", True)
        ptp_enabled = RISK_CONFIG.get("partial_tp_enabled", True)
        if (be_enabled or ts_enabled or ptp_enabled) and self.active_positions and self.mt5_bridge:
            _symbol_dfs_cache: Dict[str, pd.DataFrame] = {}
            _symbol_infos_cache: Dict[str, Any] = {}
            for pos in self.active_positions:
                oid = str(pos.get("order_id"))
                if oid not in broker_map:
                    continue
                symbol = pos.get("symbol", "")
                entry = float(pos.get("entry_price") or 0)
                direction = pos.get("direction", "")
                current_sl = float(pos.get("stop_loss") or 0)
                current_tp = float(pos.get("take_profit") or 0)
                if not symbol or entry <= 0 or direction not in ("BUY", "SELL"):
                    continue
                try:
                    if symbol not in _symbol_dfs_cache:
                        df_sym = self.mt5_bridge.get_historical_data(symbol, "1min", count=150)
                        if df_sym is None or len(df_sym) < 50:
                            df_sym = await self.market_data.get_latest_data(symbol, "1min", lookback=150)
                        _symbol_dfs_cache[symbol] = df_sym
                    df_for_exit = _symbol_dfs_cache.get(symbol)
                    if df_for_exit is None or len(df_for_exit) < 30:
                        continue
                    if symbol not in _symbol_infos_cache:
                        _symbol_infos_cache[symbol] = self.mt5_bridge.get_symbol_info(symbol) or {}
                    sym_info = _symbol_infos_cache[symbol]
                    current_price = float(df_for_exit["close"].iloc[-1])
                    exit_eval = risk_manager.evaluate_position_exit(
                        oid, current_price, entry, direction, current_sl, current_tp,
                        symbol, df_for_exit, sym_info,
                    )
                    action = (exit_eval or {}).get("action", "HOLD")
                    new_sl = float(exit_eval.get("new_sl") or 0)
                    new_tp = float(exit_eval.get("new_tp") or 0)
                    rr_state = exit_eval.get("rr_state", "")
                    reason = exit_eval.get("reason", "")
                    if action == "MOVE_SL" and (new_sl > 0 or new_tp > 0):
                        if abs(new_sl - current_sl) > 1e-9 or abs(new_tp - current_tp) > 1e-9:
                            mod_res = await self.broker.modify_position(int(oid), stop_loss=new_sl, take_profit=new_tp)
                            if mod_res.get("success"):
                                pos["stop_loss"] = new_sl if new_sl > 0 else current_sl
                                pos["take_profit"] = new_tp if new_tp > 0 else current_tp
                                self._log_activity(
                                    symbol,
                                    f"#{oid} EXIT-MGR MOVE_SL/TP — {reason} — state={rr_state} "
                                    f"new SL={new_sl:.6f} TP={new_tp:.6f}",
                                )
                            else:
                                self.logger.warning(f"modify_position failed for {symbol} #{oid}: {mod_res.get('error')}")
                    elif action in ("PARTIAL_CLOSE_1", "PARTIAL_CLOSE_2"):
                        close_fraction = float(exit_eval.get("close_fraction") or 0.0)
                        current_size = float(pos.get("size") or 0)
                        if close_fraction > 0 and current_size > 0:
                            partial_size_raw = current_size * close_fraction
                            vol_step = float(sym_info.get("volume_step") or 0.01)
                            partial_size = partial_size_raw
                            if vol_step > 0:
                                partial_size = round(partial_size_raw / vol_step) * vol_step
                                decimals = len(str(vol_step).split(".")[-1]) if "." in str(vol_step) else 2
                                partial_size = round(partial_size, decimals)
                            volume_min = float(sym_info.get("volume_min") or 0.01)
                            remaining_after = current_size - partial_size
                            if partial_size >= volume_min and remaining_after >= volume_min:
                                close_res = await self.broker.close_partial(int(oid), partial_size)
                                if close_res.get("success"):
                                    pos["size"] = max(remaining_after, 0.0)
                                    self._log_activity(
                                        symbol,
                                        f"#{oid} EXIT-MGR {action} @ +{exit_eval.get('rr_level','')}R — "
                                        f"closed {partial_size:.4f} lots ({close_fraction*100:.0f}%) — "
                                        f"remaining {pos['size']:.4f} lots — reason: {reason}",
                                    )
                                else:
                                    self.logger.warning(f"partial close failed for {symbol} #{oid}: {close_res.get('error')}")
                            else:
                                self.logger.debug(
                                    f"Skipping partial for {symbol} #{oid}: partial={partial_size} "
                                    f"remaining={remaining_after} (min per-side={volume_min})"
                                )
                except Exception as exit_err:
                    self.logger.warning(f"Exit-state eval skipped for pos {oid} ({symbol}): {exit_err}")

        if not self.active_positions and not open_orders:
            return

        for position in self.active_positions[:]:
            try:
                order_id = str(position.get("order_id"))
                if order_id in broker_ids:
                    continue

                symbol = position.get("symbol", "")
                pnl = 0.0
                exit_price = position.get("entry_price", 0)
                bot_initiated = position.pop("bot_initiated_close", False)
                preset_reason = position.pop("close_reason", "")

                closed = await self.broker.get_close_deal(int(order_id))
                if not closed:
                    history = await self.broker.get_order_history(limit=500)
                    closed = next((o for o in history if str(o.get("order_id")) == order_id), None)

                if closed:
                    pnl = closed.get("pnl_usd", 0) or closed.get("profit", 0)
                    exit_price = closed.get("price", exit_price)
                    if preset_reason:
                        closed["close_reason"] = preset_reason
                else:
                    # MT5 deal-history sync has been unreliable on this
                    # terminal — falling back to "unknown" every time meant
                    # classify_close()'s own SL/TP-proximity heuristic (which
                    # is quite good) never got a chance to run, since it only
                    # runs when a deal dict is present. A live market-price
                    # snapshot is a solid stand-in for the real exit price
                    # here, and the position's last-known floating P&L (from
                    # the update loop above, while it was still open) is a
                    # much better pnl estimate than the $0 placeholder — this
                    # also unblocks online learning, which only trains on
                    # bot_sl/bot_tp/bot_manual_close outcomes.
                    self.logger.warning(f"Position {order_id} closed — no deal history found, falling back to market-price proxy")
                    pnl = float(position.get("floating_pnl", 0) or 0)
                    proxy_price = 0.0
                    if self.mt5_bridge and symbol:
                        try:
                            tick = self.mt5_bridge.get_market_price(symbol)
                            direction = position.get("direction", "")
                            if tick:
                                proxy_price = float(tick.get("ask" if direction == "SELL" else "bid", 0) or 0)
                        except Exception as price_err:
                            self.logger.warning(f"Could not fetch proxy exit price for {symbol}: {price_err}")
                    if proxy_price > 0:
                        exit_price = proxy_price
                        closed = {"price": proxy_price}

                closed_by, close_reason = classify_close(
                    position, closed or {},
                    sl_price=float(position.get("stop_loss", 0) or 0),
                    tp_price=float(position.get("take_profit", 0) or 0),
                    bot_initiated=bot_initiated,
                )

                # resolve_orphan_trade handles both cases correctly: if the
                # order_id already has an open row (the normal case), it just
                # closes it; if not (e.g. a position reconciled from broker
                # state at startup after a restart, which was never opened
                # via record_open_trade), it creates the row first. Calling
                # record_close_trade directly here used to silently no-op
                # for reconciled positions — its UPDATE matches zero rows
                # and there's no fallback — so those closes vanished from
                # the ledger entirely instead of at least landing as
                # unattributed like a normal orphan.
                resolve_orphan_trade(
                    order_id, self.account_id, position.get("symbol", ""),
                    position.get("direction", ""), position.get("entry_price", 0),
                    position.get("size", 0), pnl,
                    closed_by=closed_by, close_reason=close_reason,
                    exit_price=exit_price,
                )
                outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
                self._log_activity(
                    position.get("symbol", "?"),
                    f"position closed — {outcome} ${pnl:.2f} — closed_by={closed_by} — {close_reason}",
                )

                if pnl > 0:
                    self.win_streak += 1
                else:
                    self.win_streak = 0

                # Online learning — exclude user manual closes
                if closed_by in TRAINABLE_CLOSED_BY:
                    actual_dir = 1 if pnl > 0 else 0
                    try:
                        sym = position.get("symbol", "EURUSD")
                        tf = position.get("timeframe", "5min")
                        df = await self.market_data.get_latest_data(sym, tf, lookback=50)
                        if df is not None:
                            await self.ai_predictor.update_with_new_data(
                                df, actual_dir, tf,
                                trade_sample={
                                    "order_id": order_id,
                                    "symbol": sym,
                                    "features": json.loads(position.get("features_json", "[]") or "[]") if position.get("features_json") else [],
                                    "label": int(actual_dir),
                                    "timeframe": tf,
                                    "recorded_at": datetime.now().isoformat(),
                                    "closed_by": closed_by,
                                },
                            )
                    except Exception as learn_err:
                        self.logger.warning(f"Online learning skipped for {order_id}: {learn_err}")
                else:
                    self.logger.info(
                        f"Trade {order_id} excluded from training — closed_by={closed_by}"
                    )

                risk_manager.record_trade({"pnl": pnl, "symbol": position.get("symbol")})
                # Always tear down per-position exit state when the position fully closes
                try:
                    risk_manager.clear_position_exit_state(order_id)
                except Exception:
                    pass
                self.logger.info(f"Position {order_id} closed: {outcome} P&L=${pnl:.2f} [{closed_by}]")
                self.active_positions.remove(position)
            except Exception as e:
                self.logger.error(
                    f"Error resolving closed position {position.get('order_id')}: {e} — will retry next cycle"
                )
    
    async def close_position(self, position: dict):
        order_id = position.get("order_id")
        if order_id:
            await self.broker.close_order(order_id)

    async def shutdown(self):
        self.logger.info("Shutting down...")
        for pos in self.active_positions:
            await self.close_position(pos)
        await self.ai_predictor.stop_retraining_scheduler()
        await self.broker.disconnect()
        self.running = False
        self.logger.info(f"{APP_NAME} shutdown complete")

    def stop(self):
        self.running = False


# Backward-compatible alias
HybridTradingBot = AeroTradeBot


async def main():
    bot = AeroTradeBot()
    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        print("\nBot stopped")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
