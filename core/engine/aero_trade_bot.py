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
from ai_engine.training.model_status import get_best_timeframe_for_symbol
from core.sessions import is_favorable_session
from decision_maker.signal_fusion import SignalFusion, Signal, SignalSource, FusedSignal
from core.broker.mt5_broker_client import MT5BrokerClient
from core.broker.credential_store import get_account_meta
from core.strategies.sniper_logic import sniper_logic
from core.risk_management.risk_manager import risk_manager
from core.ledger.trade_ledger import (
    init_ledger, record_open_trade, record_close_trade, resolve_orphan_trade,
)
from core.symbols import symbol_asset_class, price_distance_to_pips, get_tick_value_from_broker
from data.market_data import MarketDataFeed
from data.market_analysis import market_analysis
from data.binance_data import check_btc_divergence
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
        if len(self.activity_log) > 500:
            self.activity_log = self.activity_log[-500:]

    def get_activity(self, limit: int = 50) -> List[Dict]:
        return self.activity_log[-limit:]

    async def initialize(self, creds: Dict[str, Any] | None = None):
        self.logger.info(f"Initializing {APP_NAME}...")
        init_ledger()
        await self._initialize_ai()
        await self.ai_predictor.start_retraining_scheduler()

        if creds:
            await self.broker.connect_with_credentials(creds)
        else:
            await self.broker.connect(demo=True)

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
            all_symbols = self.mt5_bridge.get_all_symbols()
            available = [symbol for symbol in configured_pairs if symbol in all_symbols]
            missing = [symbol for symbol in configured_pairs if symbol not in all_symbols]
            self.available_symbols = available
            if missing:
                self.logger.warning(f"Configured symbols not available from broker: {', '.join(missing)}")
        else:
            self.available_symbols = configured_pairs

        await self.market_data.connect()
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

    async def trading_iteration(self):
        try:
            stop, reason = risk_manager.should_stop_trading(
                self.account_info.get("balance", 0),
                self.initial_balance,
                self.is_demo,
            )
            if stop:
                self.logger.warning(f"Trading stopped: {reason}")
                self._log_activity("-", f"cycle aborted — {reason}")
                return

            if len(self.active_positions) >= TRADING_CONFIG.get("max_positions", 10):
                await self.monitor_positions()
                self._log_activity("-", f"cycle skipped — max positions reached ({len(self.active_positions)})")
                return

            self.account_info = await self.broker.get_account_info()
            if self.broker.use_mt5 and self.broker.mt5_bridge and not self.broker.mt5_bridge.connected:
                self.logger.warning("MT5 connection lost — attempting reconnect")
                try:
                    from core.broker.credential_store import load_broker_credentials
                    creds = load_broker_credentials(self.account_id)
                    await self.broker.connect_with_credentials(creds)
                except Exception as e:
                    self.logger.error(f"MT5 reconnect failed: {e}")
            symbols_analyzed = 0
            for symbol in self.available_symbols[:TRADING_CONFIG.get("max_symbols", 50)]:
                try:
                    await self._process_symbol(symbol)
                    symbols_analyzed += 1
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {e}")
                    self._log_activity(symbol, f"error — {e}")
            await self.monitor_positions()
            self._log_activity("-", f"cycle complete — {symbols_analyzed} symbols analyzed — next cycle in {1 if TRADING_CONFIG.get('scalping_enabled') else 5}s")
        except Exception as e:
            self.logger.error(f"Iteration error: {e}")
            self._log_activity("-", f"cycle error — {e}")

    def _has_active_trading_model(self, symbol: str, timeframe: str = "1min") -> bool:
        family = "gbm" if timeframe == "15min" else "rf"
        model_key = make_model_key(symbol, timeframe, family)
        return model_key in self.ai_predictor.models

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
        if self.mt5_bridge:
            df_1h = self.mt5_bridge.get_historical_data(symbol, "1h", count=200)
            df_4h = self.mt5_bridge.get_historical_data(symbol, "4h", count=200)
        macro = get_macro_bias(symbol, df_1h, df_4h)
        sniper_signal = sniper_logic.get_sniper_signal(symbol, df_1min, df_5min, macro_bias=macro["bias"])
        if sniper_signal["direction"] is None:
            self._log_activity(symbol, "analyzed — no price-action signal — skipped")
            return

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

        # AI signal
        try:
            df_map = {"1min": df_1min, "5min": df_5min, "15min": df_15min}
            df_for_ai = df_map.get(active_timeframe, df_1min)
            ai_pred = await self.ai_predictor.predict(df_for_ai, timeframe=active_timeframe, symbol=symbol)
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

        fused = await self.signal_fusion.fuse_signals(symbol)
        min_conf = AI_CONFIG.get("min_confidence", 0.55)
        if fused.action == "HOLD" or fused.confidence < min_conf:
            self._log_activity(symbol, f"analyzed — fused confidence {fused.confidence:.2f} (below {min_conf} threshold) — no trade")
            return

        if any(p.get("symbol") == symbol for p in self.active_positions):
            self._log_activity(symbol, f"analyzed — fused confidence {fused.confidence:.2f} — signal: {fused.action} — already in position — skipped")
            return

        sentiment_agrees = (
            (fused.action == "BUY" and market_bias.get("bias") != "bearish") or
            (fused.action == "SELL" and market_bias.get("bias") != "bullish") or
            market_bias.get("bias") == "neutral"
        )

        # ADX/trend-strength filter
        try:
            adx_ok, adx_msg = risk_manager.check_trend_filter(symbol, df_1min, adx_threshold=10.0)
            if not adx_ok:
                self._log_activity(symbol, f"analyzed — {adx_msg} — continuing anyway")
        except Exception as e:
            self.logger.warning(f"ADX check failed for {symbol}: {e}")

        self._log_activity(symbol, f"analyzed — fused signal {fused.action} @ {fused.confidence:.2f}")
        await self.execute_trade(fused, symbol, df_1min, market_bias, sniper_signal, sentiment_agrees)

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
    ):
        min_conf = AI_CONFIG.get("min_confidence", 0.55)
        if decision.confidence < min_conf:
            return

        balance = self.account_info.get("balance", 0)
        symbol_info = self.mt5_bridge.get_symbol_info(symbol) if self.mt5_bridge else {}

        # Stop loss distance
        current_price = market_data_df["close"].iloc[-1]
        
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
        
        if sniper_signal and sniper_signal.get("stop_loss"):
            sl_price = sniper_signal["stop_loss"]
            sl_pips = price_distance_to_pips(symbol, abs(current_price - sl_price))
        else:
            cls = symbol_asset_class(symbol)
            if cls == "crypto":
                sl_pips = RISK_CONFIG.get("crypto_sl_dollars", 200)
            elif cls == "metal":
                sl_pips = RISK_CONFIG.get("metal_sl_pips", 30)
            else:
                sl_pips = RISK_CONFIG.get("scalping_sl_pips", 50)

        position_size = risk_manager.calculate_dynamic_position_size(
            balance, sl_pips, decision.confidence, sentiment_agrees, symbol, symbol_info,
        )
        
        # Slightly reduce position size if there's divergence or high news volume
        if has_divergence:
            position_size = position_size * 0.7  # 30% reduction
            self._log_activity(symbol, "reducing position size by 30% due to price divergence")
        if has_high_news_volume:
            position_size = position_size * 0.8  # 20% reduction for high news volume
            self._log_activity(symbol, "reducing position size by 20% due to high news volume")

        volume_min = symbol_info.get("volume_min", 0.01)

        if position_size < volume_min:
            max_safe_risk_pct = 0.02
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

        order_type = "BUY" if decision.action == "BUY" else "SELL"
        cls = symbol_asset_class(symbol)
        if cls == "crypto":
            tp_pips = RISK_CONFIG.get("crypto_tp_dollars", 150)
        elif cls == "metal":
            tp_pips = RISK_CONFIG.get("metal_tp_pips", 20)
        else:
            tp_pips = RISK_CONFIG.get("scalping_tp_pips", 30)

        self._log_activity(symbol, f"signal: {decision.action} — confidence {decision.confidence:.2f} — size {position_size:.2f} lots — placing order")
        self.logger.info(
            f"[{APP_NAME}] {order_type} {symbol} | Size: {position_size:.2f} | Conf: {decision.confidence:.2f}"
        )

        order_result = await self.broker.place_order(
            symbol=symbol,
            order_type=order_type,
            volume=position_size,
            stop_loss=sl_pips,
            take_profit=tp_pips,
            comment=f"AeroTrade-{decision.action}",
        )

        if order_result.get("success"):
            order_id = str(order_result.get("order_id", order_result.get("ticket", "")))
            features_json = "{}"
            try:
                feats = await self.ai_predictor.prepare_features(market_data_df)
                if feats.size > 0:
                    features_json = json.dumps(feats.flatten().tolist()[:50])
            except Exception:
                pass

            record_open_trade(
                order_id, self.account_id, symbol, order_type,
                order_result.get("price", current_price), position_size,
                decision.confidence, features_json, sentiment_agrees,
                decision.details.get("model", "fusion"),
            )
            self.active_positions.append({
                "order_id": order_id,
                "symbol": symbol,
                "direction": order_type,
                "entry_price": order_result.get("price", current_price),
                "size": position_size,
                "confidence": decision.confidence,
                "features_json": features_json,
            })
            self._log_activity(symbol, f"order placed — {order_type} {position_size:.2f} lots #{order_id}")
            self.logger.success(f"Order placed: {order_type} {symbol} #{order_id}")
        else:
            self._log_activity(symbol, f"order FAILED — {order_result.get('error')}")
            self.logger.error(f"Order failed: {order_result.get('error')}")

    async def monitor_positions(self):
        if not self.active_positions:
            return
        open_orders = await self.broker.get_open_orders()
        broker_ids = {str(o.get("order_id")) for o in open_orders}

        for position in self.active_positions[:]:
            order_id = str(position.get("order_id"))
            if order_id not in broker_ids:
                pnl = 0.0
                exit_price = position.get("entry_price", 0)
                history = await self.broker.get_order_history(limit=200)
                closed = next((o for o in history if str(o.get("order_id")) == order_id), None)
                if closed:
                    pnl = closed.get("pnl_usd", 0) or closed.get("profit", 0)
                    exit_price = closed.get("price", exit_price)
                else:
                    self.logger.warning(f"Position {order_id} closed — resolving via ledger")
                    resolve_orphan_trade(
                        order_id, self.account_id, position.get("symbol", ""),
                        position.get("direction", ""), position.get("entry_price", 0),
                        position.get("size", 0), pnl,
                    )

                outcome = record_close_trade(order_id, exit_price, pnl, position.get("confidence", 0))
                if pnl > 0:
                    self.win_streak += 1
                else:
                    self.win_streak = 0

                # Online learning from outcome
                actual_dir = 1 if pnl > 0 else 0
                try:
                    sym = position.get("symbol", "EURUSD")
                    df = await self.market_data.get_latest_data(sym, "5min", lookback=50)
                    if df is not None:
                        await self.ai_predictor.update_with_new_data(
                            df,
                            actual_dir,
                            "5min",
                            trade_sample={
                                "order_id": order_id,
                                "symbol": sym,
                                "features": json.loads(position.get("features_json", "[]") or "[]") if position.get("features_json") else [],
                                "label": int(actual_dir),
                                "timeframe": "5min",
                                "recorded_at": datetime.now().isoformat(),
                            },
                        )
                except Exception:
                    pass

                risk_manager.record_trade({"pnl": pnl, "symbol": position.get("symbol")})
                self.logger.info(f"Position {order_id} closed: {outcome} P&L=${pnl:.2f}")
                self.active_positions.remove(position)

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
