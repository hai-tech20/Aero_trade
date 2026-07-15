"""Stress-test the strategy against known historical crash events."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger
from data.yfinance_source import fetch_crash_window, save_supplementary_history
import importlib.util
spec = importlib.util.spec_from_file_location("backtest", Path(__file__).parent.parent / "backtest.py")
backtest_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest_mod)
BacktestEngine = backtest_mod.BacktestEngine
from core.risk_management.risk_manager import risk_manager


STRESS_TEST_EVENTS = {
    "2008 Financial Crisis Trough": "2008-10-08",
    "2010 Flash Crash": "2010-05-06",
    "2015 Swiss Franc Peg Shock": "2015-01-15",
    "2020 COVID-19 Crash": "2020-03-12",
    "2022 FTX Collapse": "2022-11-21",
}

BALANCE_TIERS = [100.0, 1_000.0, 10_000.0]


def _safe_max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max * 100
    return float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0


def _safe_sharpe(equity_curve: np.ndarray) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / equity_curve[:-1]
    std = np.std(returns)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(252))


async def run_stress_test_for_event(
    event_name: str,
    event_date: str,
    symbol: str = "EURUSD",
    window_days: int = 10,
    initial_balances: Optional[List[float]] = None,
) -> Dict:
    balances = initial_balances or BALANCE_TIERS
    logger.info(f"[STRESS] {event_name} ({event_date})")
    df = fetch_crash_window(symbol, event_date, window_days)
    if df is None or len(df) < 5:
        logger.error(f"[STRESS] Insufficient data for {event_name}")
        return {"event": event_name, "error": "insufficient data"}
    save_supplementary_history(f"{symbol}_stress_{event_date.replace('-', '')}", df)
    results = {}
    for balance in balances:
        engine = BacktestEngine(initial_balance=balance)
        engine.balance = balance
        engine.equity_curve = [balance]
        engine.trade_history = []
        engine.positions = []
        lookback = 20
        for i in range(lookback, len(df)):
            current_data = df.iloc[:i + 1]
            current_price = df["close"].iloc[i]
            current_time = df.index[i]
            from data.market_data import market_data
            market_data.data_cache[f"{symbol}_1min"] = current_data
            try:
                from ai_engine.models.predictor import ai_predictor
                from decision_maker.signal_fusion import signal_fusion
                ai_prediction = await ai_predictor.predict(current_data, timeframe="5min")
                if ai_prediction.direction != 0 and not engine.positions:
                    sl_pips = 50.0
                    pos_size = risk_manager.calculate_position_size(
                        balance, 0.02, sl_pips, symbol, ai_prediction.confidence,
                    )
                    if pos_size >= 0.01:
                        trade = {
                            "entry_time": current_time,
                            "entry_price": current_price,
                            "symbol": symbol,
                            "direction": "BUY" if ai_prediction.direction == 1 else "SELL",
                            "size": pos_size,
                            "stop_loss": current_price * (0.98 if ai_prediction.direction == 1 else 1.02),
                            "take_profit": current_price * (1.03 if ai_prediction.direction == 1 else 0.97),
                        }
                        engine.positions.append(trade)
                for pos in engine.positions[:]:
                    if pos["direction"] == "BUY":
                        if current_price <= pos["stop_loss"]:
                            pnl = (current_price - pos["entry_price"]) * 10000 * 10 * pos["size"]
                            engine.balance += pnl
                            engine.trade_history.append({"pnl": pnl, "reason": "SL"})
                            engine.positions.remove(pos)
                        elif current_price >= pos["take_profit"]:
                            pnl = (current_price - pos["entry_price"]) * 10000 * 10 * pos["size"]
                            engine.balance += pnl
                            engine.trade_history.append({"pnl": pnl, "reason": "TP"})
                            engine.positions.remove(pos)
                    else:
                        if current_price >= pos["stop_loss"]:
                            pnl = (pos["entry_price"] - current_price) * 10000 * 10 * pos["size"]
                            engine.balance += pnl
                            engine.trade_history.append({"pnl": pnl, "reason": "SL"})
                            engine.positions.remove(pos)
                        elif current_price <= pos["take_profit"]:
                            pnl = (pos["entry_price"] - current_price) * 10000 * 10 * pos["size"]
                            engine.balance += pnl
                            engine.trade_history.append({"pnl": pnl, "reason": "TP"})
                            engine.positions.remove(pos)
            except Exception:
                pass
            engine.equity_curve.append(engine.balance)
        for pos in engine.positions:
            exit_price = df["close"].iloc[-1]
            pnl = (exit_price - pos["entry_price"]) * 10000 * 10 * pos["size"] if pos["direction"] == "BUY" else (pos["entry_price"] - exit_price) * 10000 * 10 * pos["size"]
            engine.balance += pnl
            engine.trade_history.append({"pnl": pnl, "reason": "END"})
        equity_arr = np.array(engine.equity_curve)
        max_dd = _safe_max_drawdown(equity_arr)
        sharpe = _safe_sharpe(equity_arr)
        daily_limit = -0.05 * balance
        max_dd_limit = -0.10 * balance
        limit_triggered = any(
            (np.array(engine.equity_curve) - balance) <= daily_limit
        ) or max_dd <= -10.0
        survived = engine.balance > 0
        results[f"balance_{balance}"] = {
            "initial_balance": balance,
            "final_balance": round(engine.balance, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "trades": len(engine.trade_history),
            "limit_triggered": bool(limit_triggered),
            "survived": bool(survived),
        }
    logger.success(f"[STRESS] {event_name} complete")
    return {"event": event_name, "date": event_date, "symbol": symbol, "results": results}


async def run_all_stress_tests(symbol: str = "EURUSD") -> List[Dict]:
    results = []
    for event_name, event_date in STRESS_TEST_EVENTS.items():
        try:
            res = await run_stress_test_for_event(event_name, event_date, symbol)
            results.append(res)
        except Exception as e:
            logger.error(f"[STRESS] Failed for {event_name}: {e}")
    return results


def print_stress_report(results: List[Dict]):
    logger.info("=" * 70)
    logger.info(" STRESS TEST REPORT")
    logger.info("=" * 70)
    for res in results:
        event = res.get("event", "Unknown")
        date = res.get("date", "")
        symbol = res.get("symbol", "")
        logger.info(f"\nEvent: {event} ({date}) | Symbol: {symbol}")
        if "error" in res:
            logger.error(f"  Error: {res['error']}")
            continue
        for key, val in res.get("results", {}).items():
            logger.info(
                f"  {key}: initial=${val['initial_balance']:,.2f} -> "
                f"final=${val['final_balance']:,.2f} | DD={val['max_drawdown_pct']:.1f}% | "
                f"trades={val['trades']} | limit_triggered={val['limit_triggered']} | "
                f"survived={val['survived']}"
            )
    logger.info("=" * 70)


async def main():
    logger.info("Starting stress tests...")
    results = await run_all_stress_tests("EURUSD")
    print_stress_report(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())
