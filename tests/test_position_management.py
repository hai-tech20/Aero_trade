"""Tests for position management, attribution, and self-analysis."""

import sys
from pathlib import Path
import uuid

import pytest

sys.path.append(str(Path(__file__).parent.parent))


def test_max_positions_is_18():
    from config.settings import TRADING_CONFIG
    assert TRADING_CONFIG["max_positions"] == 18


def test_allow_early_close_defaults_off():
    from config.settings import RISK_CONFIG
    assert RISK_CONFIG.get("allow_early_close") is False


def test_close_attribution_sl():
    from core.ledger.close_attribution import classify_close, TRAINABLE_CLOSED_BY
    pos = {"direction": "BUY", "stop_loss": 1.1000, "take_profit": 1.1100}
    deal = {"reason": 4, "price": 1.1000, "comment": ""}  # DEAL_REASON_SL
    closed_by, reason = classify_close(pos, deal)
    assert closed_by == "bot_sl"
    assert closed_by in TRAINABLE_CLOSED_BY


def test_close_attribution_user_manual():
    from core.ledger.close_attribution import classify_close, TRAINABLE_CLOSED_BY
    pos = {"direction": "BUY", "stop_loss": 1.0900, "take_profit": 1.1100}
    deal = {"reason": 0, "price": 1.1050, "comment": "manual"}  # DEAL_REASON_CLIENT
    closed_by, _ = classify_close(pos, deal)
    assert closed_by == "user_manual_close"
    assert closed_by not in TRAINABLE_CLOSED_BY


def test_ledger_closed_by_columns():
    from core.ledger.trade_ledger import init_ledger, record_open_trade, record_close_trade, get_closed_trades
    init_ledger()
    oid = f"test-{uuid.uuid4().hex[:8]}"
    aid = "test-attribution"
    record_open_trade(oid, aid, "EURUSD", "BUY", 1.1, 0.01, 0.7, "{}", True, "rf_5min", timeframe="5min")
    record_close_trade(oid, 1.105, 5.0, 0.7, closed_by="bot_tp", close_reason="Take profit hit")
    trades = get_closed_trades(aid, limit=5)
    t = next(x for x in trades if x["order_id"] == oid)
    assert t["closed_by"] == "bot_tp"
    assert "Take profit" in t["close_reason"]


def test_training_excludes_user_manual():
    from core.ledger.trade_ledger import (
        init_ledger, record_open_trade, record_close_trade, get_recent_closed_trades_for_training,
    )
    init_ledger()
    oid_bot = f"bot-{uuid.uuid4().hex[:6]}"
    oid_user = f"user-{uuid.uuid4().hex[:6]}"
    for oid, cb in [(oid_bot, "bot_sl"), (oid_user, "user_manual_close")]:
        record_open_trade(oid, "train-test", "EURUSD", "BUY", 1.1, 0.01, 0.7, "[1,2,3]", True, "rf")
        record_close_trade(oid, 1.1, -1.0, 0.7, closed_by=cb, close_reason="test")
    training = get_recent_closed_trades_for_training(limit=100)
    ids = {t["order_id"] for t in training}
    assert oid_bot in ids
    assert oid_user not in ids


def test_self_analysis_structure():
    from core.ledger.trade_ledger import build_self_analysis
    report = build_self_analysis("nonexistent-account", days=30)
    assert "overall" in report
    assert "by_symbol" in report
    assert "by_timeframe" in report
    assert "flags" in report
    assert "excluded_from_analysis" in report


def test_floating_pnl_sizing_reduction():
    from core.risk_management.risk_manager import risk_manager
    normal = risk_manager.calculate_dynamic_position_size(
        10000, 50, 0.7, True, "EURUSD", {"volume_min": 0.01, "volume_step": 0.01, "volume_max": 100},
        floating_pnl=0, floating_pnl_pct=0,
    )
    underwater = risk_manager.calculate_dynamic_position_size(
        10000, 50, 0.7, True, "EURUSD", {"volume_min": 0.01, "volume_step": 0.01, "volume_max": 100},
        floating_pnl=-350, floating_pnl_pct=-3.5,
    )
    assert underwater < normal


def test_execute_trade_no_hardcoded_symbol():
    import inspect
    from core.engine.aero_trade_bot import AeroTradeBot
    sig = inspect.signature(AeroTradeBot.execute_trade)
    assert "symbol" in sig.parameters
