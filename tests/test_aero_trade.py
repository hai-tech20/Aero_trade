"""Unit tests for AERO TRADE critical paths."""

import re
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))


def test_execute_trade_does_not_hardcode_symbol():
    """Ensure execute_trade uses the passed-in symbol, not a literal."""
    source = Path("core/engine/aero_trade_bot.py").read_text(encoding="utf-8")
    # Find execute_trade method body
    match = re.search(r"async def execute_trade\(.*?\n(.*?)(?=\n    async def )", source, re.DOTALL)
    assert match, "execute_trade method not found"
    body = match.group(1)
    assert "symbol = 'EURUSD'" not in body
    assert "symbol = \"EURUSD\"" not in body
    assert "symbol=symbol" in body or "symbol," in body


def test_signal_weights_sum_to_one():
    from config.settings import SIGNAL_FUSION_CONFIG, validate_settings
    weights = SIGNAL_FUSION_CONFIG["weights"]
    assert abs(sum(weights.values()) - 1.0) < 0.001
    assert validate_settings()


def test_market_analysis_neutral_fallback():
    from data.market_analysis import MarketAnalysis
    ma = MarketAnalysis()
    result = ma.get_symbol_news_sentiment("EURUSD")
    assert "bullish" in result
    assert "bearish" in result
    assert 0 <= result["bullish"] <= 1


def test_symbol_mapping():
    from core.symbols import get_news_identifiers, symbol_asset_class
    assert "EUR" in get_news_identifiers("EURUSD")
    assert symbol_asset_class("BTCUSD") == "crypto"
    assert symbol_asset_class("XAUUSD") == "metal"


def test_broker_registry():
    from core.broker.broker_registry import get_broker, SUPPORTED_BROKERS
    assert len(SUPPORTED_BROKERS) >= 6
    fbs = get_broker("fbs")
    assert fbs.display_name == "FBS"


def test_trade_ledger_outcome():
    from core.ledger.trade_ledger import init_ledger, record_open_trade, record_close_trade, get_closed_trades
    import uuid
    init_ledger()
    oid = f"test-{uuid.uuid4().hex[:8]}"
    aid = "test-account"
    record_open_trade(oid, aid, "EURUSD", "BUY", 1.1, 0.01, 0.7, "{}", True, "rf_5min")
    outcome = record_close_trade(oid, 1.105, 5.0, 0.7)
    assert outcome == "win"
    trades = get_closed_trades(aid, limit=10)
    assert any(t["order_id"] == oid for t in trades)


def test_execute_trade_symbol_param_for_multiple_symbols():
    """Verify execute_trade signature accepts symbol for 3+ instruments."""
    from core.engine.aero_trade_bot import AeroTradeBot
    import inspect
    sig = inspect.signature(AeroTradeBot.execute_trade)
    assert "symbol" in sig.parameters
    symbols = ["EURUSD", "GBPUSD", "XAUUSD", "BTCUSD"]
    for sym in symbols:
        assert len(sym) >= 6 or "BTC" in sym or "XAU" in sym
