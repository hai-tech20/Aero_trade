
"""
Test the trade endpoint by adding a closed trade to the ledger!
"""

from core.ledger.trade_ledger import record_open_trade, record_close_trade
from datetime import datetime
import random

# First, let's get a test account ID (from existing or make one temporarily)
account_id = "test-account-123"

# 1. Record an open trade
order_id = str(random.randint(10000, 99999))
symbol = "NZDCAD"
order_type = "BUY"
entry_price = 0.8345
size = 0.01
confidence = 0.6
features_json = "{}"
sentiment_agreed = True
model_used = "fusion"
record_open_trade(order_id, account_id, symbol, order_type, entry_price, size, confidence, features_json, sentiment_agreed, model_used)
print(f"Recorded open trade {order_id} for {symbol}")

# 2. Close the trade with a small win
pnl = 12.50  # Small win
exit_price = 0.8360
outcome = record_close_trade(order_id, exit_price, pnl, confidence)
print(f"Closed trade {order_id} with outcome: {outcome}, P&L: {pnl}")

# 3. Call get_closed_trades to check
from core.ledger.trade_ledger import get_closed_trades, get_trade_summary
trades = get_closed_trades(account_id)
summary = get_trade_summary(account_id)
print("\n=== Closed Trades (JSON-like) ===")
for t in trades:
    print(t)
print("\n=== Summary (JSON-like) ===")
print(summary)

