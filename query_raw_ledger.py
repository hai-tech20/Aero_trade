
"""
Query the raw SQLite ledger database and show all real trades!
"""

import sqlite3
from pathlib import Path

DB_PATH = Path("data/aero_trade_ledger.db")

if not DB_PATH.exists():
    print(f"Database {DB_PATH} doesn't exist yet!")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=== Raw Trades Table ===")
cursor.execute("SELECT * FROM trades")
rows = cursor.fetchall()
col_names = [desc[0] for desc in cursor.description]
print(f"Columns: {col_names}")
print("\nRows:")
for row in rows:
    print(row)

print("\n=== Closed Trades (as JSON-like) ===")
from core.ledger.trade_ledger import get_closed_trades, get_trade_summary
# Use your real account ID here (replace if you have one, use test-account-123 for now)
for account_id in set(r[1] for r in rows):
    print(f"\n--- Account {account_id} ---")
    print(get_closed_trades(account_id))
    print("Summary:", get_trade_summary(account_id))

conn.close()

