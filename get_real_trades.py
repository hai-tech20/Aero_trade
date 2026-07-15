
import sqlite3
from pathlib import Path

db_path = Path("data/aero_trade_ledger.db")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check all trades where account id is NOT test-account or test-account-123
    print("=== Non-test trades in ledger")
    cursor.execute("SELECT * FROM trades WHERE account_id NOT LIKE 'test%';")
    col_names = [desc[0] for desc in cursor.description]
    print(f"Columns: {col_names}")
    for row in cursor.fetchall():
        print(row)
        # Also decode features_json if exists
        if row[col_names.index("features_json")] != "{}":
            print(f"Features JSON:", row[col_names.index("features_json")])
    conn.close()

# Check git status/commit
print("\n=== Git Status:")
import subprocess

try:
    result = subprocess.run(["git", "status"], capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(e)
