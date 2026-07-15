
import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

print("=== Section 1: Initial Training & Model Files ===")
model_dir = Path("ai_engine/models/saved")
if model_dir.exists():
    for f in model_dir.glob("*.joblib"):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        print(f"{f.name:<18} | Timestamp: {mtime} | Size: {f.stat().st_size} bytes")

data_dir = Path("data")
history_path = data_dir / "EURUSD_5min_history.parquet"
if history_path.exists():
    df = pd.read_parquet(history_path)
    print(f"\n✅ EURUSD_5min_history.parquet found: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
else:
    print("\n❌ EURUSD_5min_history.parquet NOT FOUND!")

print("\n=== Section 2: Database Tables ===")
db_path = Path("data/aero_trade_ledger.db")
if db_path.exists():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables in DB: {tables}")

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        print(f"  Table '{table}': {count} rows")
        if count > 0:
            cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
            col_names = [desc[0] for desc in cursor.description]
            print(f"    Columns: {col_names}")
            for row in cursor.fetchall():
                print(f"    {row}")
    conn.close()
else:
    print("❌ DB not found!")

print("\n=== Section 3: Feature Computation Check ===")
try:
    from ai_engine.models.predictor import get_ai_predictor
    from data.market_data import market_data

    predictor = get_ai_predictor()
    print("  Predictor initialized successfully")

    # Let's get some real market data
    import asyncio

    async def check_features():
        await market_data.connect()
        df = await market_data.get_latest_data("EURUSD", "5min", lookback=50)
        if df is not None and len(df) > 20:
            features = await predictor.prepare_features(df)
            print(f"\n  ✅ Feature vector computed! Shape: {features.shape}")
            print(f"  Features: {features.flatten().tolist()}")
        else:
            print("\n  ❌ Could not get market data for feature test")

    asyncio.run(check_features())
except Exception as e:
    print(f"\n  ❌ Error checking features: {e}")
    import traceback

    traceback.print_exc()

print("\n=== Done ===")
