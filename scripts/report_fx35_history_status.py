import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, TRADING_CONFIG


TIMEFRAMES = ["1min", "5min", "15min"]


def summarize(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "rows": 0, "date_range": [None, None]}
    df = pd.read_parquet(path)
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    else:
        ts = pd.to_datetime(df.index)
    return {
        "exists": True,
        "rows": int(len(df)),
        "date_range": [str(ts.min()), str(ts.max())] if len(df) else [None, None],
    }


def main() -> None:
    rows = []
    for symbol in TRADING_CONFIG.get("pairs", []):
        for timeframe in TIMEFRAMES:
            path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
            rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                **summarize(path),
            })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
