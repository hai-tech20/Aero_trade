import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, TRADING_CONFIG, DATA_CONFIG
from core.broker.credential_store import list_accounts, load_broker_credentials
from core.broker.mt5_broker_client import MT5BrokerClient
from data.history_downloader import download_full_history

TIMEFRAMES = ["1min", "5min", "15min", "30min", "1h", "4h", "1d"]


def _expected_history_cutoff() -> datetime:
    years = int(DATA_CONFIG.get("history_years", 15))
    return datetime.now() - timedelta(days=365 * years)


def _expected_min_rows_for_timeframe(timeframe: str) -> int:
    thresholds = {
        "1min": 200000,
        "5min": 50000,
        "15min": 20000,
        "30min": 15000,
        "1h": 90000,
        "4h": 20000,
        "1d": 3600,
    }
    return thresholds.get(timeframe, 2000)


def _history_file_complete(symbol: str, timeframe: str) -> bool:
    path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
    if not path.exists():
        return False

    try:
        df = pd.read_parquet(path)
    except Exception:
        return False

    if df is None or len(df) == 0:
        return False

    if "time" in df.columns:
        time_index = pd.to_datetime(df["time"])
    else:
        time_index = pd.to_datetime(df.index)

    if len(time_index) < 200:
        return False

    if len(time_index) < _expected_min_rows_for_timeframe(timeframe):
        return False

    earliest = time_index.min()
    if getattr(earliest, 'tzinfo', None) is not None:
        earliest = earliest.tz_convert(None)
    cutoff = _expected_history_cutoff()
    if earliest > cutoff + timedelta(days=30):
        return False

    return True


def needs_download(symbol: str, timeframe: str) -> bool:
    return not _history_file_complete(symbol, timeframe)


async def main():
    async def _connect_with_any_account() -> Optional[MT5BrokerClient]:
        accounts = list_accounts()
        if not accounts:
            print(json.dumps({"error": "no_accounts"}, indent=2))
            return None

        for acct in accounts:
            account_id = acct["account_id"]
            creds = load_broker_credentials(account_id)
            client = MT5BrokerClient(account_id)
            print(json.dumps({
                "trying_account": account_id,
                "login": creds.get("login"),
                "server": creds.get("server"),
            }, indent=2))
            ok = await client.connect_with_credentials(creds)
            if ok and client.mt5_bridge and client.mt5_bridge.connected:
                print(json.dumps({"using_account": account_id}, indent=2))
                return client
            await client.disconnect()

        print(json.dumps({"error": "no_valid_accounts"}, indent=2))
        return None

    client = await _connect_with_any_account()
    if not client:
        return

    report = {
        "account_id": client.account_id,
        "connected": client.connected,
        "use_mt5": client.use_mt5,
        "downloads": [],
    }

    symbols = []
    for symbol in TRADING_CONFIG.get("pairs", []):
        for timeframe in TIMEFRAMES:
            if needs_download(symbol, timeframe):
                symbols.append((symbol, timeframe))

    for symbol, timeframe in symbols:
        try:
            df = download_full_history(client.mt5_bridge, symbol, timeframe)
            report["downloads"].append({
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "downloaded" if df is not None and len(df) > 0 else "empty",
                "rows": 0 if df is None else int(len(df)),
            })
        except Exception as e:
            report["downloads"].append({
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "failed",
                "error": str(e),
            })

    await client.disconnect()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
