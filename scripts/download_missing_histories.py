import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, TRADING_CONFIG
from core.broker.credential_store import list_accounts, load_broker_credentials
from core.broker.mt5_broker_client import MT5BrokerClient
from data.history_downloader import download_full_history

TIMEFRAMES = ["1min", "5min", "15min"]


def needs_download(symbol: str, timeframe: str) -> bool:
    return not (DATA_DIR / f"{symbol}_{timeframe}_history.parquet").exists()


async def main():
    accounts = list_accounts()
    if not accounts:
        print(json.dumps({"error": "no_accounts"}, indent=2))
        return

    acct = accounts[-1]
    creds = load_broker_credentials(acct["account_id"])
    client = MT5BrokerClient(acct["account_id"])
    ok = await client.connect_with_credentials(creds)
    report = {
        "account_id": acct["account_id"],
        "connected": ok,
        "use_mt5": client.use_mt5,
        "downloads": [],
    }

    if not ok or not client.mt5_bridge:
        print(json.dumps(report, indent=2))
        return

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
