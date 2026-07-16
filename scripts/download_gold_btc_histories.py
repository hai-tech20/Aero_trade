import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.broker.credential_store import list_accounts, load_broker_credentials
from core.broker.mt5_broker_client import MT5BrokerClient
from data.history_downloader import download_full_history

TARGETS = [
    ("XAUUSD", "1min"),
    ("XAUUSD", "5min"),
    ("XAUUSD", "15min"),
    ("BTCUSD", "1min"),
    ("BTCUSD", "5min"),
    ("BTCUSD", "15min"),
]


async def main():
    accounts = list_accounts()
    acct = accounts[-1]
    creds = load_broker_credentials(acct["account_id"])
    client = MT5BrokerClient(acct["account_id"])
    ok = await client.connect_with_credentials(creds)
    report = []

    if not ok or not client.mt5_bridge:
        print(json.dumps({"connected": ok, "use_mt5": client.use_mt5, "report": report}, indent=2))
        return

    for symbol, timeframe in TARGETS:
        try:
            df = download_full_history(client.mt5_bridge, symbol, timeframe, chunk_size=1500, chunk_delay_seconds=0.75)
            if df is not None and len(df) > 0:
                report.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "downloaded",
                    "rows": int(len(df)),
                    "start": str(df.index.min()),
                    "end": str(df.index.max()),
                })
            else:
                report.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "empty",
                    "last_error": client.mt5_bridge._get_last_error_details(),
                })
        except Exception as e:
            report.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "failed",
                "error": str(e),
                "last_error": client.mt5_bridge._get_last_error_details(),
            })

    await client.disconnect()
    print(json.dumps({"connected": ok, "use_mt5": client.use_mt5, "report": report}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
