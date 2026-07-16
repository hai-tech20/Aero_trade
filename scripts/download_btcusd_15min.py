import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.broker.credential_store import list_accounts, load_broker_credentials
from core.broker.mt5_broker_client import MT5BrokerClient
from data.history_downloader import download_full_history


async def main():
    acct = list_accounts()[-1]
    creds = load_broker_credentials(acct["account_id"])
    client = MT5BrokerClient(acct["account_id"])
    ok = await client.connect_with_credentials(creds)
    report: dict[str, object] = {"connected": ok, "use_mt5": client.use_mt5}
    if not ok or not client.mt5_bridge:
        print(json.dumps(report, indent=2))
        return
    try:
        df = download_full_history(client.mt5_bridge, "BTCUSD", "15min", chunk_size=1500, chunk_delay_seconds=0.75)
        report["rows"] = int(len(df)) if df is not None else 0
        report["start"] = str(df.index.min()) if df is not None and len(df) else None
        report["end"] = str(df.index.max()) if df is not None and len(df) else None
        report["last_error"] = client.mt5_bridge._get_last_error_details()
    except Exception as e:
        report["error"] = str(e)
        report["last_error"] = client.mt5_bridge._get_last_error_details()
    await client.disconnect()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
