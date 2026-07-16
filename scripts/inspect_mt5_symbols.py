import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.broker.credential_store import list_accounts, load_broker_credentials
from core.broker.mt5_broker_client import MT5BrokerClient


async def main():
    accounts = list_accounts()
    if not accounts:
        print(json.dumps({"error": "no_accounts"}, indent=2))
        return

    acct = accounts[-1]
    creds = load_broker_credentials(acct["account_id"])
    client = MT5BrokerClient(acct["account_id"])
    ok = await client.connect_with_credentials(creds)
    payload = {
        "account_id": acct["account_id"],
        "server": acct.get("server"),
        "login": acct.get("login"),
        "connected": ok,
        "use_mt5": client.use_mt5,
        "symbols": [],
    }
    if client.mt5_bridge and client.connected:
        syms = client.mt5_bridge.get_all_symbols()
        payload["symbol_count"] = len(syms)
        payload["symbols"] = [
            s for s in syms
            if "BTC" in s.upper() or "XAU" in s.upper() or s.upper().startswith("GBPUSD") or s.upper().startswith("USDJPY")
        ]
        await client.disconnect()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
