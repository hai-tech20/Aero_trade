import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.broker.credential_store import list_accounts, load_broker_credentials
from core.broker.mt5_broker_client import MT5BrokerClient
from core.symbols import normalize_symbol
from data.history_downloader import download_full_history

DEFAULT_TARGET_PAIRS = [
    "EURCAD", "GBPCHF", "GBPJPY", "USDCAD",
    "AUDUSD", "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "EURAUD", "EURNZD", "EURCHF",
    "GBPAUD", "GBPCAD", "GBPNZD", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    "AUDCAD", "AUDNZD", "AUDCHF", "NZDCAD", "NZDCHF", "CADCHF",
    "USDZAR", "USDMXN", "USDSGD", "USDNOK", "USDSEK",
]
TIMEFRAMES = ["1min", "5min", "15min"]


def parse_targets(argv: list[str]) -> list[str]:
    return argv if argv else DEFAULT_TARGET_PAIRS


def resolve_broker_symbol(target: str, broker_symbols: List[str]) -> Optional[str]:
    target_upper = target.upper()
    exact = next((symbol for symbol in broker_symbols if symbol.upper() == target_upper), None)
    if exact:
        return exact

    normalized_matches = [symbol for symbol in broker_symbols if normalize_symbol(symbol) == target_upper]
    if normalized_matches:
        normalized_matches.sort(key=lambda symbol: (len(symbol), symbol))
        return normalized_matches[0]

    prefix_matches = [symbol for symbol in broker_symbols if symbol.upper().startswith(target_upper)]
    if prefix_matches:
        prefix_matches.sort(key=lambda symbol: (len(symbol), symbol))
        return prefix_matches[0]

    return None


def summarize_history(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None or len(df) == 0:
        return {"rows": 0, "date_range": [None, None]}
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    else:
        ts = pd.to_datetime(df.index)
    return {
        "rows": int(len(df)),
        "date_range": [str(ts.min()), str(ts.max())],
    }


async def main() -> None:
    accounts = list_accounts()
    if not accounts:
        print(json.dumps({"error": "no_accounts"}, indent=2))
        return

    targets = parse_targets(sys.argv[1:])

    acct = accounts[-1]
    creds = load_broker_credentials(acct["account_id"])
    client = MT5BrokerClient(acct["account_id"])
    ok = await client.connect_with_credentials(creds)

    report: Dict[str, Any] = {
        "account_id": acct["account_id"],
        "server": acct.get("server"),
        "connected": ok,
        "use_mt5": client.use_mt5,
        "symbol_resolution": [],
        "downloads": [],
    }

    if not ok or not client.mt5_bridge:
        print(json.dumps(report, indent=2))
        return

    broker_symbols = client.mt5_bridge.get_all_symbols()
    resolved_map: Dict[str, str] = {}

    for target in targets:
        resolved = resolve_broker_symbol(target, broker_symbols)
        report["symbol_resolution"].append(
            {
                "requested": target,
                "resolved": resolved,
                "status": "resolved" if resolved else "missing",
            }
        )
        if resolved:
            resolved_map[target] = resolved

    for requested, resolved in resolved_map.items():
        for timeframe in TIMEFRAMES:
            try:
                df = download_full_history(client.mt5_bridge, resolved, timeframe)
                summary = summarize_history(df)
                report["downloads"].append(
                    {
                        "requested_symbol": requested,
                        "broker_symbol": resolved,
                        "timeframe": timeframe,
                        "status": "downloaded" if summary["rows"] > 0 else "empty",
                        **summary,
                    }
                )
            except Exception as e:
                report["downloads"].append(
                    {
                        "requested_symbol": requested,
                        "broker_symbol": resolved,
                        "timeframe": timeframe,
                        "status": "failed",
                        "error": str(e),
                    }
                )

    await client.disconnect()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
