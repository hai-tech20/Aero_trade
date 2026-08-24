"""Download all historical data for enabled trading pairs."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from core.broker.mt5_broker_client import MT5BrokerClient
from core.broker.credential_store import list_accounts, load_broker_credentials
from data.history_downloader import download_all_for_bridge
from config.settings import TRADING_CONFIG, APP_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Download full MT5 history for configured trading pairs")
    parser.add_argument("--years", type=int, default=None, help="Number of years of history to download")
    parser.add_argument("--pairs", nargs="*", help="Subset of pairs to download (default: configured pairs)")
    parser.add_argument("--timeframes", nargs="*", help="Subset of timeframes to download (default: configured timeframes)")
    return parser.parse_args()


async def _connect_with_any_account() -> Optional[MT5BrokerClient]:
    accounts = list_accounts()
    if not accounts:
        logger.error("No accounts configured. Add an account via the API first.")
        return None

    for acct in accounts:
        account_id = acct["account_id"]
        creds = load_broker_credentials(account_id)
        client = MT5BrokerClient(account_id)
        logger.info(f"Trying MT5 account {account_id} login={creds.get('login')} server={creds.get('server')}")
        connected = await client.connect_with_credentials(creds)
        if connected and client.mt5_bridge and client.mt5_bridge.connected:
            logger.success(f"Using MT5 account {account_id}")
            return client
        await client.disconnect()

    logger.error("No MT5 account could connect; aborting historical download")
    return None


async def main():
    args = parse_args()
    logger.info(f"{APP_NAME}: Starting historical data download")
    client = await _connect_with_any_account()
    if not client:
        return

    download_all_for_bridge(
        client.mt5_bridge,
        pairs=args.pairs or None,
        timeframes=args.timeframes or None,
        years=args.years,
    )
    await client.disconnect()
    logger.success("Historical download complete")


if __name__ == "__main__":
    asyncio.run(main())
