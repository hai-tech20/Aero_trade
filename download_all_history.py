"""Download all historical data for enabled trading pairs."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from core.broker.mt5_broker_client import MT5BrokerClient
from core.broker.credential_store import list_accounts, load_broker_credentials
from data.history_downloader import download_all_for_bridge
from config.settings import TRADING_CONFIG, APP_NAME


async def main():
    logger.info(f"{APP_NAME}: Starting historical data download")
    accounts = list_accounts()
    if not accounts:
        logger.error("No accounts configured. Add an account via the API first.")
        return

    account_id = accounts[0]["account_id"]
    creds = load_broker_credentials(account_id)
    client = MT5BrokerClient(account_id)
    await client.connect_with_credentials(creds)

    if not client.mt5_bridge:
        logger.error("MT5 not connected")
        return

    download_all_for_bridge(client.mt5_bridge)
    await client.disconnect()
    logger.success("Historical download complete")


if __name__ == "__main__":
    asyncio.run(main())
