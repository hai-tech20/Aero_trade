import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from api.server import start_trading, get_activity, _bots

ACCOUNT_ID = "c1743f71-80e7-4c6e-88c5-1b9a715e90bd"


async def main() -> None:
    result = await start_trading(ACCOUNT_ID)
    print("START", json.dumps(result))
    await asyncio.sleep(20)
    bot = _bots.get(ACCOUNT_ID)
    symbols = bot.available_symbols if bot is not None else []
    print("SYMBOLS", json.dumps(symbols))
    print("ACTIVITY", json.dumps(get_activity(ACCOUNT_ID, limit=12), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
