"""AERO TRADE — entry point (delegates to AeroTradeBot)."""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.engine.aero_trade_bot import AeroTradeBot, main

# Backward compatibility
HybridTradingBot = AeroTradeBot

if __name__ == "__main__":
    asyncio.run(main())
