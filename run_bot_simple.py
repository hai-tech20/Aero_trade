"""
Simple script to run the trading bot
Press Ctrl+C to stop
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main import HybridTradingBot

async def main():
    print("=" * 60)
    print("  AERO TRADE — Starting...")
    print("=" * 60)
    print("\nNOTE: Bot runs in DEMO mode by default (safe for testing)")
    print("Press Ctrl+C to stop the bot\n")
    
    bot = HybridTradingBot()
    
    try:
        await bot.initialize()
        print(f"\n✓ Bot initialized!")
        print(f"✓ Account Balance: ${bot.account_info.get('balance', 0):,.2f}")
        print(f"✓ Starting trading loop...\n")
        
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Bot stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
