"""
Run the bot with REAL FBS connection via MT5
Positions will appear in your MT5 app/phone
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main import HybridTradingBot

async def main():
    print("=" * 70)
    print("  FBS AI TRADING BOT - REAL CONNECTION")
    print("=" * 70)
    print("\n⚠️  IMPORTANT:")
    print("  - Bot will connect to your FBS DEMO account")
    print("  - Trades will appear in your MT5 app/phone")
    print("  - Make sure MT5 terminal is installed and running")
    print("  - Press Ctrl+C to stop the bot\n")
    
    bot = HybridTradingBot()
    
    try:
        print("[1/2] Initializing bot and connecting to FBS...")
        await bot.initialize()
        
        if bot.account_info:
            print(f"\n✓ Connected successfully!")
            print(f"  Account: #{bot.account_info.get('login', 'N/A')}")
            print(f"  Balance: ${bot.account_info.get('balance', 0):,.2f}")
            print(f"  Equity: ${bot.account_info.get('equity', 0):,.2f}")
            print(f"  Server: {bot.account_info.get('server', 'N/A')}")
            print(f"  Leverage: 1:{bot.account_info.get('leverage', 100)}")
        else:
            print("\n⚠️  Warning: Could not get account info")
        
        print("\n[2/2] Starting trading loop...")
        print("  The bot will now:")
        print("  - Monitor market data")
        print("  - Make AI predictions")
        print("  - Execute trades when confidence is high")
        print("  - All trades will appear in your MT5 app!\n")
        print("=" * 70)
        print("Bot is running... Press Ctrl+C to stop\n")
        
        await bot.run()
        
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Bot stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        await bot.shutdown()
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
