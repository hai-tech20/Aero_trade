"""Quick test to run the bot"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from main import HybridTradingBot

async def test_bot():
    print("=" * 50)
    print("Starting FBS AI Trading Bot")
    print("=" * 50)
    
    bot = HybridTradingBot()
    
    try:
        print("\n[1/4] Initializing bot...")
        await bot.initialize()
        
        print("\n[2/4] Bot initialized successfully!")
        print(f"Account Balance: ${bot.account_info.get('balance', 0):,.2f}")
        print(f"Demo Mode: {bot.account_info.get('name', 'Unknown')}")
        
        print("\n[3/4] Running one trading iteration...")
        await bot.trading_iteration()
        
        print("\n[4/4] Test complete!")
        print("\nTo run the full bot, use: python main.py")
        
        await bot.shutdown()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(test_bot())
