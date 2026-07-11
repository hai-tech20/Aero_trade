"""Test MT5 connection to FBS"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.credentials import ACTIVE_CREDENTIALS, USE_DEMO
from core.broker.mt5_bridge import MT5Bridge

print("Testing MT5 Connection to FBS...")
print("=" * 50)

creds = ACTIVE_CREDENTIALS
print(f"Account: {creds['login']}")
print(f"Server: {creds['server']}")
print(f"Demo: {USE_DEMO}")
print()

bridge = MT5Bridge()
success = bridge.connect(
    login=creds['login'],
    password=creds['password'],
    server=creds['server'],
    demo=USE_DEMO
)

if success:
    print("\n[SUCCESS] Connection successful!")
    account = bridge.get_account_info()
    print(f"  Balance: ${account.get('balance', 0):,.2f}")
    print(f"  Equity: ${account.get('equity', 0):,.2f}")
    print(f"  Leverage: 1:{account.get('leverage', 100)}")
    
    # Test getting a price
    price = bridge.get_market_price('EURUSD')
    if price:
        print(f"\n[SUCCESS] Market data working!")
        print(f"  EURUSD Bid: {price.get('bid', 0):.5f}")
        print(f"  EURUSD Ask: {price.get('ask', 0):.5f}")
        print(f"  Spread: {price.get('spread', 0):.1f} pips")
    
    bridge.disconnect()
    print("\n[READY] Ready to run the bot!")
else:
    print("\n✗ Connection failed!")
    print("  Make sure:")
    print("  1. MT5 terminal is installed")
    print("  2. MT5 terminal is running")
    print("  3. Your FBS account credentials are correct")
    print("  4. You can log into MT5 manually with these credentials")
