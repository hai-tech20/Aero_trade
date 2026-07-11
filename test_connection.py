from core.broker.mt5_bridge import mt5_bridge
from config.credentials import ACTIVE_CREDENTIALS

connected = mt5_bridge.connect(
    login=ACTIVE_CREDENTIALS['login'],
    password=ACTIVE_CREDENTIALS['password'],
    server=ACTIVE_CREDENTIALS['server'],
    demo=ACTIVE_CREDENTIALS['demo']
)

if connected:
    print("✅ Connected to FBS!")
    info = mt5_bridge.get_account_info()
    print(f"Account: {info.get('login')}")
    print(f"Balance: ${info.get('balance')}")
    print(f"Server: {info.get('server')}")
else:
    print("❌ Connection failed")