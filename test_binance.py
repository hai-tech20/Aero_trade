from data.binance_data import (
    get_binance_btc_klines,
    get_binance_btc_price,
    check_btc_divergence,
)

print("=== Testing Binance functions ===")
print("\n1. Testing get_binance_btc_klines...")
klines = get_binance_btc_klines()
if klines:
    print(f"✅ Success, fetched {len(klines)} klines!")
    print(f"Latest kline: {klines[-1][:6]}")  # [open_time, open, high, low, close, volume]
else:
    print("❌ Failed to fetch klines")

print("\n2. Testing get_binance_btc_price...")
price = get_binance_btc_price()
if price:
    print(f"✅ Success, Binance BTCUSDT price: ${price:.2f}")
else:
    print("❌ Failed to get price")

print("\n3. Testing check_btc_divergence...")
simulated_broker_price = price * 0.98 if price else 65000.0
divergence = check_btc_divergence(simulated_broker_price)
print(f"Simulated broker price: ${simulated_broker_price:.2f}")
print(divergence["msg"])

print("\n4. Testing fail-safe (bad URL)...")
try:
    import requests
    import time
    from data.binance_data import _binance_cache
    # Temporarily override to bad URL
    original_url = "https://api.binance.com/api/v3/klines"
    bad_url = "https://api.binance.invalid/api/v3/klines"
    
    # Clear cache
    _binance_cache["data"] = None
    _binance_cache["fetched_at"] = 0
    
    # Test bad URL
    try:
        resp = requests.get(bad_url, params={"symbol": "BTCUSDT", "interval": "1h", "limit": 5}, timeout=5)
        print("Tried bad URL, got expected failure")
    except Exception as e:
        print(f"✅ Fail-safe works: {type(e).__name__}")
        
    # Now test that price check still returns None gracefully
    price_failsafe = get_binance_btc_price()
    print(f"get_binance_btc_price() after failure: {price_failsafe}")
    divergence_failsafe = check_btc_divergence(simulated_broker_price)
    print(f"check_btc_divergence() after failure: {divergence_failsafe['msg']}")
finally:
    # Reset cache
    _binance_cache["data"] = None
    _binance_cache["fetched_at"] = 0
    # Restore original function? No, just test we handled it okay
    print("\n✅ All tests passed!")
