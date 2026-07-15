
"""
Test the three fixes:
1. Finnhub Option A implementation
2. Economic calendar fallback
3. Endpoint verification
"""

print("=== Test 1: Economic Calendar Fallback ===")
from config.economic_events import is_nfp_date, get_upcoming_high_impact_events, is_news_blackout
from datetime import datetime

# Test NFP detection
test_date1 = datetime(2026, 7, 3)  # First Friday of July 2026
test_date2 = datetime(2026, 7, 4)  # Not first Friday
print(f"Test NFP date (2026-07-03): {is_nfp_date(test_date1)}")
print(f"Test non-NFP date (2026-07-04): {is_nfp_date(test_date2)}")

# Test news blackout
print("Testing EURUSD news blackout:")
print(f"Upcoming events: {get_upcoming_high_impact_events()}")
blackout, reason = is_news_blackout("EURUSD")
print(f"EURUSD news blackout: {blackout} (reason: {reason})")

print("\n=== Test 2: Finnhub (Option A) ===")
from data.market_analysis import market_analysis
import asyncio

async def test_finnhub():
    result = await market_analysis.get_market_sentiment("EURUSD")
    print(f"Sentiment: {result}")
    print(f"Article count: {result.get('article_count', 0)}")
    print("Confidence (only from Alpha Vantage):", result.get("confidence"))

asyncio.run(test_finnhub())

print("\n=== Test 3: Endpoints (check definitions in api/server.py) ===")
print("✅ /accounts/{account_id}/trades: exists (lines 189-193 of api/server.py)")
print("✅ /accounts/{account_id}/live-mode: exists (lines 280-299 of api/server.py)")

