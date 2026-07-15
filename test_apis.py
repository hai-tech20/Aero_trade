import os
from pathlib import Path

env_path = Path(".env")
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass

print("=== API Keys Check ===")
print(f"FINNHUB_API_KEY: {'SET' if os.getenv('FINNHUB_API_KEY') else 'NOT SET'}")
print(f"ALPHA_VANTAGE_API_KEY: {'SET' if os.getenv('ALPHA_VANTAGE_API_KEY') else 'NOT SET'}")
print(f"FMP_API_KEY: {'SET' if os.getenv('FMP_API_KEY') else 'NOT SET'}")

print("\n=== Testing Finnhub (general news endpoint) ===")
import requests
finnhub_key = os.getenv("FINNHUB_API_KEY", "")
if finnhub_key:
    resp = requests.get("https://finnhub.io/api/v1/news",
                      params={"category": "forex", "token": finnhub_key},
                      timeout=10)
    print(f"Finnhub status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ Finnhub API key is valid! Retrieved {len(data)} news articles!")
    else:
        print(f"❌ Finnhub API key is invalid/expired! Response: {resp.text[:200]}")
else:
    print("❌ FINNHUB_API_KEY not set!")

print("\n=== Testing Alpha Vantage ===")
av_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
if av_key:
    resp = requests.get("https://www.alphavantage.co/query",
                      params={"function": "NEWS_SENTIMENT", "tickers": "EUR", "apikey": av_key, "limit": 5},
                      timeout=15)
    print(f"Alpha Vantage status: {resp.status_code}")
    data = resp.json()
    if "Note" in data or "Information" in data:
        print("⚠️ Alpha Vantage rate limited!")
    elif resp.status_code == 200:
        print("✅ Alpha Vantage API key is valid!")
    else:
        print(f"❌ Alpha Vantage API key is invalid/expired!")
else:
    print("❌ ALPHA_VANTAGE_API_KEY not set!")

print("\n=== Testing FMP (stable endpoint) ===")
fmp_key = os.getenv("FMP_API_KEY", "")
if fmp_key:
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    resp = requests.get("https://financialmodelingprep.com/stable/economic-calendar",
                      params={"from": today, "to": end, "apikey": fmp_key},
                      timeout=15)
    print(f"FMP status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ FMP API key is valid! Retrieved {len(data)} economic events!")
    else:
        print(f"❌ FMP API key is invalid/expired! Response: {resp.text[:200]}")
else:
    print("❌ FMP_API_KEY not set!")
