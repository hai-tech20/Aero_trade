"""Verify all data sources are reachable and returning data."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

RESULTS = []


def check(name: str, fn):
    try:
        result = fn()
        ok = result is not None and result != {} and result != []
        RESULTS.append((name, ok, result))
        status = "OK" if ok else "EMPTY/NEUTRAL"
        print(f"[{status}] {name}: {str(result)[:120]}")
    except Exception as e:
        RESULTS.append((name, False, str(e)))
        print(f"[FAIL] {name}: {e}")


print("=" * 60)
print("AERO TRADE — Data Source Verification")
print("=" * 60)

check("VIX", lambda: __import__("data.vix_source", fromlist=["get_current_vix"]).get_current_vix())
check("VIX high regime", lambda: __import__("data.vix_source", fromlist=["is_high_volatility_regime"]).is_high_volatility_regime())
check("FRED FEDFUNDS", lambda: __import__("data.fred_source", fromlist=["get_series_latest"]).get_series_latest("FEDFUNDS"))
check("FRED rates rising", lambda: __import__("data.fred_source", fromlist=["is_rates_rising"]).is_rates_rising("FEDFUNDS"))
check("COT EUR", lambda: __import__("data.cot_source", fromlist=["get_latest_cot_positioning"]).get_latest_cot_positioning("EURO FX"))
check("Marketaux EURUSD", lambda: __import__("data.marketaux_source", fromlist=["get_symbol_sentiment"]).get_symbol_sentiment("EURUSD"))
check("Finnhub sentiment", lambda: __import__("data.market_analysis", fromlist=["market_analysis"]).market_analysis.get_symbol_news_sentiment("EURUSD"))
check("Alpha Vantage sentiment", lambda: __import__("data.market_analysis", fromlist=["market_analysis"]).market_analysis.get_alpha_vantage_sentiment("EURUSD"))
check("FMP calendar", lambda: __import__("data.market_analysis", fromlist=["market_analysis"]).market_analysis.get_economic_calendar())
check("Binance BTC divergence", lambda: __import__("data.binance_data", fromlist=["check_btc_divergence"]).check_btc_divergence(95000.0))

print("=" * 60)
passed = sum(1 for _, ok, _ in RESULTS if ok)
print(f"Result: {passed}/{len(RESULTS)} sources returned usable data")
print("Note: EMPTY/NEUTRAL may indicate rate limits or missing API keys — check .env")
