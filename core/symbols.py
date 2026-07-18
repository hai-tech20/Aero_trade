"""Symbol mapping and pip/tick utilities for FX, metals, and crypto."""

import re
from typing import Dict, List

# MT5 symbol -> Finnhub/Alpha Vantage query identifiers
SYMBOL_NEWS_MAP: Dict[str, List[str]] = {
    "EURUSD": ["EUR", "USD", "FOREX:EURUSD"],
    "GBPUSD": ["GBP", "USD", "FOREX:GBPUSD"],
    "USDJPY": ["USD", "JPY", "FOREX:USDJPY"],
    "AUDUSD": ["AUD", "USD", "FOREX:AUDUSD"],
    "USDCAD": ["USD", "CAD", "FOREX:USDCAD"],
    "USDCHF": ["USD", "CHF", "FOREX:USDCHF"],
    "NZDUSD": ["NZD", "USD", "FOREX:NZDUSD"],
    "EURCAD": ["EUR", "CAD", "FOREX:EURCAD"],
    "GBPCHF": ["GBP", "CHF", "FOREX:GBPCHF"],
    "GBPJPY": ["GBP", "JPY", "FOREX:GBPJPY"],
    "EURGBP": ["EUR", "GBP", "FOREX:EURGBP"],
    "EURJPY": ["EUR", "JPY", "FOREX:EURJPY"],
    "EURAUD": ["EUR", "AUD", "FOREX:EURAUD"],
    "EURNZD": ["EUR", "NZD", "FOREX:EURNZD"],
    "EURCHF": ["EUR", "CHF", "FOREX:EURCHF"],
    "GBPAUD": ["GBP", "AUD", "FOREX:GBPAUD"],
    "GBPCAD": ["GBP", "CAD", "FOREX:GBPCAD"],
    "GBPNZD": ["GBP", "NZD", "FOREX:GBPNZD"],
    "AUDJPY": ["AUD", "JPY", "FOREX:AUDJPY"],
    "NZDJPY": ["NZD", "JPY", "FOREX:NZDJPY"],
    "CADJPY": ["CAD", "JPY", "FOREX:CADJPY"],
    "CHFJPY": ["CHF", "JPY", "FOREX:CHFJPY"],
    "AUDCAD": ["AUD", "CAD", "FOREX:AUDCAD"],
    "AUDNZD": ["AUD", "NZD", "FOREX:AUDNZD"],
    "AUDCHF": ["AUD", "CHF", "FOREX:AUDCHF"],
    "NZDCAD": ["NZD", "CAD", "FOREX:NZDCAD"],
    "NZDCHF": ["NZD", "CHF", "FOREX:NZDCHF"],
    "CADCHF": ["CAD", "CHF", "FOREX:CADCHF"],
    "USDZAR": ["USD", "ZAR", "FOREX:USDZAR"],
    "USDMXN": ["USD", "MXN", "FOREX:USDMXN"],
    "USDSGD": ["USD", "SGD", "FOREX:USDSGD"],
    "USDNOK": ["USD", "NOK", "FOREX:USDNOK"],
    "USDSEK": ["USD", "SEK", "FOREX:USDSEK"],
    "XAUUSD": ["gold", "XAU", "FOREX:XAUUSD"],
    "BTCUSD": ["bitcoin", "BTC", "CRYPTO:BTC"],
}


def normalize_symbol(symbol: str) -> str:
    """Strip broker suffixes like .m, m, #."""
    s = symbol.upper()
    s = re.sub(r"[.#]$", "", s)
    s = re.sub(r"M$", "", s)
    return s


def get_news_identifiers(symbol: str) -> List[str]:
    base = normalize_symbol(symbol)
    if base in SYMBOL_NEWS_MAP:
        return SYMBOL_NEWS_MAP[base]
    # Generic: split 6-char FX pair
    if len(base) == 6 and base.isalpha():
        return [base[:3], base[3:], f"FOREX:{base}"]
    if base.startswith("XAU"):
        return ["gold", "XAU"]
    if "BTC" in base:
        return ["bitcoin", "BTC"]
    return [base]


def get_currencies_from_symbol(symbol: str) -> List[str]:
    base = normalize_symbol(symbol)
    if base.startswith("XAU"):
        return ["USD", "XAU"]
    if "BTC" in base:
        return ["USD", "BTC"]
    if len(base) >= 6:
        return [base[:3], base[3:6]]
    return [base]


def symbol_asset_class(symbol: str) -> str:
    base = normalize_symbol(symbol)
    if "BTC" in base or "ETH" in base:
        return "crypto"
    if base.startswith("XAU") or base.startswith("XAG"):
        return "metal"
    return "forex"


def get_pip_size(symbol: str, point: float = 0.00001) -> float:
    """Return price increment for one pip."""
    cls = symbol_asset_class(symbol)
    if cls == "crypto":
        return 1.0
    if cls == "metal":
        return 0.01
    # Standard FX: 4-decimal pairs use 0.0001, JPY pairs use 0.01
    base = normalize_symbol(symbol)
    if "JPY" in base:
        return 0.01
    return 0.0001


def price_distance_to_pips(symbol: str, distance: float) -> float:
    pip = get_pip_size(symbol)
    if pip <= 0:
        return distance
    return abs(distance) / pip


def pips_to_price_distance(symbol: str, pips: float) -> float:
    return pips * get_pip_size(symbol)


def get_tick_value_from_broker(symbol_info: dict, default: float = 10.0) -> float:
    """Extract pip/tick value per lot from MT5 symbol_info dict."""
    if not symbol_info:
        return default
    tick_value = symbol_info.get("trade_tick_value", 0)
    tick_size = symbol_info.get("trade_tick_size", 0)
    if tick_value and tick_size:
        return tick_value / tick_size * get_pip_size(symbol_info.get("name", ""))
    return default
