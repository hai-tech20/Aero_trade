"""Simple UTC-based trading session favorability check."""

from datetime import datetime, timezone

# Approximate real-world session windows in UTC
SESSIONS = {
    "sydney": (21, 6),
    "tokyo": (0, 9),
    "london": (7, 16),
    "new_york": (12, 21),
}

# Which sessions matter most per currency
CURRENCY_SESSIONS = {
    "AUD": ["sydney"],
    "NZD": ["sydney"],
    "JPY": ["tokyo"],
    "EUR": ["london"],
    "GBP": ["london"],
    "CHF": ["london"],
    "USD": ["new_york", "london"],
    "CAD": ["new_york"],
    "ZAR": ["london"],
    "MXN": ["new_york"],
    "SGD": ["tokyo"],
    "NOK": ["london"],
    "SEK": ["london"],
    "XAU": ["london", "new_york"],
    "BTC": None,
}


def _in_window(hour: int, window: tuple[int, int]) -> bool:
    start, end = window
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def is_favorable_session(symbol: str) -> bool:
    """True if current UTC time is within a favorable session for this symbol."""
    hour = datetime.now(timezone.utc).hour
    base = symbol[:3].upper()
    quote = symbol[3:6].upper() if len(symbol) >= 6 else ""

    relevant_sessions = set()
    for code in (base, quote):
        if code == "BTC":
            return True
        sessions_for_code = CURRENCY_SESSIONS.get(code)
        if sessions_for_code:
            relevant_sessions.update(sessions_for_code)

    if not relevant_sessions:
        return True

    return any(_in_window(hour, SESSIONS[s]) for s in relevant_sessions)
PYEOF:any
Output:any

"""Simple UTC-based trading session favorability check."""

from datetime import datetime, timezone

# Approximate real-world session windows in UTC
SESSIONS = {
    "sydney": (21, 6),
    "tokyo": (0, 9),
    "london": (7, 16),
    "new_york": (12, 21),
}

# Which sessions matter most per currency
CURRENCY_SESSIONS = {
    "AUD": ["sydney"],
    "NZD": ["sydney"],
    "JPY": ["tokyo"],
    "EUR": ["london"],
    "GBP": ["london"],
    "CHF": ["london"],
    "USD": ["new_york", "london"],
    "CAD": ["new_york"],
    "ZAR": ["london"],
    "MXN": ["new_york"],
    "SGD": ["tokyo"],
    "NOK": ["london"],
    "SEK": ["london"],
    "XAU": ["london", "new_york"],
    "BTC": None,
}


def _in_window(hour: int, window: tuple[int, int]) -> bool:
    start, end = window
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def is_favorable_session(symbol: str) -> bool:
    """True if current UTC time is within a favorable session for this symbol."""
    hour = datetime.now(timezone.utc).hour
    base = symbol[:3].upper()
    quote = symbol[3:6].upper() if len(symbol) >= 6 else ""

    relevant_sessions = set()
    for code in (base, quote):
        if code == "BTC":
            return True
        sessions_for_code = CURRENCY_SESSIONS.get(code)
        if sessions_for_code:
            relevant_sessions.update(sessions_for_code)

    if not relevant_sessions:
        return True

    return any(_in_window(hour, SESSIONS[s]) for s in relevant_sessions)
