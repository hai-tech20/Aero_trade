
"""
Free, zero-API-cost fallback for high-impact economic events (USD only, manual updates).
Update quarterly from the Fed's published calendar: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
"""

from datetime import datetime, timedelta
from typing import List, Dict


KNOWN_HIGH_IMPACT_EVENTS_2026 = [
    {"name": "FOMC Rate Decision", "date": "2026-07-29", "currencies": ["USD"], "time": "14:00", "timezone": "US/Eastern"},
    {"name": "FOMC Rate Decision", "date": "2026-09-17", "currencies": ["USD"], "time": "14:00", "timezone": "US/Eastern"},
    {"name": "FOMC Rate Decision", "date": "2026-11-04", "currencies": ["USD"], "time": "14:00", "timezone": "US/Eastern"},
    {"name": "FOMC Rate Decision", "date": "2026-12-16", "currencies": ["USD"], "time": "14:00", "timezone": "US/Eastern"},
]


def is_nfp_date(check_datetime: datetime) -> bool:
    """
    Check if the given datetime is NFP release day (first Friday of the month at 8:30am ET).
    Returns True if it's the first Friday of the month, regardless of time (for simplicity).
    """
    first_day = check_datetime.replace(day=1)
    # Calculate first Friday: weekday() is 0=Monday, 4=Friday
    days_to_first_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_to_first_friday)
    return check_datetime.date() == first_friday.date()


def get_upcoming_high_impact_events(check_window_minutes: int = 15) -> List[Dict]:
    """
    Get list of high-impact events within the next check_window_minutes.
    Uses manual list for FOMC plus NFP detection.
    """
    now_utc = datetime.utcnow()
    # Simplify: assume events are in UTC for now (or use ET conversion)
    upcoming = []

    # Check known events
    for event in KNOWN_HIGH_IMPACT_EVENTS_2026:
        try:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            # For simplicity, treat event date as entire day (or add time if needed)
            if abs((event_date.date() - now_utc.date()).days) <= 1:
                upcoming.append(event)
        except Exception:
            pass

    # Check NFP
    if is_nfp_date(now_utc):
        upcoming.append({
            "name": "US Non-Farm Payrolls (NFP)",
            "date": now_utc.strftime("%Y-%m-%d"),
            "currencies": ["USD"],
            "time": "08:30",
            "timezone": "US/Eastern"
        })

    return upcoming


def is_news_blackout(symbol: str, window_minutes: int = 15) -> tuple:
    """
    Check if any high-impact event is upcoming that affects the symbol's currencies.
    Returns (is_blackout: bool, reason: str)
    """
    from core.symbols import get_currencies_from_symbol
    symbol_currencies = get_currencies_from_symbol(symbol)
    events = get_upcoming_high_impact_events(window_minutes)

    for event in events:
        event_currencies = event.get("currencies", [])
        # If any currency in symbol is affected by event
        if any(curr in symbol_currencies for curr in event_currencies):
            return True, f"High-impact event upcoming: {event['name']} ({event['date']})"

    return False, ""

