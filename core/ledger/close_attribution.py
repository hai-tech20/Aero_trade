"""Determine who closed a trade and why — used by monitor_positions."""

from typing import Dict, Optional, Tuple

# MT5 deal reason codes (MetaTrader5 constants)
try:
    import MetaTrader5 as mt5
    REASON_SL = getattr(mt5, "DEAL_REASON_SL", 4)
    REASON_TP = getattr(mt5, "DEAL_REASON_TP", 5)
    REASON_SO = getattr(mt5, "DEAL_REASON_SO", 6)
    REASON_CLIENT = getattr(mt5, "DEAL_REASON_CLIENT", 0)
    REASON_EXPERT = getattr(mt5, "DEAL_REASON_EXPERT", 3)
except ImportError:
    REASON_SL, REASON_TP, REASON_SO = 4, 5, 6
    REASON_CLIENT, REASON_EXPERT = 0, 3

# Outcomes eligible for ML training — user manual closes excluded
TRAINABLE_CLOSED_BY = frozenset({"bot_sl", "bot_tp", "bot_manual_close"})


def _price_near(a: float, b: float, tolerance_pct: float = 0.001) -> bool:
    if not a or not b:
        return False
    return abs(a - b) / max(abs(b), 1e-9) <= tolerance_pct


def classify_close(
    position: Dict,
    deal: Optional[Dict],
    sl_price: float = 0.0,
    tp_price: float = 0.0,
    bot_initiated: bool = False,
) -> Tuple[str, str]:
    """
    Classify closure attribution.

    Heuristic priority:
    1. MT5 deal `reason` field when available (most reliable)
    2. Exit price proximity to recorded SL/TP (±0.1% tolerance)
    3. bot_initiated flag for early-close feature
    4. Fallback: user_manual_close or unknown

    Limitations: SL/TP heuristic fails if broker uses variable slippage or
    partial closes; reason field is preferred when present.
    """
    if bot_initiated:
        return "bot_manual_close", deal.get("close_reason", "Closed by bot early-exit logic") if deal else "Closed by bot"

    if not deal:
        return "unknown", "No deal history found — outcome recorded but attribution unknown"

    reason_code = deal.get("reason")
    comment = str(deal.get("comment", "") or "")
    exit_price = float(deal.get("price", 0) or deal.get("exit_price", 0) or 0)

    if reason_code == REASON_SL:
        return "bot_sl", f"Stop loss hit @ {exit_price:.5f}"
    if reason_code == REASON_TP:
        return "bot_tp", f"Take profit hit @ {exit_price:.5f}"
    if reason_code == REASON_SO:
        return "bot_sl", f"Stop out @ {exit_price:.5f}"
    if reason_code == REASON_EXPERT or "AeroTrade" in comment or "AI" in comment:
        return "bot_manual_close", f"Expert/EA close — {comment or 'no comment'}"

    # Heuristic: compare exit to SL/TP prices stored on position
    pos_sl = float(position.get("stop_loss", 0) or sl_price or 0)
    pos_tp = float(position.get("take_profit", 0) or tp_price or 0)
    if pos_sl and _price_near(exit_price, pos_sl):
        return "bot_sl", f"Exit near SL ({exit_price:.5f} ≈ SL {pos_sl:.5f})"
    if pos_tp and _price_near(exit_price, pos_tp):
        return "bot_tp", f"Exit near TP ({exit_price:.5f} ≈ TP {pos_tp:.5f})"

    if reason_code == REASON_CLIENT:
        return "user_manual_close", f"Manual close from terminal @ {exit_price:.5f}"

    return "user_manual_close", f"Close reason indeterminate (reason={reason_code}) — treated as user/manual"
