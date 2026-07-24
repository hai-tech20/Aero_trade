"""Persistent trade ledger — every trade resolves to win/loss/breakeven."""

import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config.settings import AI_CONFIG

DB_PATH = Path("data/aero_trade_ledger.db")

# Columns added via migration
_MIGRATION_COLUMNS = [
    ("closed_by", "TEXT DEFAULT 'unknown'"),
    ("close_reason", "TEXT DEFAULT ''"),
    ("timeframe", "TEXT DEFAULT ''"),
]


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _migrate(conn: sqlite3.Connection):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(trades)").fetchall()}
    for col, typedef in _MIGRATION_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typedef}")


def init_ledger():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            order_id TEXT PRIMARY KEY,
            account_id TEXT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            volume REAL,
            open_time TEXT,
            close_time TEXT,
            pnl_usd REAL,
            outcome TEXT,
            confidence REAL,
            features_json TEXT,
            sentiment_agreed INTEGER,
            model_used TEXT,
            closed_by TEXT DEFAULT 'unknown',
            close_reason TEXT DEFAULT '',
            timeframe TEXT DEFAULT ''
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            symbol TEXT,
            confidence REAL,
            features_json TEXT,
            recorded_at TEXT
        )
    """)
    _migrate(conn)
    conn.commit()
    conn.close()


def record_open_trade(
    order_id: str,
    account_id: str,
    symbol: str,
    direction: str,
    entry_price: float,
    volume: float,
    confidence: float,
    features_json: str,
    sentiment_agreed: bool,
    model_used: str,
    timeframe: str = "",
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
):
    init_ledger()
    conn = _conn()
    conn.execute(
        """INSERT OR REPLACE INTO trades
           (order_id, account_id, symbol, direction, entry_price, volume,
            open_time, confidence, features_json, sentiment_agreed, model_used, timeframe)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (order_id, account_id, symbol, direction, entry_price, volume,
         datetime.now().isoformat(), confidence, features_json,
         int(sentiment_agreed), model_used, timeframe),
    )
    conn.commit()
    conn.close()


def record_close_trade(
    order_id: str,
    exit_price: float,
    pnl_usd: float,
    confidence_at_entry: float = 0.0,
    closed_by: str = "unknown",
    close_reason: str = "",
):
    outcome = "win" if pnl_usd > 0 else ("loss" if pnl_usd < 0 else "breakeven")
    init_ledger()
    conn = _conn()
    conn.execute(
        """UPDATE trades SET exit_price = ?, close_time = ?, pnl_usd = ?, outcome = ?,
           closed_by = ?, close_reason = ?
           WHERE order_id = ?""",
        (exit_price, datetime.now().isoformat(), pnl_usd, outcome, closed_by, close_reason, order_id),
    )

    threshold = float(AI_CONFIG.get("min_confidence", 0.30))
    from core.ledger.close_attribution import TRAINABLE_CLOSED_BY
    if outcome == "loss" and float(confidence_at_entry) >= threshold and closed_by in TRAINABLE_CLOSED_BY:
        row = conn.execute(
            "SELECT symbol, features_json, confidence FROM trades WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row:
            existing = conn.execute("SELECT 1 FROM mistakes WHERE order_id = ?", (order_id,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO mistakes (order_id, symbol, confidence, features_json, recorded_at) VALUES (?,?,?,?,?)",
                    (order_id, row[0], row[2], row[1], datetime.now().isoformat()),
                )
    conn.commit()
    conn.close()
    return outcome


def get_closed_trades(account_id: str, limit: int = 100) -> List[Dict]:
    init_ledger()
    conn = _conn()
    rows = conn.execute(
        """SELECT order_id, symbol, direction, entry_price, exit_price, volume,
                  open_time, close_time, pnl_usd, outcome, confidence,
                  closed_by, close_reason, timeframe, model_used
           FROM trades WHERE account_id = ? AND close_time IS NOT NULL
           ORDER BY close_time DESC LIMIT ?""",
        (account_id, limit),
    ).fetchall()
    conn.close()
    cols = ["order_id", "symbol", "direction", "entry_price", "exit_price", "volume",
            "open_time", "close_time", "pnl_usd", "outcome", "confidence",
            "closed_by", "close_reason", "timeframe", "model_used"]
    return [dict(zip(cols, r)) for r in rows]


def get_trade_summary(account_id: str, exclude_user_closes: bool = True) -> Dict:
    trades = get_closed_trades(account_id, limit=10000)
    if exclude_user_closes:
        trades = [t for t in trades if t.get("closed_by") != "user_manual_close"]
    if not trades:
        return {"total_pnl": 0, "win_rate": 0, "wins": 0, "losses": 0, "profit_factor": None}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_profit = sum(t["pnl_usd"] for t in wins)
    total_loss = abs(sum(t["pnl_usd"] for t in losses))
    return {
        "total_pnl": sum(t["pnl_usd"] for t in trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "wins": len(wins),
        "losses": len(losses),
        "profit_factor": total_profit / total_loss if total_loss > 0 else None,
    }


def get_recent_closed_trades_for_training(limit: int = 500) -> List[Dict]:
    from core.ledger.close_attribution import TRAINABLE_CLOSED_BY
    init_ledger()
    conn = _conn()
    placeholders = ",".join("?" * len(TRAINABLE_CLOSED_BY))
    rows = conn.execute(
        f"""SELECT order_id, symbol, outcome, confidence, features_json, close_time, closed_by, timeframe
           FROM trades
           WHERE close_time IS NOT NULL AND outcome IN ('win', 'loss')
             AND features_json IS NOT NULL AND features_json != ''
             AND closed_by IN ({placeholders})
           ORDER BY close_time DESC
           LIMIT ?""",
        (*TRAINABLE_CLOSED_BY, limit),
    ).fetchall()
    conn.close()
    cols = ["order_id", "symbol", "outcome", "confidence", "features_json", "close_time", "closed_by", "timeframe"]
    return [dict(zip(cols, r)) for r in rows]


def get_mistake_rows(limit: int = 200) -> List[Dict]:
    init_ledger()
    conn = _conn()
    rows = conn.execute(
        "SELECT id, order_id, symbol, confidence, features_json, recorded_at FROM mistakes ORDER BY recorded_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    cols = ["id", "order_id", "symbol", "confidence", "features_json", "recorded_at"]
    return [dict(zip(cols, r)) for r in rows]


def count_closed_trades_since(since_iso: Optional[str]) -> int:
    init_ledger()
    conn = _conn()
    if since_iso:
        count = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE close_time IS NOT NULL AND close_time > ?",
            (since_iso,),
        ).fetchone()[0]
    else:
        count = conn.execute("SELECT COUNT(*) FROM trades WHERE close_time IS NOT NULL").fetchone()[0]
    conn.close()
    return int(count)


def resolve_orphan_trade(
    order_id: str, account_id: str, symbol: str, direction: str,
    entry_price: float, volume: float, pnl_usd: float = 0.0,
    closed_by: str = "unknown", close_reason: str = "Orphan resolution — no broker history",
):
    init_ledger()
    conn = _conn()
    existing = conn.execute("SELECT order_id FROM trades WHERE order_id = ?", (order_id,)).fetchone()
    if existing:
        record_close_trade(order_id, entry_price, pnl_usd, 0, closed_by, close_reason)
    else:
        record_open_trade(order_id, account_id, symbol, direction, entry_price, volume, 0, "{}", False, "")
        record_close_trade(order_id, entry_price, pnl_usd, 0, closed_by, close_reason)
    conn.close()


def build_self_analysis(account_id: str, days: int = 30) -> Dict:
    """Structured self-analysis report from real ledger data."""
    init_ledger()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = _conn()
    rows = conn.execute(
        """SELECT symbol, timeframe, outcome, pnl_usd, closed_by, model_used, confidence
           FROM trades WHERE account_id = ? AND close_time IS NOT NULL AND close_time >= ?""",
        (account_id, cutoff),
    ).fetchall()
    conn.close()

    all_trades = rows
    user_manual = [r for r in rows if r[4] == "user_manual_close"]
    bot_trades = [r for r in rows if r[4] != "user_manual_close"]

    def _stats(subset):
        if not subset:
            return {"trades": 0, "wins": 0, "losses": 0, "breakeven": 0, "win_rate": 0, "pnl_usd": 0, "profit_factor": None}
        wins = [r for r in subset if r[2] == "win"]
        losses = [r for r in subset if r[2] == "loss"]
        be = [r for r in subset if r[2] == "breakeven"]
        tp = sum(r[3] for r in wins)
        tl = abs(sum(r[3] for r in losses))
        return {
            "trades": len(subset),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": len(be),
            "win_rate": len(wins) / len(subset) if subset else 0,
            "pnl_usd": round(sum(r[3] for r in subset), 2),
            "profit_factor": round(tp / tl, 2) if tl > 0 else None,
        }

    overall = _stats(bot_trades)
    overall["total_trades"] = overall.pop("trades")

    # By symbol
    sym_map: Dict[str, list] = {}
    for r in bot_trades:
        sym_map.setdefault(r[0], []).append(r)
    by_symbol = sorted(
        [{"symbol": s, **_stats(v)} for s, v in sym_map.items()],
        key=lambda x: x["trades"], reverse=True,
    )

    # By timeframe
    tf_map: Dict[str, list] = {}
    for r in bot_trades:
        tf = r[1] or "unknown"
        tf_map.setdefault(tf, []).append(r)
    by_timeframe = sorted(
        [{"timeframe": tf, **_stats(v)} for tf, v in tf_map.items()],
        key=lambda x: x["trades"], reverse=True,
    )

    account_wr = overall["win_rate"]
    flags = []
    for item in by_timeframe:
        if item["trades"] >= 10 and item["win_rate"] < account_wr - 0.10:
            flags.append(
                f"{item['timeframe']} timeframe is underperforming ({item['win_rate']:.2f} win rate, "
                f"P&L ${item['pnl_usd']:.2f}) — consider disabling or raising confidence threshold"
            )
    for item in by_symbol:
        if item["trades"] >= 8 and item["win_rate"] < account_wr - 0.15:
            flags.append(
                f"{item['symbol']} has a materially lower win rate ({item['win_rate']:.2f}) than account average "
                f"({account_wr:.2f}) — consider reviewing or pausing this pair"
            )
    if overall["profit_factor"] is not None and overall["profit_factor"] < 1.0 and overall["total_trades"] >= 20:
        flags.append(
            f"Overall profit factor {overall['profit_factor']:.2f} below 1.0 — strategy is net losing; "
            "review position sizing and underperforming symbols/timeframes"
        )

    return {
        "period": f"last_{days}_days",
        "overall": overall,
        "by_symbol": by_symbol,
        "by_timeframe": by_timeframe,
        "excluded_from_analysis": {
            "user_manual_closes": len(user_manual),
            "note": "excluded from win/loss stats — these reflect user decisions, not bot decisions",
        },
        "flags": flags,
    }
