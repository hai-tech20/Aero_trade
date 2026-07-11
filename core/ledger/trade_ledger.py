"""Persistent trade ledger — every trade resolves to win/loss/breakeven."""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

DB_PATH = Path("data/aero_trade_ledger.db")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


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
            model_used TEXT
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
):
    init_ledger()
    conn = _conn()
    conn.execute(
        """INSERT OR REPLACE INTO trades
           (order_id, account_id, symbol, direction, entry_price, volume,
            open_time, confidence, features_json, sentiment_agreed, model_used)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (order_id, account_id, symbol, direction, entry_price, volume,
         datetime.now().isoformat(), confidence, features_json,
         int(sentiment_agreed), model_used),
    )
    conn.commit()
    conn.close()


def record_close_trade(order_id: str, exit_price: float, pnl_usd: float, confidence_at_entry: float = 0.0):
    outcome = "win" if pnl_usd > 0 else ("loss" if pnl_usd < 0 else "breakeven")
    init_ledger()
    conn = _conn()
    conn.execute(
        """UPDATE trades SET exit_price = ?, close_time = ?, pnl_usd = ?, outcome = ?
           WHERE order_id = ?""",
        (exit_price, datetime.now().isoformat(), pnl_usd, outcome, order_id),
    )
    # Log mistakes: confident but wrong
    if outcome == "loss" and confidence_at_entry > 0.65:
        row = conn.execute(
            "SELECT symbol, features_json FROM trades WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row:
            conn.execute(
                "INSERT INTO mistakes (order_id, symbol, confidence, features_json, recorded_at) VALUES (?,?,?,?,?)",
                (order_id, row[0], confidence_at_entry, row[1], datetime.now().isoformat()),
            )
    conn.commit()
    conn.close()
    return outcome


def get_closed_trades(account_id: str, limit: int = 100) -> List[Dict]:
    init_ledger()
    conn = _conn()
    rows = conn.execute(
        """SELECT order_id, symbol, direction, entry_price, exit_price, volume,
                  open_time, close_time, pnl_usd, outcome, confidence
           FROM trades WHERE account_id = ? AND close_time IS NOT NULL
           ORDER BY close_time DESC LIMIT ?""",
        (account_id, limit),
    ).fetchall()
    conn.close()
    cols = ["order_id", "symbol", "direction", "entry_price", "exit_price", "volume",
            "open_time", "close_time", "pnl_usd", "outcome", "confidence"]
    return [dict(zip(cols, r)) for r in rows]


def get_trade_summary(account_id: str) -> Dict:
    trades = get_closed_trades(account_id, limit=10000)
    if not trades:
        return {"total_pnl": 0, "win_rate": 0, "wins": 0, "losses": 0, "profit_factor": 0}
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_profit = sum(t["pnl_usd"] for t in wins)
    total_loss = abs(sum(t["pnl_usd"] for t in losses))
    return {
        "total_pnl": sum(t["pnl_usd"] for t in trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0,
        "wins": len(wins),
        "losses": len(losses),
        "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
    }


def resolve_orphan_trade(order_id: str, account_id: str, symbol: str, direction: str,
                         entry_price: float, volume: float, pnl_usd: float = 0.0):
    """Ensure every closed position gets a recorded outcome — never silent loss."""
    init_ledger()
    conn = _conn()
    existing = conn.execute("SELECT order_id FROM trades WHERE order_id = ?", (order_id,)).fetchone()
    if existing:
        record_close_trade(order_id, entry_price, pnl_usd)
    else:
        record_open_trade(order_id, account_id, symbol, direction, entry_price, volume, 0, "{}", False, "")
        record_close_trade(order_id, entry_price, pnl_usd)
    conn.close()
