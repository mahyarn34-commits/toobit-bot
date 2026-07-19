"""
SIGNAL STORE — persists every signal sent, tracks its real outcome,
and gives you real (not backtested) win-rate stats.

Uses sqlite3 (stdlib, no extra dependency) in a single local file,
signals.db, next to this script.

-----------------------------------------------------------------
IMPORTANT — Railway persistence
-----------------------------------------------------------------
Railway's default filesystem is EPHEMERAL: signals.db is wiped on
every redeploy (not on a simple restart/crash, but definitely on a
new deployment). If you want history to survive redeploys, add a
Railway Volume mounted at the working directory, or point DB_PATH
(env var) at a mounted volume path. Without a volume, treat this as
"since the last deploy" stats, not permanent history.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get("SIGNAL_DB_PATH", "signals.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    market_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit_1 REAL NOT NULL,
    take_profit_2 REAL NOT NULL,
    score REAL,
    leverage REAL,
    opened_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',   -- OPEN, TP1_BE, TP2, SL, BE_EXIT, INVALIDATED
    breakeven_moved INTEGER NOT NULL DEFAULT 0,
    closed_at REAL,
    exit_price REAL,
    close_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_open ON signals(status) WHERE status = 'OPEN';
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def record_signal(chat_id: int, sig) -> int:
    """sig: a toobit_stage2_signals.Signal instance."""
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO signals
               (chat_id, symbol, market_type, direction, entry_price, stop_loss,
                take_profit_1, take_profit_2, score, leverage, opened_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')""",
            (chat_id, sig.symbol, sig.market_type, sig.direction, sig.entry_price,
             sig.stop_loss, sig.take_profit_1, sig.take_profit_2, sig.score,
             sig.suggested_leverage, time.time()),
        )
        return cur.lastrowid


def get_open_signals(chat_id: Optional[int] = None) -> list[sqlite3.Row]:
    with _conn() as conn:
        if chat_id is not None:
            return conn.execute(
                "SELECT * FROM signals WHERE status IN ('OPEN','TP1_BE') AND chat_id = ?", (chat_id,)
            ).fetchall()
        return conn.execute(
            "SELECT * FROM signals WHERE status IN ('OPEN','TP1_BE')"
        ).fetchall()


def open_chat_ids() -> list[int]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT chat_id FROM signals WHERE status IN ('OPEN','TP1_BE')"
        ).fetchall()
        return [r["chat_id"] for r in rows]


def mark_breakeven(signal_id: int) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE signals SET status = 'TP1_BE', breakeven_moved = 1 WHERE id = ?", (signal_id,)
        )


def close_signal(signal_id: int, status: str, exit_price: float, reason: str) -> None:
    with _conn() as conn:
        conn.execute(
            """UPDATE signals SET status = ?, exit_price = ?, close_reason = ?, closed_at = ?
               WHERE id = ?""",
            (status, exit_price, reason, time.time(), signal_id),
        )


def get_stats(chat_id: Optional[int] = None) -> dict:
    with _conn() as conn:
        q = "SELECT status, COUNT(*) as n FROM signals WHERE status NOT IN ('OPEN','TP1_BE')"
        params: tuple = ()
        if chat_id is not None:
            q += " AND chat_id = ?"
            params = (chat_id,)
        q += " GROUP BY status"
        rows = conn.execute(q, params).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        closed_total = sum(counts.values())
        # TP2 = full win. BE_EXIT = reached TP1 (partial win) before price
        # came back to entry — not a loss, counted as a win here.
        wins = counts.get("TP2", 0) + counts.get("BE_EXIT", 0)
        return {
            "closed_total": closed_total,
            "by_status": counts,
            "win_rate_pct": round(wins / closed_total * 100, 1) if closed_total else None,
        }
