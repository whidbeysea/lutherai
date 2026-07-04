"""SQLite logging for chat sessions/messages. Deliberately not Postgres -- this is a
single low-traffic app with one writer process, none of the concurrent-write
contention that justified Postgres for the trading algos. Plain files, zero ops."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "luther.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    question TEXT NOT NULL,
    response TEXT NOT NULL,
    retrieved_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_session(session_id: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, _now()),
        )


def log_message(session_id: str, question: str, response: str, retrieved_json: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, question, response, retrieved_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, question, response, retrieved_json, _now()),
        )


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    """Reconstruct recent conversation turns for a session, oldest first, as
    {"role": ..., "content": ...} records suitable for the Claude messages API."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT question, response FROM messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    history = []
    for row in reversed(rows):
        history.append({"role": "user", "content": row["question"]})
        history.append({"role": "assistant", "content": row["response"]})
    return history


def get_stats() -> dict:
    with get_conn() as conn:
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        per_day = conn.execute(
            "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count "
            "FROM messages GROUP BY day ORDER BY day DESC LIMIT 30"
        ).fetchall()
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "messages_per_day": [dict(row) for row in per_day],
    }


def list_messages(limit: int = 50, offset: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, session_id, question, response, retrieved_json, created_at "
            "FROM messages ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]
