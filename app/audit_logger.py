"""
SQLite audit logger — persists all queries, responses, and guardrail flags.
"""
import sqlite3
import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("AUDIT_DB_PATH", "./logs/audit.db")


def init_db():
    """Create the audit table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            question        TEXT    NOT NULL,
            answer          TEXT    NOT NULL,
            sources         TEXT,
            confidence      TEXT,
            blocked         INTEGER DEFAULT 0,
            block_reason    TEXT,
            response_time_ms INTEGER
        )
    """)
    conn.commit()
    conn.close()


def log_query(question: str, answer: str, sources: list,
              confidence: str = "N/A", blocked: bool = False,
              block_reason: str = "", response_time_ms: int = 0):
    """Insert a query/response record into the audit log."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO audit_log
          (timestamp, question, answer, sources, confidence,
           blocked, block_reason, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        question,
        answer,
        json.dumps(sources),
        confidence,
        1 if blocked else 0,
        block_reason,
        response_time_ms,
    ))
    conn.commit()
    conn.close()


def get_recent_logs(limit: int = 50) -> list:
    """Fetch the most recent audit log entries."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM audit_log
        ORDER BY id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats() -> dict:
    """Return summary statistics for the admin dashboard."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    stats = {}
    stats["total"]   = conn.execute(
        "SELECT COUNT(*) FROM audit_log").fetchone()[0]
    stats["blocked"] = conn.execute(
        "SELECT COUNT(*) FROM audit_log WHERE blocked=1").fetchone()[0]
    stats["allowed"] = stats["total"] - stats["blocked"]
    stats["avg_response_ms"] = conn.execute(
        "SELECT AVG(response_time_ms) FROM audit_log WHERE blocked=0"
    ).fetchone()[0] or 0
    block_reasons = conn.execute("""
        SELECT block_reason, COUNT(*) as cnt
        FROM audit_log WHERE blocked=1
        GROUP BY block_reason ORDER BY cnt DESC
    """).fetchall()
    stats["block_breakdown"] = {r[0]: r[1] for r in block_reasons}
    conn.close()
    return stats


if __name__ == "__main__":
    init_db()
    log_query(
        question="Test query",
        answer="Test answer",
        sources=["CNC_Lathe_Safety_Manual.pdf"],
        confidence="High",
        blocked=False,
        response_time_ms=1234,
    )
    logs = get_recent_logs(1)
    print("Audit log test entry:", logs[0])
    print("Stats:", get_stats())


def export_to_csv(filepath: str = "./logs/audit_export.csv"):
    """Export full audit log to CSV — useful for compliance review."""
    import csv
    logs = get_recent_logs(limit=10000)
    if not logs:
        print("No logs to export.")
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)
    print(f"Exported {len(logs)} records to {filepath}")


def clear_test_logs():
    """Remove the test entry inserted by __main__ — keeps log clean."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM audit_log WHERE question = 'Test query'")
    conn.commit()
    conn.close()
    print("Test entries removed from audit log.")
