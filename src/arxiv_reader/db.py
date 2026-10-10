import os
import sqlite3
from pathlib import Path

from arxiv_reader.ingest import Paper

DATA_DIR = Path(os.environ.get("ARXIV_READER_HOME", Path.home() / ".local/share/arxiv-reader"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    categories TEXT NOT NULL,
    announced TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS digests (
    date TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    first_opened_at TEXT
);
CREATE TABLE IF NOT EXISTS digest_items (
    date TEXT NOT NULL REFERENCES digests,
    paper_id TEXT NOT NULL REFERENCES papers,
    rank INTEGER NOT NULL,
    bullets TEXT NOT NULL,  -- JSON list
    reason TEXT NOT NULL,
    headline TEXT,  -- NULL on digests made before headlines existed
    rejected_at TEXT,
    PRIMARY KEY (date, paper_id)
);
"""


def connect(path: str | Path = DATA_DIR / "app.db") -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript(SCHEMA)
    if "headline" not in {r["name"] for r in db.execute("PRAGMA table_info(digest_items)")}:
        db.execute("ALTER TABLE digest_items ADD COLUMN headline TEXT")
    return db


def items(db: sqlite3.Connection, where: str, *args) -> list[sqlite3.Row]:
    """Digest items joined with their paper and digest, filtered by `where`."""
    return db.execute(
        f"""SELECT i.*, p.title, p.abstract, p.categories, d.first_opened_at
            FROM digest_items i JOIN papers p ON p.id = i.paper_id JOIN digests d USING (date)
            WHERE {where}""",
        args,
    ).fetchall()


def paper(row: sqlite3.Row) -> Paper:
    return Paper(row["paper_id"], row["title"], row["abstract"], row["categories"])
