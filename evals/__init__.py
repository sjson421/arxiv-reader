import re
import sqlite3
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from arxiv_reader import db as appdb
from arxiv_reader import llm
from arxiv_reader.db import DATA_DIR

EVALS_DB = DATA_DIR / "evals.db"
EVALS_MEMORY = DATA_DIR / "evals-memory.db"  # a copy of the live profile, so evals never touch memory.db

SCHEMA = """
CREATE TABLE IF NOT EXISTS feed (  -- every paper of a day's feed with its stage 1 score
    date TEXT NOT NULL,
    paper_id TEXT NOT NULL REFERENCES papers,
    score REAL NOT NULL,
    PRIMARY KEY (date, paper_id)
);
CREATE TABLE IF NOT EXISTS grades (
    rubric TEXT, date TEXT, paper_id TEXT, grade TEXT NOT NULL,
    PRIMARY KEY (rubric, date, paper_id)
);
"""


def connect() -> sqlite3.Connection:
    """evals.db: the app's tables plus the ones only evals use."""
    db = appdb.connect(EVALS_DB)
    db.executescript(SCHEMA)
    return db


def keep_feed(db: sqlite3.Connection, day: date, papers, scores) -> None:
    db.executemany("INSERT INTO feed VALUES (?, ?, ?)", [(day.isoformat(), p.id, s) for p, s in zip(papers, scores)])


_LIMIT = re.compile(r"session limit.*resets (\d{1,2})(?::(\d{2}))?\s*([ap]m) \(([^)]+)\)", re.I | re.S)


def seconds_until_reset(message: str, now: datetime | None = None) -> float | None:
    """Parses \"You've hit your session limit · resets 5pm (America/New_York)\". None if it is not that error."""
    m = _LIMIT.search(message)
    if not m:
        return None
    hour, minute, ampm, tz = int(m[1]) % 12 + (12 if m[3].lower() == "pm" else 0), int(m[2] or 0), m[3], ZoneInfo(m[4])
    now = (now or datetime.now(tz)).astimezone(tz)
    reset = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= now:
        reset += timedelta(days=1)
    return (reset - now).total_seconds()


def resuming[T](call: Callable[[], T]) -> T:
    """Runs `call`; when the usage limit stops it, sleeps until the limit resets and runs it again."""
    while True:
        try:
            return call()
        except llm.LLMError as e:
            wait = seconds_until_reset(str(e))
            if wait is None:
                raise
            print(f"Usage limit hit; resuming in {wait / 60:.0f} minutes.", flush=True)
            time.sleep(wait + 120)  # a margin past the reset
