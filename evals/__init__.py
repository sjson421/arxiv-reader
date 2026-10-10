import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from arxiv_reader import db as appdb
from arxiv_reader import llm
from arxiv_reader.db import DATA_DIR

WORKERS = 4
DIMENSIONS = ("relevance", "correctness", "helpfulness")  # the 1-5 grades; `safe` is a separate check
EVALS_DB = DATA_DIR / "evals.db"
EVALS_MEMORY = DATA_DIR / "evals-memory.db"  # a copy of the live profile, so evals never touch memory.db

SCHEMA = """
CREATE TABLE IF NOT EXISTS grades (
    rubric TEXT, date TEXT, paper_id TEXT, grade TEXT NOT NULL,
    PRIMARY KEY (rubric, date, paper_id)
);
"""


def connect() -> sqlite3.Connection:
    """evals.db: the app's tables plus the grades table."""
    db = appdb.connect(EVALS_DB)
    db.executescript(SCHEMA)
    return db


def stored_grades(db: sqlite3.Connection, rubric: str) -> dict[tuple[str, str], str]:
    """The JSON grades saved under `rubric`, keyed by (date, paper_id)."""
    rows = db.execute("SELECT date, paper_id, grade FROM grades WHERE rubric = ?", (rubric,))
    return {(d, p): g for d, p, g in rows}


def grade_missing(db: sqlite3.Connection, rubric: str, rows: Iterable[sqlite3.Row], grade: Callable[[sqlite3.Row], str]):
    """Grades the rows not yet saved under `rubric`, WORKERS at a time, and returns every saved grade.
    `grade` returns the grade as JSON. A rerun picks up where a crash left off."""
    done = stored_grades(db, rubric)
    lock = threading.Lock()  # one connection is shared by the workers

    def run(r: sqlite3.Row) -> None:
        g = grade(r)
        with lock, db:
            db.execute("INSERT INTO grades VALUES (?, ?, ?, ?)", (rubric, r["date"], r["paper_id"], g))

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(run, [r for r in rows if (r["date"], r["paper_id"]) not in done]))
    return stored_grades(db, rubric)


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
