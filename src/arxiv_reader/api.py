"""Serves the digest and keeps the interest memory in step with feedback as it happens.

Endpoints are async so every database and memory call runs on the event loop thread, one at a time.
"""

import json
import sqlite3
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from interest_memory import InterestMemory

from arxiv_reader import db as appdb

KEPT = 0.15
REJECTED = -1.0
STATIC = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def _sync(mem: InterestMemory, row: sqlite3.Row) -> None:
    """Rejected papers are strong negatives, papers seen in an opened digest are mild positives."""
    if row["rejected_at"] is not None:
        weight = REJECTED
    elif row["first_opened_at"] is not None:
        weight = KEPT
    else:
        return mem.forget(row["paper_id"])
    mem.record(row["paper_id"], appdb.paper(row).text, weight, datetime.fromisoformat(row["date"]))


def create_app(open_db: Callable[[], sqlite3.Connection], open_mem: Callable[[], InterestMemory]) -> FastAPI:
    """Takes factories so the connections are opened on the server thread."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db = open_db()
        app.state.mem = open_mem()
        app.state.mem.embed([""])  # load the embedding model now, not on the first click
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/digest/today")
    async def today():
        db = app.state.db
        latest = db.execute("SELECT date FROM digests ORDER BY date DESC LIMIT 1").fetchone()
        if not latest:
            raise HTTPException(404, "No digest yet")
        now = date.today()
        return {
            "date": latest["date"],
            "stale": now.weekday() < 5 and latest["date"] != now.isoformat(),
            "items": [
                {
                    "id": r["paper_id"],
                    "title": r["title"],
                    "headline": r["headline"] or r["title"],
                    "link": appdb.paper(r).link,
                    "bullets": json.loads(r["bullets"]),
                    "reason": r["reason"],
                    "rejected": r["rejected_at"] is not None,
                }
                for r in appdb.items(db, "i.date = ? ORDER BY i.rank", latest["date"])
            ],
        }

    @app.put("/digest/{day}/opened")
    async def opened(day: str):
        db = app.state.db
        digest = db.execute("SELECT first_opened_at FROM digests WHERE date = ?", (day,)).fetchone()
        if not digest:
            raise HTTPException(404, "No such digest")
        if digest["first_opened_at"] is None:
            # One transaction: if a sync fails part way, the digest stays unopened and the next open retries.
            with db:
                db.execute("UPDATE digests SET first_opened_at = CURRENT_TIMESTAMP WHERE date = ?", (day,))
                for r in appdb.items(db, "i.date = ?", day):
                    _sync(app.state.mem, r)
        return {"ok": True}

    def mark(paper_id: str, rejected: bool) -> dict:
        db = app.state.db
        rows = appdb.items(db, "i.paper_id = ? ORDER BY i.date DESC LIMIT 1", paper_id)
        if not rows:
            raise HTTPException(404, "Paper is not in any digest")
        with db:
            db.execute(
                "UPDATE digest_items SET rejected_at = CASE WHEN ? THEN CURRENT_TIMESTAMP END "
                "WHERE date = ? AND paper_id = ?",
                (rejected, rows[0]["date"], paper_id),
            )
        _sync(app.state.mem, appdb.items(db, "i.date = ? AND i.paper_id = ?", rows[0]["date"], paper_id)[0])
        return {"ok": True}

    @app.put("/papers/{paper_id}/not-interested")
    async def reject(paper_id: str):
        return mark(paper_id, True)

    @app.delete("/papers/{paper_id}/not-interested")
    async def undo(paper_id: str):
        return mark(paper_id, False)

    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=STATIC, html=True))
    return app
