"""The daily run, plus the one-time setup and the profile rewrite."""

import json
import sqlite3
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta

from interest_memory import InterestMemory

from arxiv_reader import db as appdb
from arxiv_reader import llm, rank
from arxiv_reader.ingest import Paper

STAGE1_KEEP = 40
PROFILE_WINDOW_DAYS = 35  # the profile timer runs monthly

type Item = tuple[Paper, list[str], str, str]  # paper, bullets, reason, headline


def run(
    db: sqlite3.Connection,
    mem: InterestMemory,
    day: date,
    papers: Sequence[Paper],
    review: Callable = llm.review,
    scored: Callable[[sqlite3.Connection, date, Sequence[Paper], Sequence[float]], None] | None = None,
) -> list[Item]:
    """Builds and stores the digest for `day`. `scored` sees every paper's stage 1 score inside the same
    transaction, so a caller such as evals can keep them without the app storing them."""
    if db.execute("SELECT 1 FROM digests WHERE date = ?", (day.isoformat(),)).fetchone():
        print(f"The digest for {day} already exists.")
        return []
    if not papers:
        print("The feed is empty, so the last digest stays.")
        return []
    profile = mem.profile()
    if not profile:
        raise SystemExit("No profile yet. Run `arxiv-reader init` first.")

    scores = mem.score([p.text for p in papers], datetime.combine(day, time()))
    top = [p for _, p in sorted(zip(scores, papers), key=lambda sp: -sp[0])][:STAGE1_KEEP]

    count = 7 if day.weekday() == 1 else 5  # Tuesday's feed carries Friday-to-Monday submissions
    try:
        reviews = review(top, profile)
        if not any(p.id in reviews for p in top):
            raise llm.LLMError("Claude returned no reviews for the stage 1 papers")
    except llm.LLMError as e:
        print(f"Claude review failed, using stage 1 order: {e}")
        items = [(p, p.first_sentences(2), "", p.title) for p in top[:count]]
    else:
        items = _choose(mem, top, reviews, count)

    with db:
        db.executemany(
            "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?)",
            [(p.id, p.title, p.abstract, p.categories, day.isoformat()) for p in papers],
        )
        if scored:
            scored(db, day, papers, scores)
        db.execute("INSERT INTO digests (date) VALUES (?)", (day.isoformat(),))
        db.executemany(
            "INSERT INTO digest_items (date, paper_id, rank, bullets, reason, headline) VALUES (?, ?, ?, ?, ?, ?)",
            [(day.isoformat(), p.id, n, json.dumps(b), r, h) for n, (p, b, r, h) in enumerate(items)],
        )
    return items


def _choose(mem: InterestMemory, top: list[Paper], reviews: dict[str, llm.Review], count: int) -> list[Item]:
    reviewed = [p for p in top if p.id in reviews]
    ordered = rank.blend(reviewed, {p: reviews[p.id].score for p in reviewed})
    vecs = mem.embed([p.text for p in ordered])
    picks = [ordered[i] for i in rank.pick_diverse(vecs, count)]
    return [(p, reviews[p.id].bullets, reviews[p.id].reason, reviews[p.id].headline) for p in picks]


def init(mem: InterestMemory, paragraph: str) -> llm.Interests:
    """Cold start: Claude turns the paragraph into a profile and seed phrases."""
    interests = llm.expand_interests(paragraph)
    mem.set_profile(interests.profile)
    mem.add_seeds(interests.seeds)
    return interests


def rewrite_profile(
    db: sqlite3.Connection, mem: InterestMemory, today: date, rewrite: Callable = llm.rewrite_profile
) -> str | None:
    """Updates the profile from recently opened digests. Returns None when there is no feedback."""
    since = (today - timedelta(days=PROFILE_WINDOW_DAYS)).isoformat()
    rows = appdb.items(db, "d.date >= ? AND d.first_opened_at IS NOT NULL", since)
    if not rows:
        return None
    kept = [r["title"] for r in rows if r["rejected_at"] is None]
    rejected = [r["title"] for r in rows if r["rejected_at"] is not None]
    profile = rewrite(mem.profile(), kept, rejected)
    mem.set_profile(profile)
    return profile
