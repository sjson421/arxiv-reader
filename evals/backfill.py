"""Builds past digests into evals.db from arXiv's API, one real pipeline run per weekday. Safe to rerun: days
already stored are skipped. Run: uv run python -m evals.backfill --days 200"""

import argparse
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

from interest_memory import InterestMemory

from arxiv_reader import db as appdb
from arxiv_reader import llm, pipeline
from arxiv_reader.ingest import FEED, Paper
from evals import EVALS_DB, EVALS_MEMORY, resuming

API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
CATEGORIES = FEED.rsplit("/", 1)[-1].split("+")
PAGE = 1000
NY, UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")


def prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def window(day: date) -> tuple[datetime, datetime]:
    """Submissions behind `day`'s announcement: the previous two 14:00 ET deadlines (Tuesday spans the weekend)."""
    end = prev_weekday(day)
    start = prev_weekday(end)
    return tuple(datetime.combine(d, dtime(14), NY).astimezone(UTC) for d in (start, end))


def fetch(day: date) -> list[Paper]:
    start, end = window(day)
    query = "(" + " OR ".join(f"cat:{c}" for c in CATEGORIES) + ") AND submittedDate:" \
        f"[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    papers: dict[str, Paper] = {}
    while True:
        params = urllib.parse.urlencode(
            {"search_query": query, "start": len(papers), "max_results": PAGE, "sortBy": "submittedDate"}
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(f"{API}?{params}", timeout=120) as resp:
                    entries = ET.fromstring(resp.read()).findall("a:entry", ATOM)
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 4:
                    raise
                time.sleep(30 * (attempt + 1))
        time.sleep(3)  # arXiv asks for one request per 3 seconds
        for e in entries:
            pid = e.findtext("a:id", "", ATOM).rsplit("/", 1)[-1].rsplit("v", 1)[0]
            papers.setdefault(pid, Paper(
                id=pid,
                title=" ".join(e.findtext("a:title", "", ATOM).split()),
                abstract=" ".join(e.findtext("a:summary", "", ATOM).split()),
                categories=", ".join(c.get("term") for c in e.findall("a:category", ATOM)),
            ))
        if len(entries) < PAGE:
            return list(papers.values())


class ReviewFailed(Exception):
    """Not an LLMError, so the pipeline aborts the day instead of storing its stage 1 fallback."""


def strict_review(papers, profile):
    try:
        return resuming(lambda: llm.review(papers, profile))  # llm.ask already retries once
    except llm.LLMError as e:
        raise ReviewFailed(e) from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=200, help="how many weekday digests to have in evals.db")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today(), help="newest day, exclusive")
    parser.add_argument("--shard", default="0/1", help="i/N: handle every Nth day, so N processes can run at once")
    args = parser.parse_args()
    index, shards = map(int, args.shard.split("/"))

    if not EVALS_MEMORY.exists():
        shutil.copy(appdb.DATA_DIR / "memory.db", EVALS_MEMORY)
    mem, db = InterestMemory(str(EVALS_MEMORY)), appdb.connect(EVALS_DB)
    days, d = [], args.end
    while len(days) < args.days:
        d = prev_weekday(d)
        days.append(d)
    for n, day in enumerate(sorted(days), 1):
        if n % shards != index:
            continue
        if db.execute("SELECT 1 FROM digests WHERE date = ?", (day.isoformat(),)).fetchone():
            continue
        papers = fetch(day)
        print(f"[{n}/{len(days)}] {day} {len(papers)} papers", flush=True)
        try:
            pipeline.run(db, mem, day, papers, review=strict_review)
        except ReviewFailed as e:
            print(f"{day} skipped, rerun to retry: {e}", flush=True)


if __name__ == "__main__":
    main()
