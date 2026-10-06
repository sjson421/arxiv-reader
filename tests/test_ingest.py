from datetime import date
from pathlib import Path

from arxiv_reader.ingest import parse

FEED = (Path(__file__).parent / "fixtures/feed.xml").read_bytes()


def test_parse_keeps_new_and_cross_once():
    day, papers = parse(FEED)
    assert day == date(2026, 10, 6)
    assert len(papers) == 5  # 3 new + 2 cross; replace, replace-cross and the duplicate are dropped
    first = papers[0]
    assert first.id == "2610.03769"
    assert first.abstract.startswith("Multi-agent LLM systems")
    assert "cs.MA" in first.categories


def test_parse_skips_items_missing_fields():
    broken = FEED.replace(b"<link>https://arxiv.org/abs/2610.03812</link>", b"")
    _, papers = parse(broken)
    assert len(papers) == 4 and "2610.03812" not in {p.id for p in papers}
