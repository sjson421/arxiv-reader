from datetime import date

from arxiv_reader import pipeline
from conftest import FakeClaude, run

MONDAY, TUESDAY = date(2026, 10, 5), date(2026, 10, 6)


def test_tuesday_shows_7_and_other_days_5(app_db, mem, papers):
    assert len(run(app_db, mem, TUESDAY, papers)) == 7
    assert len(run(app_db, mem, MONDAY, papers)) == 5


def test_picks_come_from_stage_1_top(app_db, mem, papers):
    items = run(app_db, mem, MONDAY, papers)
    assert all(p.title.startswith("agent") for p, *_ in items)


def test_claude_failure_falls_back_to_stage_1(app_db, mem, papers):
    items = run(app_db, mem, MONDAY, papers, FakeClaude(fail=True))
    assert len(items) == 5
    assert items[0][1] == [items[0][0].abstract.split(". ")[0] + "."] + ["Result " + items[0][0].id[1:] + "."]


def test_rerun_for_the_same_day_does_nothing(app_db, mem, papers):
    run(app_db, mem, MONDAY, papers)
    assert run(app_db, mem, MONDAY, papers) == []
    assert app_db.execute("SELECT COUNT(*) FROM digest_items").fetchone()[0] == 5


def test_rewrite_profile_uses_opened_digests_only(app_db, mem, papers):
    run(app_db, mem, MONDAY, papers)
    seen = []
    rewrite = lambda profile, kept, rejected: seen.append((kept, rejected)) or "new profile"
    assert pipeline.rewrite_profile(app_db, mem, MONDAY, rewrite=rewrite) is None
    app_db.execute("UPDATE digests SET first_opened_at = CURRENT_TIMESTAMP")
    app_db.execute("UPDATE digest_items SET rejected_at = CURRENT_TIMESTAMP WHERE rank = 0")
    assert pipeline.rewrite_profile(app_db, mem, MONDAY, rewrite=rewrite) == "new profile"
    assert len(seen[0][0]) == 4 and len(seen[0][1]) == 1
    assert mem.profile() == "new profile"


def test_headline_is_stored_and_falls_back_to_the_title(app_db, mem, papers):
    run(app_db, mem, MONDAY, papers)
    assert {r[0] for r in app_db.execute("SELECT headline FROM digest_items")} == {"h"}
    items = run(app_db, mem, TUESDAY, papers, FakeClaude(fail=True))
    assert all(h == p.title for p, _, _, h in items)
