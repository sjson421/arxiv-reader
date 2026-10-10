"""Grades the same reviews with two judge models and compares them.
Run: uv run python -m evals.compare_judges --digests 20"""

import argparse
import hashlib
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from statistics import correlation, mean

from interest_memory import InterestMemory

from arxiv_reader import db as appdb
from evals import EVALS_DB, EVALS_MEMORY, resuming
from evals.claude_judge import DIMENSIONS, RUBRIC, Grade, judge

MODELS = ("claude-sonnet-5-5", "claude-haiku-5-5")
WORKERS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digests", type=int, default=20, help="digests to grade, evenly spread over the backfill")
    parser.add_argument("--low", type=int, help="instead, N reviews per dimension that the full Sonnet set scored <= 2")
    args = parser.parse_args()

    db = appdb.connect(EVALS_DB)
    db.execute(
        "CREATE TABLE IF NOT EXISTS grades (rubric TEXT, date TEXT, paper_id TEXT, grade TEXT NOT NULL, "
        "PRIMARY KEY (rubric, date, paper_id))"
    )
    items = appdb.items(db, "i.reason != ''")
    if args.low:
        full = db.execute(
            "SELECT rubric FROM grades WHERE rubric NOT LIKE '%-claude-%' GROUP BY rubric ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()[0]
        old = {(d, p): json.loads(g) for d, p, g in db.execute("SELECT date, paper_id, grade FROM grades WHERE rubric = ?", (full,))}
        random.seed(0)
        picked = set()
        for d in DIMENSIONS:
            low = sorted(k for k, g in old.items() if g[d] <= 2)
            picked |= set(random.sample(low, min(args.low, len(low))))
        rows = [r for r in items if (r["date"], r["paper_id"]) in picked]
        chosen = {r["date"] for r in rows}
    else:
        dates = [r[0] for r in db.execute("SELECT date FROM digests ORDER BY date")]
        step = len(dates) / args.digests
        chosen = {dates[int(i * step)] for i in range(args.digests)}
        rows = [r for r in items if r["date"] in chosen]
    profile = InterestMemory(str(EVALS_MEMORY)).profile()
    base = hashlib.sha256(RUBRIC.encode()).hexdigest()[:12]
    lock = threading.Lock()

    def run(model: str, r) -> None:
        key = f"{base}-{model}"  # the model is part of the cache key, so judges never mix
        with lock:
            if db.execute("SELECT 1 FROM grades WHERE rubric=? AND date=? AND paper_id=?", (key, r["date"], r["paper_id"])).fetchone():
                return
        g = resuming(lambda: judge(r["abstract"], profile, r["reason"], json.loads(r["bullets"]), model))
        with lock, db:
            db.execute("INSERT INTO grades VALUES (?, ?, ?, ?)", (key, r["date"], r["paper_id"], g.model_dump_json()))

    jobs = [(m, r) for r in rows for m in MODELS]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda j: run(*j), jobs))

    grades = {
        m: {
            (d, p): Grade.model_validate_json(g)
            for d, p, g in db.execute("SELECT date, paper_id, grade FROM grades WHERE rubric = ?", (f"{base}-{m}",))
        }
        for m in MODELS
    }
    keys = [(r["date"], r["paper_id"]) for r in rows]
    a, b = MODELS
    print(f"rubric={base} digests={len(chosen)} reviews={len(keys)}")
    print(f"{'':12} {a:>20} {b:>20}  diff  exact  within1  corr")
    for d in DIMENSIONS:
        x = [getattr(grades[a][k], d) for k in keys]
        y = [getattr(grades[b][k], d) for k in keys]
        exact = mean(i == j for i, j in zip(x, y))
        near = mean(abs(i - j) <= 1 for i, j in zip(x, y))
        corr = correlation(x, y) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")
        print(f"{d:12} {mean(x):20.2f} {mean(y):20.2f} {mean(y) - mean(x):+5.2f} {exact:6.0%} {near:8.0%} {corr:5.2f}")
    sa, sb = [grades[a][k].safe for k in keys], [grades[b][k].safe for k in keys]
    print(f"{'safe':12} {sum(sa):>17}/{len(keys)} {sum(sb):>17}/{len(keys)}  agree {mean(i == j for i, j in zip(sa, sb)):.0%}")
    for d in DIMENSIONS:
        low_a = {k for k in keys if getattr(grades[a][k], d) <= 2}
        low_b = {k for k in keys if getattr(grades[b][k], d) <= 2}
        print(f"flags <=2 {d}: both={len(low_a & low_b)} {a.split('-')[1]}-only={len(low_a - low_b)} {b.split('-')[1]}-only={len(low_b - low_a)}")


if __name__ == "__main__":
    main()
