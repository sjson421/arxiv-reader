"""Compares prompt variants for the headline, bullets and reason. Each variant reviews the same sample of stored
papers and Jev grades the result; grades are cached per variant, so a rerun only does what is missing.
Run: uv run python -m evals.optimize [--size 60] [--seed 0] [variant ...]"""

import argparse
import hashlib
import json
import random
from statistics import mean

from interest_memory import InterestMemory

from arxiv_reader import db as appdb
from arxiv_reader import llm
from evals import DIMENSIONS, EVALS_MEMORY, connect, grade_missing, resuming, stored_grades
from evals.judge import VERSION, api_key, grade

BATCH = 30

# The prompt before this tuning, kept as the baseline to beat.
PREVIOUS = (
    "Give each paper a headline, written from its abstract and not copied from its title, so the reader knows "
    "right away what the paper is about and what it does or found. "
    "Write 1 to 5 bullets per paper so that a reader who sees only the bullets, with no title and no abstract, "
    "understands what the paper is about. Put them in this order: the problem it tackles, what it does about it, "
    "what it found (with the key number when there is one), and why it matters. "
    "Each bullet is one short, self-contained sentence in plain words. "
)

# Add a candidate here, then: uv run python -m evals.optimize current <name>
VARIANTS = {"previous": PREVIOUS, "current": llm.WRITING}


def sample(db, size: int, seed: int):
    rows = appdb.items(db, "i.reason != ''")
    return random.Random(seed).sample(rows, size)


def generate(rows, profile: str, writing: str) -> dict[tuple[str, str], llm.Review]:
    """Reviews for `rows`, batched by day like the pipeline does."""
    out = {}
    for i in range(0, len(rows), BATCH):
        chunk = rows[i : i + BATCH]
        got = resuming(lambda: llm.review([appdb.paper(r) for r in chunk], profile, writing))
        out |= {(r["date"], r["paper_id"]): got[r["paper_id"]] for r in chunk if r["paper_id"] in got}
    return out


def run_variant(db, rows, profile: str, name: str, key: str) -> dict:
    writing = VARIANTS[name]
    picked = "".join(r["date"] + r["paper_id"] for r in rows)
    rubric = f"opt-{name}-{hashlib.sha256((writing + picked).encode()).hexdigest()[:8]}-{VERSION}"
    reviews = {}
    if len(stored_grades(db, rubric)) < len(rows):
        reviews = generate(rows, profile, writing)
    todo = [r for r in rows if (r["date"], r["paper_id"]) in reviews or (r["date"], r["paper_id"]) in stored_grades(db, rubric)]

    def one(r):
        rv = reviews[r["date"], r["paper_id"]]
        g = grade(r["abstract"], profile, rv.reason, rv.bullets, key)
        return json.dumps(g | {"review": rv.model_dump()})

    grades = {k: json.loads(g) for k, g in grade_missing(db, rubric, todo, one).items()}
    return grades


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0, help="which random sample of stored papers to use")
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args()
    key, db = api_key(), connect()
    profile = InterestMemory(str(EVALS_MEMORY)).profile()
    rows = sample(db, args.size, args.seed)
    results = {n: run_variant(db, rows, profile, n, key) for n in args.variants}
    print(f"sample={len(rows)} judge={VERSION}")
    for n, grades in results.items():
        means = {d: mean(g[d] for g in grades.values()) for d in DIMENSIONS}
        print(f"{n:10} " + "  ".join(f"{d}={v:.2f}" for d, v in means.items()) + f"  safe={mean(g['safe'] for g in grades.values()):.2f}  n={len(grades)}")
    for n, grades in results.items():
        worst = sorted(grades.items(), key=lambda kv: min(kv[1][d] for d in DIMENSIONS))[:3]
        for (date, pid), g in worst:
            print(f"LOW {n} {pid}: { {d: round(g[d], 2) for d in DIMENSIONS} } bullets={g['review']['bullets']} reason={g['review']['reason']}")


if __name__ == "__main__":
    main()
