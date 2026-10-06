"""Replays past feedback: do kept papers outscore rejected ones? Run: uv run python -m evals.stage1"""

import tempfile
from datetime import datetime
from pathlib import Path

from interest_memory import InterestMemory

from arxiv_reader import db as appdb
from arxiv_reader.api import KEPT, REJECTED

SPLIT = 0.8  # train on the oldest 80% of days, test on the rest


def main() -> None:
    rows = appdb.items(appdb.connect(), "d.first_opened_at IS NOT NULL ORDER BY i.date")
    cut = int(len(rows) * SPLIT)
    train, test = rows[:cut], rows[cut:]
    with tempfile.TemporaryDirectory() as tmp:
        mem = InterestMemory(str(Path(tmp) / "memory.db"))
        for r in train:
            weight = REJECTED if r["rejected_at"] else KEPT
            mem.record(r["paper_id"], appdb.paper(r).text, weight, datetime.fromisoformat(r["date"]))
        scores = mem.score([appdb.paper(r).text for r in test], datetime.fromisoformat(test[-1]["date"]))
    kept = [s for s, r in zip(scores, test) if not r["rejected_at"]]
    rejected = [s for s, r in zip(scores, test) if r["rejected_at"]]
    pairs = [(k, x) for k in kept for x in rejected]
    auc = sum((k > x) + 0.5 * (k == x) for k, x in pairs) / len(pairs)
    print(f"train={len(train)} test={len(test)} kept={len(kept)} rejected={len(rejected)} AUC={auc:.3f}")


if __name__ == "__main__":
    main()
