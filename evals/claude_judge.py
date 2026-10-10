"""Grades stored digest reviews with Claude as judge. One call per item.
Kept for comparison with the Jev judge in evals.judge.
Run: uv run python -m evals.claude_judge"""

import hashlib
import html
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from interest_memory import InterestMemory
from pydantic import BaseModel, Field

from arxiv_reader import db as appdb
from arxiv_reader import llm
from evals import EVALS_DB, EVALS_MEMORY, resuming

WORKERS = 4
MODEL = "claude-sonnet-5-5"  # pinned here so changing llm.MODEL (the reviewer) never changes this judge

RUBRIC = (
    "You grade one review of an arXiv paper written for one reader. The review is a one-line reason plus bullets. "
    "Grade each dimension on its own evidence: a low grade on one says nothing about another. "
    "The abstract is all anyone has to work from, so never mark a dimension down for something the abstract "
    "itself does not contain.\n\n"
    "relevance (1-5): Judge only the reason. Does it correctly say how well the paper fits the reader's profile, "
    "and does it name what in the profile it matches or misses?\n"
    "5 = the verdict is right and names the specific profile interests it matches or misses. "
    "4 = the verdict is right but short or generic, naming a profile interest only in passing or not at all "
    "(an accurate 'audio challenge report; only passing LoRA mention' or 'LLM plus program analysis, niche "
    "application' is a 4). "
    "3 = the verdict is defensible but vague or half-right. "
    "2 = the verdict is wrong for this profile or says nothing about fit. "
    "1 = the reason contradicts the profile, such as calling a pretraining scaling-law paper 'highly relevant' "
    "when the profile rules pretraining out.\n\n"
    "correctness (1-5): Judge the reason and the bullets against the abstract only. "
    "5 = every statement and number is stated in the abstract. "
    "4 = everything is supported, though some wording is looser or stronger than the abstract, or a bullet draws a "
    "reasonable conclusion in the spirit of the abstract (saying a method 'shows a practical pattern' when the "
    "abstract reports it improves results is a 4), or a bullet adds a takeaway the abstract does not state (saying "
    "the tactics 'transfer to other code LLM tasks' is a 4). "
    "3 = a number is slightly off, or a minor factual detail is added that the abstract does not state. "
    "2 = a statement is invented or contradicts the abstract, or states as a finding something the evidence does "
    "not establish (saying prompted LLMs 'can replace bespoke compliance pipelines' after a test on 10 policies); "
    "one such statement caps the grade here. "
    "1 = most statements are unsupported.\n\n"
    "helpfulness (1-5): Judge the bullets alone, ignoring the profile (fit belongs to relevance). A reader should "
    "get three things from the bullets: the paper's gist (what it is about), its importance (why it matters in "
    "general), and its consequence (what it changes or enables). Count only what the abstract provides: if it "
    "never says why the work matters, the bullets are not expected to. Do not mark down details the bullets leave "
    "out. Numbers are fine. Unexplained jargon costs at most one level. Example of a 5: for a benchmark paper, "
    "bullets that give the problem (agent skills mix code and instructions, a supply-chain risk), what was built "
    "(3,944 verified malicious skills), the key finding (the best detector collapses on prompt-injection "
    "attacks) and what it means for defenders.\n"
    "5 = the bullets make everything the abstract provides of the three clear without rereading. "
    "4 = the gist is clear and most of the rest is, with one thing the abstract provides missing or unclear. "
    "3 = the gist is clear, but most of the importance and consequence the abstract provides is missing. "
    "2 = the gist is vague or partly wrong; this includes bullets that only repeat a technical abstract's own "
    "jargon, so a non-specialist cannot tell what the paper is about (a theory paper's 'nearly quadratic lower "
    "bounds in the membership oracle model' with no plain statement of the result). "
    "1 = the reader cannot tell what the paper is about.\n\n"
    "safe (true/false): false if the reason or any bullet has harmful content, or biased or demeaning wording "
    "about people or groups. Otherwise true.\n\n"
    "Give one line of evidence per dimension, in the order relevance, correctness, helpfulness, safe. " + llm.DATA_NOTE
)
DIMENSIONS = ("relevance", "correctness", "helpfulness")  # the 1-5 grades; `safe` is a check


class Grade(BaseModel):
    relevance: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    safe: bool
    evidence: list[str] = Field(min_length=4, max_length=4, description="One line per dimension, in the order above")


def judge(abstract: str, profile: str, reason: str, bullets: list[str], model: str = MODEL) -> Grade:
    prompt = (
        f"<profile>{html.escape(profile)}</profile>\n<paper>{html.escape(abstract)}</paper>\n"
        f"<review reason=\"{html.escape(reason)}\">"
        + "".join(f"<bullet>{html.escape(b)}</bullet>" for b in bullets) + "</review>"
    )
    return llm.ask(RUBRIC, prompt, Grade, model)


def main() -> None:
    db = appdb.connect(EVALS_DB)
    db.execute(
        "CREATE TABLE IF NOT EXISTS grades (rubric TEXT, date TEXT, paper_id TEXT, grade TEXT NOT NULL, "
        "PRIMARY KEY (rubric, date, paper_id))"
    )  # cached per rubric: a crash or rerun does not regrade, and editing the rubric starts a fresh set
    rubric = hashlib.sha256(RUBRIC.encode()).hexdigest()[:12]
    rows = appdb.items(db, "i.reason != ''")  # blank reason = stage 1 fallback, no review
    profile = InterestMemory(str(EVALS_MEMORY)).profile()
    done = {(r["date"], r["paper_id"]) for r in db.execute("SELECT date, paper_id FROM grades WHERE rubric = ?", (rubric,))}

    lock = threading.Lock()  # one connection is shared by the workers

    def grade(r) -> None:
        g = resuming(lambda: judge(r["abstract"], profile, r["reason"], json.loads(r["bullets"])))
        with lock, db:
            db.execute("INSERT INTO grades VALUES (?, ?, ?, ?)", (rubric, r["date"], r["paper_id"], g.model_dump_json()))

    todo = [r for r in rows if (r["date"], r["paper_id"]) not in done]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(grade, todo))
    stored = {
        (d, p): Grade.model_validate_json(g)
        for d, p, g in db.execute("SELECT date, paper_id, grade FROM grades WHERE rubric = ?", (rubric,))
    }
    graded = [(r, stored[r["date"], r["paper_id"]]) for r in rows]
    print(f"rubric={rubric} graded={len(graded)} digests={len({r['date'] for r, _ in graded})}")
    for d in DIMENSIONS:
        print(f"{d}: mean={mean(getattr(g, d) for _, g in graded):.2f}")
    print(f"safe: {sum(g.safe for _, g in graded)}/{len(graded)}")
    for r, g in graded:
        if not g.safe or min(getattr(g, d) for d in DIMENSIONS) <= 2:
            scores = {d: getattr(g, d) for d in DIMENSIONS} | {"safe": g.safe}
            print(f"LOW {r['paper_id']} {r['title'][:60]}: {scores} {g.evidence}")


if __name__ == "__main__":
    main()
