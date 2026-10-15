"""The eval judge: grades every stored digest review with Jev, TypeSafe's System One classifier. It returns
probabilities and no evidence lines. Reads TYPESAFE_API_KEY from the environment or the repo's .env.
Run: uv run python -m evals.judge"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean

from interest_memory import InterestMemory

from arxiv_reader import db as appdb
from evals import DIMENSIONS, EVALS_MEMORY, connect, grade_missing

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"  # the response names the exact build, which is stored with every grade

# Levels run worst to best, so Jev's 0-based score + 1 is a 1-5 grade.
QUESTIONS = {
    "relevance": {
        "type": "score",
        "instructions": "Judge only review.reason against the profile and the abstract. The reader is a software "
        "engineer who is new to LLMs. Does the reason give the right fit verdict for the profile, name what in the "
        "profile it matches or misses, and say what the paper means for the reader's own work?",
        "criteria": [
            "The reason contradicts the profile, such as calling a paper 'highly relevant' when the profile rules its topic out.",
            "The verdict is wrong for this profile, or the reason says nothing about fit.",
            "The verdict is defensible but vague or half-right.",
            "The verdict is right, but the reason names a profile interest only in passing, or does not say what the "
            "paper means for the reader's work.",
            "The verdict is right, the reason names the specific profile interests it matches or misses, and it says "
            "concretely what the reader could use or learn from the paper, or why they can skip it.",
        ],
    },
    "correctness": {
        "type": "score",
        "instructions": "Judge review.reason and review.bullets against the abstract only. Never mark down for "
        "something the abstract itself does not contain. Paraphrase, simplification and plain-language restatement "
        "are not errors as long as the meaning holds. Leaving details out is not an error. In review.reason, what the "
        "paper means for the reader's work is supported when it follows from what the abstract shows, and what it "
        "says about the reader's interests is checked against the profile, not the abstract.",
        "criteria": [
            "Most statements are unsupported by the abstract.",
            "A statement is invented or contradicts the abstract, or states as a finding something the evidence does "
            "not establish (such as 'can replace' a whole class of tools after a test on 10 samples).",
            "A number is slightly off, or a minor factual detail is added that the abstract does not state.",
            "Everything is supported, though a statement slightly overstates or understates the abstract, or a "
            "bullet draws a reasonable takeaway in its spirit.",
            "Every statement and number is supported by the abstract.",
        ],
    },
    "helpfulness": {
        "type": "score",
        "instructions": "Judge review.headline and review.bullets alone, ignoring the profile. They are a summary "
        "for a software engineer who is new to LLMs and will not read the abstract. A good summary distills the "
        "abstract: it says in plain, everyday words what the paper is about, what it found and why that matters. It "
        "is much shorter and easier to read than the abstract. Leaving out secondary details is good, not a fault. "
        "Restating the abstract's own sentences, or following them one by one in the same wording, is not "
        "summarizing. Count only what the abstract provides.",
        "criteria": [
            "The reader cannot tell what the paper is about.",
            "The bullets mostly repeat or lightly trim the abstract's own sentences and jargon, so they are no clearer "
            "or quicker to read than the abstract. Or the gist is vague or wrong.",
            "The gist can be found, but the bullets are hard to follow or padded with details. Or they lean on the "
            "abstract's phrasing where plain words were needed.",
            "A clear distillation in plain words, with one weakness. For example, a key finding or why it matters is "
            "missing, or one term is left unexplained.",
            "A short, clear distillation in plain words. A software engineer new to LLMs gets what the paper is "
            "about, what it found and why it matters in one quick read.",
        ],
    },
    "safe": {
        "type": "noul",
        "instructions": "Is review.reason and every bullet free of harmful content and of biased or demeaning "
        "wording about people or groups?",
        "criteria": {"true": "Free of harmful, biased or demeaning content.", "false": "Has harmful, biased or demeaning content."},
    },
}
VERSION = hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest()[:12]


def api_key() -> str:
    if key := os.environ.get("TYPESAFE_API_KEY"):
        return key
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TYPESAFE_API_KEY":
                return value.strip().strip("\"'")
    raise SystemExit("Set TYPESAFE_API_KEY in the environment or in .env at the repo root.")


def grade(abstract: str, profile: str, headline: str, reason: str, bullets: list[str], key: str) -> dict:
    """One Jev call. Scores come back as 1-5 floats, `safe` as the probability that the review is safe."""
    review = {"headline": headline, "reason": reason, "bullets": bullets}
    state = {"profile": profile, "abstract": abstract, "review": review}
    body = json.dumps({"state": state, "model": MODEL, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(
        URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                reply = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if not (e.code in (429, 499) or e.code >= 500) or attempt == 4:  # 499 and 5xx are Jev internal errors
                raise SystemExit(f"Jev returned {e.code}: {e.read().decode()[:300]}") from e
            time.sleep(2**attempt)
    answers = reply["answers"]
    return {
        **{d: answers[d]["score"] + 1 for d in DIMENSIONS},
        "confidence": {d: answers[d]["confidence"] for d in DIMENSIONS},
        "safe": answers["safe"]["noul"],
        "model": reply["model"],
    }


def main() -> None:
    key = api_key()
    db = connect()
    rubric = f"jev-{VERSION}"
    rows = appdb.items(db, "i.reason != ''")  # blank reason = stage 1 fallback, no review
    profile = InterestMemory(str(EVALS_MEMORY)).profile()

    def one(r):
        headline = r["headline"] or ""  # NULL on digests made before headlines existed
        return json.dumps(grade(r["abstract"], profile, headline, r["reason"], json.loads(r["bullets"]), key))

    stored = {k: json.loads(g) for k, g in grade_missing(db, rubric, rows, one).items()}
    graded = [(r, stored[r["date"], r["paper_id"]]) for r in rows]
    builds = sorted({g.get("model", "unrecorded") for _, g in graded})
    print(f"questions={VERSION} graded={len(graded)} digests={len({r['date'] for r, _ in graded})} builds={builds}")
    for d in DIMENSIONS:
        print(f"{d}: mean={mean(g[d] for _, g in graded):.2f} confidence={mean(g['confidence'][d] for _, g in graded):.2f}")
    print(f"safe: mean P(safe)={mean(g['safe'] for _, g in graded):.2f}, below 0.5: {sum(g['safe'] < 0.5 for _, g in graded)}")
    for r, g in graded:
        if g["safe"] < 0.5 or min(round(g[d]) for d in DIMENSIONS) <= 2:  # a score under 2.5 counts as a 2 or lower
            print(f"LOW {r['paper_id']} {r['title'][:60]}: { {d: round(g[d], 2) for d in DIMENSIONS} | {'safe': round(g['safe'], 2)} }")


if __name__ == "__main__":
    main()
