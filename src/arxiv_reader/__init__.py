import argparse
import sys
import urllib.error
from datetime import date

from arxiv_reader.db import DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(prog="arxiv-reader", description="Daily arXiv digest that learns your interests.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("run", help="build today's digest (the default)")
    init = sub.add_parser("init", help="set up your profile and seed phrases from a paragraph")
    init.add_argument("paragraph", nargs="?", help="your interests; read from stdin if omitted")
    sub.add_parser("rewrite-profile", help="update the profile from the last 5 weeks of feedback")
    serve = sub.add_parser("serve", help="serve the page on 127.0.0.1")
    serve.add_argument("--port", type=int, default=8732)
    args = parser.parse_args()

    # Imported here so `--help` stays fast; these pull in numpy, fastembed and FastAPI.
    from interest_memory import InterestMemory

    from arxiv_reader import db, ingest, pipeline

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def memory() -> InterestMemory:
        return InterestMemory(str(DATA_DIR / "memory.db"))

    match args.cmd:
        case "init":
            interests = pipeline.init(memory(), args.paragraph or sys.stdin.read())
            print(interests.profile, "", *(f"- {s}" for s in interests.seeds), sep="\n")
        case "rewrite-profile":
            profile = pipeline.rewrite_profile(db.connect(), memory(), date.today())
            print(profile or "No opened digests in the last 5 weeks, so the profile is unchanged.")
        case "serve":
            import uvicorn

            from arxiv_reader.api import create_app

            uvicorn.run(create_app(db.connect, memory), host="127.0.0.1", port=args.port)
        case _:
            try:
                day, papers = ingest.fetch()
            except (urllib.error.URLError, TimeoutError) as e:
                sys.exit(f"Could not fetch the arXiv feed, so the last digest stays: {e}")
            for paper, bullets, reason in pipeline.run(db.connect(), memory(), day or date.today(), papers):
                print(f"\n{paper.title}\n{paper.link}", *(f"  - {b}" for b in bullets), sep="\n")
