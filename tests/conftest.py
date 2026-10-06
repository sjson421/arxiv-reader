import zlib

import numpy as np
import pytest
from interest_memory import InterestMemory

from arxiv_reader import db, llm, pipeline
from arxiv_reader.ingest import Paper


def bag_of_words(texts):
    """Offline stand-in for fastembed: texts sharing words are similar."""
    vecs = np.zeros((len(texts), 64))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vecs[i, zlib.crc32(word.encode()) % 64] += 1
    return vecs


@pytest.fixture
def papers():
    """30 on-profile papers, then 30 off-profile ones. p59 ranks last in stage 1."""
    topics = ["agent tool use", "protein folding"]
    return [Paper(f"p{i}", f"{topics[i // 30]} study {i}", f"We study {topics[i // 30]}. Result {i}.", "cs.AI")
            for i in range(60)]


def open_mem(tmp_path):
    return InterestMemory(str(tmp_path / "memory.db"), embed=bag_of_words)


@pytest.fixture
def mem(tmp_path):
    m = open_mem(tmp_path)
    m.set_profile("Applied AI engineering")
    m.add_seeds(["agent tool use"])
    return m


@pytest.fixture
def app_db(tmp_path):
    return db.connect(tmp_path / "app.db")


class FakeClaude:
    def __init__(self, fail=False):
        self.fail = fail

    def review(self, papers, profile):
        if self.fail:
            raise llm.LLMError("down")
        return {p.id: llm.Review(id=p.id, score=5, reason="r", bullets=["a", "b", "c"])
                for p in papers}


def run(app_db, mem, day, papers, claude=None):
    claude = claude or FakeClaude()
    return pipeline.run(app_db, mem, day, papers, review=claude.review)
