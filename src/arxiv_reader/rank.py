from collections.abc import Sequence

import numpy as np

DIVERSITY = 0.5  # how much similarity to an already-picked paper costs, against relevance in [0, 1]


def blend[T](stage1: Sequence[T], claude: dict[T, int]) -> list[T]:
    """Orders by 0.5 x stage-1 rank + 0.5 x Claude rank. Ranks, because embedding scores cluster high."""
    r1 = {x: n for n, x in enumerate(stage1)}
    r2 = {x: n for n, x in enumerate(sorted(stage1, key=lambda x: -claude[x]))}
    return sorted(stage1, key=lambda x: (r1[x] + r2[x], r1[x]))


def pick_diverse(vecs: np.ndarray, count: int) -> list[int]:
    """Greedy pick over rows in relevance order, minus a penalty for similarity to rows already picked."""
    n = len(vecs)
    relevance = 1 - np.arange(n) / n
    sims = vecs @ vecs.T
    picked: list[int] = []
    for _ in range(min(count, n)):
        penalty = sims[:, picked].max(axis=1) if picked else np.zeros(n)
        value = relevance - DIVERSITY * penalty
        value[picked] = -np.inf
        picked.append(int(value.argmax()))
    return picked
