"""Frozen research policies over cosine-ranked neighbors; no target labels."""


def vote(neighbors):
    """Sum nonnegative cosine weights; ties follow nearest-neighbor order."""
    totals = {}
    for category, similarity in neighbors:
        totals[category] = totals.get(category, 0.0) + max(0.0, similarity)
    if not totals or max(totals.values()) <= 0:
        return None
    return max(totals, key=totals.get)


def needs_laya(neighbors):
    """Choose the model branch using only retrieval output."""
    return (
        len(neighbors) < 2
        or neighbors[0][0] != neighbors[1][0]
        or vote(neighbors) is None
    )


def disagreement_hybrid(neighbors, laya):
    return laya if needs_laya(neighbors) else vote(neighbors)
