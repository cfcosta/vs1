import re


def shingles(text):
    words = re.findall(r"\w+", re.sub(r"\d+", "NUMBER", text.lower()))
    return {tuple(words[i : i + 5]) for i in range(max(0, len(words) - 4))}


def independent(train, test):
    senders = {x["sender"] for x in train}
    bodies = [shingles(x["body"]) for x in train]
    return [
        x
        for x in test
        if x["sender"] not in senders
        and all(
            len(shingles(x["body"]) & b) / max(1, len(shingles(x["body"]) | b)) < 0.5
            for b in bodies
        )
    ]


def nearest(vectors, target, candidates, count=2):
    if target in candidates:
        raise ValueError("Self retrieval is forbidden")
    return sorted(
        candidates,
        key=lambda i: (-sum(a * b for a, b in zip(vectors[target], vectors[i])), i),
    )[:count]
