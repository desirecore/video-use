"""Fuse timestamped ASR paths while preserving independent overlapping speech."""

import itertools
import math
import re
import statistics
from collections import Counter, defaultdict


CONTEXT_WINDOW_S = 0.20


TIGHT_MATCH_S = 0.06


def token(text):
    value = text.lower().replace("’", "'").strip(' \n\t.,!?;:"()[]')
    return value if re.fullmatch(r"[a-z0-9]+(?:['-][a-z0-9]+)*", value) else None


def midpoint(word):
    return (word["start"] + word["end"]) / 2


def fuse(sources):
    observations, owner, paths = [], [], {}
    for source, words in sources.items():
        paths[source] = []
        previous = 0.0
        for word in words:
            start, end = word["start"], word["end"]
            if (
                not all(map(math.isfinite, (start, end)))
                or not previous <= start <= end
            ):
                raise ValueError(f"Invalid time sequence in {source}")
            previous = start
            paths[source].append(len(observations))
            observations.append(word)
            owner.append(source)

    clusters = {i: {i} for i in range(len(observations))}
    membership = list(range(len(observations)))
    marks = [token(word["text"]) for word in observations]
    centers = [midpoint(word) for word in observations]

    def combine(first, second):
        a, b = sorted((membership[first], membership[second]))
        if a == b:
            return
        left, right = clusters[a], clusters[b]
        if {owner[i] for i in left} & {owner[i] for i in right}:
            return
        joined = left | right
        times = [centers[i] for i in joined]
        if max(times) - min(times) > CONTEXT_WINDOW_S:
            return
        for member in right:
            membership[member] = a
        clusters[a] = joined
        del clusters[b]

    bigrams = defaultdict(list)
    for path in paths.values():
        for a, b in zip(path, path[1:]):
            if marks[a] and marks[b]:
                bigrams[marks[a], marks[b]].append((a, b))
    evidence = Counter()
    for sequences in bigrams.values():
        for (a, b), (c, d) in itertools.combinations(sequences, 2):
            if (
                owner[a] != owner[c]
                and abs(centers[a] - centers[c]) <= CONTEXT_WINDOW_S
                and abs(centers[b] - centers[d]) <= CONTEXT_WINDOW_S
            ):
                evidence[tuple(sorted((a, c)))] += 1
                evidence[tuple(sorted((b, d)))] += 1
    for a, b in sorted(
        evidence,
        key=lambda pair: (
            -evidence[pair],
            abs(centers[pair[0]] - centers[pair[1]]),
            pair,
        ),
    ):
        combine(a, b)

    # Lone matching words need nearly identical physical intervals.
    same_token = defaultdict(list)
    for i, mark in enumerate(marks):
        if mark:
            same_token[mark].append(i)
    for group in same_token.values():
        ordered = sorted(group, key=centers.__getitem__)
        for a, b in zip(ordered, ordered[1:]):
            if owner[a] == owner[b] or centers[b] - centers[a] > TIGHT_MATCH_S:
                continue
            if all(
                abs(observations[a][edge] - observations[b][edge]) <= TIGHT_MATCH_S
                for edge in ("start", "end")
            ):
                combine(a, b)

    # A disagreement on one word between two shared neighbors refers to one slot.
    # Resolve it by independent source support, then prefer the mixed transcript.
    slots = defaultdict(set)
    for path in paths.values():
        for i in range(1, len(path) - 1):
            left, word, right = (membership[item] for item in path[i - 1 : i + 2])
            slots[left, right].add(word)
    priority = lambda cluster: (
        len(clusters[cluster]),
        any(owner[i] == "mix" for i in clusters[cluster]),
        -cluster,
    )
    aliases = {}
    for group in slots.values():
        if len(group) < 2:
            continue
        times = [
            statistics.median(centers[i] for i in clusters[cluster])
            for cluster in group
        ]
        if max(times) - min(times) > CONTEXT_WINDOW_S:
            continue
        winner = max(group, key=priority)
        for cluster in group - {winner}:
            if cluster not in aliases or priority(winner) > priority(aliases[cluster]):
                aliases[cluster] = winner

    def canonical(index):
        index = membership[index]
        while index in aliases:
            index = aliases[index]
        return index

    fused_paths = {
        source: list(dict.fromkeys(canonical(i) for i in path))
        for source, path in paths.items()
    }
    words = []
    for cluster in sorted({i for path in fused_paths.values() for i in path}):
        members = sorted(clusters[cluster], key=lambda i: (owner[i] != "mix", i))
        supported = sorted({owner[i] for i in members})
        stems = [source for source in supported if source != "mix"]
        word = {
            "id": cluster,
            "type": "word",
            "text": observations[members[0]]["text"],
            "start": statistics.median(observations[i]["start"] for i in members),
            "end": statistics.median(observations[i]["end"] for i in members),
            "sources": supported,
        }
        if len(stems) == 1:
            word["speaker_id"] = stems[0]
        words.append(word)
    words.sort(key=lambda word: (word["start"], word["end"], word["id"]))
    return {
        "text": " ".join(word["text"].strip() for word in words),
        "words": words,
        "paths": fused_paths,
        "fusion": {
            "input_words": len(observations),
            "distinct_words": len(words),
            "merged_duplicate_observations": len(observations) - len(clusters),
            "resolved_single_word_alternatives": len(aliases),
        },
    }
