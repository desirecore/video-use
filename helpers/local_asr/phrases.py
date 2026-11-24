"""Verify same-voice contextual alternatives against both original and separated audio."""

import statistics
from collections import defaultdict

import torch
from torchaudio.functional import forced_align, merge_tokens

from .fusion import midpoint, token


def unique_triples(words):
    positions = defaultdict(list)
    names = [token(word["text"]) for word in words]
    for i in range(1, len(words) - 1):
        context = tuple(names[i - 1 : i + 2])
        if all(context):
            positions[context].append(words[i]["start"])
    return {key: times[0] for key, times in positions.items() if len(times) == 1}


def align_model_clocks(document, whisper, independent):
    words = {word["id"]: word for word in document["words"]}
    offsets = []
    for window, a, b in zip(
        whisper["windows"], whisper["results"], independent["results"]
    ):
        first, second = unique_triples(a), unique_triples(b)
        for context in first.keys() & second.keys():
            if any(
                words[i]["start"] - 0.25 <= first[context] <= words[i]["end"] + 0.25
                for i in window["targets"]
            ):
                continue
            offset = second[context] - first[context]
            if abs(offset) <= 0.45:
                offsets.append(offset)
    median = statistics.median(offsets) if offsets else 0.0
    mad = (
        statistics.median(abs(value - median) for value in offsets) if offsets else 0.0
    )
    applied = (
        median if len(offsets) >= 5 and abs(median) <= 0.25 and mad <= 0.12 else 0.0
    )
    results = []
    for window, stream in zip(independent["windows"], independent["results"]):
        start = window["start_sample"] / 16000
        results.append(
            [
                {
                    **word,
                    "start": max(start, word["start"] - applied),
                    "end": max(start, word["end"] - applied),
                }
                for word in stream
            ]
        )
    return {**independent, "results": results}, {
        "anchors": len(offsets),
        "median_offset_s": median,
        "mad_s": mad,
        "applied_s": applied,
    }


def word_timestamps(result, offset, duration):
    tokens = list(result.tokens)
    times = list(result.timestamps)
    durations = list(getattr(result, "durations", []))
    if len(tokens) != len(times):
        raise ValueError(
            "Parakeet returned token and timestamp lists of different lengths"
        )
    output = []
    pieces = []
    for i, (text, time) in enumerate(zip(tokens, times)):
        starts_word = text.startswith(("▁", " ", "Ġ"))
        if starts_word and pieces:
            output.append(pieces)
            pieces = []
        stripped = text.lstrip("▁ Ġ")
        if stripped:
            elapsed = (
                durations[i]
                if len(durations) == len(tokens) and durations[i] > 0
                else 0.04
            )
            pieces.append((stripped, float(time), float(elapsed)))
    if pieces:
        output.append(pieces)
    return [
        {
            "text": "".join(text for text, _, _ in word),
            "start": offset + min(duration, max(0, word[0][1])),
            "end": offset + min(duration, max(word[0][1], word[-1][1] + word[-1][2])),
        }
        for word in output
    ]


def anchor_pair(path, words, position, weak):
    target = words[path[position]]
    speaker = target.get("speaker_id")
    if not speaker:
        return None

    def anchor(positions, edge, opposite):
        for index in positions:
            word = words[path[index]]
            if (
                word.get("speaker_id") not in (speaker, None)
                or abs(target[edge] - word[opposite]) > 1.15
            ):
                break
            if (
                path[index] not in weak
                and word.get("speaker_id") == speaker
                and token(word["text"])
            ):
                return index
        return None

    left = anchor(range(position - 1, max(-1, position - 6), -1), "start", "end")
    right = anchor(range(position + 1, min(len(path), position + 6)), "end", "start")
    return (left, right) if left is not None and right is not None else None


def recognized_phrase(words, first, last):
    a = [
        (i, abs(word["start"] - first["start"]))
        for i, word in enumerate(words)
        if token(word["text"]) == token(first["text"])
        and abs(word["start"] - first["start"]) <= 0.28
    ]
    b = [
        (i, abs(word["start"] - last["start"]))
        for i, word in enumerate(words)
        if token(word["text"]) == token(last["text"])
        and abs(word["start"] - last["start"]) <= 0.28
    ]
    options = [(x + y, i, j) for i, x in a for j, y in b if i + 1 < j]
    if not options:
        return None
    options.sort()
    if len(options) > 1 and options[1][0] - options[0][0] < 0.05:
        return None
    _, begin, end = options[0]
    phrase = words[begin : end + 1]
    return (
        [word["text"] for word in phrase]
        if all(token(word["text"]) for word in phrase)
        else None
    )


def plan_phrases(document, whisper, parakeet):
    words = {word["id"]: word for word in document["words"]}
    targets = {int(i): source for i, source in whisper["targets"].items()}
    independent, clock = align_model_clocks(document, whisper, parakeet)
    decodes = {
        name: dict(
            zip([window["source"] for window in model["windows"]], model["results"])
        )
        for name, model in (("large-v3", whisper), ("parakeet", independent))
    }
    plans = {}
    for target, stem in targets.items():
        for source, path in document["paths"].items():
            if target not in path:
                continue
            pair = anchor_pair(path, words, path.index(target), targets)
            if not pair:
                continue
            left, right = pair
            ids = path[left : right + 1]
            first, last = words[ids[0]], words[ids[-1]]
            original = [words[i]["text"] for i in ids]
            if first["end"] >= last["start"] or not all(
                token(text) for text in original
            ):
                continue
            identity = (tuple(ids), stem)
            if identity in plans:
                continue
            baseline = tuple(map(token, original))
            options = {baseline: {"words": original, "sources": ["original"]}}
            protected = [token(words[i]["text"]) for i in ids[1:-1] if i not in targets]
            for engine, streams in decodes.items():
                for audio in ("mix", stem):
                    candidate = recognized_phrase(streams[audio], first, last)
                    if not candidate:
                        continue
                    key = tuple(map(token, candidate))
                    remaining = iter(key[1:-1])
                    if not all(
                        any(expected == observed for observed in remaining)
                        for expected in protected
                    ):
                        continue
                    options.setdefault(key, {"words": candidate, "sources": []})[
                        "sources"
                    ].append(f"{engine}:{audio}")
            if len(options) > 1:
                plans[identity] = {
                    "ids": ids,
                    "stem": stem,
                    "speaker_id": first["speaker_id"],
                    "start": first["start"],
                    "end": last["end"],
                    "protected_ids": [i for i in ids if i not in targets],
                    "candidates": list(options.values()),
                }
    return {
        "clock_alignment": clock,
        "low_confidence_count": len(targets),
        "phrases": list(plans.values()),
    }


def acoustic_phrase(words, logp, vocab, hop, offset, blank):
    labels, owners = [], []
    text = [token(word) for word in words]
    for i, word in enumerate(text):
        if not word or any(char not in vocab for char in word):
            return None
        if i and "|" in vocab:
            labels.append(vocab["|"])
            owners.append(None)
        labels.extend(vocab[char] for char in word)
        owners.extend([i] * len(word))
    steps = len(logp)
    if steps < len(labels) + sum(a == b for a, b in zip(labels, labels[1:])):
        return None
    target = torch.tensor([labels], dtype=torch.long)
    loss = torch.nn.functional.ctc_loss(
        logp.unsqueeze(1), target, [steps], [len(labels)], blank=blank, reduction="sum"
    )
    if not torch.isfinite(loss):
        return None
    path, conf = forced_align(logp.unsqueeze(0), target.to(torch.int32), blank=blank)
    spans = merge_tokens(path[0], conf[0], blank=blank)
    if [span.token for span in spans] != labels:
        raise ValueError("Phrase alignment did not recover the target sequence")
    intervals = {}
    for owner, span in zip(owners, spans):
        if owner is not None:
            begin, end = offset + span.start * hop, offset + span.end * hop
            intervals[owner] = (intervals.get(owner, (begin, begin))[0], end)
    return {
        "log_probability_per_frame": -float(loss) / steps,
        "times": [intervals[i] for i in range(len(words))],
    }


def rank_phrase(phrase):
    candidates = phrase["candidates"]
    sources = ("mix", phrase["stem"])
    valid = [
        (i, [value["acoustic"][src]["log_probability_per_frame"] for src in sources])
        for i, value in enumerate(candidates)
        if all(value["acoustic"][src] is not None for src in sources)
    ]
    ranked = sorted(valid, key=lambda item: ((sum(item[1]), -item[0])), reverse=True)
    if len(ranked) < 2 or ranked[0][0] == 0 or 0 not in dict(valid):
        return None
    winner, score = ranked[0]
    gain = [new - old for new, old in zip(score, dict(valid)[0])]
    if (
        min(gain) < -0.015
        or sum(gain) / 2 < 0.045
        or (sum(score) - sum(ranked[1][1])) / 2 < 0.02
    ):
        return None
    return winner, sum(gain) / 2


def ranked_patches(scored):
    proposed = []
    for phrase in scored:
        best = rank_phrase(phrase)
        if best:
            proposed.append((best[1], phrase, phrase["candidates"][best[0]]))
    return sorted(proposed, key=lambda item: -item[0])


def reuse_existing(document, scored):
    result = {
        **document,
        "paths": {key: path.copy() for key, path in document["paths"].items()},
    }
    by_id = {word["id"]: word for word in document["words"]}
    changes = []
    used = set()
    for gain, phrase, option in ranked_patches(scored):
        ids = phrase["ids"]
        if used.intersection(ids):
            continue
        left, right = by_id[ids[0]], by_id[ids[-1]]
        spans = option["acoustic"]
        reused = []
        for text, mix, stem in zip(
            option["words"][1:-1],
            spans["mix"]["times"][1:-1],
            spans[phrase["stem"]]["times"][1:-1],
        ):
            near = [
                word["id"]
                for word in document["words"]
                if word.get("speaker_id") == phrase["speaker_id"]
                and token(word["text"]) == token(text)
                and left["end"] <= word["start"] < word["end"] <= right["start"]
                and all(
                    abs(midpoint(word) - (a + b) / 2) <= 0.18 for a, b in (mix, stem)
                )
            ]
            if len(near) != 1:
                break
            reused.append(near[0])
        if len(reused) != len(option["words"]) - 2 or len(set(reused)) != len(reused):
            continue
        if any(by_id[a]["end"] > by_id[b]["start"] for a, b in zip(reused, reused[1:])):
            continue
        if not (set(phrase["protected_ids"]) & set(ids[1:-1])) <= set(reused):
            continue
        updated = []
        for source, path in result["paths"].items():
            if ids[0] not in path:
                continue
            start = path.index(ids[0])
            if path[start : start + len(ids)] != ids:
                continue
            before = path[: start + 1]
            after = path[start + len(ids) - 1 :]
            if set(reused).intersection(before + after):
                continue
            result["paths"][source] = before + reused + after
            updated.append(source)
        if updated:
            used.update(ids)
            used.update(reused)
            changes.append(
                {
                    "anchors": [ids[0], ids[-1]],
                    "old_inner_ids": ids[1:-1],
                    "reused_ids": reused,
                    "updated_paths": updated,
                    "mean_log_probability_gain": gain,
                }
            )
    if changes:
        retained = {i for path in result["paths"].values() for i in path}
        result["words"] = [word for word in document["words"] if word["id"] in retained]
        result["text"] = " ".join(word["text"].strip() for word in result["words"])
    result["phrase_reuse"] = changes
    return result
