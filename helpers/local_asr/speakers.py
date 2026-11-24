"""Assign anonymous voices using diarization, across-track evidence and recording-wide clusters."""

from collections import defaultdict

import numpy as np
import sherpa_onnx as sherpa

from .fusion import midpoint, token


RATE = 16000


TICKS = 100


def load_models(segmentation, speaker_model, num_speakers):
    embedding = sherpa.SpeakerEmbeddingExtractorConfig(
        model=str(speaker_model), num_threads=2
    )
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation), window_shift_ratio=0.1
            ),
            num_threads=2,
        ),
        embedding=embedding,
        clustering=sherpa.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=0.5
        ),
        min_duration_on=0.15,
        min_duration_off=0.2,
    )
    if not config.validate():
        raise ValueError("Diarization model configuration is invalid")
    return sherpa.OfflineSpeakerDiarization(config), sherpa.SpeakerEmbeddingExtractor(
        embedding
    )


def normalized_vector(items):
    v = np.asarray(items, dtype=np.float32)
    length = np.linalg.norm(v)
    return v / length if length > 0 else None


def embedding(extractor, samples):
    if len(samples) < round(0.32 * RATE) or np.sqrt(np.mean(samples**2)) < 0.004:
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(RATE, samples)
    stream.input_finished()
    return (
        normalized_vector(extractor.compute(stream))
        if extractor.is_ready(stream)
        else None
    )


def diarization_mask(segments, duration):
    speakers = sorted({f"speaker-{int(seg.speaker)}" for seg in segments})
    ticks = round(duration * TICKS)
    active = np.zeros((len(speakers), ticks), dtype=bool)
    rows = {speaker: i for i, speaker in enumerate(speakers)}
    for seg in segments:
        a, b = max(0, round(seg.start * TICKS)), min(ticks, round(seg.end * TICKS))
        active[rows[f"speaker-{int(seg.speaker)}"], a:b] = True
    return speakers, active


def speaker_prototypes(extractor, audio, speakers, activity):
    result = {}
    for speaker, row in zip(speakers, activity):
        exclusive = row & (activity.sum(axis=0) == 1)
        ends = np.diff(np.r_[False, exclusive, False].astype(np.int8))
        regions = [
            (a, b)
            for a, b in zip(np.flatnonzero(ends == 1), np.flatnonzero(ends == -1))
            if b - a >= 38
        ]
        longest = sorted(regions, key=lambda interval: -(interval[1] - interval[0]))[:4]
        embeddings = [
            embedding(
                extractor, audio[(a + 4) * RATE // TICKS : (b - 4) * RATE // TICKS]
            )
            for a, b in longest
        ]
        usable = [anchor for anchor in embeddings if anchor is not None]
        prototype = normalized_vector(np.mean(usable, axis=0)) if usable else None
        if prototype is not None:
            result[speaker] = prototype
    return result


def acoustic_strength(observation):
    return max(observation.get("fit", 0.0), 0.65 * observation.get("agreement", 0.0))


def assign_words(
    words, paths, acoustic, speakers, activity, prototypes, extractor, audio
):
    stems = {name: wave for name, wave in audio.items() if name != "mix"}
    windows = {}
    output = []
    for original in words:
        word = {key: value for key, value in original.items() if key != "speaker_id"}
        a = max(0, min(activity.shape[1] - 1, round((word["start"] - 0.02) * TICKS)))
        b = min(activity.shape[1], max(a + 1, round((word["end"] + 0.02) * TICKS)))
        support = dict(zip(speakers, activity[:, a:b].mean(axis=1)))
        active = [speaker for speaker, share in support.items() if share >= 0.20]
        if len(active) == 1 and support[active[0]] >= 0.60:
            word.update(speaker_id=active[0], speaker_assignment="exclusive_activity")
        elif len(active) > 1:
            observations = acoustic.get(str(word["id"]), {})
            rank = []
            center = round(midpoint(word) / 0.12) * 0.12
            for source, wave in stems.items():
                strength = acoustic_strength(observations.get(source, {}))
                if source not in word["sources"] and strength < 0.28:
                    continue
                window = (source, center)
                if window not in windows:
                    lo, hi = (
                        max(0, round((center - 0.5) * RATE)),
                        round((center + 0.5) * RATE),
                    )
                    windows[window] = embedding(extractor, wave[lo:hi])
                vector = windows[window]
                if vector is None:
                    continue
                for speaker in active:
                    if speaker not in prototypes:
                        continue
                    similarity = float(vector @ prototypes[speaker])
                    score = (
                        similarity
                        + 0.20 * strength
                        + 0.04 * (source in word["sources"])
                    )
                    rank.append((score, similarity, speaker))
            if rank:
                score, similarity, speaker = max(rank)
                rival = max(
                    (value for value, _, label in rank if label != speaker),
                    default=-1.0,
                )
                if similarity >= 0.45 and score - rival >= 0.06:
                    word.update(
                        speaker_id=speaker, speaker_assignment="cross_track_voice"
                    )
        output.append(word)

    # Bridge only short uncertain words bracketed by the same assigned voice.
    by_id = {word["id"]: word for word in output}
    for word in output:
        if word.get("speaker_id"):
            continue
        evidence = set()
        for path in paths.values():
            if word["id"] not in path:
                continue
            i = path.index(word["id"])
            if i == 0 or i == len(path) - 1:
                continue
            previous, following = by_id[path[i - 1]], by_id[path[i + 1]]
            voice = previous.get("speaker_id")
            if (
                voice
                and following.get("speaker_id") == voice
                and word["start"] - previous["end"] <= 0.25
                and following["start"] - word["end"] <= 0.25
            ):
                evidence.add(voice)
        if len(evidence) == 1:
            word.update(
                speaker_id=evidence.pop(), speaker_assignment="neighboring_context"
            )
    return output


def protected_context(words, paths, acoustic):
    by_id = {word["id"]: word for word in words}
    strong = {
        int(i)
        for i, observations in acoustic.items()
        if any(
            value["fit"] >= 0.30 or value["agreement"] >= 0.80
            for value in observations.values()
        )
    }
    protected = set()
    for path in paths.values():
        for a, b, c in zip(path, path[1:], path[2:]):
            first, middle, last = by_id[a], by_id[b], by_id[c]
            speaker = middle.get("speaker_id")
            if (
                speaker
                and first.get("speaker_id") == last.get("speaker_id") == speaker
                and a in strong
                and c in strong
                and -0.08 <= middle["start"] - first["end"] <= 0.20
                and -0.08 <= last["start"] - middle["end"] <= 0.20
            ):
                protected.add(b)
    return protected


def filter_words(words, paths, acoustic, protect_context=False):
    rejected = {}
    confidence = {}
    protected = protected_context(words, paths, acoustic) if protect_context else set()
    for word in words:
        scores = list(acoustic.get(str(word["id"]), {}).values())
        if not scores:
            continue
        top_fit = max(score["fit"] for score in scores)
        top_agreement = max(score["agreement"] for score in scores)
        confidence[word["id"]] = (len(word["sources"]), top_fit, top_agreement)
        if (
            len(word["sources"]) == 1
            and top_fit < 0.16
            and top_agreement < 0.65
            and word["id"] not in protected
        ):
            rejected[word["id"]] = "weak_acoustic_support"

    membership = defaultdict(set)
    for name, path in paths.items():
        for i in path:
            membership[i].add(name)
    groups = defaultdict(list)
    for word in words:
        speaker = word.get("speaker_id")
        name = token(word["text"])
        if speaker and name and word["id"] in confidence and word["id"] not in rejected:
            groups[speaker, name].append(word)
    for group in groups.values():
        remaining = []
        for word in sorted(
            group, key=lambda item: confidence[item["id"]], reverse=True
        ):
            if any(
                min(other["end"], word["end"]) - max(other["start"], word["start"])
                >= 0.025
                and abs(midpoint(other) - midpoint(word)) <= 0.15
                and not membership[other["id"]] & membership[word["id"]]
                for other in remaining
            ):
                rejected[word["id"]] = "duplicate_word_from_same_voice"
            else:
                remaining.append(word)

    selected = [word for word in words if word["id"] not in rejected]
    by_id = {word["id"]: word for word in selected}
    selected_paths = {}
    for source, path in paths.items():
        groups = [[]]
        removed = False
        for i in path:
            if i in rejected:
                removed = True
                continue
            if (
                removed
                and groups[-1]
                and by_id[i]["start"] - by_id[groups[-1][-1]]["end"] > 0.25
            ):
                groups.append([])
            groups[-1].append(i)
            removed = False
        selected_paths.update(
            {f"{source}:{i}": group for i, group in enumerate(groups) if group}
        )
    return selected, selected_paths, rejected


STRONG_VOICE = {"exclusive_activity", "cross_track_voice"}


def overlapping_activity(word, speaker, turns):
    start, end = word["start"] - 0.025, word["end"] + 0.025
    return max(
        (
            max(0, min(end, row["end"]) - max(start, row["start"])) / (end - start)
            for row in turns
            if row["speaker_id"] == speaker
        ),
        default=0.0,
    )


def voice_continuity(document):
    original = {word["id"]: word for word in document["words"]}
    proposals = defaultdict(list)
    for source, path in document["paths"].items():
        for i in range(1, len(path) - 1):
            word = original[path[i]]
            if word.get("speaker_assignment") == "exclusive_activity":
                continue
            a = next(
                (
                    original[j]
                    for j in reversed(path[max(0, i - 5) : i])
                    if original[j].get("speaker_assignment") in STRONG_VOICE
                    and -0.09 <= word["start"] - original[j]["end"] <= 0.90
                ),
                None,
            )
            b = next(
                (
                    original[j]
                    for j in path[i + 1 : i + 6]
                    if original[j].get("speaker_assignment") in STRONG_VOICE
                    and -0.09 <= original[j]["start"] - word["end"] <= 0.90
                ),
                None,
            )
            if not a or not b or a.get("speaker_id") != b.get("speaker_id"):
                continue
            speaker = a.get("speaker_id")
            if (
                not speaker
                or overlapping_activity(word, speaker, document["speaker_turns"]) < 0.48
            ):
                continue
            exclusive = (
                a["speaker_assignment"]
                == b["speaker_assignment"]
                == "exclusive_activity"
            )
            proposals[word["id"]].append((speaker, source, exclusive))

    revised = []
    changes = []
    for original_word in document["words"]:
        word = original_word.copy()
        votes = proposals.get(word["id"], [])
        choices = {vote[0] for vote in votes}
        if len(choices) == 1:
            speaker = next(iter(choices))
            previous = word.get("speaker_id")
            confident = len({vote[1] for vote in votes}) >= 2 or any(
                vote[2] for vote in votes
            )
            if speaker != previous and (
                word.get("speaker_assignment") != "cross_track_voice" or confident
            ):
                word.update(
                    speaker_id=speaker, speaker_assignment="two_sided_continuity"
                )
                changes.append(
                    {"id": word["id"], "previous": previous, "speaker": speaker}
                )
        revised.append(word)
    return {**document, "words": revised, "continuity_changes": changes}


def whole_clip_windows(extractor, audio, streams, activity):
    duration = len(audio["mix"]) / RATE
    result = []
    for source, waveform in audio.items():
        words = sorted(streams[source], key=lambda word: word["start"])
        left = 0
        for center in np.arange(0.2, duration, 0.40):
            start, end = max(0, center - 0.65), min(duration, center + 0.65)
            while left < len(words) and words[left]["end"] <= start:
                left += 1
            vocal = 0
            for i in range(left, len(words)):
                word = words[i]
                if word["start"] >= end:
                    break
                vocal += max(0, min(end, word["end"]) - max(start, word["start"]))
            if vocal < 0.28:
                continue
            if source == "mix":
                active = activity[:, round(100 * start) : round(100 * end)]
                if (active.sum(axis=0) == 1).mean() < 0.70:
                    continue
            vector = embedding(
                extractor, waveform[round(start * RATE) : round(end * RATE)]
            )
            if vector is not None:
                result.append(
                    {
                        "source": source,
                        "center": float(center),
                        "voice": vector,
                        "weight": min(1.0, vocal),
                    }
                )
    return result


def fit_clusters(windows, prototypes):
    labels = sorted(prototypes)
    anchors = np.stack([prototypes[label] for label in labels])
    prototypes = anchors.copy()
    samples = np.stack([row["voice"] for row in windows])
    weights = np.asarray([row["weight"] for row in windows], dtype=np.float32)
    for _ in range(8):
        similarity = samples @ prototypes.T
        best = similarity.argmax(axis=1)
        if len(labels) > 1:
            largest = np.sort(similarity, axis=1)[:, -2:]
            reliable = (largest[:, 1] >= 0.43) & (largest[:, 1] - largest[:, 0] >= 0.05)
        else:
            reliable = similarity[:, 0] >= 0.43
        for i, anchor in enumerate(anchors):
            chosen = reliable & (best == i)
            prototypes[i] = normalized_vector(
                6 * anchor + np.sum(samples[chosen] * weights[chosen, None], axis=0)
            )

    similarity = samples @ prototypes.T
    result = defaultdict(list)
    for i, window in enumerate(windows):
        ranking = np.argsort(similarity[i])
        best = ranking[-1]
        score = similarity[i, best]
        rival = similarity[i, ranking[-2]] if len(labels) > 1 else 0.0
        if score >= 0.44 and score - rival >= 0.045:
            result[window["source"]].append(
                (window["center"], labels[best], float(score))
            )
    return result


def infer_label(word, windows, turns, evidence):
    center = midpoint(word)
    current = word.get("speaker_id")
    voices = defaultdict(list)
    for source, observations in windows.items():
        if source == "mix":
            continue
        strength = acoustic_strength(evidence.get(source, {}))
        present = source in word["sources"]
        if not present and strength < 0.40:
            continue
        local = defaultdict(float)
        for at, label, similarity in observations:
            distance = abs(center - at)
            if distance <= 0.55:
                local[label] += (similarity - 0.36) * (1 - distance / 0.75)
        if not local:
            continue
        label = max(local, key=local.get)
        if local[label] / sum(local.values()) < 0.65:
            continue
        weight = min(1.0, local[label]) * (0.65 + 0.85 * strength + 0.27 * present)
        voices[label].append(weight)
    if not voices:
        return current
    rank = []
    for label, votes in voices.items():
        votes.sort(reverse=True)
        score = (
            votes[0]
            + 0.20 * sum(votes[1:])
            + 0.12 * overlapping_activity(word, label, turns)
        )
        score += 0.09 * (label == current)
        rank.append((score, label))
    score, label = max(rank)
    rival = max((v for v, candidate in rank if candidate != label), default=0)
    if label == current or score < 0.19 or score - rival < 0.10:
        return current
    if word.get("speaker_assignment") == "exclusive_activity" and (
        len(voices[label]) < 2 or score - rival < 0.24
    ):
        return current
    if overlapping_activity(word, label, turns) < 0.20 and len(voices[label]) < 2:
        return current
    return label
