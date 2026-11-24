"""Calibrate word cuts and test candidates against local CTC acoustic evidence."""

import itertools
import math
import statistics
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import torch
from opencc import OpenCC
from torchaudio.functional import forced_align, merge_tokens


SAMPLE_RATE = 16_000


def align_words(audio, words, language, processor, model):
    converter = OpenCC("t2s") if language == "zh" else None
    vocabulary = {
        token.casefold(): index
        for token, index in processor.tokenizer.get_vocab().items()
        if len(token) == 1
    }
    delimiter = (
        processor.tokenizer.word_delimiter_token_id if language == "en" else None
    )
    blank = model.config.pad_token_id
    targets = []
    owners = []
    for i, word in enumerate(words):
        text = unicodedata.normalize("NFKC", word["text"]).casefold().replace("’", "'")
        text = converter.convert(text) if converter else text
        labels = [
            vocabulary[char]
            for char in text
            if (char.isalnum() or char == "'") and char in vocabulary
        ]
        if labels:
            if targets and delimiter is not None and delimiter != blank:
                targets.append(delimiter)
                owners.append(None)
            targets.extend(labels)
            owners.extend([i] * len(labels))
    if not targets:
        return words.copy(), set()

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        log_probs = model(**inputs).logits.log_softmax(dim=-1)
        minimum_frames = len(targets) + sum(
            a == b for a, b in zip(targets, targets[1:])
        )
        if log_probs.size(1) < minimum_frames:
            return words.copy(), set()
        path, scores = forced_align(
            log_probs, torch.tensor([targets], dtype=torch.int32), blank=blank
        )
    spans = merge_tokens(path[0], scores[0], blank=blank)
    if [span.token for span in spans] != targets:
        raise ValueError("CTC alignment did not recover the original target sequence")

    # Frame timestamps use the feature extractor's actual hop and receptive field.
    hop = 1
    window = 1
    for kernel, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        window += (kernel - 1) * hop
        hop *= stride
    duration = len(audio) / SAMPLE_RATE
    intervals = {}
    for owner, span in zip(owners, spans):
        if owner is None:
            continue
        start = span.start * hop / SAMPLE_RATE
        end = min(duration, ((span.end - 1) * hop + window) / SAMPLE_RATE)
        intervals[owner] = (intervals.get(owner, (start, start))[0], end)

    # Consecutive receptive fields overlap slightly. Keep word intervals ordered,
    # and constrain words outside the acoustic model's vocabulary to the gap.
    next_anchor = duration
    right_bounds = {}
    output = words.copy()
    for index in reversed(range(len(words))):
        if index not in intervals:
            right_bounds[index] = next_anchor
            continue
        start, end = intervals[index]
        output[index] = {**words[index], "start": start, "end": min(next_anchor, end)}
        next_anchor = start
    previous_end = 0.0
    for index, word in enumerate(output):
        if index in right_bounds:
            right = right_bounds[index]
            start = max(previous_end, min(right, word["start"]))
            output[index] = {
                **word,
                "start": start,
                "end": max(start, min(right, word["end"])),
            }
        previous_end = output[index]["end"]
    return output, set(intervals)


def quiet_intervals(audio):
    """Find sustained low-energy gaps relative to the recording's noise floor."""
    frame = SAMPLE_RATE // 200
    count = len(audio) // frame
    if count < 3:
        return []
    power = np.square(audio[: count * frame].reshape(count, frame)).mean(axis=1)
    smooth = np.convolve(np.pad(power, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    floor, voiced = np.quantile(smooth, [0.2, 0.9])
    threshold = floor + max(0.5 * floor, 0.003 * (voiced - floor))
    if voiced <= threshold:
        return []
    changes = np.diff(np.r_[False, smooth <= threshold, False].astype(np.int8))
    timescale = frame / SAMPLE_RATE
    return [
        (start * timescale, end * timescale)
        for start, end in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        )
        if (end - start) * timescale >= 0.25
    ]


def snap_to_pauses(audio, words, aligned):
    output = [word.copy() for word in words]
    if not output:
        return output, 0
    maximum_shift = 0.35
    changes = 0
    duration = len(audio) / SAMPLE_RATE
    for left, right in quiet_intervals(audio):
        if left == 0:
            word = output[0]
            if (
                0 in aligned
                and abs(word["start"] - right) <= maximum_shift
                and right < word["end"]
            ):
                word["start"] = right
                changes += 1
        elif duration - right < 0.01:
            word = output[-1]
            if (
                len(words) - 1 in aligned
                and abs(word["end"] - left) <= maximum_shift
                and word["start"] < left
            ):
                word["end"] = left
                changes += 1
        else:
            candidates = (
                (abs(a["end"] - left) + abs(b["start"] - right), i)
                for i, (a, b) in enumerate(zip(output, output[1:]))
                if i in aligned
                and i + 1 in aligned
                and abs(a["end"] - left) <= maximum_shift
                and abs(b["start"] - right) <= maximum_shift
                and a["start"] < left
                and right < b["end"]
            )
            closest = min(candidates, default=None)
            if closest is not None:
                _, i = closest
                output[i]["end"] = left
                output[i + 1]["start"] = right
                changes += 2
    return output, changes


def examine_window(log_probs, labels, blank, letters, target):
    if log_probs.size(0) < len(labels) + sum(
        a == b for a, b in zip(labels, labels[1:])
    ):
        return {"fit": 0.0, "agreement": 0.0}
    path, scores = forced_align(
        log_probs.unsqueeze(0), torch.tensor([labels], dtype=torch.int32), blank=blank
    )
    spans = merge_tokens(path[0], scores[0], blank=blank)
    fit = math.exp(statistics.mean(span.score for span in spans)) if spans else 0.0
    best_ids = [
        i for i, _ in itertools.groupby(log_probs.argmax(dim=1).tolist()) if i != blank
    ]
    phrase = "".join(letters.get(i, "") for i in best_ids)
    alternatives = [part for part in phrase.split("|") if part]
    agreement = max(
        (
            SequenceMatcher(None, target, part, autojunk=False).ratio()
            for part in alternatives
        ),
        default=0.0,
    )
    return {"fit": round(fit, 5), "agreement": round(agreement, 5)}
