"""Resolve a single input's timed words, parallel paths and anonymous speakers."""

import math
import wave

import numpy as np
import sherpa_onnx as sherpa
import torch
from faster_whisper import WhisperModel
from speechbrain.inference.separation import SepformerSeparation
from torchaudio.functional import resample
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .alignment import align_words, examine_window, snap_to_pauses
from .fusion import fuse, midpoint, token
from .models import model_dir
from .phrases import acoustic_phrase, plan_phrases, reuse_existing, word_timestamps
from .speakers import (
    acoustic_strength,
    assign_words,
    diarization_mask,
    filter_words,
    fit_clusters,
    infer_label,
    load_models,
    speaker_prototypes,
    voice_continuity,
    whole_clip_windows,
)


RATE = 16000
# SepFormer's attention cost grows with input length. Decode overlapping windows,
# retaining the center region of each, then cluster speakers over the recording.
CORE_SECONDS = 24
CONTEXT_SECONDS = 2


def read_pcm(path):
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
            1,
            2,
            RATE,
        ):
            raise ValueError(f"Expected 16 kHz mono PCM: {path}")
        data = source.readframes(source.getnframes())
    return np.frombuffer(data, dtype="<i2").astype("float32") / 32768


def separate(separator, mixture):
    audio = torch.from_numpy(mixture)
    input_rate = separator.hparams.sample_rate
    samples = resample(audio, RATE, input_rate)
    if len(samples) < input_rate // 2:
        samples = torch.nn.functional.pad(samples, (0, input_rate // 2 - len(samples)))
    with torch.inference_mode():
        tracks = separator.separate_batch(samples.unsqueeze(0))[0].transpose(0, 1)
        stems = resample(tracks, input_rate, RATE).numpy()
    if (
        stems.shape[0] != 3
        or stems.shape[1] < len(audio)
        or not np.isfinite(stems).all()
    ):
        raise RuntimeError("Speaker separation returned invalid audio")
    # Match the tested 16-bit shared waveform used by both recognizers and CTC.
    return [
        np.round(stem[: len(audio)] / max(1.0, np.max(np.abs(stem))) * 32767)
        .astype(np.int16)
        .astype(np.float32)
        / 32768
        for stem in stems
    ]


def whisper_words(model, samples, language, full_context=False):
    options = (
        {"condition_on_previous_text": False, "temperature": 0.0}
        if full_context
        else {}
    )
    segments, info = model.transcribe(
        samples,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=False,
        **options,
    )
    words = [
        {
            "text": word.word,
            "start": float(word.start),
            "end": float(word.end),
            "probability": float(word.probability),
            "type": "word",
        }
        for segment in segments
        for word in (segment.words or [])
    ]
    duration = len(samples) / RATE
    for word in words:
        word["start"] = min(duration, max(0.0, word["start"]))
        word["end"] = min(duration, max(word["start"], word["end"]))
    return words, info.language


def parakeet_model(path):
    return sherpa.OfflineRecognizer.from_transducer(
        encoder=str(path / "encoder.int8.onnx"),
        decoder=str(path / "decoder.int8.onnx"),
        joiner=str(path / "joiner.int8.onnx"),
        tokens=str(path / "tokens.txt"),
        num_threads=3,
        provider="cpu",
        model_type="nemo_transducer",
        decoding_method="greedy_search",
    )


def decode_parakeet(model, samples):
    stream = model.create_stream()
    stream.accept_waveform(RATE, samples)
    model.decode_stream(stream)
    return word_timestamps(stream.result, 0.0, len(samples) / RATE)


class Calibration:
    def __init__(self, path, language):
        self.language = language
        self.processor = Wav2Vec2Processor.from_pretrained(path, local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(path, local_files_only=True).eval()
        self.hop = self.model.config.inputs_to_logits_ratio / RATE
        self.blank = self.model.config.pad_token_id
        self.vocab = {
            char.lower(): i
            for char, i in self.processor.tokenizer.get_vocab().items()
            if len(char) == 1 and i != self.blank
        }

    def calibrate(self, signal, words):
        if not words or len(signal) < RATE // 10:
            return words
        revised, aligned = align_words(
            signal, words, self.language, self.processor, self.model
        )
        return snap_to_pauses(signal, revised, aligned)[0]

    def logits(self, audio):
        with torch.inference_mode():
            return {
                source: self.model(
                    **self.processor(waveform, sampling_rate=RATE, return_tensors="pt")
                )
                .logits[0]
                .log_softmax(dim=-1)
                for source, waveform in audio.items()
            }

    def word_evidence(self, words, logits):
        letters = {i: char for char, i in self.vocab.items()}
        result = {str(word["id"]): {} for word in words}
        for word in words:
            target = "".join(char for char in token(word["text"]) if char in self.vocab)
            if not target:
                continue
            labels = [self.vocab[char] for char in target]
            start = max(0, int((word["start"] - 0.08) / self.hop))
            end = max(start + 1, math.ceil((word["end"] + 0.08) / self.hop))
            for source, frames in logits.items():
                result[str(word["id"])][source] = examine_window(
                    frames[start:end], labels, self.blank, letters, target
                )
        return result

    def phrase_evidence(self, plans, logits):
        for phrase in plans["phrases"]:
            start = max(0, math.floor((phrase["start"] - 0.06) / self.hop))
            end = math.ceil((phrase["end"] + 0.06) / self.hop)
            for candidate in phrase["candidates"]:
                candidate["acoustic"] = {
                    source: acoustic_phrase(
                        candidate["words"],
                        logits[source][start:end],
                        self.vocab,
                        self.hop,
                        start * self.hop,
                        self.blank,
                    )
                    for source in ("mix", phrase["stem"])
                }
        return plans["phrases"]


def weak_targets(document, evidence):
    targets = {}
    for word in document["words"]:
        votes = evidence.get(str(word["id"]), {})
        weak = (
            len(word["sources"]) == 1
            and votes
            and max(vote["fit"] for vote in votes.values()) < 0.16
            and max(vote["agreement"] for vote in votes.values()) < 0.65
        )
        if not weak:
            continue
        source = word["sources"][0]
        if source == "mix":
            stems = {key: vote for key, vote in votes.items() if key != "mix"}
            if not stems:
                continue
            source = max(stems, key=lambda key: acoustic_strength(stems[key]))
        targets[str(word["id"])] = source
    return targets


def refine_phrases(document, signals, asr, independent, calibration, evidence):
    targets = weak_targets(document, evidence)
    if not targets or not document["speaker_turns"]:
        return {**document, "phrase_reuse": [], "phrase_candidates": 0}
    selected = ["mix", *dict.fromkeys(targets.values())]
    windows = [
        {
            "source": source,
            "start_sample": 0,
            "end_sample": len(signals[source]),
            "targets": [
                int(i) for i, track in targets.items() if source in ("mix", track)
            ],
        }
        for source in selected
    ]
    whisper = {
        "targets": targets,
        "windows": windows,
        "results": [
            whisper_words(asr, signals[source], "en", full_context=True)[0]
            for source in selected
        ],
    }
    parakeet = {
        "windows": windows,
        "results": [
            decode_parakeet(independent, signals[source]) for source in selected
        ],
    }
    plans = plan_phrases(document, whisper, parakeet)
    used = {"mix", *(phrase["stem"] for phrase in plans["phrases"])}
    logits = (
        calibration.logits({key: signals[key] for key in used})
        if plans["phrases"]
        else {}
    )
    output = reuse_existing(document, calibration.phrase_evidence(plans, logits))
    output.update(
        phrase_candidates=len(plans["phrases"]), low_confidence_count=len(targets)
    )
    return output


def clip_turns(turns, start, end):
    return [
        {
            **row,
            "start": max(start, row["start"]) - start,
            "end": min(end, row["end"]) - start,
        }
        for row in turns
        if row["start"] < end and row["end"] > start
    ]


def merge_windows(pieces):
    output = {
        "words": [],
        "paths": {},
        "phrase_reuse": [],
        "phrase_candidates": 0,
        "low_confidence_count": 0,
        "speaker_reassignments": 0,
    }
    next_id = 0
    for index, (result, start, core_start, core_end) in enumerate(pieces):
        id_map = {}
        for word in result["words"]:
            # Like the main ASR, words at the padding belong to only one window.
            if core_start <= start + midpoint(word) < core_end:
                id_map[word["id"]] = next_id
                output["words"].append(
                    {
                        **word,
                        "id": next_id,
                        "start": start + word["start"],
                        "end": start + word["end"],
                    }
                )
                next_id += 1
        for name, path in result["paths"].items():
            ids = [id_map[i] for i in path if i in id_map]
            if ids:
                output["paths"][f"{index}:{name}"] = ids
        for repair in result.get("phrase_reuse", []):
            anchors = repair["anchors"]
            if all(i in id_map for i in (*anchors, *repair["reused_ids"])):
                output["phrase_reuse"].append(
                    {
                        "anchors": [id_map[i] for i in anchors],
                        "reused_ids": [id_map[i] for i in repair["reused_ids"]],
                        "updated_paths": [
                            f"{index}:{path}" for path in repair["updated_paths"]
                        ],
                        "mean_log_probability_gain": repair[
                            "mean_log_probability_gain"
                        ],
                    }
                )
        for key in (
            "phrase_candidates",
            "low_confidence_count",
            "speaker_reassignments",
        ):
            output[key] += result.get(key, 0)
    output["words"].sort(key=lambda word: (word["start"], word["end"], word["id"]))
    output["text"] = " ".join(word["text"].strip() for word in output["words"])
    return output


def transcribe(audio_path, language=None, num_speakers=None, offline=False):
    torch.set_num_threads(4)
    mixture = read_pcm(audio_path)
    duration = len(mixture) / RATE
    asr = WhisperModel(
        str(model_dir("large-v3", offline)),
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        local_files_only=True,
    )
    separator = SepformerSeparation.from_hparams(
        source=str(model_dir("sepformer", offline)),
        savedir=None,
        run_opts={"device": "cpu"},
    )
    diarizer, embedder = load_models(
        model_dir("segmentation", offline) / "pyannote.onnx",
        model_dir("embedding", offline) / "campplus-en.onnx",
        num_speakers,
    )
    segments = diarizer.process(mixture).sort_by_start_time()
    turns = [
        {
            "start": max(0.0, seg.start),
            "end": min(duration, seg.end),
            "speaker_id": f"speaker-{int(seg.speaker)}",
        }
        for seg in segments
        if seg.start < duration
    ]
    speakers, activity = diarization_mask(segments, duration)
    prototypes = speaker_prototypes(embedder, mixture, speakers, activity)
    calibration = independent = None
    windows, chunks = [], []
    for core in range(0, len(mixture), CORE_SECONDS * RATE):
        endcore = min(len(mixture), core + CORE_SECONDS * RATE)
        begin = max(0, core - CONTEXT_SECONDS * RATE)
        end = min(len(mixture), endcore + CONTEXT_SECONDS * RATE)
        signal = mixture[begin:end]
        print(
            f"  local separation and recognition: {core / RATE:.1f}–{endcore / RATE:.1f}s",
            flush=True,
        )
        if np.max(np.abs(signal)) < 0.001:
            continue
        signals = {
            "mix": signal,
            **{
                f"track-{i}": stem
                for i, stem in enumerate(separate(separator, signal), 1)
            },
        }
        mix_words, detected = whisper_words(asr, signal, language)
        language = language or detected
        if calibration is None and language in ("en", "zh"):
            calibration = Calibration(
                model_dir(f"alignment-{language}", offline), language
            )
            if language == "en":
                independent = parakeet_model(model_dir("parakeet", offline))
        streams = {"mix": mix_words}
        for source in signals:
            if source != "mix":
                streams[source] = whisper_words(asr, signals[source], language)[0]
        if calibration is not None:
            streams = {
                key: calibration.calibrate(signals[key], words)
                for key, words in streams.items()
            }
        fused = fuse(streams)
        evidence = (
            calibration.word_evidence(fused["words"], calibration.logits(signals))
            if language == "en" and len(signal) >= RATE // 10
            else {}
        )
        local_activity = activity[
            :, round(100 * begin / RATE) : round(100 * end / RATE)
        ]
        local_turns = clip_turns(turns, begin / RATE, end / RATE)
        assigned = (
            assign_words(
                fused["words"],
                fused["paths"],
                evidence,
                speakers,
                local_activity,
                prototypes,
                embedder,
                signals,
            )
            if local_activity.size
            else [
                {key: value for key, value in word.items() if key != "speaker_id"}
                for word in fused["words"]
            ]
        )
        words, paths, _ = filter_words(
            assigned, fused["paths"], evidence, protect_context=True
        )
        document = voice_continuity(
            {"words": words, "paths": paths, "speaker_turns": local_turns}
        )
        window_id = len(chunks)
        for window in whole_clip_windows(embedder, signals, streams, local_activity):
            at = window["center"]
            if core <= begin + at * RATE < endcore:
                windows.append({**window, "source": (window_id, window["source"])})
        chunks.append(
            (document, evidence, signals, begin / RATE, core / RATE, endcore / RATE)
        )
    del separator, diarizer, embedder

    clustered = fit_clusters(windows, prototypes) if prototypes and windows else {}
    completed = []
    for i, (document, evidence, signals, start, core_start, core_end) in enumerate(
        chunks
    ):
        votes = {source: clustered.get((i, source), []) for source in signals}
        changes = 0
        for word in document["words"]:
            speaker = infer_label(
                word,
                votes,
                document["speaker_turns"],
                evidence.get(str(word["id"]), {}),
            )
            if speaker and speaker != word.get("speaker_id"):
                word.update(
                    speaker_id=speaker, speaker_assignment="global_embedding_cluster"
                )
                changes += 1
        document["speaker_reassignments"] = changes
        if independent is not None and len(signals["mix"]) >= RATE // 10:
            print(
                f"  local phrase verification: {core_start:.1f}–{core_end:.1f}s",
                flush=True,
            )
            document = refine_phrases(
                document, signals, asr, independent, calibration, evidence
            )
        completed.append((document, start, core_start, core_end))
    result = merge_windows(completed)
    result.update(
        language_code=language,
        speaker_turns=turns,
        models=["large-v3", "sepformer", "pyannote", "campplus"],
        speaker_cluster_windows=len(windows),
    )
    if calibration is not None:
        result["models"].append(f"alignment-{language}")
    if independent is not None:
        result["models"].append("parakeet")
    return result
