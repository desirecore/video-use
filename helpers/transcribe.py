"""Transcribe a local media file with the selected word-level backend."""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import requests


BACKENDS = ("local", "scribe")
SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def load_api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY in the runtime environment before transcribing.")
    return key


def count_audio_tracks(video_path: Path) -> int:
    """How many audio streams the container holds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return len([ln for ln in out.stdout.splitlines() if ln.strip()])


def peak_dbfs(wav_path: Path) -> float:
    """Peak level of a 16-bit PCM wav, in dBFS. -inf for digital silence."""
    peak = 0
    with wave.open(str(wav_path), "rb") as w:
        # A chunk at a time: batch mode runs several of these at once, and a two-hour
        # take is 230 MB of 16 kHz mono before the array copy doubles it.
        while frames := w.readframes(1 << 16):
            samples = array.array("h", frames)
            peak = max(peak, max(samples), -min(samples))
    return 20 * math.log10(peak / 32768) if peak > 0 else float("-inf")


def extract_audio(video_path: Path, dest: Path, audio_track: int = 0) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:a:{audio_track}",
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    data: dict[str, str] = {
        "model_id": "scribe_v2",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
            allow_redirects=False,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned HTTP {resp.status_code}")

    return resp.json()


def transcript_path(edit_dir: Path, video: Path, audio_track: int = 0) -> Path:
    """Where a video's transcript lands.

    The track belongs in the name, or a rerun with --audio-track hands back the transcript of
    the track it is meant to replace. Track 0 keeps the plain name, so transcripts made before
    the flag existed stay valid. Batch mode tests its cache with this too — one function, so
    the two cannot drift apart.
    """
    suffix = "" if audio_track == 0 else f".track{audio_track}"
    return edit_dir / "transcripts" / f"{video.stem}{suffix}.json"


def add_backend_arguments(parser):
    parser.add_argument("--backend", choices=BACKENDS, default="local",
                        help="Transcription backend (default: local).")
    parser.add_argument("--offline", action="store_true",
                        help="Require pre-downloaded local models.")
    parser.add_argument("--upload-approved", action="store_true",
                        help="Operator approved uploading the selected audio to ElevenLabs Scribe.")


def validate_backend_arguments(parser, args):
    if args.upload_approved and args.backend != "scribe":
        parser.error("--upload-approved requires --backend scribe")
    if args.offline and args.backend != "local":
        parser.error("--offline requires --backend local")
    if args.audio_track < 0:
        parser.error("--audio-track must be nonnegative")
    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be positive")


def cache_options(video, backend, language, num_speakers):
    stat = video.stat()
    return {"file": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "backend": backend, "language": language, "num_speakers": num_speakers,
            "revision": 1}


def backend_cache_path(output, backend):
    return output.parent / ".backends" / backend / output.name


def read_transcript(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return result if isinstance(result, dict) and isinstance(result.get("words"), list) else None


def is_legacy(data):
    return data is not None and "video_use" not in data


def cached_transcript(output, options):
    backend = options["backend"]
    for candidate in (output, backend_cache_path(output, backend)):
        data = read_transcript(candidate)
        if data is not None and data.get("video_use") == options:
            return candidate
        if (backend == "scribe" and is_legacy(data)
                and options["language"] is None and options["num_speakers"] is None):
            return candidate
    return None


def save_transcript(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    # All readers see a fully written transcript, including during a backend switch.
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        staging = Path(temp) / path.name
        staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.replace(path)


def call_local(audio, language, num_speakers, offline):
    try:
        from local_asr.pipeline import transcribe
    except ImportError as error:
        raise RuntimeError(
            "Local dependencies are unavailable. Run with uv run --locked --python 3.12 --extra local."
        ) from error
    return transcribe(audio, language, num_speakers, offline)


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str | None = None,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
    audio_track: int = 0,
    *,
    upload_approved: bool = False,
    backend: str = "local",
    offline: bool = False,
) -> Path:
    """Publish the selected backend's transcript at the editor's canonical path."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown transcription backend: {backend}")
    if (upload_approved and backend != "scribe") or (offline and backend != "local"):
        raise ValueError("Upload approval belongs to Scribe and offline mode belongs to local transcription.")
    video = video.resolve()
    out_path = transcript_path(edit_dir, video, audio_track)
    options = cache_options(video, backend, language, num_speakers)
    cached = cached_transcript(out_path, options)
    if cached == out_path:
        if verbose:
            print(f"cached: {out_path.name} ({backend})")
        return out_path

    # Preserve the active backend before publishing a different one. Untagged
    # transcripts come from the previous Scribe-only releases.
    previous = read_transcript(out_path)
    if is_legacy(previous):
        legacy_cache = backend_cache_path(out_path, "scribe")
        if not legacy_cache.exists():
            save_transcript(legacy_cache, previous)
    if cached is not None:
        data = read_transcript(cached)
        if data is not None:
            save_transcript(out_path, data)
            if verbose:
                print(f"cached: {out_path.name} ({backend})")
            return out_path

    if backend == "scribe":
        if not upload_approved:
            raise ValueError("Confirm sending the selected audio to ElevenLabs before uploading.")
        if api_key is None:
            api_key = load_api_key()

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    n_tracks = count_audio_tracks(video)
    if not 0 <= audio_track < n_tracks:
        raise ValueError(f"{video.name} has {n_tracks} audio tracks. Select an available --audio-track.")
    if n_tracks > 1 and verbose:
        print(f"  note: {video.name} has {n_tracks} audio tracks, using track "
              f"{audio_track + 1} (--audio-track to change)", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio, audio_track)

        # Uploading silence costs the same as uploading speech and returns
        # nothing, so catch the wrong-track case before paying for it.
        peak = peak_dbfs(audio)
        if peak < -60.0 and backend == "scribe":
            raise RuntimeError(
                f"track {audio_track + 1} of {video.name} is silent "
                f"(peak {peak:.1f} dBFS) - not uploading. "
                + (f"The file has {n_tracks} audio tracks; try --audio-track "
                   + " or ".join(str(i) for i in range(n_tracks) if i != audio_track) + "."
                   if n_tracks > 1 else "Check the source audio.")
            )

        if backend == "scribe":
            if verbose:
                size_mb = audio.stat().st_size / (1024 * 1024)
                print(f"  uploading {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
            payload = call_scribe(audio, api_key, language, num_speakers)
        elif peak < -60.0:
            payload = {"language_code": language, "words": [], "paths": {}, "text": ""}
        else:
            payload = call_local(audio, language, num_speakers, offline)

    if cache_options(video, backend, language, num_speakers) != options:
        raise RuntimeError(f"Source changed during transcription: {video.name}")
    payload["video_use"] = options
    save_transcript(backend_cache_path(out_path, backend), payload)
    save_transcript(out_path, payload)
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video locally or with explicitly selected Scribe")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe. OBS writes the game on track 0 "
             "and the mic on track 1; without this ffmpeg applies its default audio "
             "stream selection, which picks the track with the most channels.",
    )
    add_backend_arguments(ap)
    args = ap.parse_args()
    validate_backend_arguments(ap, args)

    video = args.video.resolve()
    if not video.is_file():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    try:
        transcribe_one(
            video=video, edit_dir=edit_dir, language=args.language, num_speakers=args.num_speakers,
            audio_track=args.audio_track, backend=args.backend, offline=args.offline,
            upload_approved=args.upload_approved,
        )
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        ap.error(str(error))


if __name__ == "__main__":
    main()
