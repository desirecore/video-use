"""Inventory local videos, then transcribe an explicitly approved selection.

Usage:
    python helpers/transcribe_batch.py <videos_dir>
    python helpers/transcribe_batch.py <videos_dir> --file clip.mp4 --upload-approved
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transcribe import load_api_key, transcribe_one, transcript_path


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def find_videos(videos_dir: Path) -> list[Path]:
    videos = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in VIDEO_EXTS
    )
    return videos


def describe_pending(videos: list[Path], audio_track: int) -> None:
    print(f"Pending audio for ElevenLabs Scribe (track {audio_track + 1}, usage charges may apply):")
    for video in videos:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True,
        )
        duration = result.stdout.strip() if result.returncode == 0 else ""
        print(f"  {json.dumps(video.name, ensure_ascii=False)}: {duration or 'unknown'} seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview or transcribe selected video audio with ElevenLabs Scribe")
    ap.add_argument("videos_dir", type=Path, help="Directory containing source videos")
    ap.add_argument(
        "--file", action="append", default=[], metavar="NAME",
        help="Exact filename in videos_dir. Repeat for every selected file.",
    )
    ap.add_argument(
        "--upload-approved", action="store_true",
        help="Operator approved sending the selected files' audio to ElevenLabs.",
    )
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <videos_dir>/edit)",
    )
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code. Omit to auto-detect per file.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers. Improves diarization when known.",
    )
    ap.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Zero-based audio track to transcribe (OBS: 0 = game, 1 = mic).",
    )
    args = ap.parse_args()
    if args.upload_approved and not args.file:
        ap.error("list the approved files with --file before uploading")

    videos_dir = args.videos_dir.resolve()
    if not videos_dir.is_dir():
        sys.exit(f"not a directory: {videos_dir}")

    edit_dir = (args.edit_dir or (videos_dir / "edit")).resolve()
    videos = find_videos(videos_dir)
    if not videos:
        sys.exit(f"no videos found in {videos_dir}")
    if args.file:
        by_name = {video.name: video for video in videos}
        for name in args.file:
            if name not in by_name:
                ap.error(f"not a supported video in the directory: {name!r}")
        videos = [by_name[name] for name in dict.fromkeys(args.file)]

    already_cached = [v for v in videos
                      if transcript_path(edit_dir, v, args.audio_track).exists()]
    pending = [v for v in videos if v not in already_cached]

    print(f"selected {len(videos)} videos ({len(already_cached)} cached, {len(pending)} to transcribe)")
    if not pending:
        print("nothing to do")
        return

    if not args.upload_approved:
        describe_pending(pending, args.audio_track)
        return

    if len({transcript_path(edit_dir, video, args.audio_track) for video in videos}) != len(videos):
        ap.error("selected files must have distinct names before their extensions")
    api_key = load_api_key()

    print(f"transcribing {len(pending)} files with {args.workers} parallel workers")
    t0 = time.time()

    errors: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                transcribe_one,
                video=v,
                edit_dir=edit_dir,
                api_key=api_key,
                language=args.language,
                num_speakers=args.num_speakers,
                verbose=False,
                audio_track=args.audio_track,
                upload_approved=True,
            ): v
            for v in pending
        }
        for fut in as_completed(futures):
            v = futures[fut]
            try:
                out = fut.result()
                print(f"  + {v.stem}  →  {out.name}")
            except Exception as e:
                errors.append((v, str(e)))
                print(f"  x {v.stem}  FAILED: {e}")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s")
    if errors:
        print(f"{len(errors)} failures:")
        for v, msg in errors:
            print(f"  {v.name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
