# Install video-use

Install **video-use** with one click in the DesireCore Skill Market. On another skill host, register a published checkout that contains `SKILL.md`. Both install the skill and its Python lockfile.

## Local transcription

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and put `ffmpeg` and `ffprobe` on the runtime PATH. The command below uses Python 3.12 and sets up the locked CPU dependencies. It downloads pinned model files on first use and runs transcription on the machine with the media. No account or API key is required.

```sh
uv run --locked --python 3.12 --extra local --project "$SKILL_DIR" \
  python "$SKILL_DIR/helpers/transcribe_batch.py" "$VIDEOS_DIR"
```

`SKILL_DIR` is the directory containing `SKILL.md`. Repeat `--file=NAME` to limit a batch to specific sources. For an individual file, call `helpers/transcribe.py` with its path. `--audio-track` takes a zero-based stream index.

Allow about 4 GB for the English models and another 1.3 GB for Chinese alignment, plus space for the environment and temporary downloads. Network access to the public model hosts is needed to populate the cache. The model manifest records individual hashes and licenses. The model cache defaults to `$XDG_CACHE_HOME/video-use/models` or `~/.cache/video-use/models`. Set `VIDEO_USE_MODEL_DIR` to use an existing shared cache. Add `--offline` to require cached models. The first run can take significant CPU time, and local transcription uses a single batch worker by default.

Local transcription runs large-v3 on the mix and up to three estimated sources per window. It fuses overlapping paths, assigns anonymous speakers by clustering voice embeddings across the recording, and aligns cuts for English and Chinese. English also uses Parakeet and large-v3 for same-speaker phrase candidates, with local acoustic checks on both the mix and corresponding stem. Weak or ambiguous phrase evidence keeps the original words. Other supported Whisper languages use Whisper's timing. Speaker labels are scoped to each recording. The local JSON supplies speech words and timestamps. Scribe audio-event tags remain available via the Scribe backend.

## Explicitly select Scribe

Scribe is optional. Configure `ELEVENLABS_API_KEY` through the runtime environment or a secret manager outside chat. You need an ElevenLabs account with speech-to-text access and network access to `api.elevenlabs.io`. Usage may incur fees under the [service terms](https://elevenlabs.io/terms-of-use). Audio sent to Scribe is subject to the provider's [privacy policy](https://elevenlabs.io/privacy-policy).

Preview the pending list with `--backend scribe`:

```sh
uv run --locked --project "$SKILL_DIR" python "$SKILL_DIR/helpers/transcribe_batch.py" \
  "$VIDEOS_DIR" --backend scribe
```

After the user explicitly approves sending the specific files and selected audio track, add one `--file=NAME` per approved file and `--upload-approved`. For a single file, use `transcribe.py "$VIDEO" --backend scribe --upload-approved` with the same locked base environment. Selecting `--backend local` on a later run switches back to local. The selected backend supplies the active transcript for packing and rendering, and the compatible cache for each backend is retained.

Optional tasks use separately configured tools. URL acquisition uses `yt-dlp`. Animation can use HyperFrames, Remotion, or Manim as described in their bundled guides.
