# Install video-use

Install **video-use** in the DesireCore Skill Market to register the skill and its locked sources. The market installs the skill files. Run the helpers on a machine with Python 3.10 or newer, [uv](https://docs.astral.sh/uv/getting-started/installation/), and both `ffmpeg` and `ffprobe` on the runtime PATH.

For another skill host, take a checkout of a published release and register the checkout directory containing `SKILL.md`.

## Prepare the environment

From the skill checkout, run:

```sh
uv sync --locked
```

This installs the dependencies recorded in `uv.lock` into the project's virtual environment. The first `uv run --locked` can do the same setup. When running from elsewhere, add `--project "$SKILL_DIR"`, where `SKILL_DIR` is the directory containing `SKILL.md`.

For cloud transcription, set `ELEVENLABS_API_KEY` in the environment of the process running the skill. Configure it through the operator's secret manager or runtime environment settings, outside chat. A directory `.env` does not configure this version. Setup requires no API request.

ElevenLabs Scribe is a separate service. You need an ElevenLabs account with speech-to-text access, network access to `api.elevenlabs.io`, and an API key. Usage may incur separate fees under the [service terms](https://elevenlabs.io/terms-of-use). Review the provider's [privacy policy](https://elevenlabs.io/privacy-policy) before sending audio.

## Review the upload

Use local inventory to see the uncached files and their durations:

```sh
uv run --locked --project "$SKILL_DIR" python "$SKILL_DIR/helpers/transcribe_batch.py" "$VIDEOS_DIR"
```

After the operator explicitly approves sending the selected files and track to ElevenLabs, add one `--file=NAME` per approved file and `--upload-approved`. By default, track 1 is selected. Use `--audio-track 1` to select the second track and include that selection in the request for approval. If the operator declines or a key is unavailable, use existing compatible cached transcripts or report that transcription is unavailable.

Optional tasks use separately configured tools. URL acquisition uses `yt-dlp`. Animation may use a version-pinned HyperFrames or Remotion install, or a local Manim install with the prerequisites described in its bundled guide.
