# video-use

Video editing by conversation. Point your agent at a folder of footage and review its editing strategy before it renders `edit/final.mp4`.

This maintenance fork is published by DesireCore and derived from [Browser Use's video-use](https://github.com/browser-use/video-use). See [third-party notices](THIRD_PARTY_NOTICES.md).

## What it does

- **Cuts out filler words** (`umm`, `uh`, false starts) and dead space between takes
- **Auto color grades** every segment (warm cinematic, neutral punch, or any custom ffmpeg chain)
- **30ms audio fades** at every cut so you never hear a pop
- **Burns subtitles** in your style — 2-word UPPERCASE chunks by default, fully customizable
- **Generates animation overlays** via [HyperFrames](https://github.com/heygen-com/hyperframes), [Remotion](https://www.remotion.dev/), [Manim](https://www.manim.community/), or PIL — spawned in parallel sub-agents, one per animation
- **Self-evaluates the rendered output** at every cut boundary before showing you anything
- **Persists session memory** in `project.md` so next week's session picks up where you left off

## Install and use

In DesireCore, install **video-use** from the Skill Market. The market installs the skill and its dependency lockfile. You need Python 3.10+, `uv`, `ffmpeg` and `ffprobe` on the machine running the skill. See [install.md](install.md) for dependency setup and manual installation.

Transcription sends the selected source audio to ElevenLabs Scribe, a separately operated service that requires an API key and may charge for usage. Configure `ELEVENLABS_API_KEY` in the runtime environment. The skill asks which uncached files and audio track it may send before the first upload.

All editing artifacts go into `<videos_dir>/edit/`. The source videos stay intact.

## How it works

The LLM never watches the video. It **reads** it — through two layers that together give it everything it needs to cut with word-boundary precision.

<p align="center">
  <img src="static/timeline-view.svg" alt="timeline_view composite — filmstrip + speaker track + waveform + word labels + silence-gap cut candidates" width="100%">
</p>

**Layer 1 — Audio transcript (always loaded).** One approved ElevenLabs Scribe call per uncached source gives word-level timestamps, speaker diarization, and audio events (`(laughter)`, `(applause)`, `(sigh)`). All takes pack into a single ~12KB `takes_packed.md` — the LLM's primary reading view.

```
## C0103  (duration: 43.0s, 8 phrases)
  [002.52-005.36] S0 Ninety percent of what a web agent does is completely wasted.
  [006.08-006.74] S0 We fixed this.
```

**Layer 2 — Visual composite (on demand).** `timeline_view` produces a filmstrip + waveform + word labels PNG for any time range. Called only at decision points — ambiguous pauses, retake comparisons, cut-point sanity checks.

> Naive approach: 30,000 frames × 1,500 tokens = **45M tokens of noise**.
> Video Use: **12KB text + a handful of PNGs**.

Same idea as browser-use giving an LLM a structured DOM instead of a screenshot — but for video.

## Pipeline

```
Approve upload ──> Transcribe ──> Pack ──> LLM Reasons ──> EDL ──> Render ──> Self-Eval
                                                              │
                                                              └─ issue? fix + re-render (max 3)
```

The self-eval loop runs `timeline_view` on the _rendered output_ at every cut boundary — catches visual jumps, audio pops, hidden subtitles. You see the preview only after it passes.

## Design principles

1. **Text + on-demand visuals.** No frame-dumping. The transcript is the surface.
2. **Audio is primary, visuals follow.** Cuts come from speech boundaries and silence gaps.
3. **Ask → confirm → execute → self-eval → persist.** Never touch the cut without strategy approval.
4. **Zero assumptions about content type.** Look, ask, then edit.
5. **12 hard rules, artistic freedom elsewhere.** Production-correctness is non-negotiable. Taste isn't.

See [`SKILL.md`](./SKILL.md) for the full production rules and editing craft.
