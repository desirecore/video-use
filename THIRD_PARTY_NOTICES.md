# Third-party notices

- **video-use** is derived from [Browser Use's video-use](https://github.com/browser-use/video-use/tree/43cfc566833548093455c9d4a46bd69913b7e552). Copyright © 2026 Browser Use. See [LICENSE](LICENSE).
- **Manim Video** in `skills/manim-video/` comes from [SHL0MS and Hermes Agent](https://github.com/NousResearch/hermes-agent/tree/ef75bc708867e3884cb40a91363be6dc6fddc034/skills/creative/manim-video), with an adapted skill description. Copyright © 2025 Nous Research. See [the included MIT license](skills/manim-video/LICENSE).

## Local transcription models

The runtime downloads these weights separately. The pinned source and hashes are in [models.json](helpers/local_asr/models.json).

- [Systran's CTranslate2 conversion of OpenAI Whisper large-v3](https://huggingface.co/Systran/faster-whisper-large-v3), MIT.
- [SpeechBrain SepFormer Libri3Mix](https://huggingface.co/speechbrain/sepformer-libri3mix), Apache-2.0.
- [pyannote segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0), copyright © 2022 CNRS, MIT. The [sherpa-onnx conversion](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models) supplies a copy of the license in the model cache.
- [3D-Speaker CAM++](https://github.com/modelscope/3D-Speaker), Apache-2.0, in the [sherpa-onnx ONNX format](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models).
- [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), in the [sherpa-onnx INT8 ONNX format](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models).
- [Meta Wav2Vec2 Base 960h](https://huggingface.co/facebook/wav2vec2-base-960h) and [Jonatas Grosman's Chinese XLSR-53 Wav2Vec2](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn), Apache-2.0.
