# Changelog

## 0.1.0a1 (unreleased)

- First release. Android: offline speech to text with whisper.cpp 1.9.5 (pinned source, sha256-checked,
  built with the app). ggml is built as shared libraries with one CPU backend per instruction set level on
  arm64 and x86_64; the fastest one the phone's CPU runs is loaded at start (1.6x faster than a generic armv8.0
  build on a Galaxy S23) and named in `system` (`BACKEND = ...`). Every library is called `libpyxpo_speech_*.so`,
  so another module's ggml (llama.cpp) does not clash. The microphone is recorded into memory at 16 kHz (no file), then
  cleared after transcribing; nothing is logged or sent anywhere.
- Calls: `status`, `get_permission`, `request_permission`, `memory`, `load_model` (with a free-memory check),
  `unload_model`, `start`, `stop`, `cancel`, `transcribe_file` (PCM/float WAV, any rate), state and level
  listeners. `stop` and `transcribe_file` take `language`, `translate`, `prompt`, `threads`, `beam_size` and
  `audio_ctx` (`"fit"` encodes only the clip: faster, less reliable). Works while the app runs headless, so an input method can use it.
- iOS, the web and the desktop: `status` reports `supported: False`; the other calls raise `NotSupportedError`.
