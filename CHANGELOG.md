# Changelog

## 0.1.0a1 (unreleased)

- First release. Android: offline speech to text with whisper.cpp 1.9.5 (pinned source, sha256-checked,
  built with the app; ggml symbols hidden). The microphone is recorded into memory at 16 kHz (no file), then
  cleared after transcribing; nothing is logged or sent anywhere.
- Calls: `status`, `get_permission`, `request_permission`, `memory`, `load_model` (with a free-memory check),
  `unload_model`, `start`, `stop`, `cancel`, `transcribe_file` (PCM/float WAV, any rate), state and level
  listeners. Works while the app runs headless, so an input method can use it.
- iOS, the web and the desktop: `status` reports `supported: False`; the other calls raise `NotSupportedError`.
