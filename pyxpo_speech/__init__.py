"""Offline speech to text for pyxpo apps: the microphone and whisper.cpp, or the phone's own on-device recognizer.

    import pyxpo_speech as Speech

    if not (await Speech.request_permission_async())["granted"]:
        return
    await Speech.load_model_async(path_to_ggml_model)   # a whisper.cpp ggml file, for example ggml-base-q5_1.bin
    await Speech.start_async()                           # recording into memory
    ...
    t = await Speech.stop_async(language="auto")         # {"text", "language", "segments", ...}

Audio is recorded into memory (no file), transcribed on the CPU, then cleared. Nothing is sent anywhere and
nothing is logged. The app brings the model file; this package does not download anything at run time.
Android only in this version: on iOS, the web and the desktop `status_async` reports `supported: False` and
the other calls raise `NotSupportedError`.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Optional, Union

from pyxpo.module_api import NotSupportedError, call, on_event, permission_response, platform, require_linked

__version__ = "0.1.0a1"

#: `add_state_listener` states: recording, transcribing, idle, and limit (recording hit `max_seconds` and stopped;
#: call `stop_async` for the text).
STATES = ("recording", "transcribing", "idle", "limit")

PathLike = Union[str, "os.PathLike[str]"]

_LANG = re.compile(r"^(auto|[a-z]{2,3})$")
_LOCALE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")


def _native() -> bool:
    if platform() != "android":
        return False
    require_linked("Speech", "speech", package="pyxpo-speech")
    return True


def _need_native(what: str) -> None:
    if not _native():
        raise NotSupportedError("%s needs Android: pyxpo-speech does not support %s yet" % (what, platform()))


async def status_async() -> Dict[str, Any]:
    """`supported`, `loaded`, `model` (the loaded file's path), `multilingual`, `recording`, `transcribing`."""
    if not _native():
        return {"supported": False, "loaded": False, "model": None, "multilingual": False, "recording": False,
                "transcribing": False}
    return await call("Speech", "status", reply=True)


# permission

def _permission(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = raw or {}
    return permission_response(str(raw.get("status", "denied")), bool(raw.get("canAskAgain", False)))


async def get_permission_async() -> Dict[str, Any]:
    """The microphone permission, without asking (Expo's `PermissionResponse`)."""
    _need_native("get_permission_async")
    return _permission(await call("Speech", "getPermission", reply=True))


async def request_permission_async() -> Dict[str, Any]:
    """Asks for the microphone. Needs the app on screen: a keyboard or other service cannot ask."""
    _need_native("request_permission_async")
    return _permission(await call("Speech", "requestPermission", reply=True))


# models

async def memory_async() -> Dict[str, Any]:
    """The phone's memory in bytes: `total`, `available`, `threshold` (Android starts killing apps below it)
    and `low`. Use it to pick a model size."""
    _need_native("memory_async")
    return await call("Speech", "memory", reply=True)


async def load_model_async(path: PathLike, *, check_memory: bool = True,
                           flash_attention: bool = True) -> Dict[str, Any]:
    """Loads a whisper.cpp ggml model file (and unloads the one before). Returns `status_async()` plus `load_ms`
    and `system` (the CPU features whisper uses). Loading the loaded file again does nothing.

    `check_memory` refuses a model that would push the phone below its low-memory line (the app would be
    killed). Raises `RuntimeError` for a missing, damaged or non-whisper file."""
    _need_native("load_model_async")
    p = os.fspath(path)
    if not isinstance(p, str) or not p:
        raise ValueError("path must be a model file path")
    r = await call("Speech", "load", p, {"checkMemory": bool(check_memory), "flashAttention": bool(flash_attention)},
                   reply=True)
    if "loadMs" in r:
        r["load_ms"] = r.pop("loadMs")
    return r


async def unload_model_async() -> None:
    """Frees the model's memory. A running transcription stops first."""
    if _native():
        await call("Speech", "unload", reply=True)


# recording and transcribing

async def start_async(*, max_seconds: float = 30, levels: bool = False) -> None:
    """Starts recording the microphone into memory. Recording stops by itself after `max_seconds` (1 to 120)
    and the state listeners get `limit`. `levels=True` sends `add_level_listener` the loudness ten times a second."""
    _need_native("start_async")
    if not 1 <= max_seconds <= 120:
        raise ValueError("max_seconds must be 1 to 120")
    await call("Speech", "start", {"maxSeconds": float(max_seconds), "levels": bool(levels)}, reply=True)


def _options(language: str, translate: bool, prompt: Optional[str], threads: Optional[int],
             beam_size: Optional[int], audio_ctx: Union[None, int, str] = None) -> Dict[str, Any]:
    if not isinstance(language, str) or not _LANG.match(language):
        raise ValueError("language must be 'auto' or a whisper language code like 'en' or 'hi', not %r" % (language,))
    if threads is not None and not (isinstance(threads, int) and 1 <= threads <= 16):
        raise ValueError("threads must be 1 to 16")
    if beam_size is not None and not (isinstance(beam_size, int) and 1 <= beam_size <= 8):
        raise ValueError("beam_size must be 1 to 8")
    if not (audio_ctx is None or audio_ctx == "fit"
            or (isinstance(audio_ctx, int) and not isinstance(audio_ctx, bool) and 64 <= audio_ctx <= 1500)):
        raise ValueError("audio_ctx must be None, 'fit' or 64 to 1500")
    o: Dict[str, Any] = {"language": language, "translate": bool(translate)}
    if prompt:
        o["prompt"] = str(prompt)
    if threads:
        o["threads"] = threads
    if beam_size:
        o["beamSize"] = beam_size
    if audio_ctx is not None:
        o["audioCtx"] = -1 if audio_ctx == "fit" else audio_ctx
    return o


def _transcript(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"text": r.get("text", ""), "language": r.get("language", ""), "segments": list(r.get("segments") or []),
            "no_speech": float(r.get("noSpeech", 0.0)), "audio_ms": int(r.get("audioMs", 0)),
            "decode_ms": int(r.get("decodeMs", 0))}


async def stop_async(*, language: str = "auto", translate: bool = False, prompt: Optional[str] = None,
                     threads: Optional[int] = None, beam_size: Optional[int] = None,
                     audio_ctx: Union[None, int, str] = None, keep_audio: bool = False) -> Dict[str, Any]:
    """Stops recording and transcribes it. Returns `text`, `language` (detected, or the one given), `segments`,
    `no_speech` (0..1; high means it was likely silence), `audio_ms` and `decode_ms`.

    `language`: `"auto"` detects it, or a code like `"en"`/`"hi"`. `translate=True` makes whisper write English.
    `prompt` primes spelling and style (names, Hinglish words). `threads` defaults to up to 4 CPU cores,
    `beam_size` > 1 searches more (slower, sometimes better).

    `audio_ctx`: whisper encodes a 30-second window (1500 steps of 20 ms) whatever the clip's length. `"fit"`
    encodes only as much as was recorded, which makes short clips several times faster; check its accuracy for
    your model and languages. An int sets the window (64 to 1500).

    `keep_audio=True` keeps the recording in memory so `transcribe_kept_async` can write it again, for example
    in Hindi and then in English. It is cleared by `forget_audio_async`, the next `start_async`, `cancel_async`,
    `unload_model_async`, or two minutes after stopping, whichever comes first."""
    _need_native("stop_async")
    o = _options(language, translate, prompt, threads, beam_size, audio_ctx)
    if keep_audio:
        o["keepAudio"] = True
    return _transcript(await call("Speech", "stop", o, reply=True))


async def transcribe_kept_async(*, language: str = "auto", translate: bool = False, prompt: Optional[str] = None,
                                threads: Optional[int] = None, beam_size: Optional[int] = None,
                                audio_ctx: Union[None, int, str] = None) -> Dict[str, Any]:
    """Transcribes the recording kept by `stop_async(keep_audio=True)` again, with other options (`translate=True`
    for English). Raises `RuntimeError` when nothing is kept. The recording stays kept."""
    _need_native("transcribe_kept_async")
    o = _options(language, translate, prompt, threads, beam_size, audio_ctx)
    return _transcript(await call("Speech", "transcribeKept", o, reply=True))


async def forget_audio_async() -> None:
    """Clears a kept recording from memory now."""
    if _native():
        await call("Speech", "forgetAudio", reply=True)


async def cancel_async() -> None:
    """Stops recording (whisper's or live) without text, or stops a running transcription (its call raises
    "cancelled"). Also clears a kept recording."""
    if _native():
        await call("Speech", "cancel", reply=True)


async def transcribe_file_async(path: PathLike, *, language: str = "auto", translate: bool = False,
                                prompt: Optional[str] = None, threads: Optional[int] = None,
                                beam_size: Optional[int] = None,
                                audio_ctx: Union[None, int, str] = None) -> Dict[str, Any]:
    """Transcribes a WAV file (PCM 8/16/24/32-bit or 32-bit float, any rate or channel count) like `stop_async`."""
    _need_native("transcribe_file_async")
    o = _options(language, translate, prompt, threads, beam_size, audio_ctx)
    return _transcript(await call("Speech", "transcribeFile", os.fspath(path), o, reply=True))


# live: the phone's own on-device recognizer

def _locale(language: str) -> str:
    if not isinstance(language, str) or not _LOCALE.match(language):
        raise ValueError("language must be a BCP 47 tag like 'hi-IN' or 'en-US', not %r" % (language,))
    return language


async def live_status_async(language: str = "en-US") -> Dict[str, Any]:
    """Whether the phone's on-device recognizer can listen in `language` (a tag like "hi-IN").

    `available` is False without one (Android 11 and older, or none installed). On Android 13+ it also has
    `ready` (the language is on the phone), and the `installed`, `pending` (downloading), `downloadable` and
    `online` language lists. On Android 12 only `available` is known: try `start_live_async`."""
    if not _native():
        return {"available": False}
    return await call("Speech", "liveStatus", _locale(language), reply=True)


async def download_live_language_async(language: str) -> None:
    """Asks the on-device recognizer to download `language` (Android 13+). The phone downloads it in the
    background and may show a notification; poll `live_status_async` for `ready`."""
    _need_native("download_live_language_async")
    await call("Speech", "liveDownload", _locale(language), reply=True)


async def start_live_async(language: str = "en-US", *, max_seconds: float = 30, levels: bool = False) -> None:
    """Starts listening with the phone's on-device recognizer: no model to load, words arrive while the person
    speaks (`add_partial_listener`), and it writes languages like Hindi in their own script. It never uses the
    network (Android's `createOnDeviceSpeechRecognizer`); it raises `RuntimeError` when the phone has none or
    lacks the language. Pauses do not end it: it listens until `stop_live_async`, `cancel_async` or
    `max_seconds` (1 to 120; the state listeners get `limit`). `levels=True` sends the loudness like
    `start_async`. Whisper's recording and this one cannot run at once."""
    _need_native("start_live_async")
    if not 1 <= max_seconds <= 120:
        raise ValueError("max_seconds must be 1 to 120")
    await call("Speech", "liveStart", {"language": _locale(language), "maxSeconds": float(max_seconds),
                                       "levels": bool(levels)}, reply=True)


async def stop_live_async() -> Dict[str, Any]:
    """Stops listening and returns `text` (everything heard, empty for silence), `language`, `audio_ms` and
    `decode_ms` (the wait for the last words)."""
    _need_native("stop_live_async")
    r = await call("Speech", "liveStop", reply=True)
    return {"text": r.get("text", ""), "language": r.get("language", ""), "audio_ms": int(r.get("audioMs", 0)),
            "decode_ms": int(r.get("decodeMs", 0))}


# events

def add_state_listener(listener: Callable[[str], Any]) -> Callable[[], None]:
    """Calls `listener(state)` (see `STATES`) as recording and transcribing start and end. Returns a remove function."""
    return on_event("Speech.state", lambda d: listener(str((d or {}).get("state", ""))))


def add_level_listener(listener: Callable[[float], Any]) -> Callable[[], None]:
    """Calls `listener(level)` with the loudness, 0 (quiet) to 1, while recording with `levels=True`."""
    return on_event("Speech.level", lambda d: listener(float((d or {}).get("level", 0.0))))


def add_partial_listener(listener: Callable[[str], Any]) -> Callable[[], None]:
    """Calls `listener(text)` with everything heard so far while `start_live_async` listens."""
    return on_event("Speech.partial", lambda d: listener(str((d or {}).get("text", ""))))


statusAsync = status_async
getPermissionAsync = get_permission_async
requestPermissionAsync = request_permission_async
memoryAsync = memory_async
loadModelAsync = load_model_async
unloadModelAsync = unload_model_async
startAsync = start_async
stopAsync = stop_async
cancelAsync = cancel_async
transcribeFileAsync = transcribe_file_async
transcribeKeptAsync = transcribe_kept_async
forgetAudioAsync = forget_audio_async
addStateListener = add_state_listener
addLevelListener = add_level_listener
liveStatusAsync = live_status_async
downloadLiveLanguageAsync = download_live_language_async
startLiveAsync = start_live_async
stopLiveAsync = stop_live_async
addPartialListener = add_partial_listener

__all__ = [
    "STATES", "status_async", "get_permission_async", "request_permission_async", "memory_async",
    "load_model_async", "unload_model_async", "start_async", "stop_async", "cancel_async", "transcribe_file_async",
    "transcribe_kept_async", "forget_audio_async", "transcribeKeptAsync", "forgetAudioAsync",
    "add_state_listener", "add_level_listener", "statusAsync", "getPermissionAsync", "requestPermissionAsync",
    "memoryAsync", "loadModelAsync", "unloadModelAsync", "startAsync", "stopAsync", "cancelAsync",
    "transcribeFileAsync", "addStateListener", "addLevelListener", "live_status_async",
    "download_live_language_async", "start_live_async", "stop_live_async", "add_partial_listener",
    "liveStatusAsync", "downloadLiveLanguageAsync", "startLiveAsync", "stopLiveAsync", "addPartialListener",
]
