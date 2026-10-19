"""Offline speech to text for pyxpo apps: the microphone and whisper.cpp, on the phone.

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
             beam_size: Optional[int]) -> Dict[str, Any]:
    if not isinstance(language, str) or not _LANG.match(language):
        raise ValueError("language must be 'auto' or a whisper language code like 'en' or 'hi', not %r" % (language,))
    if threads is not None and not (isinstance(threads, int) and 1 <= threads <= 16):
        raise ValueError("threads must be 1 to 16")
    if beam_size is not None and not (isinstance(beam_size, int) and 1 <= beam_size <= 8):
        raise ValueError("beam_size must be 1 to 8")
    o: Dict[str, Any] = {"language": language, "translate": bool(translate)}
    if prompt:
        o["prompt"] = str(prompt)
    if threads:
        o["threads"] = threads
    if beam_size:
        o["beamSize"] = beam_size
    return o


def _transcript(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"text": r.get("text", ""), "language": r.get("language", ""), "segments": list(r.get("segments") or []),
            "no_speech": float(r.get("noSpeech", 0.0)), "audio_ms": int(r.get("audioMs", 0)),
            "decode_ms": int(r.get("decodeMs", 0))}


async def stop_async(*, language: str = "auto", translate: bool = False, prompt: Optional[str] = None,
                     threads: Optional[int] = None, beam_size: Optional[int] = None) -> Dict[str, Any]:
    """Stops recording and transcribes it. Returns `text`, `language` (detected, or the one given), `segments`,
    `no_speech` (0..1; high means it was likely silence), `audio_ms` and `decode_ms`.

    `language`: `"auto"` detects it, or a code like `"en"`/`"hi"`. `translate=True` makes whisper write English.
    `prompt` primes spelling and style (names, Hinglish words). `threads` defaults to up to 4 CPU cores,
    `beam_size` > 1 searches more (slower, sometimes better)."""
    _need_native("stop_async")
    o = _options(language, translate, prompt, threads, beam_size)
    return _transcript(await call("Speech", "stop", o, reply=True))


async def cancel_async() -> None:
    """Stops recording without transcribing, or stops a running transcription (its call raises "cancelled")."""
    if _native():
        await call("Speech", "cancel", reply=True)


async def transcribe_file_async(path: PathLike, *, language: str = "auto", translate: bool = False,
                                prompt: Optional[str] = None, threads: Optional[int] = None,
                                beam_size: Optional[int] = None) -> Dict[str, Any]:
    """Transcribes a WAV file (PCM 8/16/24/32-bit or 32-bit float, any rate or channel count) like `stop_async`."""
    _need_native("transcribe_file_async")
    o = _options(language, translate, prompt, threads, beam_size)
    return _transcript(await call("Speech", "transcribeFile", os.fspath(path), o, reply=True))


# events

def add_state_listener(listener: Callable[[str], Any]) -> Callable[[], None]:
    """Calls `listener(state)` (see `STATES`) as recording and transcribing start and end. Returns a remove function."""
    return on_event("Speech.state", lambda d: listener(str((d or {}).get("state", ""))))


def add_level_listener(listener: Callable[[float], Any]) -> Callable[[], None]:
    """Calls `listener(level)` with the loudness, 0 (quiet) to 1, while recording with `levels=True`."""
    return on_event("Speech.level", lambda d: listener(float((d or {}).get("level", 0.0))))


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
addStateListener = add_state_listener
addLevelListener = add_level_listener

__all__ = [
    "STATES", "status_async", "get_permission_async", "request_permission_async", "memory_async",
    "load_model_async", "unload_model_async", "start_async", "stop_async", "cancel_async", "transcribe_file_async",
    "add_state_listener", "add_level_listener", "statusAsync", "getPermissionAsync", "requestPermissionAsync",
    "memoryAsync", "loadModelAsync", "unloadModelAsync", "startAsync", "stopAsync", "cancelAsync",
    "transcribeFileAsync", "addStateListener", "addLevelListener",
]
