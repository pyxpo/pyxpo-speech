# pyxpo-speech

Offline speech to text for [pyxpo](https://github.com/pyxpo/pyxpo) apps. The microphone is recorded into memory,
then [whisper.cpp](https://github.com/ggml-org/whisper.cpp) turns it into text on the phone's CPU. There is no
cloud service, no API key and no network call. Audio is never written to a file, and it is cleared as soon as the
text is ready. It works from an app screen, and also from an Android keyboard or another service with no screen.

<p align="center">
  <img src="https://raw.githubusercontent.com/pyxpo/pyxpo-speech/main/docs/images/hero.png" width="820" alt="The AI Keyboard voice check on the Android emulator: Hinglish and English recordings turned into text on the phone, and the keyboard listening after its microphone button was tapped">
</p>

```python
import pyxpo_speech as Speech

if (await Speech.request_permission_async())["granted"]:
    await Speech.load_model_async(model_path)        # a whisper.cpp ggml file, for example ggml-base-q5_1.bin
    await Speech.start_async()                        # recording into memory
    ...
    t = await Speech.stop_async(language="en")        # {"text", "language", "segments", "no_speech", ...}
```

## Features

- **On the phone, offline:**
  - whisper.cpp 1.9.5, built from pinned source (sha256-checked) together with your app.
  - Runs on the CPU. The app carries one whisper build per CPU generation, and the phone loads the fastest one
    its CPU can run (dot product, fp16, int8 matrix multiply). No GPU, no Google services.
  - No network permission, no downloads at run time. Your app brings the model file.
- **Private by design:**
  - The microphone is recorded into memory at 16 kHz, never to a file.
  - The recording and the audio buffer are zeroed after each transcription or cancel.
  - Neither audio nor text is logged; whisper's own logging is turned off.
- **Record or transcribe a file:**
  - `start_async` / `stop_async` for the microphone, with a time limit and live loudness levels.
  - `transcribe_file_async` for WAV files (8/16/24/32-bit PCM or 32-bit float, any sample rate or channel count).
- **Any whisper language:**
  - Language detection, or a fixed language such as `"en"` or `"hi"`.
  - `translate=True` writes English from other languages.
  - A `prompt` steers spelling and style, for example Hinglish written in Roman letters.
- **Live words while speaking (Android 12+):** `start_live_async` uses the phone's own on-device recognizer
  (`createOnDeviceSpeechRecognizer`, never the network). Words arrive while the person speaks, and Hindi comes out
  in Devanagari. No whisper model needed. See [Live recognition](#live-recognition).
- **Memory-aware:** `memory_async` reports the phone's free memory, and `load_model_async` refuses a model that
  would push the phone into its low-memory zone (where Android kills apps).
- **Headless:** works while the app's Python runs without a screen, for example behind a keyboard toolbar button
  from [pyxpo-keyboard](https://pypi.org/project/pyxpo-keyboard/).
- **Typed:** ships `py.typed`. Expo-style camelCase names (`stopAsync`, ...) work too.

## Contents

| If you are... | Start here |
| --- | --- |
| Adding speech to text to your app | [Install](#install) → [Quick start](#quick-start) |
| Choosing and shipping a model | [Models](#models) |
| Recording the microphone | [Microphone permission](#microphone-permission) · [Recording](#recording) |
| Getting better text for your language | [Language, translate and prompts](#language-translate-and-prompts) |
| Showing words while the person speaks | [Live recognition](#live-recognition) |
| Transcribing a saved WAV file | [Transcribing files](#transcribing-files) |
| Adding voice typing to a keyboard | [Recipe: voice typing in a keyboard](#recipe-voice-typing-in-a-keyboard) |
| Shipping to the stores | [Production checklist](#production-checklist) |
| Looking for exact signatures | [API reference](#api-reference) · [Errors](#errors) |
| Stuck | [Testing](#testing) · [Troubleshooting](#troubleshooting) · [FAQ](#faq) |

## Platform support

| Target | Speech |
| --- | --- |
| Android, API 24+ | Everything. Built for the ABIs your app ships (`arm64-v8a`, `armeabi-v7a`, `x86_64`). |
| iOS | Not yet. `status_async()` returns `supported: False`; the other calls raise `NotSupportedError` (`unload_model_async` and `cancel_async` do nothing). |
| Web page and desktop window (`pyxpo run web` / `pyxpo run desktop`), Mac Catalyst, Windows | Same as iOS. See [Web and desktop](#web-and-desktop). |

Requires Python 3.9+ and a pyxpo core with native contract 4 (headless Python and C/C++ in modules; see
`pyxpo.module_api.NATIVE_CONTRACT`). Older cores refuse to build the module and say so.

## Install

```sh
pip install pyxpo-speech
```

Add the module to your app's `pyproject.toml`:

```toml
[tool.pyxpo]
modules = ["speech"]
```

The package has no config plugin. When the module is linked, it adds the following to the Android manifest:

- The `RECORD_AUDIO` permission. It is a runtime permission, so your app asks for it (see
  [Microphone permission](#microphone-permission)).
- `android.hardware.microphone` as an optional feature, so Google Play still offers the app on devices without a
  microphone. `transcribe_file_async` works there.

The module has C++ code, so the first build needs the Android NDK (the version pyxpo uses,
`pyxpo.build.autolink.NDK_VERSION`; Android Studio's SDK Manager installs it) and CMake. That build downloads the
whisper.cpp 1.9.5 source once, checks its sha256, and keeps it in `~/.cache/pyxpo` (or `$XDG_CACHE_HOME/pyxpo`).
To build offline, point `PYXPO_WHISPER_DIR` at an unpacked whisper.cpp 1.9.5 source tree. Rebuild the app after
adding the module (`pyxpo run android`).

## Quick start

A screen that asks for the microphone, loads a model and turns a recording into text:

```python
import os

import pyxpo_speech as Speech
from pyxpo import Button, Text, View, component, use_async, use_state

MODEL = os.path.join(os.path.expanduser("~"), "models", "whisper", "ggml-base-q5_1.bin")


@component
def Dictate():
    status = use_async(Speech.status_async, [])
    recording, set_recording = use_state(False)
    text, set_text = use_state("")
    error, set_error = use_state(None)

    async def toggle():
        set_error(None)
        try:
            if recording:
                set_recording(False)
                t = await Speech.stop_async(language="en")
                set_text(t["text"] if t["no_speech"] < 0.8 else "(nothing heard)")
            else:
                if not (await Speech.request_permission_async())["granted"]:
                    set_error("The microphone is off for this app.")
                    return
                await Speech.load_model_async(MODEL)   # does nothing when it is already loaded
                await Speech.start_async(max_seconds=30)
                set_recording(True)
        except Exception as e:
            set_recording(False)
            set_error(str(e))

    s = status.value
    if s is not None and not s["supported"]:
        return Text("Speech to text works on Android.")
    return View(
        Button("Stop" if recording else "Speak", on_press=toggle),
        Text(text),
        Text(error, style={"color": "#c62828"}) if error else None,
        style={"flex": 1, "justifyContent": "center", "padding": 24, "gap": 16},
    )
```

`os.path.expanduser("~")` is the app's private files folder on Android, so this model lives at
`files/models/whisper/` (see [Models](#models) for how it gets there).

## Models

The package does not include or download a model: your app brings one. Use the whisper.cpp ggml files from the
[ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) repository on Hugging Face (MIT). Quantized
files are smaller and faster:

| File | Size | Notes |
| --- | --- | --- |
| `ggml-tiny-q5_1.bin` | ~31 MB | Fastest. Fine for short English commands, weak for other languages. |
| `ggml-base-q5_1.bin` | ~57 MB | A good default for English. Hindi and Hinglish are rough (see below). |
| `ggml-small-q5_1.bin` | ~181 MB | Much better for Hindi, Hinglish and accents. About 3x slower than base. |
| `ggml-*.en.bin` | | English-only models. `language` must be `"en"` or `"auto"`. |

Ways to put a model on the phone:

- **Download it on first use** to the app's files, after the user agrees (it is large). Check the file's sha256
  against the value published with the model before you load it. `pyxpo-file-system` has
  `File.download_file_async(url, destination)`.
- **Ship it in the app** for a small model, then copy it out of the app's assets on first launch.
- **While developing:** `adb push` it, then copy it in with `run-as` (see [Testing](#testing)).

```python
info = await Speech.load_model_async(path)   # {"loaded": True, "model": path, "load_ms": 830, "system": "... | BACKEND = ..."}
await Speech.unload_model_async()             # frees the memory
```

- One model is loaded at a time. Loading another one unloads the first; loading the same file again does nothing.
- `load_model_async` raises `RuntimeError` for a missing, damaged or non-whisper file, or when there is not
  enough free memory. Its message says how much the model needs and how much is free.
- `check_memory=False` skips the memory check. `flash_attention=False` turns off flash attention (on by default, it
  is faster on most phones).
- `memory_async()` returns `total`, `available`, `threshold` (Android starts killing apps below it) and `low`, all
  in bytes, so you can pick a model size before loading one:

```python
async def pick_model(small_path, base_path):
    m = await Speech.memory_async()
    free = m["available"] - m["threshold"]
    return small_path if free > 400 * 1024 * 1024 else base_path
```

## Microphone permission

```python
p = await Speech.get_permission_async()       # asks nothing
p = await Speech.request_permission_async()   # shows the system dialog when it can
if p["granted"]:
    ...
```

Both return Expo's permission response: `status` (`"granted"`, `"denied"` or `"undetermined"`), `granted`,
`can_ask_again` and `expires` (`"never"`).

- The dialog needs a screen. Called from a keyboard or another service with no activity,
  `request_permission_async` returns the status when it is already granted and raises `RuntimeError` otherwise.
  Ask in your app first.
- Once the user picks "Don't allow" twice, Android stops showing the dialog and `can_ask_again` is `False`. Send
  them to your app's page in Settings instead.

## Recording

```python
await Speech.start_async(max_seconds=30, levels=True)
...
t = await Speech.stop_async(language="en")
print(t["text"], t["no_speech"], t["audio_ms"], t["decode_ms"])
```

| Call | What it does |
| --- | --- |
| `start_async(max_seconds=30, levels=False)` | Starts recording into memory. `max_seconds` is 1 to 120. Raises `RuntimeError` when already recording or the permission is missing. |
| `stop_async(language="auto", ...)` | Stops recording and transcribes it. Takes the [options below](#language-translate-and-prompts). |
| `cancel_async()` | Stops recording without transcribing (the audio is cleared), or stops a running transcription, whose call then raises `RuntimeError("cancelled")`. |
| `stop_async(..., keep_audio=True)` | Also keeps the recording in memory so it can be written again. |
| `transcribe_kept_async(language="auto", ...)` | Transcribes the kept recording again with other options, for example `translate=True` for English after a Hindi pass. Raises `RuntimeError` when nothing is kept. |
| `forget_audio_async()` | Clears the kept recording now. |

When recording reaches `max_seconds` it stops by itself and the state listeners get `"limit"`. Call `stop_async`
then to get the text. Under a quarter of a second of audio returns empty text without running whisper.

**One recording, two texts.** Keep the audio to show what was said and an English version without asking the
user to speak twice:

```python
said = await Speech.stop_async(language="hi", keep_audio=True)
english = await Speech.transcribe_kept_async(language="hi", translate=True)
await Speech.forget_audio_async()
```

A kept recording lives only in memory and is zeroed by `forget_audio_async`, the next `start_async`,
`cancel_async`, `unload_model_async`, or two minutes after stopping, whichever comes first.

`stop_async` and `transcribe_file_async` return:

| Key | Type | Meaning |
| --- | --- | --- |
| `text` | `str` | The whole transcript on one line. |
| `language` | `str` | The language whisper detected, or the one you gave. |
| `segments` | `list[str]` | The transcript in whisper's segments. |
| `no_speech` | `float` | 0 to 1. Above about 0.8 it was probably silence or noise; treat the text as empty. |
| `audio_ms` | `int` | The length of the audio. |
| `decode_ms` | `int` | How long whisper took. `decode_ms / audio_ms` below 1 is faster than real time. |

**Events.** Listeners return a function that removes them:

```python
off = Speech.add_state_listener(lambda state: print(state))   # "recording", "transcribing", "idle", "limit"
off_level = Speech.add_level_listener(lambda level: None)      # 0..1, ten times a second with levels=True
```

`STATES` lists the four states. Listeners keep working while the app runs without a screen.

## Language, translate and prompts

`stop_async` and `transcribe_file_async` take the same options:

| Option | Default | What it does |
| --- | --- | --- |
| `language` | `"auto"` | `"auto"` detects the language. Or a whisper code such as `"en"`, `"hi"`, `"es"` (lower case, 2 or 3 letters). |
| `translate` | `False` | Writes English, whatever was spoken. Needs a multilingual model. |
| `prompt` | `None` | Text that primes spelling and style: names, product words, how to write a mixed language. |
| `threads` | up to 4 | CPU threads, 1 to 16. |
| `beam_size` | greedy | 1 to 8. Above 1 searches more: slower, sometimes better. |
| `audio_ctx` | `None` (30 s) | How much audio the encoder looks at. `None` is whisper's full 30-second window. `"fit"` sizes it to the clip, and an `int` from 64 to 1500 sets it in steps of 20 ms. A smaller window is much faster but less reliable: see [How fast is it?](#faq). |

Bad values raise `ValueError` before anything runs.

**Hinglish and Hindi.** Measured on the voice check app in the pictures, with `ggml-base` and `ggml-small`:

- `"auto"` and `"hi"` write Hindi speech in Urdu script more often than not.
- `language="en"` with a Roman Hinglish prompt writes Hinglish in Roman letters, and leaves English as it is:

```python
HINGLISH = "Haan bhai, main kal office aa raha hoon. Meeting ke baad baat karte hain, theek hai?"
t = await Speech.stop_async(language="en", prompt=HINGLISH)
# "Bhai kal meeting cancelled ka do, main busy hoon, next week karte hain."
```

- `language="hi"` with a Devanagari prompt keeps Hindi in Devanagari. `ggml-base` gets Hindi words wrong often;
  use `ggml-small` or larger for Hindi.
- `language="hi", translate=True` with `ggml-small` gives good English from Hindi speech.

**Clean up before you type it.** whisper writes tags like `[BLANK_AUDIO]` or `(music)` for noise, "you" or
"Thank you." for silence, and with a short or quiet recording it can repeat your prompt. Drop those, and treat a
high `no_speech` as nothing heard. The [recipe](#recipe-voice-typing-in-a-keyboard) does this.

## Live recognition

Android 12 and later can carry an on-device speech recognizer (on most phones it comes with Google's
"Speech Recognition & Synthesis" or Android System Intelligence). `start_live_async` uses it through
`SpeechRecognizer.createOnDeviceSpeechRecognizer`, which never sends audio to a server. Words arrive while the
person speaks, which whisper cannot do, and Hindi is written in Devanagari.

```python
from pyxpo_speech import Speech


async def listen_hindi(show):
    s = await Speech.live_status_async("hi-IN")
    if not s["available"]:
        return None                                  # no on-device recognizer: use whisper instead
    if not s.get("ready", True):
        await Speech.download_live_language_async("hi-IN")   # the phone shows the download
        return None
    remove = Speech.add_partial_listener(show)       # show(text): everything heard so far
    await Speech.start_live_async("hi-IN", max_seconds=30, levels=True)
    return remove


async def finish():
    r = await Speech.stop_live_async()
    return r["text"]                                 # "मैं कल ऑफिस आऊँगा"
```

- `language` is a locale such as `"hi-IN"`, `"en-IN"` or `"en-US"` (not whisper's `"hi"`).
- The recognizer ends its session at a pause. The module starts a new one until `stop_live_async`, so one
  recording can hold several sentences. `max_seconds` ends it (the state listener hears `"limit"`).
- `live_status_async(language)` returns `{"available": False}` without a recognizer. On Android 13+ it also returns
  `installed`, `pending`, `downloadable` and `online` language lists, and `ready` (the language is installed).
  On Android 12 it returns only `available`.
- `download_live_language_async` asks the recognizer to download a language (Android 13+). The phone shows its own
  progress; call `live_status_async` again later.
- The recognizer keeps no audio, so `translate` and `transcribe_kept_async` do not apply. For English, send the
  text to a language model.
- Nothing is logged. `cancel_async` stops listening and drops the text.

## Transcribing files

```python
t = await Speech.transcribe_file_async(wav_path, language="en")
```

- WAV only: 8, 16, 24 or 32-bit PCM, or 32-bit float, at any sample rate, mono or more channels (mixed to mono and
  resampled to 16 kHz).
- The file is read into memory and the buffer is cleared after transcribing. The file itself is not changed.
- It needs no microphone permission, so it works on devices without a microphone.

## Errors

| Error | When | What to do |
| --- | --- | --- |
| `ValueError` | A bad `language` (`"Hindi"`, `"EN"`, `""`), `threads` outside 1 to 16 or not an `int`, `beam_size` outside 1 to 8, `audio_ctx` not `None`, `"fit"` or 64 to 1500, `max_seconds` outside 1 to 120, an empty model path, a live `language` that is not a locale (`"Hindi"`, `"hi_IN"`). | Use whisper's lower-case codes and the ranges in [Language, translate and prompts](#language-translate-and-prompts). |
| `RuntimeError` | No model file at the path, a damaged file, not enough free memory, no model loaded, no microphone permission, already recording, `stop_async` while not recording, a language the model does not know, a non-English language with an `.en` model, `"cancelled"`, live recognition without a recognizer, or a language the phone has not downloaded ("download it first"), asking for the permission without a screen. | Show the message; it says what is missing. For memory, load a smaller model. |
| `NotSupportedError` (from `pyxpo.module_api`) | Any call other than `status_async`, `unload_model_async` and `cancel_async` on iOS, the web or the desktop. | Check `status["supported"]` first. |
| `ModuleNotLinkedError` (from `pyxpo.module_api`) | An Android app built without the module. | Add `"speech"` to `[tool.pyxpo] modules` and rebuild. |

## Web and desktop

There is no speech engine on the web, the desktop, Mac Catalyst or Windows in this version:

- `status_async()` returns `{"supported": False, "loaded": False, "model": None, "multilingual": False,
  "recording": False, "transcribing": False}`.
- `unload_model_async()` and `cancel_async()` do nothing.
- Every other call raises `NotSupportedError`.

Hide your voice buttons when `status["supported"]` is `False`. The Quick start screen shows "Speech to text works
on Android." there.

## Recipe: voice typing in a keyboard

<p align="center">
  <img src="https://raw.githubusercontent.com/pyxpo/pyxpo-speech/main/docs/images/keyboard.png" width="560" alt="The AI Keyboard on the Android emulator: listening after its microphone button was tapped, then saying nothing was heard after a silent recording">
</p>

This is how the AI Keyboard app (in the pictures) adds a 🎤 button to its
[pyxpo-keyboard](https://pypi.org/project/pyxpo-keyboard/) toolbar. A tap starts the app's Python in the background
when it is not running, so everything below happens without a screen:

- Tap 🎤: load a model if none is loaded (the largest one that fits in memory), start recording.
- Tap ⏹, or reach the time limit: transcribe, clean up the text, type it into the field.
- The keyboard closes: drop the recording.

```python
import asyncio
import os
import re

import pyxpo_keyboard as Keyboard
import pyxpo_speech as Speech

MIC = "mic"
MODELS = os.path.join(os.path.expanduser("~"), "models", "whisper")
PREFERRED = ("ggml-small-q5_1.bin", "ggml-base-q5_1.bin")   # better first; a model that does not fit is skipped
PROMPT = "Haan bhai, main kal office aa raha hoon. Meeting ke baad baat karte hain, theek hai?"
NOISE = re.compile(r"\[[^\]]*\]|\((?:music|silence|noise|applause|laughs?)\)|\*[^*]*\*", re.I)
SILENCE = {"you", "thank you", "thanks", "thanks for watching", "bye", "okay", "so"}

phase = "idle"   # idle, starting, recording, writing


def clean(text):
    text = re.sub(r"\s+", " ", NOISE.sub(" ", text)).strip()
    words = " ".join(re.findall(r"\w+", text.lower()))
    return "" if words in SILENCE or (len(words) > 30 and words in PROMPT.lower()) else text


async def toolbar(active):
    await Keyboard.set_toolbar_async([{"id": MIC, "label": "⏹" if active else "🎤", "active": active,
                                       "description": "Finish voice typing" if active else "Voice typing"}])


async def ensure_model():
    if (await Speech.status_async())["loaded"]:
        return
    last = RuntimeError("No speech model yet: open the app to get one")
    for name in PREFERRED:
        path = os.path.join(MODELS, name)
        if os.path.isfile(path):
            try:
                await Speech.load_model_async(path)
                return
            except RuntimeError as e:   # not enough memory: try the next one
                last = e
    raise last


async def start():
    global phase
    phase = "starting"
    try:
        if not (await Speech.get_permission_async())["granted"]:
            raise RuntimeError("Allow the microphone in the app first")
        await Keyboard.set_status_async("Loading speech…")
        await ensure_model()
        if phase != "starting":   # the keyboard closed meanwhile
            return
        await Speech.start_async(max_seconds=30)
        phase = "recording"
        await toolbar(True)
        await Keyboard.set_status_async("Listening… tap ⏹ when done")
    except Exception as e:
        phase = "idle"
        await Keyboard.set_status_async(str(e))


async def finish():
    global phase
    phase = "writing"
    try:
        await toolbar(False)
        await Keyboard.set_status_async("Writing it down…")
        t = await Speech.stop_async(language="en", prompt=PROMPT)
        text = clean(t["text"]) if t["no_speech"] < 0.8 else ""
        if text:
            await Keyboard.commit_text_async(text + " ")
            await Keyboard.set_status_async("")
        else:
            await Keyboard.set_status_async("Nothing heard. Tap 🎤 and speak")
    except Exception as e:
        await Keyboard.set_status_async(str(e))
    finally:
        phase = "idle"


async def cancel():
    global phase
    phase = "idle"
    await Speech.cancel_async()
    await toolbar(False)


def run(coro):
    asyncio.get_event_loop().create_task(coro)


def on_tap(event):
    if event["id"] == MIC and phase == "idle":
        run(start())
    elif event["id"] == MIC and phase == "recording":
        run(finish())


def on_state(state):
    if state == "limit" and phase == "recording":
        run(finish())


def on_input(event):
    if event["state"] == "finish" and phase in ("starting", "recording"):
        run(cancel())


Keyboard.add_toolbar_listener(on_tap)
Keyboard.add_input_listener(on_input)
Speech.add_state_listener(on_state)
```

Import this module from your root layout so the listeners are registered when the keyboard starts your Python, and
call `toolbar(False)` once when the app opens (for example `use_async(lambda: toolbar(False), [])`). The keyboard
shows no toolbar buttons in private fields (passwords, PINs, incognito tabs) and does not type app text there, so
voice typing never runs in them.

## Production checklist

- Say in plain words, before you ask for the microphone, that speech is turned into text on the phone and nothing
  is sent anywhere.
- Ask for the microphone in your app. A keyboard or other service cannot show the dialog.
- Download a model only after the user agrees, show its size, and check its sha256 before loading it. Prefer
  Wi-Fi for the large ones.
- Pick the model from `memory_async()`, and handle the `RuntimeError` from `load_model_async` by trying a smaller
  one. Phones with 3 to 4 GB of memory often cannot keep `ggml-small` loaded next to other apps.
- Unload the model (`unload_model_async`) when the user leaves voice features for a while; it holds 60 to 250 MB.
- Treat `no_speech` above about 0.8 as silence, and drop whisper's noise tags and silence words before you use the
  text.
- Measure on a real mid-range phone: `decode_ms / audio_ms` tells you how long people wait after they stop talking.
  Emulators are much slower and their microphones often do not work.
- Google Play: declare the microphone use and say in the data safety form that audio is processed on the device
  and not collected. Keep a privacy policy that says so.
- Hide voice features on iOS, the web and the desktop (`status["supported"]` is `False` there).

## Testing

**On the Android emulator.** A WAV file and `transcribe_file_async` test the whole pipeline without a
microphone. Copy a model and a WAV into the app's files (`run-as` works with debug builds):

```sh
adb push ggml-base-q5_1.bin /data/local/tmp/
adb shell run-as com.example.app mkdir -p files/models/whisper
adb shell run-as com.example.app cp /data/local/tmp/ggml-base-q5_1.bin files/models/whisper/
adb shell rm /data/local/tmp/ggml-base-q5_1.bin
```

The emulator's microphone forwards your computer's microphone only when you allow it in the emulator's extended
controls (Microphone > "Virtual microphone uses host audio input"); otherwise it records silence. Recording on
macOS hosts can also hang the emulator. Start it with `-no-audio` to test the recording flow with silence (it
should end in empty text), and test real speech on a phone.

**Unit tests** run without a device. `pyxpo.module_testing.FakeBridge` pretends to be the native side, and
`pyxpo.modules.dispatch` sends events to your listeners:

```python
import asyncio

import pyxpo_speech as Speech
from pyxpo import modules
from pyxpo.module_testing import FakeBridge


def test_stop_returns_the_transcript():
    with FakeBridge("android", modules=("Speech",)) as fb:
        fb.on("Speech", "start", lambda options: None)
        fb.on("Speech", "stop", lambda options: {"text": "hello there", "language": "en", "noSpeech": 0.01,
                                                 "segments": ["hello there"], "audioMs": 1500, "decodeMs": 420})
        asyncio.run(Speech.start_async(max_seconds=10))
        t = asyncio.run(Speech.stop_async(language="en"))
        assert t["text"] == "hello there" and t["audio_ms"] == 1500
        assert fb.calls[-1][2] == ({"language": "en", "translate": False},)


def test_limit_event_reaches_listeners():
    seen = []
    off = Speech.add_state_listener(seen.append)
    modules.dispatch({"name": "Speech.state", "data": {"state": "limit"}})
    off()
    assert seen == ["limit"]
```

Without a `FakeBridge`, the calls behave as on the desktop. A screen test can also
`monkeypatch.setattr(Speech, "stop_async", fake)` with an `async def fake(**options)`.

## Troubleshooting

| Problem | Cause and fix |
| --- | --- |
| Hindi comes out in Urdu script | whisper writes Hindi in Urdu script with `"auto"` or `"hi"` and no prompt. Use `language="en"` with a Roman Hinglish prompt for Hinglish, or `"hi"` with a Devanagari prompt for Hindi. See [Language, translate and prompts](#language-translate-and-prompts). |
| A silent recording types "you" or "Thank you." | That is what whisper hears in silence. Treat `no_speech` above about 0.8 as nothing, and drop those words (see the recipe). |
| The text repeats the prompt | whisper copies the prompt when it hears little. Drop a transcript that is part of the prompt, and keep prompts short. |
| `not enough free memory for this model` | The model would push the phone into its low-memory zone. Load a smaller model, or close other apps. `check_memory=False` skips the check, but Android may then kill the app. |
| `could not load ...` | The file is not a whisper.cpp ggml model, or it is damaged (a partial download). Check its size and sha256. |
| `this model knows English only` | `.en` models know English only. Use `language="en"` or a multilingual model. |
| `asking for the microphone needs the app on screen` | `request_permission_async` was called from a keyboard or service. Ask in your app. |
| Transcribing takes much longer than the recording | The phone is slow for that model, or it is an emulator. Use a smaller or quantized (`q5_1`) model; leave `threads` at the default. |
| The emulator hangs or closes when recording starts | A host audio problem on macOS. Start the emulator with `-no-audio`, and test real speech on a phone. |
| The first build fails downloading whisper.cpp | The build needs the network once. Retry, or set `PYXPO_WHISPER_DIR` to an unpacked whisper.cpp 1.9.5 source tree. |
| `NotSupportedError` | iOS, the web and the desktop have no speech engine in this version. Check `status["supported"]`. |
| `ModuleNotLinkedError` | The app was built without the module. Add `"speech"` to `[tool.pyxpo] modules` and rebuild. |

## FAQ

**Does any audio or text leave the phone?**
No. Recording, transcription and the model all stay in your app's process. The package has no network code; the
manifest it adds holds only the microphone permission. Audio buffers are zeroed after use, and nothing is logged.

**Why does the app need to bring the model?**
Models are 30 MB to 1.5 GB, and apps need different ones (English only, or many languages; fast or accurate). The
package stays small, and you decide how and when to download, with the user's consent.

**How fast is it?**
It depends on the phone and the model. On a Galaxy S23 with the default 30-second window:

| Clip | `ggml-base-q5_1` | `ggml-small-q5_1` |
| --- | --- | --- |
| 11 s English | 1.5 s | 6.3 s |
| 5 s Hinglish (Roman) | 1.5 s | 6.6 s |
| 4 s Hindi (Devanagari) | 5.5 s, wrong text | 7.0 s, close |

Whisper encodes a 30-second window, so a short clip costs almost as much as a long one. `audio_ctx="fit"`
encodes only the clip: on the S23 it wrote the same clips in 0.6 to 2.3 s, but it dropped words, punctuation,
or once all but one word, and when the text came out unsure whisper decoded again and took longer (12.8 s for
the Hindi clip). Keep the default for typing, and measure `"fit"` with your own model and languages before
using it. The numbers in the pictures come from an emulator on a busy computer and are slower than a phone.
Use `decode_ms / audio_ms` to measure your own.

**Which CPU build does it use?**
The end of `system` (returned by `load_model_async`) names it, for example
`BACKEND = libpyxpo_speech_ggml_cpu_android_armv8.6_1.so` on a Galaxy S23. ggml checks the CPU's features and skips
builds the phone cannot run. On a 64-bit ARM or x86_64 phone the app carries one build per CPU generation (about
8 MB of native code per ABI in all); 32-bit ARM gets one plain build.

**Can it stream text while I talk?**
Not in this version. It transcribes when recording stops. A 30-second limit keeps the wait short.

**Does it use Android's built-in speech recognizer?**
No. The system recognizer depends on Google services and may send audio to a server. This package runs whisper
inside your app.

**When does iOS get it?**
whisper.cpp runs on iOS too; the iOS side of this module is planned. Until then, `status_async()` reports
`supported: False` on iOS.

## API reference

**Functions**

| Name | Description |
| --- | --- |
| `status_async() -> dict` | `{"supported", "loaded", "model", "multilingual", "recording", "transcribing"}`. |
| `get_permission_async() -> dict` | The microphone permission, without asking. Raises `NotSupportedError` off Android. |
| `request_permission_async() -> dict` | Asks for the microphone (needs a screen). |
| `memory_async() -> dict` | `{"total", "available", "threshold", "low"}` in bytes. |
| `load_model_async(path, *, check_memory=True, flash_attention=True) -> dict` | Loads a whisper.cpp ggml file. Returns the status plus `load_ms` and `system`. |
| `unload_model_async() -> None` | Frees the model. Does nothing off Android. |
| `start_async(*, max_seconds=30, levels=False) -> None` | Starts recording into memory. |
| `stop_async(*, language="auto", translate=False, prompt=None, threads=None, beam_size=None, audio_ctx=None, keep_audio=False) -> dict` | Stops and transcribes. Returns `{"text", "language", "segments", "no_speech", "audio_ms", "decode_ms"}`. |
| `cancel_async() -> None` | Drops the recording, or stops a transcription. Does nothing off Android. |
| `transcribe_kept_async(*, language="auto", translate=False, prompt=None, threads=None, beam_size=None, audio_ctx=None) -> dict` | Transcribes the recording kept by `stop_async(keep_audio=True)`; returns the same as `stop_async`. |
| `forget_audio_async() -> None` | Clears a kept recording. Does nothing off Android. |
| `transcribe_file_async(path, *, language="auto", translate=False, prompt=None, threads=None, beam_size=None, audio_ctx=None) -> dict` | Transcribes a WAV file; returns the same as `stop_async`. |
| `live_status_async(language="en-US") -> dict` | The on-device recognizer: `{"available"}`, plus `installed`, `pending`, `downloadable`, `online` and `ready` on Android 13+. `{"available": False}` off Android. |
| `download_live_language_async(language) -> None` | Asks the recognizer to download a language (Android 13+). |
| `start_live_async(language="en-US", *, max_seconds=30, levels=False) -> None` | Starts live recognition. Partial text goes to `add_partial_listener`. |
| `stop_live_async() -> dict` | Stops and returns `{"text", "language", "audio_ms", "decode_ms"}`. |
| `add_partial_listener(fn) -> remove` | `fn(text)`: everything heard so far, while live recognition runs. |
| `add_state_listener(fn) -> remove` | `fn(state)`, one of `STATES`. |
| `add_level_listener(fn) -> remove` | `fn(level)`, 0 to 1, while recording with `levels=True`. |
| `statusAsync`, `getPermissionAsync`, `requestPermissionAsync`, `memoryAsync`, `loadModelAsync`, `unloadModelAsync`, `startAsync`, `stopAsync`, `cancelAsync`, `transcribeFileAsync`, `transcribeKeptAsync`, `forgetAudioAsync`, `addStateListener`, `addLevelListener`, `liveStatusAsync`, `downloadLiveLanguageAsync`, `startLiveAsync`, `stopLiveAsync`, `addPartialListener` | Expo-style names for the functions above. |

**Constants**

| Name | Description |
| --- | --- |
| `STATES` | `("recording", "transcribing", "idle", "limit")`. |

## License

MIT. whisper.cpp is MIT-licensed; check the license of the model you ship. The screenshots show the AI Keyboard
app running on the Android emulator.
