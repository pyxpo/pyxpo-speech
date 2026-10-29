"""pyxpo-speech: the calls, argument checks, other-platform behaviour and the Android native wiring."""
import asyncio
import json
import re
from pathlib import Path

import pytest

from pyxpo import modules
from pyxpo.build import autolink
from pyxpo.module_api import ModuleNotLinkedError, NotSupportedError
from pyxpo.module_testing import FakeBridge

import pyxpo_speech as Speech

PKG = Path(Speech.__file__).parent
MAIN = PKG / "android" / "src" / "main"
KT = MAIN / "java" / "dev" / "pyxpo" / "modules" / "speech"


def run(coro):
    return asyncio.run(coro)


RAW = {"text": "hello there", "language": "en", "noSpeech": 0.01, "segments": ["hello there"], "audioMs": 1500,
       "decodeMs": 420}
TRANSCRIPT = {"text": "hello there", "language": "en", "segments": ["hello there"], "no_speech": 0.01,
              "audio_ms": 1500, "decode_ms": 420}


@pytest.mark.parametrize("os_name", ["desktop", "web", "ios"])
def test_other_platforms_report_unsupported(os_name):
    def check():
        assert run(Speech.status_async())["supported"] is False
        assert run(Speech.live_status_async("hi-IN")) == {"available": False}
        run(Speech.unload_model_async())
        run(Speech.cancel_async())
        run(Speech.forget_audio_async())
        for coro in (Speech.get_permission_async(), Speech.request_permission_async(), Speech.memory_async(),
                     Speech.load_model_async("m.bin"), Speech.start_async(), Speech.stop_async(),
                     Speech.transcribe_file_async("a.wav"), Speech.transcribe_kept_async(),
                     Speech.start_live_async("hi-IN"), Speech.stop_live_async(),
                     Speech.download_live_language_async("hi-IN")):
            with pytest.raises(NotSupportedError):
                run(coro)

    if os_name == "desktop":
        check()
    else:
        with FakeBridge(os_name) as fb:
            check()
            assert fb.calls == []


def test_android_calls():
    with FakeBridge("android", modules=("Speech",)) as fb:
        fb.on("Speech", "status", lambda: {"supported": True, "loaded": False})
        fb.on("Speech", "getPermission", lambda: {"status": "undetermined", "canAskAgain": True})
        fb.on("Speech", "requestPermission", lambda: {"status": "granted", "canAskAgain": True})
        fb.on("Speech", "memory", lambda: {"total": 8 << 30, "available": 3 << 30, "threshold": 1 << 28, "low": False})
        fb.on("Speech", "load", lambda p, o: {"supported": True, "loaded": True, "model": p, "loadMs": 900, "system": "NEON = 1"})
        fb.on("Speech", "unload", lambda: None)
        fb.on("Speech", "start", lambda o: None)
        fb.on("Speech", "stop", lambda o: RAW)
        fb.on("Speech", "cancel", lambda: None)
        fb.on("Speech", "transcribeFile", lambda p, o: RAW)
        fb.on("Speech", "transcribeKept", lambda o: RAW)
        fb.on("Speech", "forgetAudio", lambda: None)
        assert run(Speech.status_async())["supported"] is True
        assert run(Speech.get_permission_async())["status"] == "undetermined"
        assert run(Speech.request_permission_async())["granted"] is True
        assert run(Speech.memory_async())["low"] is False
        loaded = run(Speech.load_model_async(Path("/data/m/ggml-base.bin")))
        assert loaded["model"] == "/data/m/ggml-base.bin" and loaded["load_ms"] == 900 and "loadMs" not in loaded
        run(Speech.start_async(max_seconds=10, levels=True))
        assert run(Speech.stop_async(language="hi", prompt="Hinglish", threads=4, beam_size=2)) == TRANSCRIPT
        assert run(Speech.transcribe_file_async("/data/a.wav", translate=True)) == TRANSCRIPT
        run(Speech.transcribe_file_async("/data/a.wav", audio_ctx="fit"))
        run(Speech.transcribe_file_async("/data/a.wav", audio_ctx=768))
        assert run(Speech.stop_async(language="hi", keep_audio=True)) == TRANSCRIPT
        assert run(Speech.transcribe_kept_async(language="hi", translate=True)) == TRANSCRIPT
        run(Speech.forget_audio_async())
        run(Speech.cancel_async())
        run(Speech.unload_model_async())
        names = [c[1] for c in fb.calls]
        assert names == ["status", "getPermission", "requestPermission", "memory", "load", "start", "stop",
                         "transcribeFile", "transcribeFile", "transcribeFile", "stop", "transcribeKept",
                         "forgetAudio", "cancel", "unload"]
        files = [c[2][1] for c in fb.calls if c[1] == "transcribeFile"]
        assert [f.get("audioCtx") for f in files] == [None, -1, 768]   # -1: sized to the clip natively
        stops = [c[2] for c in fb.calls if c[1] == "stop"]
        assert stops[1] == ({"language": "hi", "translate": False, "keepAudio": True},)
        args = {c[1]: c[2] for c in fb.calls if c[1] not in ("transcribeFile", "stop")}
        assert args["transcribeKept"] == ({"language": "hi", "translate": True},)
        assert args["load"] == ("/data/m/ggml-base.bin", {"checkMemory": True, "flashAttention": True})
        assert args["start"] == ({"maxSeconds": 10.0, "levels": True},)
        assert stops[0] == ({"language": "hi", "translate": False, "prompt": "Hinglish", "threads": 4, "beamSize": 2},)
        assert files[0] == {"language": "auto", "translate": True}


def test_android_live_calls():
    with FakeBridge("android", modules=("Speech",)) as fb:
        fb.on("Speech", "liveStatus", lambda lang: {"available": True, "ready": True, "installed": [lang]})
        fb.on("Speech", "liveDownload", lambda lang: None)
        fb.on("Speech", "liveStart", lambda o: None)
        fb.on("Speech", "liveStop", lambda: {"text": "नमस्ते", "language": "hi-IN", "audioMs": 2100, "decodeMs": 80})
        assert run(Speech.live_status_async("hi-IN"))["ready"] is True
        run(Speech.download_live_language_async("hi-IN"))
        run(Speech.start_live_async("hi-IN", max_seconds=20, levels=True))
        assert run(Speech.stop_live_async()) == {"text": "नमस्ते", "language": "hi-IN", "audio_ms": 2100,
                                                  "decode_ms": 80}
        assert [c[1] for c in fb.calls] == ["liveStatus", "liveDownload", "liveStart", "liveStop"]
        assert fb.calls[2][2] == ({"language": "hi-IN", "maxSeconds": 20.0, "levels": True},)


@pytest.mark.parametrize("language", ["hindi", "HI-IN", "", "hi_IN", "auto x"])
def test_bad_live_language(language):
    with FakeBridge("android", modules=("Speech",)) as fb:
        for coro in (Speech.live_status_async(language), Speech.start_live_async(language),
                     Speech.download_live_language_async(language)):
            with pytest.raises(ValueError):
                run(coro)
        with pytest.raises(ValueError, match="max_seconds"):
            run(Speech.start_live_async("hi-IN", max_seconds=0))
        assert fb.calls == []


@pytest.mark.parametrize("kwargs", [{"language": "Hindi"}, {"language": ""}, {"language": "EN"}, {"threads": 0},
                                    {"threads": 64}, {"beam_size": 9}, {"threads": 2.5},
                                    {"audio_ctx": 32}, {"audio_ctx": 1501}, {"audio_ctx": "auto"},
                                    {"audio_ctx": True}])
def test_bad_transcribe_options(kwargs):
    with FakeBridge("android", modules=("Speech",)) as fb:
        with pytest.raises(ValueError):
            run(Speech.stop_async(**kwargs))
        with pytest.raises(ValueError):
            run(Speech.transcribe_file_async("a.wav", **kwargs))
        with pytest.raises(ValueError):
            run(Speech.transcribe_kept_async(**kwargs))
        assert fb.calls == []


def test_bad_start_and_load():
    with FakeBridge("android", modules=("Speech",)) as fb:
        for s in (0, 0.5, 121):
            with pytest.raises(ValueError, match="max_seconds"):
                run(Speech.start_async(max_seconds=s))
        with pytest.raises(ValueError):
            run(Speech.load_model_async(""))
        assert fb.calls == []


def test_not_linked():
    with FakeBridge("android"):
        with pytest.raises(ModuleNotLinkedError, match="pyxpo-speech"):
            run(Speech.status_async())


def test_state_and_level_listeners():
    seen = []
    off_state = Speech.add_state_listener(seen.append)
    off_level = Speech.add_level_listener(seen.append)
    off_partial = Speech.add_partial_listener(seen.append)
    modules.dispatch({"name": "Speech.state", "data": {"state": "recording"}})
    modules.dispatch({"name": "Speech.level", "data": {"level": 0.25}})
    modules.dispatch({"name": "Speech.partial", "data": {"text": "मैं कल"}})
    off_state()
    off_level()
    off_partial()
    modules.dispatch({"name": "Speech.state", "data": {"state": "idle"}})
    modules.dispatch({"name": "Speech.partial", "data": {"text": "x"}})
    assert seen == ["recording", 0.25, "मैं कल"]
    assert set(Speech.STATES) == {"recording", "transcribing", "idle", "limit"}


def test_module_manifest_builds_native_code():
    meta = json.loads((PKG / "pyxpo-module.json").read_text())
    assert meta["native"] == "Speech" and meta["name"] == "speech" and "speech" in autolink.OFFICIAL
    assert meta["nativeContract"] == 4 and "ios" not in meta
    assert meta["android"]["classes"] == ["dev.pyxpo.modules.speech.SpeechModule"]
    cmake = PKG / "android" / meta["android"]["cmake"]["path"]
    assert cmake.is_file() and (cmake.parent / "speech_jni.cpp").is_file()
    assert "android.permission.RECORD_AUDIO" in (MAIN / "AndroidManifest.xml").read_text()


def test_whisper_source_is_pinned_and_hidden():
    cm = (MAIN / "cpp" / "CMakeLists.txt").read_text()
    sha = re.search(r"set\(WHISPER_SHA256 ([0-9a-f]+)\)", cm).group(1)
    assert len(sha) == 64 and "EXPECTED_HASH SHA256=${WHISPER_SHA256}" in cm and "TLS_VERIFY ON" in cm
    assert "PYXPO_WHISPER_DIR" in cm and "file(LOCK" in cm
    # shared ggml with one CPU backend per instruction set level, picked on the phone (a generic armv8.0 build
    # was 1.6x slower on a Galaxy S23); every library is renamed so a llama.cpp module's ggml can live alongside
    assert "set(BUILD_SHARED_LIBS ON" in cm and "set(GGML_BACKEND_DL ON" in cm and "set(GGML_CPU_ALL_VARIANTS ON" in cm
    assert "foreach(target ggml ggml-base whisper parakeet ${GGML_AVAILABLE_BACKENDS})" in cm
    assert 'string(REPLACE "-" "_" name "pyxpo_speech_${target}")' in cm and "OUTPUT_NAME ${name}" in cm
    assert "PYXPO_CPU_BACKENDS=" in cm and "CMAKE_CXX_VISIBILITY_PRESET hidden" in cm
    assert "-Wl,-z,max-page-size=16384" in cm  # 16 KB page devices
    assert "set(GGML_NATIVE OFF" in cm and "set(GGML_OPENMP OFF" in cm
    # debug app builds would run whisper about ten times slower unoptimized
    assert "set(CMAKE_BUILD_TYPE Release)" in cm
    jni = (MAIN / "cpp" / "speech_jni.cpp").read_text()
    assert "whisper_log_set(quiet, nullptr)" in jni  # no transcript or audio detail reaches logcat
    # fastest CPU backend first; ggml refuses one whose instructions the CPU lacks, and no backend fails init
    assert "ggml_backend_load(name.c_str())" in jni and "load_cpu_backend();" in jni
    assert "speech cannot run on this phone's CPU" in jni and "BACKEND = " in jni
    for name in ("nativeInit", "nativeFree", "nativeAbort", "nativeIsMultilingual", "nativeKnowsLanguage",
                 "nativeSystemInfo", "nativeTranscribe"):
        assert "Java_dev_pyxpo_modules_speech_Whisper_" + name in jni
        assert "external fun " + name in (KT / "Whisper.kt").read_text()


def test_android_keeps_audio_private():
    rec = (KT / "Recorder.kt").read_text()
    mod = (KT / "SpeechModule.kt").read_text()
    live = (KT / "Live.kt").read_text()
    for src in (rec, mod, live, (KT / "Audio.kt").read_text(), (KT / "Whisper.kt").read_text()):
        assert "android.util.Log" not in src and "Log." not in src and "println" not in src
        assert "FileOutputStream" not in src and "writeBytes" not in src
    assert "pcm.fill(0)" in rec and "audio.fill(0f)" in mod
    # a kept recording (stop_async(keep_audio=True)) is zeroed by every way out, and after two minutes at most
    assert "kept?.fill(0f)" in mod and "KEEP_MS = 120_000L" in mod and "postDelayed(forgetKept, KEEP_MS)" in mod
    for method in ("private fun start(", "private fun cancel(", "private fun unload(", '"forgetAudio" ->'):
        body = mod[mod.index(method):][:900]
        assert "clearKept()" in body, method
    # live listening only ever uses the on-device recognizer (never the network one), and cancel drops it
    assert "createOnDeviceSpeechRecognizer" in live and "createSpeechRecognizer(" not in live
    assert "live?.let { l -> live = null; l.cancel() }" in mod and "said.setLength(0)" in live
    # an input method has no activity to show the permission dialog
    assert "if (!ctx.hasActivity)" in mod and "PxProcess.emit(STATE" in mod
