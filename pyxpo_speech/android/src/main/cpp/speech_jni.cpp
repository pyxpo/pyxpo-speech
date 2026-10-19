// JNI over whisper.cpp for dev.pyxpo.modules.speech.Whisper. Nothing is logged: what people say stays private.
#include <jni.h>

#include <atomic>
#include <sstream>
#include <string>

#include "ggml-backend.h"
#include "whisper.h"

namespace {

struct Model {
    whisper_context *ctx;
    std::atomic<bool> abort{false};
};

void quiet(enum ggml_log_level, const char *, void *) {}

// The CPU backend in use: the first of PYXPO_CPU_BACKENDS (fastest first) that loads and whose instructions
// this CPU has (ggml asks each library). Empty when none loaded.
std::string cpu_backend;

void load_cpu_backend() {
    std::stringstream names(PYXPO_CPU_BACKENDS);
    std::string name;
    while (std::getline(names, name, ',')) {
        // a bare name: the linker searches the app's native libraries, also when they stay inside the APK
        if (ggml_backend_load(name.c_str()) != nullptr) {
            cpu_backend = name;
            return;
        }
    }
}

bool aborted(void *user) { return static_cast<Model *>(user)->abort.load(); }

std::string utf8(JNIEnv *env, jstring s) {
    if (s == nullptr) return "";
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

void fail(JNIEnv *env, const char *message) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), message);
}

}  // namespace

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *, void *) {
    whisper_log_set(quiet, nullptr);  // also silences ggml
    load_cpu_backend();
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL Java_dev_pyxpo_modules_speech_Whisper_nativeInit(JNIEnv *env, jclass, jstring path,
                                                                          jboolean flashAttn) {
    if (cpu_backend.empty()) {
        fail(env, "speech cannot run on this phone's CPU");
        return 0;
    }
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;  // CPU only: no GPU backend is built in
    cp.flash_attn = flashAttn;
    std::string p = utf8(env, path);
    whisper_context *ctx = whisper_init_from_file_with_params(p.c_str(), cp);
    if (ctx == nullptr) {
        fail(env, "not a whisper model, or it is damaged");
        return 0;
    }
    auto *m = new Model();
    m->ctx = ctx;
    return reinterpret_cast<jlong>(m);
}

JNIEXPORT void JNICALL Java_dev_pyxpo_modules_speech_Whisper_nativeFree(JNIEnv *, jclass, jlong h) {
    auto *m = reinterpret_cast<Model *>(h);
    if (m == nullptr) return;
    whisper_free(m->ctx);
    delete m;
}

JNIEXPORT void JNICALL Java_dev_pyxpo_modules_speech_Whisper_nativeAbort(JNIEnv *, jclass, jlong h) {
    auto *m = reinterpret_cast<Model *>(h);
    if (m != nullptr) m->abort.store(true);
}

JNIEXPORT jboolean JNICALL Java_dev_pyxpo_modules_speech_Whisper_nativeIsMultilingual(JNIEnv *, jclass, jlong h) {
    return whisper_is_multilingual(reinterpret_cast<Model *>(h)->ctx) != 0;
}

JNIEXPORT jboolean JNICALL Java_dev_pyxpo_modules_speech_Whisper_nativeKnowsLanguage(JNIEnv *env, jclass,
                                                                                    jstring code) {
    std::string c = utf8(env, code);
    return whisper_lang_id(c.c_str()) >= 0;
}

JNIEXPORT jstring JNICALL Java_dev_pyxpo_modules_speech_Whisper_nativeSystemInfo(JNIEnv *env, jclass) {
    std::string info = whisper_print_system_info();
    info += "| BACKEND = " + (cpu_backend.empty() ? std::string("none") : cpu_backend);
    return env->NewStringUTF(info.c_str());
}

// Returns `language \0 no_speech \0 segment \0 segment ...` as UTF-8 bytes (whisper's text can split a
// character across tokens, which NewStringUTF must not see), or null when aborted.
JNIEXPORT jbyteArray JNICALL Java_dev_pyxpo_modules_speech_Whisper_nativeTranscribe(
        JNIEnv *env, jclass, jlong h, jfloatArray samples, jstring language, jboolean translate, jstring prompt,
        jint threads, jint beam, jint audio_ctx) {
    auto *m = reinterpret_cast<Model *>(h);
    m->abort.store(false);
    std::string lang = utf8(env, language), initial = utf8(env, prompt);

    whisper_full_params p = whisper_full_default_params(beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH
                                                                 : WHISPER_SAMPLING_GREEDY);
    p.n_threads = threads;
    p.translate = translate;
    p.language = lang.c_str();  // "auto" detects it
    p.detect_language = false;
    p.no_context = true;
    p.no_timestamps = true;
    p.suppress_nst = true;
    p.print_special = false;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    if (!initial.empty()) p.initial_prompt = initial.c_str();
    if (beam > 1) p.beam_search.beam_size = beam;
    jsize n = env->GetArrayLength(samples);
    // whisper encodes 30 s (1500 steps of 20 ms) whatever the length; -1 encodes the clip plus 1.28 s of margin
    if (audio_ctx < 0) audio_ctx = n / 320 + 64;
    if (audio_ctx > 0 && audio_ctx < 1500) p.audio_ctx = audio_ctx;
    p.abort_callback = aborted;
    p.abort_callback_user_data = m;

    jfloat *pcm = env->GetFloatArrayElements(samples, nullptr);
    int rc = whisper_full(m->ctx, p, pcm, n);
    env->ReleaseFloatArrayElements(samples, pcm, JNI_ABORT);
    if (m->abort.load()) return nullptr;
    if (rc != 0) {
        fail(env, "whisper could not transcribe this audio");
        return nullptr;
    }

    std::string out;
    int id = whisper_full_lang_id(m->ctx);
    const char *code = id >= 0 ? whisper_lang_str(id) : nullptr;
    out += code ? code : lang;
    int segments = whisper_full_n_segments(m->ctx);
    float silent = 1.0f;  // the least no-speech probability over the segments
    for (int i = 0; i < segments; i++) {
        float ns = whisper_full_get_segment_no_speech_prob(m->ctx, i);
        if (ns < silent) silent = ns;
    }
    out.push_back('\0');
    out += std::to_string(segments ? silent : 1.0f);
    for (int i = 0; i < segments; i++) {
        out.push_back('\0');
        const char *t = whisper_full_get_segment_text(m->ctx, i);
        if (t) out += t;
    }
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(out.size()));
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(out.size()), reinterpret_cast<const jbyte *>(out.data()));
    return bytes;
}

}  // extern "C"
