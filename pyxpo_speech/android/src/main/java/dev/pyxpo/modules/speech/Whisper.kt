package dev.pyxpo.modules.speech

/** A loaded whisper.cpp model (libpyxpo_speech.so). Not thread-safe: [SpeechModule] runs it on one thread. */
internal class Whisper private constructor(private var handle: Long) {
    private val lock = Any()

    class Result(val language: String, val noSpeech: Float, val segments: List<String>)

    val multilingual: Boolean get() = nativeIsMultilingual(handle)

    /** 16 kHz mono samples in [-1, 1]. Null when [abort] stopped it. */
    fun transcribe(samples: FloatArray, language: String, translate: Boolean, prompt: String?, threads: Int,
                   beam: Int): Result? {
        val raw = nativeTranscribe(handle, samples, language, translate, prompt, threads, beam) ?: return null
        // language \0 no-speech \0 segment \0 segment ...; a cut-off character decodes as U+FFFD
        val parts = String(raw, Charsets.UTF_8).split('\u0000')
        return Result(parts[0], parts.getOrNull(1)?.toFloatOrNull() ?: 0f, parts.drop(2))
    }

    /** Stops a running [transcribe] soon. Any thread. */
    fun abort() = synchronized(lock) { if (handle != 0L) nativeAbort(handle) }

    fun free() {
        val h = synchronized(lock) { handle.also { handle = 0L } }
        if (h != 0L) nativeFree(h)
    }

    companion object {
        init { System.loadLibrary("pyxpo_speech") }

        /** Throws when the file is not a whisper model. */
        fun load(path: String, flashAttn: Boolean) = Whisper(nativeInit(path, flashAttn))

        fun knowsLanguage(code: String) = nativeKnowsLanguage(code)

        fun systemInfo(): String = nativeSystemInfo()

        @JvmStatic private external fun nativeInit(path: String, flashAttn: Boolean): Long
        @JvmStatic private external fun nativeFree(handle: Long)
        @JvmStatic private external fun nativeAbort(handle: Long)
        @JvmStatic private external fun nativeIsMultilingual(handle: Long): Boolean
        @JvmStatic private external fun nativeKnowsLanguage(code: String): Boolean
        @JvmStatic private external fun nativeSystemInfo(): String
        @JvmStatic private external fun nativeTranscribe(handle: Long, samples: FloatArray, language: String,
                                                         translate: Boolean, prompt: String?, threads: Int,
                                                         beam: Int): ByteArray?
    }
}
