package dev.pyxpo.modules.speech

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognitionSupport
import android.speech.RecognitionSupportCallback
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.concurrent.Executor

/**
 * Live speech to text with the phone's own on-device recognizer (Android 12+): words arrive while the person
 * speaks, in languages whisper writes poorly (Hindi in Devanagari). `createOnDeviceSpeechRecognizer` never uses
 * the network. The recognizer ends a session at a pause; this restarts it until [stop], so one recording can
 * hold several sentences. Main thread only. Nothing is logged or kept: the text goes to [onPartial] and [stop].
 */
class Live(
    private val context: Context,
    val language: String,
    private val maxSeconds: Double,
    private val onPartial: (String) -> Unit,
    private val onLevel: ((Float) -> Unit)?,
    private val onLimit: () -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())
    private var recognizer: SpeechRecognizer? = null
    private val said = StringBuilder()   // sentences of the sessions before this one
    private var stopping = false
    private var finished = false
    private var finish: ((String?, String?) -> Unit)? = null
    private val began = SystemClock.elapsedRealtime()
    private val limit = Runnable { if (!stopping) { onLimit(); stopListening() } }

    val audioMs get() = SystemClock.elapsedRealtime() - began

    /** Starts listening; `ready(error)` once the recognizer took the request (null) or refused it. */
    fun start(ready: (String?) -> Unit) {
        if (!available(context)) return ready("this phone has no on-device speech recognizer (Android 12+ with one installed)")
        main.postDelayed(limit, (maxSeconds * 1000).toLong())
        session { e ->
            if (e != null) { finished = true; stopping = true; main.removeCallbacks(limit) }
            ready(e)
        }
    }

    private fun session(ready: ((String?) -> Unit)?) {
        val r = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
        recognizer = r
        var answered = ready == null
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                if (!answered) { answered = true; ready?.invoke(null) }
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {
                onLevel?.invoke(((rmsdB + 2f) / 12f).coerceIn(0f, 1f))
            }
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(partialResults: Bundle?) {
                onPartial(joined(best(partialResults)))
            }
            override fun onEvent(eventType: Int, params: Bundle?) {}
            override fun onResults(results: Bundle?) {
                add(best(results))
                next()
            }
            override fun onError(error: Int) {
                val quiet = error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT
                if (!answered) {
                    answered = true
                    release()
                    return ready?.invoke(message(error)) ?: Unit
                }
                when {
                    quiet -> next()
                    said.isNotEmpty() -> end(said.toString(), null)   // keep what was heard before the failure
                    else -> end(null, message(error))
                }
            }
        })
        r.startListening(intent())
    }

    /** A session ended: the next one starts unless [stop] was asked. */
    private fun next() {
        release()
        if (stopping) end(said.toString(), null)
        else main.post { if (!stopping && !finished) session(null) }
    }

    private fun add(text: String) {
        if (text.isBlank()) return
        if (said.isNotEmpty()) said.append(' ')
        said.append(text.trim())
        onPartial(said.toString())
    }

    private fun joined(partial: String) = if (said.isEmpty()) partial else if (partial.isBlank()) said.toString() else "$said $partial"

    private fun best(b: Bundle?): String =
        b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.trim() ?: ""

    private fun intent() = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        // dictation: let people pause to think before the recognizer ends the session
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2500L)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 2000L)
        if (Build.VERSION.SDK_INT >= 33) putExtra(RecognizerIntent.EXTRA_ENABLE_FORMATTING,
                                                   RecognizerIntent.FORMATTING_OPTIMIZE_QUALITY)
    }

    /** Ends listening; `done(text, error)` gets every sentence heard (at once when it already ended). */
    fun stop(done: (String?, String?) -> Unit) {
        finish = done
        if (finished) deliver() else if (!stopping) stopListening()
    }

    private fun stopListening() {
        stopping = true
        main.removeCallbacks(limit)
        val r = recognizer
        if (r == null) return end(said.toString(), null)
        r.stopListening()
        // a recognizer that never answers the stop must not hang the call
        main.postDelayed({ end(said.toString(), null) }, 4000)
    }

    /** Stops without a result and forgets the text. */
    fun cancel() {
        stopping = true
        finished = true
        main.removeCallbacks(limit)
        recognizer?.cancel()
        release()
        said.setLength(0)
        finish = null
    }

    private var text: String? = null
    private var error: String? = null

    private fun end(t: String?, e: String?) {
        if (finished) return
        finished = true
        main.removeCallbacks(limit)
        release()
        text = t
        error = e
        // ended by itself (an error): the app hears `limit` and asks stop() for what was heard
        if (finish == null && !stopping) onLimit()
        stopping = true
        deliver()
    }

    private fun deliver() {
        val f = finish ?: return
        finish = null
        val t = text
        val e = error
        text = null
        error = null
        said.setLength(0)
        f(t, e)
    }

    private fun release() {
        recognizer?.destroy()
        recognizer = null
    }

    companion object {
        fun available(c: Context) = Build.VERSION.SDK_INT >= 31 && SpeechRecognizer.isOnDeviceRecognitionAvailable(c)

        /** Which languages the on-device recognizer has (Android 13+): installed, downloading, can download. */
        fun support(c: Context, language: String, done: (Map<String, Any?>?, String?) -> Unit) {
            if (!available(c)) return done(mapOf("available" to false), null)
            if (Build.VERSION.SDK_INT < 33) return done(mapOf("available" to true), null)
            val r = SpeechRecognizer.createOnDeviceSpeechRecognizer(c)
            val main = Handler(Looper.getMainLooper())
            var answered = false
            val answer = { m: Map<String, Any?>?, e: String? ->
                main.post { if (!answered) { answered = true; r.destroy(); done(m, e) } }
                Unit
            }
            val run = Executor { main.post(it) }
            r.checkRecognitionSupport(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
            }, run, object : RecognitionSupportCallback {
                override fun onSupportResult(s: RecognitionSupport) {
                    val installed = s.installedOnDeviceLanguages
                    val pending = s.pendingOnDeviceLanguages
                    val online = s.onlineLanguages
                    val canGet = s.supportedOnDeviceLanguages
                    answer(mapOf("available" to true, "installed" to installed, "pending" to pending,
                                 "downloadable" to canGet, "online" to online,
                                 "ready" to installed.any { same(it, language) }), null)
                }
                override fun onError(error: Int) { answer(null, message(error)) }
            })
            main.postDelayed({ answer(null, "the speech recognizer did not answer") }, 5000)
        }

        /** Asks the on-device recognizer to download a language (Android 13+); the phone shows its progress. */
        fun download(c: Context, language: String): String? {
            if (!available(c)) return "this phone has no on-device speech recognizer"
            if (Build.VERSION.SDK_INT < 33) return "downloading a language needs Android 13 or later"
            val r = SpeechRecognizer.createOnDeviceSpeechRecognizer(c)
            r.triggerModelDownload(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
            })
            Handler(Looper.getMainLooper()).postDelayed({ r.destroy() }, 2000)
            return null
        }

        private fun same(a: String, b: String) = a.equals(b, ignoreCase = true) || a.replace('_', '-').equals(b, true)

        fun message(error: Int): String = when (error) {
            SpeechRecognizer.ERROR_AUDIO -> "the microphone could not record"
            SpeechRecognizer.ERROR_CLIENT -> "the speech recognizer stopped"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "the microphone permission is not granted"
            SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT,
            SpeechRecognizer.ERROR_SERVER -> "the on-device recognizer is not ready for this language"
            SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "nothing heard"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "the speech recognizer is busy: try again"
            10 -> "the speech recognizer is busy: try again"   // ERROR_TOO_MANY_REQUESTS
            11 -> "the speech recognizer service stopped: try again"   // ERROR_SERVER_DISCONNECTED
            12, 13 -> "this language is not on the phone: download it first"   // ERROR_LANGUAGE_NOT_SUPPORTED/UNAVAILABLE
            else -> "speech recognition failed ($error)"
        }
    }
}
