package dev.pyxpo.modules.speech

import android.Manifest
import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import dev.pyxpo.moduleapi.PxModule
import dev.pyxpo.moduleapi.PxModuleContext
import dev.pyxpo.moduleapi.PxProcess
import dev.pyxpo.moduleapi.PxReply
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

/**
 * Offline speech to text: the microphone recorded into memory, then whisper.cpp on the phone's CPU.
 * Audio and text never leave the app and are not logged. Works headless too (an input method calls it).
 * Whisper runs on one worker thread; recording state lives on the main thread.
 */
class SpeechModule : PxModule() {
    override val name = "Speech"

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "pyxpo-speech") }
    private val main = Handler(Looper.getMainLooper())
    @Volatile private var model: Whisper? = null
    @Volatile private var modelPath: String? = null
    private var recorder: Recorder? = null
    private var busy = 0

    override fun call(ctx: PxModuleContext, method: String, args: JSONArray, reply: PxReply) {
        val c = ctx.context
        try {
            when (method) {
                "status" -> reply(status(), null)
                "getPermission" -> reply(permission(c), null)
                "requestPermission" -> requestPermission(ctx, reply)
                "memory" -> reply(memory(c), null)
                "load" -> load(c, args.getString(0), args.optJSONObject(1) ?: JSONObject(), reply)
                "unload" -> unload(reply)
                "start" -> start(c, args.optJSONObject(0) ?: JSONObject(), reply)
                "stop" -> stop(args.optJSONObject(0) ?: JSONObject(), reply)
                "cancel" -> { cancel(); reply(null, null) }
                "transcribeFile" -> transcribeFile(args.getString(0), args.optJSONObject(1) ?: JSONObject(), reply)
                else -> reply(null, "Speech has no method $method")
            }
        } catch (e: Exception) {
            reply(null, e.message ?: e.toString())
        }
    }

    private fun status(): Map<String, Any?> {
        val m = model
        return mapOf("supported" to true, "loaded" to (m != null), "model" to modelPath,
                     "multilingual" to (m?.multilingual ?: false), "recording" to (recorder != null),
                     "transcribing" to (busy > 0))
    }

    // permission (the same shape as pyxpo-camera's)

    private fun prefs(c: Context) = c.getSharedPreferences("pyxpo.permissions", Context.MODE_PRIVATE)

    private fun granted(c: Context) = c.checkSelfPermission(MIC) == PackageManager.PERMISSION_GRANTED

    private fun permission(c: Context, activity: android.app.Activity? = null): Map<String, Any> = when {
        granted(c) -> mapOf("status" to "granted", "canAskAgain" to true)
        !prefs(c).getBoolean(MIC, false) -> mapOf("status" to "undetermined", "canAskAgain" to true)
        else -> mapOf("status" to "denied", "canAskAgain" to (activity?.shouldShowRequestPermissionRationale(MIC) ?: false))
    }

    private fun requestPermission(ctx: PxModuleContext, reply: PxReply) {
        val c = ctx.context
        if (!ctx.hasActivity) {
            if (granted(c)) reply(permission(c), null)
            else reply(null, "asking for the microphone needs the app on screen: open the app and ask there")
            return
        }
        val now = permission(c, ctx.activity)
        if (now["status"] == "granted" || now["canAskAgain"] != true) { reply(now, null); return }
        ctx.requestPermissions(arrayOf(MIC)) { _ ->
            prefs(c).edit().putBoolean(MIC, true).apply()
            reply(permission(c, ctx.activity), null)
        }
    }

    // models

    private fun memory(c: Context): Map<String, Any> {
        val info = ActivityManager.MemoryInfo()
        (c.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
        return mapOf("total" to info.totalMem, "available" to info.availMem, "threshold" to info.threshold,
                     "low" to info.lowMemory)
    }

    private fun load(c: Context, path: String, o: JSONObject, reply: PxReply) {
        val file = File(path)
        if (!file.isFile) return reply(null, "no model file at $path")
        if (o.optBoolean("checkMemory", true)) {
            val m = memory(c)
            // the weights plus whisper's work buffers; loading past the low-memory line gets the app killed
            val need = file.length() + file.length() / 5 + 64L * MB
            val free = (m["available"] as Long) - (m["threshold"] as Long)
            if (need > free && modelPath != file.absolutePath)
                return reply(null, "not enough free memory for this model: it needs about ${need / MB} MB, " +
                                   "${maxOf(0L, free) / MB} MB is free. Close other apps or pick a smaller model")
        }
        val flash = o.optBoolean("flashAttention", true)
        worker.execute {
            try {
                if (modelPath == file.absolutePath && model != null) return@execute reply(statusOnMain(), null)
                model?.free(); model = null; modelPath = null
                val t = SystemClock.elapsedRealtime()
                model = Whisper.load(file.absolutePath, flash)
                modelPath = file.absolutePath
                reply(statusOnMain() + mapOf("loadMs" to (SystemClock.elapsedRealtime() - t),
                                             "system" to Whisper.systemInfo()), null)
            } catch (e: Throwable) {
                reply(null, "could not load ${file.name}: ${e.message ?: e}")
            }
        }
    }

    private fun statusOnMain(): Map<String, Any?> {
        var s: Map<String, Any?> = emptyMap()
        val done = java.util.concurrent.CountDownLatch(1)
        main.post { s = status(); done.countDown() }
        done.await()
        return s
    }

    private fun unload(reply: PxReply) {
        model?.abort()
        worker.execute {
            model?.free(); model = null; modelPath = null
            reply(null, null)
        }
    }

    // recording

    private fun start(c: Context, o: JSONObject, reply: PxReply) {
        if (recorder != null) return reply(null, "already recording")
        if (!granted(c)) return reply(null, "the microphone permission is not granted: call request_permission_async first")
        val max = o.optDouble("maxSeconds", 30.0).coerceIn(1.0, 120.0)
        val levels = o.optBoolean("levels", false)
        val r = Recorder(max, if (levels) { l -> PxProcess.emit(LEVEL, mapOf("level" to l.toDouble())) } else null) {
            main.post { if (recorder != null) state("limit") }
        }
        r.start()
        recorder = r
        state("recording")
        reply(null, null)
    }

    private fun stop(o: JSONObject, reply: PxReply) {
        val r = recorder ?: return reply(null, "not recording: call start_async first")
        recorder = null
        busy++
        state("transcribing")
        worker.execute {
            val audio = r.stop()
            transcribe(audio, o, reply)
        }
    }

    private fun cancel() {
        recorder?.let { r -> recorder = null; worker.execute { r.stop().fill(0f) } }
        model?.abort()
        if (busy == 0) state("idle")
    }

    private fun transcribeFile(path: String, o: JSONObject, reply: PxReply) {
        val f = File(path)
        if (!f.isFile) return reply(null, "no audio file at $path")
        busy++
        state("transcribing")
        worker.execute {
            val audio = try { Audio.readWav(f) } catch (e: Exception) {
                done(); return@execute reply(null, e.message ?: e.toString())
            }
            transcribe(audio, o, reply)
        }
    }

    /** Worker thread. Clears `audio` when it is done with it. */
    private fun transcribe(audio: FloatArray, o: JSONObject, reply: PxReply) {
        try {
            val m = model ?: return reply(null, "no speech model loaded: call load_model_async first")
            val language = o.optString("language", "auto").ifEmpty { "auto" }
            if (language != "auto" && !Whisper.knowsLanguage(language)) return reply(null, "whisper has no language '$language'")
            if (language != "auto" && language != "en" && !m.multilingual)
                return reply(null, "this model knows English only: load a multilingual model for '$language'")
            val audioMs = audio.size * 1000L / Audio.RATE
            if (audio.size < Audio.RATE / 4) {   // under 0.25 s: nothing was said
                return reply(result("", language, 1f, emptyList(), audioMs, 0), null)
            }
            // whisper wants at least a second; pad short taps with silence
            val input = if (audio.size >= Audio.RATE) audio else audio.copyOf(Audio.RATE + Audio.RATE / 10)
            val threads = o.optInt("threads", 0).takeIf { it > 0 } ?: defaultThreads()
            val t = SystemClock.elapsedRealtime()
            val r = m.transcribe(input, language, o.optBoolean("translate", false),
                                 o.optString("prompt", "").ifEmpty { null }, threads, o.optInt("beamSize", 0))
            input.fill(0f)
            val ms = SystemClock.elapsedRealtime() - t
            if (r == null) reply(null, "cancelled")
            else reply(result(r.segments.joinToString("").trim().replace(Regex("\\s+"), " "), r.language, r.noSpeech,
                              r.segments.map { it.trim() }.filter { it.isNotEmpty() }, audioMs, ms), null)
        } catch (e: Throwable) {
            reply(null, e.message ?: e.toString())
        } finally {
            audio.fill(0f)
            done()
        }
    }

    private fun done() = main.post { busy--; if (busy == 0 && recorder == null) state("idle") }

    private fun result(text: String, language: String, noSpeech: Float, segments: List<String>, audioMs: Long,
                       ms: Long) = mapOf("text" to text, "language" to language, "noSpeech" to noSpeech.toDouble(),
                                         "segments" to segments, "audioMs" to audioMs, "decodeMs" to ms)

    private fun state(s: String) = PxProcess.emit(STATE, mapOf("state" to s))

    private fun defaultThreads() = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)

    companion object {
        private const val MIC = Manifest.permission.RECORD_AUDIO
        private const val MB = 1024L * 1024L
        const val STATE = "Speech.state"
        const val LEVEL = "Speech.level"
    }
}
