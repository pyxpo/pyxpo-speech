package dev.pyxpo.modules.speech

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Records the microphone into memory as 16-bit mono PCM: no file is written. [stop] hands the audio over as
 * 16 kHz floats and drops the recorder's copy. `onLevel` gets 0..1 about ten times a second, `onLimit` runs
 * once when `maxSeconds` is reached (recording stops then); both on the recorder thread.
 */
internal class Recorder(private val maxSeconds: Double, private val onLevel: ((Float) -> Unit)?,
                        private val onLimit: () -> Unit) {
    private var record: AudioRecord? = null
    private var thread: Thread? = null
    @Volatile private var running = false
    private var rate = 16000
    private var pcm = ShortArray(16000 * 4)
    private var size = 0
    private val lock = Any()

    /** Throws when the microphone cannot be opened (no permission, or no rate the device records at). */
    @SuppressLint("MissingPermission")  // SpeechModule checks RECORD_AUDIO first
    fun start() {
        var r: AudioRecord? = null
        for (hz in intArrayOf(16000, 48000, 44100)) {
            val min = AudioRecord.getMinBufferSize(hz, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            if (min <= 0) continue
            val a = try {
                AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, hz, AudioFormat.CHANNEL_IN_MONO,
                            AudioFormat.ENCODING_PCM_16BIT, maxOf(min, hz / 5 * 2))
            } catch (e: Exception) { null }
            if (a != null && a.state == AudioRecord.STATE_INITIALIZED) { r = a; rate = hz; break }
            a?.release()
        }
        val rec = r ?: throw IllegalStateException("the microphone could not be opened")
        rec.startRecording()
        if (rec.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
            rec.release()
            throw IllegalStateException("the microphone is in use by another app")
        }
        record = rec
        running = true
        thread = Thread({ loop(rec) }, "pyxpo-speech-mic").apply { start() }
    }

    private fun loop(rec: AudioRecord) {
        val chunk = ShortArray(rate / 10)
        val limit = (maxSeconds * rate).toLong()
        while (running) {
            val n = rec.read(chunk, 0, chunk.size)
            if (n <= 0) { if (n < 0) break else continue }
            val full: Boolean
            synchronized(lock) {
                val take = minOf(n.toLong(), limit - size).toInt()
                if (size + take > pcm.size) pcm = pcm.copyOf(maxOf(pcm.size * 2, size + take))
                System.arraycopy(chunk, 0, pcm, size, take)
                size += take
                full = size >= limit
            }
            onLevel?.invoke(level(chunk, n))
            if (full) { running = false; onLimit(); break }
        }
        chunk.fill(0)
    }

    /** Ends recording and returns the audio as 16 kHz mono floats. */
    fun stop(): FloatArray {
        running = false
        record?.let { r -> try { r.stop() } catch (_: IllegalStateException) {} }
        thread?.join(1000)
        record?.release()
        record = null
        thread = null
        synchronized(lock) {
            val out = Audio.resample(pcm, size, rate)
            pcm.fill(0)
            size = 0
            return out
        }
    }

    private fun level(s: ShortArray, n: Int): Float {
        var sum = 0.0
        for (i in 0 until n) { val v = s[i] / 32768.0; sum += v * v }
        val db = 20 * log10(sqrt(sum / n) + 1e-9)       // -60 dB (quiet room) .. 0 dB (clipping)
        return ((db + 60) / 60).coerceIn(0.0, 1.0).toFloat()
    }
}
