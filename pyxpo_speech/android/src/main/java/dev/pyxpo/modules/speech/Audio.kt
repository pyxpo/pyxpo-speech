package dev.pyxpo.modules.speech

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** PCM helpers: whisper wants 16 kHz mono floats in [-1, 1]. */
internal object Audio {
    const val RATE = 16000

    /** The first `n` samples of 16-bit `pcm` at `rate` Hz, as 16 kHz floats. Downsampling averages each output
     *  sample's span of input (a box filter), which keeps the speech band clean enough for recognition. */
    fun resample(pcm: ShortArray, n: Int, rate: Int): FloatArray {
        if (rate == RATE) return FloatArray(n) { pcm[it] / 32768f }
        val out = FloatArray((n.toLong() * RATE / rate).toInt())
        val step = rate.toDouble() / RATE
        for (i in out.indices) {
            val a = (i * step).toInt()
            val b = minOf(n, maxOf(a + 1, ((i + 1) * step).toInt()))
            var sum = 0f
            for (j in a until b) sum += pcm[j]
            out[i] = sum / (b - a) / 32768f
        }
        return out
    }

    /** Reads a PCM WAV file (8/16/24/32-bit integer or 32-bit float, any rate and channel count). */
    fun readWav(file: File): FloatArray {
        val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        fun tag(at: Int) = String(ByteArray(4) { buf.get(at + it) }, Charsets.US_ASCII)
        if (buf.limit() < 12 || tag(0) != "RIFF" || tag(8) != "WAVE") throw IllegalArgumentException("not a WAV file")
        var at = 12
        var format = 0; var channels = 0; var rate = 0; var bits = 0
        var data = -1; var dataSize = 0
        while (at + 8 <= buf.limit()) {
            val id = tag(at)
            val len = buf.getInt(at + 4)
            if (len < 0) break
            val body = at + 8
            if (id == "fmt " && len >= 16) {
                format = buf.getShort(body).toInt() and 0xffff
                channels = buf.getShort(body + 2).toInt()
                rate = buf.getInt(body + 4)
                bits = buf.getShort(body + 14).toInt()
                if (format == 0xfffe && len >= 26) format = buf.getShort(body + 24).toInt() and 0xffff  // extensible
            } else if (id == "data") {
                data = body
                dataSize = minOf(len, buf.limit() - body)
                break
            }
            at = body + len + (len and 1)
        }
        val int = format == 1 && bits in intArrayOf(8, 16, 24, 32)
        val float = format == 3 && bits == 32
        if (data < 0 || channels < 1 || rate < 1000 || !(int || float))
            throw IllegalArgumentException("unsupported WAV: needs PCM (8/16/24/32-bit) or 32-bit float audio")
        val width = bits / 8
        val frames = dataSize / (width * channels)
        val mono = ShortArray(frames)
        for (f in 0 until frames) {
            var sum = 0f
            for (c in 0 until channels) {
                val p = data + (f * channels + c) * width
                sum += when {
                    float -> buf.getFloat(p)
                    bits == 8 -> ((buf.get(p).toInt() and 0xff) - 128) / 128f
                    bits == 16 -> buf.getShort(p) / 32768f
                    bits == 24 -> ((buf.get(p).toInt() and 0xff) or ((buf.get(p + 1).toInt() and 0xff) shl 8) or
                                   (buf.get(p + 2).toInt() shl 16)) / 8388608f
                    else -> buf.getInt(p) / 2147483648f
                }
            }
            mono[f] = (sum / channels * 32767f).coerceIn(-32768f, 32767f).toInt().toShort()
        }
        return resample(mono, frames, rate)
    }
}
