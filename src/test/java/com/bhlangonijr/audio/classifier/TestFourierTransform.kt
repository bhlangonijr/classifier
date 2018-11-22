package com.bhlangonijr.audio.classifier

import com.badlogic.audio.analysis.FFT
import com.badlogic.audio.io.AudioDevice
import com.badlogic.audio.io.WaveDecoder
import com.badlogic.audio.visualization.Plot
import org.junit.Test
import java.awt.Color
import java.io.FileInputStream
import java.io.InputStream

class TestFourierTransform {

    @Test
    fun testAudioTransformation() {

        val device = AudioDevice()
        val decoder = WaveDecoder(FileInputStream("src/test/resources/yes2.wav") as InputStream?)
        val samples = FloatArray(1024)

        while (decoder.readSamples(samples) > 0) {
            device.writeSamples(samples)
        }

        val fft = FFT(1024, 44100f)
        fft.forward(samples)

        val plot = Plot("Spectrum", 512, 512)
        plot.plot(samples, 1f, Color.red)

        println("${fft.spectrum.size}")
        Thread.sleep(2000)
    }


}
