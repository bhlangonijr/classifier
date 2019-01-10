package com.bhlangonijr.audio.classifier

import com.badlogic.audio.analysis.FFT
import com.badlogic.audio.io.WaveDecoder
import com.github.bhlangonijr.chesslib.ml.DataSet
import com.github.bhlangonijr.chesslib.ml.FeatureSet
import com.github.bhlangonijr.chesslib.ml.NaiveBayes
import junit.framework.TestCase.assertEquals
import org.junit.Test
import java.io.FileInputStream
import java.io.InputStream

class TestClassifyAudio {

    @Test
    fun testAudioClassification() {

        val featureNames = mutableMapOf<String, Int>()
        for (i in 1..30) {
            featureNames["$i"] = i
        }

        // audio samples for training: 0.0 = YES 1.0 = NO
        val yes2 = audioToFs(0, "src/test/resources/yes2.wav", featureNames, 0.0)
        val yes3 = audioToFs(1, "src/test/resources/yes3.wav", featureNames, 0.0)
        val yes4 = audioToFs(2, "src/test/resources/yes4.wav", featureNames, 0.0)
        val no2 = audioToFs(3, "src/test/resources/no2.wav", featureNames, 1.0)
        val no3 = audioToFs(4, "src/test/resources/no3.wav", featureNames, 1.0)
        val no4 = audioToFs(5, "src/test/resources/no4.wav", featureNames, 1.0)


        val data = DataSet(arrayListOf(yes2, yes3, yes4, no2, no3, no4), featureNames.keys.toList())
        //println(data)

        val nb = NaiveBayes()

        // unknown samples
        val yes1 = audioToFs(6, "src/test/resources/yes1.wav", featureNames, -1.0)
        val no1 = audioToFs(7, "src/test/resources/no1.wav", featureNames, -1.0)
        val no5 = audioToFs(7, "src/test/resources/no5.wav", featureNames, -1.0)
        val no6 = audioToFs(7, "src/test/resources/no6.wav", featureNames, -1.0)
        val no7 = audioToFs(7, "src/test/resources/no7.wav", featureNames, -1.0)
        val no8 = audioToFs(7, "src/test/resources/no8.wav", featureNames, -1.0)
        val yes5 = audioToFs(8, "src/test/resources/yes5.wav", featureNames, -1.0)
        val yes6 = audioToFs(9, "src/test/resources/yes6.wav", featureNames, -1.0)
        val yes7 = audioToFs(10, "src/test/resources/yes7.wav", featureNames, -1.0)
        val yes8 = audioToFs(10, "src/test/resources/yes8.wav", featureNames, -1.0)

        val stats = nb.train(data)

        //println(stats)

        val classification1 = nb.classify(yes1, stats)
        val classification2 = nb.classify(no1, stats)
        val classification5 = nb.classify(no5, stats)
        val classification6 = nb.classify(yes5, stats)
        val classification7 = nb.classify(yes6, stats)
        val classification8 = nb.classify(no6, stats)
        val classification9 = nb.classify(no7, stats)
        val classification10 = nb.classify(yes7, stats)
        val classification11 = nb.classify(no8, stats)
        val classification12 = nb.classify(yes8, stats)

        println(classification1.predictions)
        println(classification1.predict())

        println(classification2.predictions)
        println(classification2.predict())

        println(classification5.predictions)
        println(classification5.predict())

        println(classification6.predictions)
        println(classification6.predict())

        println(classification7.predictions)
        println(classification7.predict())

        println(classification8.predictions)
        println(classification8.predict())

        println(classification9.predictions)
        println(classification9.predict())

        println(classification10.predictions)
        println(classification10.predict())

        println(classification11.predictions)
        println(classification11.predict())

        println(classification12.predictions)
        println(classification12.predict())

        assertEquals(0.0, classification1.predict())
        assertEquals(1.0, classification2.predict())
        assertEquals(1.0, classification5.predict())
        assertEquals(0.0, classification6.predict())
        assertEquals(0.0, classification7.predict())
        assertEquals(1.0, classification8.predict())
        assertEquals(1.0, classification9.predict())
        assertEquals(0.0, classification10.predict())
        assertEquals(1.0, classification11.predict())
        assertEquals(0.0, classification12.predict())


    }

    private val BUFFER_SIZE = 8192

    private fun audioToFs(id: Int, name: String, featureNames: Map<String, Int>, target: Double): FeatureSet {

        val decoder = WaveDecoder(FileInputStream(name) as InputStream?)
        val samples = FloatArray(BUFFER_SIZE)
        while (decoder.readSamples(samples) > 0) { }
        val fft = FFT(BUFFER_SIZE, 44100f)
        fft.forward(samples)
        val values = listOf(target) + fft.spectrum.map { it.toDouble() }.toList()
        return FeatureSet(id, values, featureNames)
    }
}
