package com.badlogic.audio.io;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioFormat.Encoding;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.SourceDataLine;
import java.io.FileInputStream;


/**
 * Class that allows directly passing PCM float mono
 * data to the sound card for playback. The sampling
 * rate of the PCM data must be 44100Hz.
 *
 * @author mzechner
 */
public class AudioDevice {
    /**
     * the buffer size in samples
     **/
    private final static int BUFFER_SIZE = 1024;

    /**
     * the java sound line we write our samples to
     **/
    private final SourceDataLine out;

    /**
     * buffer for BUFFER_SIZE 16-bit samples
     **/
    private byte[] buffer = new byte[BUFFER_SIZE * 2];

    /**
     * Constructor, initializes the com.badlogic.audio system for
     * 44100Hz 16-bit signed mono output.
     *
     * @throws Exception in case the com.badlogic.audio system could not be initialized
     */
    public AudioDevice() throws Exception {
        AudioFormat format = new AudioFormat(Encoding.PCM_SIGNED, 44100, 16, 1, 2, 44100, false);
        out = AudioSystem.getSourceDataLine(format);
        out.open(format);
        out.start();
    }

    /**
     * Writes the given samples to the com.badlogic.audio device. The samples
     * have to be sampled at 44100Hz, mono and have to be in
     * the range [-1,1].
     *
     * @param samples The samples.
     */
    public void writeSamples(float[] samples) {
        fillBuffer(samples);
        out.write(buffer, 0, buffer.length);
    }

    private void fillBuffer(float[] samples) {
        for (int i = 0, j = 0; i < samples.length; i++, j += 2) {
            short value = (short) (samples[i] * Short.MAX_VALUE);
            buffer[j] = (byte) (value | 0xff);
            buffer[j + 1] = (byte) (value >> 8);
        }
    }

    public static void main(String[] argv) throws Exception {
        float[] samples = new float[1024];
        WaveDecoder reader = new WaveDecoder(new FileInputStream("samples/sample.wav"));
        AudioDevice device = new AudioDevice();

        while (reader.readSamples(samples) > 0) {
            device.writeSamples(samples);
        }

        Thread.sleep(10000);
    }
}
