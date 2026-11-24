"""Tests of loading WAV files and turning them into messages."""

import os
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

from amsim.message import Looper, load_wav
from amsim.settings import MAX_WAV_S, SAMPLE_RATE
from tests.helpers import voice


class Wav(unittest.TestCase):
    def load(self, rate, data):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            wavfile.write(path, rate, data)
            return load_wav(path)
        finally:
            os.remove(path)

    def test_formats(self):
        for rate in (8000, 22050, 44100, 48000):
            t = np.arange(rate) / rate
            tone = np.sin(2 * np.pi * 1000 * t) * 0.5
            for data in ((tone * 32767).astype(np.int16), (tone * 127 + 128).astype(np.uint8),
                         tone.astype(np.float32),
                         np.stack([tone, tone], 1).astype(np.float32)):
                clip, truncated = self.load(rate, data)
                self.assertFalse(truncated)
                looper = Looper()
                self.assertTrue(looper.set_clip(clip))
                x = looper.sample
                self.assertAlmostEqual(len(x) / SAMPLE_RATE, 1.0, places=2)
                self.assertAlmostEqual(np.abs(clip).max(), 0.5, delta=0.02)
                self.assertAlmostEqual(np.abs(x).max(), 1.0, places=5)
                peak = np.argmax(np.abs(np.fft.rfft(x))) * SAMPLE_RATE / len(x)
                self.assertAlmostEqual(peak, 1000, delta=2)

    def test_long_file_is_trimmed(self):
        clip, truncated = self.load(8000, np.ones(8000 * 130, dtype=np.int16))
        self.assertTrue(truncated)
        self.assertEqual(len(clip), int(MAX_WAV_S * SAMPLE_RATE))

    def test_bad_samples_are_removed(self):
        data = np.full(1000, 0.5, dtype=np.float32)
        data[[10, 20, 30]] = np.nan, np.inf, -np.inf
        self.assertTrue(np.isfinite(self.load(44100, data)[0]).all())

    def test_silent_clip_is_refused_and_old_one_kept(self):
        looper = Looper()
        looper.set_clip(voice(0))
        before = looper.sample
        for clip in (np.zeros(1000), np.zeros(0)):
            self.assertFalse(looper.set_clip(clip))
            self.assertIs(looper.sample, before)
