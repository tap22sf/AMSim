"""Tests for dsp.py. Run with: python -m unittest"""

import os
import tempfile
import time
import unittest

import numpy as np
from scipy.io import wavfile

import dsp
from dsp import SAMPLE_RATE, Simulation, path_gain


def voice(seed, seconds=2.0):
    """Speech-like test signal: band-limited noise with a syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    spec = np.fft.rfft(rng.standard_normal(n)) * ((f > 150) & (f < 2700))
    return np.fft.irfft(spec, n) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))


def make(loaded=(0,), seed=1):
    sim = Simulation(seed)
    for i in loaded:
        sim.stations[i].looper.set_clip(voice(i))
        sim.stations[i].on_air = True
    return sim


def run(sim, seconds, block=512, names=("demod", "scaled_message")):
    """Simulate and return whole-run recordings of the named signals."""
    got = {n: [] for n in names}
    for _ in range(int(seconds * SAMPLE_RATE / block)):
        sim.process(block)
        for n in names:
            got[n].append(sim.capture.last(n, block))
    return {n: np.concatenate(v) for n, v in got.items()}


def quality_db(sim, seconds=3.0, settle=1.0):
    """Recovered audio vs the transmitted m·x(t): best-fit signal over the remainder."""
    got = run(sim, seconds)
    d = sim.delays["demod"]
    skip = int(settle * SAMPLE_RATE)
    y, x = got["demod"][skip:], got["scaled_message"][skip - d:len(got["scaled_message"]) - d]
    a = (x @ y) / (x @ x)
    return 10 * np.log10(a * a * (x @ x) / np.sum((y - a * x) ** 2))


class Clean(unittest.TestCase):
    def test_recovers_every_station(self):
        for i in range(3):
            sim = make((i,))
            sim.stations[0].on_air = i == 0
            sim.selected_index = i
            self.assertGreater(quality_db(sim), 20, f"station {i + 1}")

    def test_agc_restores_level_at_any_distance(self):
        for km in (1, 20, 100):
            sim = make()
            sim.stations[0].distance_km = km
            lp = run(sim, 3, names=("lowpassed",))["lowpassed"]
            self.assertAlmostEqual(lp[-SAMPLE_RATE:].mean(), 1.0, delta=0.12)


class Stations(unittest.TestCase):
    def test_other_stations_are_rejected(self):
        for i in range(3):
            sim = make((0, 1, 2))
            sim.selected_index = i
            self.assertGreater(quality_db(sim), 20, f"station {i + 1}")

    def test_strong_neighbour_bleeds_into_weak_station(self):
        sim = make((0, 1))
        sim.selected_index = 1
        sim.stations[1].distance_km = 100
        sim.stations[0].on_air = False
        run(sim, 5)
        alone = sim.snr_db
        sim.stations[0].on_air = True
        run(sim, 5)      # long enough for the switch-on transient to leave the average
        self.assertLess(sim.snr_db, alone - 6)
        sim.stations[0].distance_km = 100        # an equally weak one hardly matters
        run(sim, 8)
        self.assertGreater(sim.snr_db, alone - 1)

    def test_settings_are_independent(self):
        sim = make((0, 1))
        sim.stations[1].distance_km = 300
        sim.stations[1].modulation = 0.1
        self.assertGreater(quality_db(sim), 20)

    def test_overlapping_carriers_interfere(self):
        sim = make((0, 1))
        sim.stations[1].carrier.freq_hz = sim.stations[0].carrier.freq_hz + 1000
        self.assertLess(quality_db(sim), 5)

    def test_switching_settles_quickly(self):
        sim = make((0, 1))
        sim.stations[1].distance_km = 100
        run(sim, 1)
        sim.selected_index = 1
        lp = run(sim, 0.4, names=("lowpassed",))["lowpassed"]
        self.assertAlmostEqual(lp[-SAMPLE_RATE // 5:].mean(), 1.0, delta=0.3)


class Interference(unittest.TestCase):
    def test_distance_lowers_snr_as_modelled(self):
        snrs = []
        for km in (1, 10, 100, 300, 500):
            sim = make()
            sim.stations[0].distance_km = km
            run(sim, 2)
            snrs.append(sim.snr_db)
        self.assertEqual(snrs, sorted(snrs, reverse=True))
        # A bare carrier at 1 km is NOISE_FLOOR_DB above the noise; modulation adds a little.
        self.assertAlmostEqual(snrs[0], -dsp.NOISE_FLOOR_DB, delta=2.5)
        self.assertAlmostEqual(snrs[0] - snrs[2], -20 * np.log10(path_gain(100)), delta=1.5)
        self.assertLess(snrs[-1], 0)

    def test_weather_lowers_quality(self):
        qs = []
        for w in (0, 0.5, 1.0):
            sim = make()
            sim.stations[0].distance_km = 30
            sim.atmosphere.weather = w
            qs.append(quality_db(sim, 5))
        self.assertGreater(qs[0], qs[1] + 3)
        self.assertGreater(qs[1], qs[2] + 3)

    def test_lightning_is_impulsive_and_bounded(self):
        sim = make()
        sim.atmosphere.weather = 1.0
        got = run(sim, 10, names=("agc", "demod"))
        self.assertLessEqual(np.abs(got["agc"]).max(), dsp.AGC_CLIP)
        self.assertTrue(np.isfinite(got["demod"]).all())
        noise = np.concatenate([sim.atmosphere.render(512) for _ in range(900)])
        rms = np.sqrt((noise ** 2).mean())
        self.assertGreater(np.abs(noise).max() / rms, 8)     # Gaussian noise: ~5
        self.assertLess(len(sim.atmosphere._tail), 3 * SAMPLE_RATE)

    def test_skywave_fades_and_distorts(self):
        sim = make()
        clean = quality_db(sim, 3)
        sim.stations[0].skywave = 1.0
        tuned = run(sim, 20, names=("tuned",))["tuned"]
        seg = tuned[:len(tuned) // 4410 * 4410].reshape(-1, 4410)
        level = 20 * np.log10(np.sqrt((seg ** 2).mean(axis=1)))
        self.assertGreater(level.max() - level.min(), 6)
        self.assertLess(quality_db(sim, 6), clean - 6)


class Streaming(unittest.TestCase):
    def test_block_size_does_not_matter(self):
        outs = []
        for block in (64, 441, 4410):
            sim = make()
            sim.atmosphere.render = lambda frames: np.zeros(frames)   # only the signal path
            sim.stations[0]._gain.value = 1.0                         # skip the start-up ramp
            outs.append(run(sim, 1.4, block=block)["demod"][:44100])
        for other in outs[1:]:
            np.testing.assert_allclose(other, outs[0], atol=1e-9)

    def test_no_click_when_a_slider_moves(self):
        sim = make(())
        sim.stations[0].on_air = True             # bare carrier: demod should be ~0
        run(sim, 2)
        for name, value in (("distance_km", 1.5), ("modulation", 0.2), ("distance_km", 1.0)):
            setattr(sim.stations[0], name, value)
            y = run(sim, 0.2)["demod"]
            self.assertLess(np.abs(np.diff(y)).max(), 0.1, name)

    def test_keeps_up_with_real_time(self):
        sim = make((0, 1, 2))
        for station in sim.stations:
            station.skywave = 1.0
        sim.atmosphere.weather = 1.0
        for block in (441, 1024):
            times = []
            for _ in range(200):
                t = time.perf_counter()
                sim.process(block)
                times.append(time.perf_counter() - t)
            load = np.median(times) / (block / SAMPLE_RATE)
            print(f"\n  audio load, {block}-sample blocks: {load:.0%} median, "
                  f"{np.percentile(times, 99) / (block / SAMPLE_RATE):.0%} 99th pct")
            self.assertLess(load, 0.5)


class Modes(unittest.TestCase):
    def test_every_listen_mode(self):
        sim = make()
        for mode in dsp.LISTEN_MODES:
            sim.listen = mode
            out = np.zeros((512, 1), dtype=np.float32)
            sim.audio_out(out, 512, None, None)
            self.assertTrue(np.isfinite(out).all(), mode)

    def test_volume_is_per_station(self):
        sim = make((0, 1))
        sim.listen = "Demodulated"
        sim.stations[1].volume = 0.0
        out = np.zeros((512, 1))

        def peak(seconds=0.5):
            """Highest output, after any ramp has finished."""
            got = []
            for _ in range(int(seconds * SAMPLE_RATE / 512)):
                sim.audio_out(out, 512, None, None)
                got.append(out[:, 0].copy())
            return np.abs(np.concatenate(got)[2048:]).max()

        peak(2)
        normal = peak(2)
        self.assertGreater(normal, 0.02)
        sim.selected_index = 1
        self.assertEqual(peak(), 0.0)               # silenced station
        self.assertGreater(np.abs(sim.capture.last("demod", 4096)).max(), 0.02)
        sim.selected_index = 0
        self.assertGreater(peak(2), 0.5 * normal)   # the other one is unaffected
        sim.stations[0].volume = 0.5
        self.assertAlmostEqual(peak(2) / normal, 0.5, delta=0.05)

    def test_no_click_when_volume_moves(self):
        sim = make()
        sim.listen = "Low-passed"                    # about 1.0 throughout
        sim.stations[0].modulation = 0.0
        out = np.zeros((512, 1))
        for _ in range(100):
            sim.audio_out(out, 512, None, None)
        got = []
        for volume in (0.0, 2.0, 1.0):
            sim.stations[0].volume = volume
            sim.audio_out(out, 512, None, None)
            got.append(out[:, 0].copy())
        self.assertLess(np.abs(np.diff(np.concatenate(got))).max(), 0.003)

    def test_muted_while_any_station_records(self):
        sim = make((0, 1))
        sim.stations[1].looper.recording = True
        self.assertFalse(sim.process(512).any())


class Wav(unittest.TestCase):
    def load(self, rate, data):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            wavfile.write(path, rate, data)
            return dsp.load_wav(path)
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
                looper = dsp.Looper()
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
        self.assertEqual(len(clip), int(dsp.MAX_WAV_S * SAMPLE_RATE))

    def test_bad_samples_are_removed(self):
        data = np.full(1000, 0.5, dtype=np.float32)
        data[[10, 20, 30]] = np.nan, np.inf, -np.inf
        self.assertTrue(np.isfinite(self.load(44100, data)[0]).all())

    def test_silent_clip_is_refused_and_old_one_kept(self):
        looper = dsp.Looper()
        looper.set_clip(voice(0))
        before = looper.sample
        for clip in (np.zeros(1000), np.zeros(0)):
            self.assertFalse(looper.set_clip(clip))
            self.assertIs(looper.sample, before)


if __name__ == "__main__":
    unittest.main()
