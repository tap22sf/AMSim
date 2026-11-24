"""Test signals, and running the simulation without audio devices."""

import numpy as np

from amsim.settings import SAMPLE_RATE, LaunchSettings, StationSettings
from amsim.simulation import Simulation

# Three stations that don't overlap, close by, in clear weather.
NEUTRAL = LaunchSettings(stations=tuple(StationSettings(carrier_hz=hz)
                                        for hz in (5500.0, 11750.0, 18000.0)))


def voice(seed, seconds=2.0):
    """Speech-like test signal: band-limited noise with a syllable-rate envelope."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    spec = np.fft.rfft(rng.standard_normal(n)) * ((f > 150) & (f < 2700))
    return np.fft.irfft(spec, n) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))


def make(loaded=(0,), seed=1):
    sim = Simulation(NEUTRAL, seed)
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
