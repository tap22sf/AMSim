"""Noise from the weather and from the receiver itself."""

import numpy as np

from amsim.filters import Smooth
from amsim.settings import MESSAGE_BW_HZ, SAMPLE_RATE

# Noise, in the receiver's 6 kHz passband, relative to a carrier 1 km away.
NOISE_FLOOR_DB = -65.0          # the receiver itself, always there
STORM_STATIC_DB = 35.0          # how far a full thunderstorm raises that
STORM_FLASHES_PER_S = 6.0       # lightning at full storm


class Atmosphere:
    """Everything the antenna picks up that isn't a station."""

    def __init__(self, rng):
        self.weather = 0.0          # 0 = clear … 1 = thunderstorm
        self._rng = rng
        self._level = Smooth(self._static_level(0.0))
        self._tail = np.zeros(0)    # lightning that continues into later blocks

    @staticmethod
    def _static_level(weather):
        """RMS of white noise whose share inside the 6 kHz passband hits the target."""
        in_band_db = NOISE_FLOOR_DB + STORM_STATIC_DB * weather
        in_band_power = 0.5 * 10 ** (in_band_db / 10)       # 0.5 = power of a unit carrier
        return np.sqrt(in_band_power * (SAMPLE_RATE / 2) / (2 * MESSAGE_BW_HZ))

    def render(self, frames):
        w = self.weather
        out = self._level.block(self._static_level(w), frames) * self._rng.standard_normal(frames)
        for _ in range(self._rng.poisson(STORM_FLASHES_PER_S * w ** 2 * frames / SAMPLE_RATE)):
            self._add_flash(int(self._rng.integers(frames)), w)
        n = min(frames, len(self._tail))
        out[:n] += self._tail[:n]
        self._tail = self._tail[n:]
        return out

    def _add_flash(self, at, w):
        """One flash = a few strokes, each a burst of noise that dies away."""
        rng = self._rng
        peak_db = -42 + 34 * w + 8 * rng.standard_normal()
        for _ in range(rng.integers(1, 5)):
            decay = rng.uniform(0.003, 0.03) * SAMPLE_RATE
            length = int(6 * decay)
            burst = (10 ** (min(peak_db, 6) / 20) * np.exp(-np.arange(length) / decay)
                     * rng.standard_normal(length))
            if at + length > len(self._tail):
                self._tail = np.pad(self._tail, (0, at + length - len(self._tail)))
            self._tail[at:at + length] += burst
            at += int(rng.uniform(0.03, 0.09) * SAMPLE_RATE)
            peak_db -= rng.uniform(0, 6)
