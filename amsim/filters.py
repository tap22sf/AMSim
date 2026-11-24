"""Building blocks for processing audio block by block without clicks at the joins."""

import numpy as np
from scipy import signal as sps


def kaiser_lowpass(pass_hz, stop_hz, stop_db, rate):
    """Linear-phase low-pass. Odd length, so its delay is a whole number of samples."""
    taps, beta = sps.kaiserord(stop_db, (stop_hz - pass_hz) / (rate / 2))
    return sps.firwin(taps | 1, (pass_hz + stop_hz) / 2, window=("kaiser", beta), fs=rate)


class Smooth:
    """Turns a control value into a per-sample ramp, so moving a slider doesn't click."""

    def __init__(self, value):
        self.value = value

    def block(self, target, frames):
        start, self.value = self.value, target
        if start == target:
            return np.full(frames, float(target))
        return start + (target - start) * (np.arange(1, frames + 1) / frames)


class StreamFilter:
    """An SOS filter that keeps its state between blocks (for streaming audio)."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2))

    def __call__(self, block):
        out, self.zi = sps.sosfilt(self.sos, block, zi=self.zi)
        return out
