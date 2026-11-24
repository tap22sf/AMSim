"""The receiver's stages: tuner -> AGC -> envelope detector.

Everything runs at SAMPLE_RATE except the detector's rectifier and low-pass, which run
OVERSAMPLE times faster (see EnvelopeDetector).
"""

import numpy as np
from scipy import signal as sps

from amsim.filters import StreamFilter, kaiser_lowpass
from amsim.settings import DETECTOR_BW_HZ, MAX_CARRIER_HZ, SAMPLE_RATE

TUNER_PASS_HZ = 2900.0          # half-width of the tuner's passband …
TUNER_STOP_HZ = 3200.0          # … and where it has fully cut off (≈ -70 dB)
TUNER_LEAK_DB = -50.0           # a real tuner isn't perfect: this much of everything
                                # else on the air gets through anyway
AGC_SECONDS = 0.3
AGC_MAX_GAIN = 2000.0           # 66 dB
AGC_CLIP = 3.0                  # the IF amplifier saturates at 3 × the carrier, so
                                # lightning can't blast the speakers
OVERSAMPLE = 8          # the detector runs at 8 × 44.1 = 352.8 kHz so the
                        # rectifier's carrier harmonics (2fc, 4fc, 6fc …) don't
                        # alias back into the audio band. A real diode is analog and
                        # has no such problem; this makes the simulation behave like it.


class Tuner:
    """Linear-phase band-pass around the selected carrier; rejects other stations.

    Handles several signals at once (one per row). Retuning cross-fades from
    the old filter to the new one over a block.
    """

    def __init__(self, freq_hz, rows):
        self._lowpass = kaiser_lowpass(TUNER_PASS_HZ, TUNER_STOP_HZ, 70, SAMPLE_RATE)
        self.delay = len(self._lowpass) // 2
        self._history = np.zeros((rows, len(self._lowpass) - 1))
        self._freq = freq_hz
        self._taps = self._design(freq_hz)

    def _design(self, freq_hz):
        n = np.arange(len(self._lowpass)) - self.delay
        taps = self._lowpass * 2 * np.cos(2 * np.pi * freq_hz / SAMPLE_RATE * n)
        taps[self.delay] += 10 ** (TUNER_LEAK_DB / 20)
        return taps

    def process(self, block, freq_hz):
        x = np.concatenate([self._history, block], axis=1)
        frames = block.shape[1]
        self._history = x[:, frames:]
        out = self._filter(x)
        if freq_hz != self._freq:
            self._freq = freq_hz
            self._taps = self._design(freq_hz)
            out = out + (self._filter(x) - out) * (np.arange(1, frames + 1) / frames)
        return out

    def _filter(self, x):
        # Not mode="valid": with blocks shorter than the filter that is ambiguous
        # about which input is the longer one.
        full = sps.fftconvolve(x, self._taps[None, :], axes=1)
        return full[:, len(self._taps) - 1:x.shape[1]]


class AGC:
    """Automatic gain control: holds the average carrier level at 1.

    This is why a distant station on a real radio sounds noisy rather than quiet.
    """

    def __init__(self):
        self._level = 0.0
        self.gain = 1.0
        self.jump()

    @property
    def gain_db(self):
        """Current gain, to the nearest whole dB (an int, so never "-0")."""
        return round(20 * float(np.log10(self.gain)))

    def jump(self):
        """Adapt quickly for a moment (after switching stations)."""
        self._fast_left = int(0.08 * SAMPLE_RATE)

    def process(self, block):
        # The average of a rectified sine is 2/pi times its amplitude.
        rect = np.abs(block) * (np.pi / 2)
        fast = min(self._fast_left, len(rect))
        self._fast_left -= fast
        level = np.concatenate([self._follow(rect[:fast], 0.01),
                                self._follow(rect[fast:], AGC_SECONDS)])
        gain = 1.0 / np.maximum(level, 1.0 / AGC_MAX_GAIN)
        self.gain = float(gain[-1])
        return np.clip(block * gain, -AGC_CLIP, AGC_CLIP)

    def _follow(self, rect, seconds):
        if not len(rect):
            return rect
        a = np.exp(-1.0 / (seconds * SAMPLE_RATE))
        out, _ = sps.lfilter([1 - a], [1, -a], rect, zi=[a * self._level])
        self._level = float(out[-1])
        return out


class Upsampler:
    """Streaming OVERSAMPLE× interpolation."""

    def __init__(self):
        # Keep everything up to the highest sideband, remove the copies that
        # start at the mirror image of it (44.1 kHz minus that).
        top = MAX_CARRIER_HZ + TUNER_STOP_HZ
        taps = kaiser_lowpass(top, SAMPLE_RATE - top, 80, SAMPLE_RATE * OVERSAMPLE)
        # Pad so the delay is a whole number of 44.1 kHz samples.
        half = -(-(len(taps) // 2) // OVERSAMPLE) * OVERSAMPLE
        self._taps = np.pad(taps, half - len(taps) // 2) * OVERSAMPLE
        self.delay = half // OVERSAMPLE
        self._history = np.zeros(2 * self.delay)

    def process(self, block):
        x = np.concatenate([self._history, block])
        self._history = x[len(block):]
        start = len(self._taps) - 1
        return sps.upfirdn(self._taps, x, up=OVERSAMPLE)[start:start + len(block) * OVERSAMPLE]


class EnvelopeDetector:
    """Classic AM detector: full-wave rectify, low-pass, block DC.

    Rectifying and low-passing at OVERSAMPLE × SAMPLE_RATE keeps the carrier
    harmonics the rectifier creates (up to many × fc) from aliasing into the
    audio before the low-pass removes them.

    For a full-wave rectified sine the average is (2/pi)·amplitude, so the
    low-passed output is scaled by pi/2 to recover the envelope 1 + m·x(t).
    """

    def __init__(self):
        hi_rate = SAMPLE_RATE * OVERSAMPLE
        self._up = Upsampler()
        # Stage 2: the detector's low-pass. 8th order so the 2fc ripple is well
        # below audibility even for the lowest station (≈ -70 dB at 11 kHz).
        self._lp = StreamFilter(sps.butter(8, DETECTOR_BW_HZ, fs=hi_rate, output="sos"))
        # Only for listening to / plotting stage 1 at 44.1 kHz: anti-alias before
        # dropping samples, otherwise *that* display would be full of aliasing.
        self._rect_aa = StreamFilter(sps.ellip(10, 0.1, 100, 20000, fs=hi_rate, output="sos"))
        # Stage 3: DC block (a real radio uses a series capacitor).
        self._hp = StreamFilter(sps.butter(1, 20.0, btype="highpass",
                                           fs=SAMPLE_RATE, output="sos"))
        # Delays (in 44.1 kHz samples), so the plots can line theory up with practice.
        self.rect_delay = self._up.delay + self._delay_of(self._rect_aa.sos, hi_rate)
        self.lp_delay = self._up.delay + self._delay_of(self._lp.sos, hi_rate)

    @staticmethod
    def _delay_of(sos, rate):
        f = 500.0       # low enough to be in every passband; delay is flat there
        w = np.exp(-2j * np.pi * f / rate * np.arange(3))
        h = np.prod((sos[:, :3] @ w) / (sos[:, 3:] @ w))
        return round(-np.angle(h) / (2 * np.pi * f) * SAMPLE_RATE)

    def process(self, am):
        """Run one block through the detector. Returns every stage at 44.1 kHz."""
        rectified_hi = np.abs(self._up.process(am))                     # stage 1
        lowpassed = self._lp(rectified_hi)[::OVERSAMPLE] * (np.pi / 2)  # stage 2
        audio = self._hp(lowpassed)                                     # stage 3
        rectified = self._rect_aa(rectified_hi)[::OVERSAMPLE]
        return rectified, lowpassed, audio
