"""A station's message: reading it from a file, recording it and playing it in a loop."""

from typing import NamedTuple

import numpy as np
import sounddevice as sd
from scipy import signal as sps
from scipy.io import wavfile

from amsim.filters import kaiser_lowpass
from amsim.settings import MAX_WAV_S, MESSAGE_BW_HZ, SAMPLE_RATE


def load_wav(path):
    """Read a WAV file as mono float at SAMPLE_RATE.

    Returns (clip, truncated); at most MAX_WAV_S seconds are kept.
    """
    rate, data = wavfile.read(path)
    keep = int(MAX_WAV_S * rate)
    truncated = len(data) > keep
    data = data[:keep]
    if np.issubdtype(data.dtype, np.integer):
        offset = 128 if data.dtype == np.uint8 else 0       # 8-bit WAVs are unsigned
        data = (data.astype(np.float64) - offset) / 2.0 ** (8 * data.dtype.itemsize - 1)
    else:
        # A NaN or infinity here would stay in the receiver's filters for good.
        data = np.nan_to_num(data.astype(np.float64), posinf=0.0, neginf=0.0)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE and data.size:
        divisor = int(np.gcd(rate, SAMPLE_RATE))
        data = sps.resample_poly(data, SAMPLE_RATE // divisor, rate // divisor)
    return data, truncated


class MicRecorder:
    """Records a clip from the default microphone."""

    def __init__(self):
        self.recording = False
        self._chunks = []
        self._stream = None

    def start(self):
        self._chunks = []
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                      dtype="float32", callback=self._on_input)
        self._stream.start()
        self.recording = True

    def stop(self):
        """Stop recording and return the clip."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.recording = False
        return self.recorded_so_far()

    def recorded_so_far(self):
        chunks = list(self._chunks)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def _on_input(self, indata, frames, time_info, status):
        self._chunks.append(indata[:, 0].copy())


class LoopBlock(NamedTuple):
    """One block of a looping message."""

    message: np.ndarray             # x(t)
    delayed_message: np.ndarray     # x(t), a little earlier
    original: np.ndarray            # the recording as captured
    laps: np.ndarray                # completed passes through the clip, per sample. Tells
                                    # exactly which sample the loop wrapped on.


class Looper:
    """Holds a station's message (recorded or from a file) and plays it back in a loop.

    The stored clip is band-limited to MESSAGE_BW_HZ and normalised to a peak
    of 1, so that the modulation index m means what the textbook says.
    The unfiltered recording is kept too (same gain), for comparison.
    """

    def __init__(self):
        # (message x(t), original recording) — swapped in as one tuple so the
        # audio thread never sees a half-updated set.
        self._clip = None
        self.pos = 0
        self.lap = 0                # completed passes through the clip

    def set_clip(self, clip):
        """Make a mono clip the message. Returns False if it was silent or too short (and ignored)."""
        clip = np.asarray(clip, dtype=np.float64)
        if clip.size < 64:
            return False
        raw = clip - clip.mean()
        # Sharp, because the neighbouring station starts 250 Hz further up.
        taps = kaiser_lowpass(MESSAGE_BW_HZ - 150, MESSAGE_BW_HZ + 150, 70, SAMPLE_RATE)
        filtered = sps.fftconvolve(raw, taps)[len(taps) // 2:][:len(raw)]  # no delay
        peak = np.max(np.abs(filtered))
        if peak <= 1e-3:
            return False
        # Same scale factor for both, so the only difference you hear is the filtering.
        self.pos = 0
        self.lap = 0
        self._clip = ((filtered / peak).astype(np.float32), (raw / peak).astype(np.float32))
        return True

    @property
    def sample(self):
        """The band-limited message x(t), or None."""
        clip = self._clip
        return None if clip is None else clip[0]

    def clear(self):
        self._clip = None
        self.pos = 0

    def render(self, frames, delay, paused=False):
        """The next block. `delay` is how far delayed_message lags, in samples."""
        clip = self._clip
        if clip is None or paused:
            silence = np.zeros(frames)
            return LoopBlock(silence, silence, silence, np.full(frames, self.lap))
        sample, original = clip
        n = len(sample)
        ahead = self.pos + np.arange(frames)
        laps = self.lap + ahead // n
        self.lap += (self.pos + frames) // n
        self.pos = (self.pos + frames) % n
        return LoopBlock(sample[ahead % n], sample[(ahead - delay) % n], original[ahead % n], laps)
