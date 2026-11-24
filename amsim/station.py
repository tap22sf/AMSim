"""A radio station: its transmitter, and the path from there to the receiver."""

from dataclasses import dataclass

import numpy as np

from amsim.filters import Smooth
from amsim.message import Looper, MicRecorder
from amsim.settings import MAX_CARRIER_HZ, MIN_CARRIER_HZ, MIN_KM, SAMPLE_RATE

# Path: distance
GROUND_LOSS_DB_PER_KM = 0.04    # on top of the 1/d spreading; a simplification of
                                # the real ground-wave curves

# Path: skywave. A second copy arrives via the ionosphere, later and with a
# phase that drifts as the reflecting layer moves.
SKYWAVE_DELAY = round(0.001 * SAMPLE_RATE)  # ≈ 1 ms -> fading notches 1 kHz apart
SKYWAVE_MAX = 0.95              # strength vs the ground wave at "full night"
SKYWAVE_DRIFT_HZ = 0.2          # average phase drift: one fade about every 5 s


def path_gain(km):
    """Amplitude at the receiver relative to the same transmitter 1 km away."""
    loss_db = 20 * np.log10(km / MIN_KM) + GROUND_LOSS_DB_PER_KM * (km - MIN_KM)
    return 10 ** (-loss_db / 20)


class Carrier:
    """Unit-amplitude sine. Frequency is shared by the GUI and the audio thread."""

    def __init__(self, freq_hz):
        self.freq_hz = freq_hz
        self._phase = 0.0

    def nudge(self, delta_hz):
        self.freq_hz = float(np.clip(self.freq_hz + delta_hz, MIN_CARRIER_HZ, MAX_CARRIER_HZ))

    def phases(self, frames):
        """Phase of each sample in the next block. Carries over between blocks."""
        step = 2 * np.pi * self.freq_hz / SAMPLE_RATE
        block = self._phase + step * np.arange(frames)
        self._phase = (self._phase + step * frames) % (2 * np.pi)
        return block


@dataclass
class StationBlock:
    """One block of a station's signals, from the message to the receiver's antenna."""

    original: np.ndarray        # the recording as captured
    message: np.ndarray         # x(t): band-limited, peak 1
    scaled_message: np.ndarray  # m·x(t): what a perfect receiver would recover
    envelope: np.ndarray        # 1 + m·x(t)
    carrier: np.ndarray
    am: np.ndarray              # envelope · carrier: what leaves the transmitter
    received: np.ndarray        # what reaches the receiver, after distance and fading
    laps: np.ndarray            # completed passes of the message loop, per sample


class Station:
    """One station: its message, its transmitter, and the path to the receiver."""

    def __init__(self, number, settings, rng):
        self.number = number
        self.carrier = Carrier(settings.carrier_hz)
        self.looper = Looper()
        self.recorder = MicRecorder()
        self.on_air = False
        # The rest is explained in StationSettings.
        self.modulation = settings.modulation
        self.volume = settings.volume
        self.distance_km = settings.distance_km
        self.skywave = settings.skywave
        self._rng = rng
        self._modulation = Smooth(settings.modulation)
        self._gain = Smooth(0.0)
        self._sky = Smooth(0.0)
        self._sky_phase = rng.uniform(0, 2 * np.pi)
        self._sky_drift = SKYWAVE_DRIFT_HZ

    def stop_recording(self):
        """Make the recording the message. Returns False if it was silent (and discarded)."""
        return self.looper.set_clip(self.recorder.stop())

    def render(self, frames):
        """Simulate the next block. Returns a StationBlock."""
        theta = self.carrier.phases(frames)
        loop = self.looper.render(frames, SKYWAVE_DELAY, paused=self.recorder.recording)
        x = loop.message
        m = self._modulation.block(self.modulation, frames)
        envelope = 1.0 + m * x
        carrier = np.sin(theta)
        am = envelope * carrier                             # leaves the transmitter

        received = am
        sky = self._sky.block(SKYWAVE_MAX * self.skywave, frames)
        phi = self._drift(frames)
        if sky.any():
            # The late copy. Its carrier phase drifts against the direct one, so
            # the two alternately add up and cancel, at different moments for
            # the carrier and for each part of the sidebands.
            received = am + sky * (1.0 + m * loop.delayed_message) * np.sin(theta + phi)
        gain = self._gain.block(path_gain(self.distance_km) if self.on_air else 0.0, frames)
        return StationBlock(original=loop.original, message=x, scaled_message=m * x,
                            envelope=envelope, carrier=carrier, am=am,
                            received=gain * received, laps=loop.laps)

    def _drift(self, frames):
        # The drift rate itself wanders slowly, so fades don't come like clockwork.
        dt = frames / SAMPLE_RATE
        pull = min(dt / 4.0, 1.0)
        self._sky_drift += (pull * (SKYWAVE_DRIFT_HZ - self._sky_drift)
                            + 0.08 * np.sqrt(dt) * self._rng.standard_normal())
        step = 2 * np.pi * self._sky_drift / SAMPLE_RATE
        phi = self._sky_phase + step * np.arange(frames)
        self._sky_phase = (self._sky_phase + step * frames) % (2 * np.pi)
        return phi
