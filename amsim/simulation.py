"""The whole simulation: stations, air and receiver.

    Station (×3):  message -> modulator -> path (distance, skywave fading) --+
    Atmosphere:    receiver noise + weather ---------------------------------+-> air
    Receiver:      air -> tuner -> AGC -> rectify -> low-pass -> DC block
"""

import numpy as np

from amsim.atmosphere import Atmosphere
from amsim.filters import Smooth
from amsim.receiver import AGC, EnvelopeDetector, Tuner
from amsim.settings import FFT_N, SAMPLE_RATE
from amsim.stages import AB_MODE, AB_SOURCES, LISTEN_SIGNAL
from amsim.station import Station

OUTPUT_GAIN = 0.2       # overall speaker level; keep it modest


class Capture:
    """Ring buffer of every signal the audio thread produced, for the plots to read."""

    SIGNALS = ("carrier", "original", "message", "envelope", "scaled_message", "am", "air", "tuned",
                "agc", "rectified", "lowpassed", "demod", "out")

    def __init__(self, size):
        self.size = size
        self.idx = 0
        self.bufs = {name: np.zeros(size) for name in self.SIGNALS}

    def write(self, **blocks):
        n = len(next(iter(blocks.values())))
        where = (self.idx + np.arange(n)) % self.size
        for name, block in blocks.items():
            self.bufs[name][where] = block
        self.idx = (self.idx + n) % self.size

    def last(self, name, n, delay=0):
        """n samples of a signal, oldest first, ending `delay` samples ago."""
        where = (self.idx - delay - n + np.arange(n)) % self.size
        return self.bufs[name][where]


class Simulation:
    """Wires the stations, the air and the receiver together for the audio thread."""

    def __init__(self, settings, seed=None):
        rng = np.random.default_rng(seed)
        self.stations = [Station(number, station_settings, rng)
                         for number, station_settings in enumerate(settings.stations, start=1)]
        self.stations[0].on_air = True
        self.atmosphere = Atmosphere(rng)
        self.atmosphere.weather = settings.weather
        self.tuner = Tuner(self.stations[0].carrier.freq_hz, rows=2)
        self.agc = AGC()
        self.detector = EnvelopeDetector()
        self.capture = Capture(FFT_N)
        self.selected_index = 0                   # station being edited and received
        self.listen = settings.listen
        self.ab_playing = AB_SOURCES[0]     # which side of the A/B mode is sounding
        self.snr_db = None                  # wanted station vs everything else, after the tuner
        self._tuned_to = 0
        self._volume = Smooth(1.0)
        self._powers = None
        # How far each receiver stage lags the transmitter, for the plots.
        self.delays = {"agc": self.tuner.delay,
                       "rectified": self.tuner.delay + self.detector.rect_delay,
                       "lowpassed": self.tuner.delay + self.detector.lp_delay,
                       "demod": self.tuner.delay + self.detector.lp_delay}

    @property
    def selected_station(self):
        return self.stations[self.selected_index]

    @property
    def recording(self):
        return any(station.recorder.recording for station in self.stations)

    def process(self, frames):
        """Simulate one block; returns what goes to the speakers (before gain)."""
        selected = self.selected_index            # the GUI may change these at any time
        listen = self.listen
        if selected != self._tuned_to:
            self._tuned_to = selected
            self._powers = None
            self.agc.jump()

        signals = [station.render(frames) for station in self.stations]
        chosen = signals[selected]
        wanted = chosen.received
        unwanted = self.atmosphere.render(frames)
        for i, s in enumerate(signals):
            if i != selected:
                unwanted = unwanted + s.received

        # Wanted and unwanted go through the tuner separately only so the SNR can
        # be measured; the receiver itself uses their sum.
        parts = self.tuner.process(np.stack([wanted, unwanted]),
                                   self.stations[selected].carrier.freq_hz)
        tuned = parts.sum(axis=0)
        self._measure_snr(parts, self.stations[selected].on_air)
        agc = self.agc.process(tuned)
        rectified, lowpassed, demod = self.detector.process(agc)

        blocks = {
            # the chosen station
            "original": chosen.original, "message": chosen.message,
            "scaled_message": chosen.scaled_message, "envelope": chosen.envelope,
            "carrier": chosen.carrier, "am": chosen.am,
            # the air and the receiver
            "air": wanted + unwanted, "tuned": tuned, "agc": agc, "rectified": rectified,
            "lowpassed": lowpassed, "demod": demod,
        }
        if self.recording:
            out = np.zeros(frames)              # stay quiet so the mic hears only you
        elif listen == AB_MODE:
            # Even laps: the original recording, unfiltered, as captured.
            # Odd laps: the receiver's final output (exactly what "Demodulated"
            # plays). Switches on the sample where the loop wraps.
            laps = chosen.laps
            out = np.where(laps % 2 == 0, chosen.original, demod)
            self.ab_playing = AB_SOURCES[int(laps[-1] % 2)]
        else:
            out = blocks[LISTEN_SIGNAL[listen]]
        self.capture.write(out=out, **blocks)
        return out

    def _measure_snr(self, parts, on_air):
        power = (parts ** 2).mean(axis=1)
        if self._powers is None:
            self._powers = power
        keep = np.exp(-parts.shape[1] / SAMPLE_RATE / 0.5)
        self._powers = keep * self._powers + (1 - keep) * power
        w, u = self._powers
        self.snr_db = 10 * np.log10(max(w, 1e-20) / max(u, 1e-20)) if on_air else None

    def audio_out(self, outdata, frames, time_info, status):
        # Volume comes after everything else. Turning the transmitter down instead
        # would do nothing: the AGC would turn it back up.
        volume = self._volume.block(self.selected_station.volume, frames)
        outdata[:, 0] = np.clip(OUTPUT_GAIN * volume * self.process(frames), -1.0, 1.0)
