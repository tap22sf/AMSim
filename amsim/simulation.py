"""The whole simulation: stations, air and receiver.

    Station (×3):  message -> modulator -> path (distance, skywave fading) --+
    Atmosphere:    receiver noise + weather ---------------------------------+-> air
    Receiver:      air -> tuner -> AGC -> rectify -> low-pass -> DC block
"""

from typing import NamedTuple

import numpy as np

from amsim.atmosphere import Atmosphere
from amsim.filters import Smooth
from amsim.receiver import AGC, EnvelopeDetector, Tuner
from amsim.settings import FFT_N, SAMPLE_RATE
from amsim.stages import AB_MODE, AB_SOURCES, LISTEN_SIGNAL
from amsim.station import Station

OUTPUT_GAIN = 0.2       # overall speaker level; keep it modest
SNR_SECONDS = 0.5       # the SNR reading is averaged over about this long


class Capture:
    """Ring buffer of every signal the audio thread produced, for the plots to read."""

    SIGNALS = ("original", "message", "scaled_message", "envelope", "carrier", "am",    # station
               "air", "tuned", "agc", "rectified", "lowpassed", "demod",                # receiver
               "out")                                                                   # speakers

    def __init__(self, size):
        self.size = size
        self._next = 0          # where the next sample goes
        self._buffers = {name: np.zeros(size) for name in self.SIGNALS}

    def write(self, **blocks):
        n = len(next(iter(blocks.values())))
        where = (self._next + np.arange(n)) % self.size
        for name, block in blocks.items():
            self._buffers[name][where] = block
        self._next = (self._next + n) % self.size

    def last(self, name, n, delay=0):
        """n samples of a signal, oldest first, ending `delay` samples ago."""
        where = (self._next - delay - n + np.arange(n)) % self.size
        return self._buffers[name][where]


class Choices(NamedTuple):
    """What the user has chosen to hear, read once at the start of a block."""

    selected_index: int
    listen: str
    volume: float


class Simulation:
    """Wires the stations, the air and the receiver together.

    Two threads share this object, without locks:

    * The audio thread calls audio_out() for every block, about every 10-20 ms. It must
      finish in time or the sound breaks up, so it never waits for the window.
    * The window's thread changes settings (selected_index, listen, weather and the
      attributes of the stations) and draws what the audio thread has produced.

    The rules that make this safe:

    * The window only assigns simple values: a number, a bool or a string. It never
      calls anything that moves the simulation forward.
    * The audio thread reads each value once per block and uses that copy throughout,
      so a change takes effect at the next block (smoothly: see filters.Smooth).
    * For drawing, the window reads only from `capture` and from simple readings such as
      snr_db and agc.gain_db. If a block arrives while it reads, a plot may be a block out
      for one frame, which doesn't show.
    * A message is replaced in one assignment (see Looper.set_clip). That also rewinds
      the loop, which is the one value both threads write. If they clash the new
      message just starts somewhere other than its beginning.
    """

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

        # Set by the window.
        self.selected_index = 0             # station being edited and received
        self.listen = settings.listen

        # Set by the audio thread.
        self.ab_playing = AB_SOURCES[0]     # which side of the A/B mode is sounding
        self.snr_db = None                  # wanted station vs everything else, after the
                                            # tuner. None = the station is off the air.
        self._tuned_to = 0
        self._volume = Smooth(1.0)
        self._powers = None                 # wanted and unwanted power, averaged

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

    # --- audio thread ---
    def audio_out(self, outdata, frames, time_info, status):
        """Callback of the output stream: fill outdata with the next block."""
        choices = self._read_choices()
        out = self._simulate(frames, choices)
        # Volume comes after everything else. Turning the transmitter down instead
        # would do nothing: the AGC would turn it back up.
        volume = self._volume.block(choices.volume, frames)
        outdata[:, 0] = np.clip(OUTPUT_GAIN * volume * out, -1.0, 1.0)

    def process(self, frames):
        """Simulate one block; returns what goes to the speakers, before volume."""
        return self._simulate(frames, self._read_choices())

    def _read_choices(self):
        selected_index = self.selected_index
        return Choices(selected_index, self.listen, self.stations[selected_index].volume)

    def _simulate(self, frames, choices):
        if choices.selected_index != self._tuned_to:
            self._tuned_to = choices.selected_index
            self._powers = None
            self.agc.jump()

        # Transmitters and air.
        stations = [station.render(frames) for station in self.stations]
        chosen = stations[choices.selected_index]
        unwanted = self.atmosphere.render(frames)
        for station in stations:
            if station is not chosen:
                unwanted = unwanted + station.received

        # Receiver. Wanted and unwanted go through the tuner separately only so the SNR
        # can be measured; the receiver itself uses their sum.
        parts = self.tuner.process(np.stack([chosen.received, unwanted]), chosen.carrier_hz)
        self._measure_snr(parts, chosen.on_air)
        tuned = parts.sum(axis=0)
        agc = self.agc.process(tuned)
        rectified, lowpassed, demod = self.detector.process(agc)

        signals = {
            "original": chosen.original, "message": chosen.message,
            "scaled_message": chosen.scaled_message, "envelope": chosen.envelope,
            "carrier": chosen.carrier, "am": chosen.am,
            "air": chosen.received + unwanted, "tuned": tuned, "agc": agc,
            "rectified": rectified, "lowpassed": lowpassed, "demod": demod,
        }
        out = self._choose_output(signals, choices.listen, chosen.laps)
        self.capture.write(out=out, **signals)
        return out

    def _choose_output(self, signals, listen, laps):
        if self.recording:
            return np.zeros(len(laps))      # stay quiet so the mic hears only you
        if listen != AB_MODE:
            return signals[LISTEN_SIGNAL[listen]]
        # Even laps: the original recording, unfiltered, as captured.
        # Odd laps: the receiver's final output (exactly what "Demodulated"
        # plays). Switches on the sample where the loop wraps.
        self.ab_playing = AB_SOURCES[int(laps[-1] % 2)]
        return np.where(laps % 2 == 0, signals["original"], signals["demod"])

    def _measure_snr(self, parts, on_air):
        power = (parts ** 2).mean(axis=1)
        if self._powers is None:
            self._powers = power
        keep = np.exp(-parts.shape[1] / SAMPLE_RATE / SNR_SECONDS)
        self._powers = keep * self._powers + (1 - keep) * power
        wanted, unwanted = self._powers
        snr_db = 10 * np.log10(max(wanted, 1e-20) / max(unwanted, 1e-20))
        self.snr_db = snr_db if on_air else None
