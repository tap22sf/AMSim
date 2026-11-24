"""The plots below the block diagram.

Each class owns one plot (or a row of them) and has an update method that the window
calls once per frame.
"""

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from amsim.settings import DETECTOR_BW_HZ, FFT_N, MESSAGE_BW_HZ, SAMPLE_RATE
from amsim.stages import LISTEN_SIGNAL

CARRIER_CYCLES = 3      # carrier plot shows this many cycles, in slow motion
CARRIER_POINTS = 1000
SLOW_MOTION = 0.02      # carrier plot speed (real time would alias at 60 fps)

SPECTRUM_FLOOR_DB = -120.0
SPECTRUM_SMOOTHING = 0.7    # share of the previous frame that is kept

DASHED = QtCore.Qt.PenStyle.DashLine


def make_plot(layout, col, title, xlabel, xunits, ylabel):
    plot = layout.addPlot(row=0, col=col, title=title)
    plot.setLabel("bottom", xlabel, units=xunits)
    plot.setLabel("left", ylabel)
    plot.showGrid(x=True, y=True, alpha=0.3)
    return plot


class CarrierPlot:
    """The carrier wave, slowed down so you can see it move."""

    def __init__(self, layout, col):
        self.plot = make_plot(layout, col, "", "Time", "s", "Amplitude")
        self.plot.setYRange(-1.2, 1.2)
        self.curve = self.plot.plot(pen=pg.mkPen("y", width=2))
        self._phase = 0.0
        self._last_update = time.perf_counter()

    def update(self, station):
        now = time.perf_counter()
        elapsed = (now - self._last_update) * SLOW_MOTION
        self._last_update = now

        fc = station.carrier.freq_hz
        self._phase = (self._phase + 2 * np.pi * fc * elapsed) % (2 * np.pi)
        t = np.linspace(0.0, CARRIER_CYCLES / fc, CARRIER_POINTS)
        self.curve.setData(t, np.sin(self._phase + 2 * np.pi * fc * t))
        self.plot.setTitle(
            f"Carrier: fc = {fc:g} Hz   (↑↓ ±100 Hz, Shift ±10 Hz, PgUp/PgDn ±1 kHz)")


class MessagePlot:
    """The whole message: with a playhead while it loops, growing while it is recorded."""

    def __init__(self, layout, col):
        self.plot = make_plot(layout, col, "Message x(t): source audio, "
                              f"low-passed to {MESSAGE_BW_HZ / 1000:g} kHz",
                              "Time", "s", "x(t)")
        self.curve = self.plot.plot(pen=pg.mkPen("c", width=1))
        self.curve.setDownsampling(auto=True, method="peak")
        self.curve.setClipToView(True)
        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=2))
        self.playhead.hide()
        self.plot.addItem(self.playhead)

    def show_message(self, looper):
        """Draw a station's message. Only needed when it changes."""
        if looper.sample is None:
            self.curve.setData([], [])
            self.playhead.hide()
            return
        t = np.arange(len(looper.sample)) / SAMPLE_RATE
        self.curve.setData(t, looper.sample)
        self._set_range(t[-1])
        self.playhead.show()

    def update(self, looper):
        if looper.recording:
            clip = looper.recorded_so_far()
            self.playhead.hide()
            self.curve.setData(np.arange(len(clip)) / SAMPLE_RATE, clip)
            self._set_range(max(len(clip) / SAMPLE_RATE, 1.0))
        elif looper.sample is not None:
            self.playhead.setPos(looper.pos / SAMPLE_RATE)

    def _set_range(self, seconds):
        self.plot.setXRange(0, seconds, padding=0)
        self.plot.setYRange(-1.1, 1.1)


class SpectrumPlot:
    """Spectrum of the message, of everything on the air and of what you're listening to."""

    def __init__(self, layout, col, stations):
        self.plot = make_plot(layout, col, "", "Frequency", "Hz", "Level (dB)")
        self.plot.setXRange(0, SAMPLE_RATE / 2, padding=0)
        self.plot.setYRange(-100, 5)
        # The part of the spectrum the tuner lets through.
        self.tuned_band = pg.LinearRegionItem(movable=False, brush=(255, 220, 0, 30),
                                              pen=pg.mkPen(None))
        self.plot.addItem(self.tuned_band)
        # Later curves are drawn on top of earlier ones.
        self.air_curve = self.plot.plot(pen=pg.mkPen((150, 150, 150), width=1))
        self.message_curve = self.plot.plot(pen=pg.mkPen("c", width=1))
        self.listen_curve = self.plot.plot(pen=pg.mkPen("y", width=1))
        self.carrier_lines = []
        for station in stations:
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 255, 255, 80)),
                                   label=f"CH {station.number}", labelOpts={"position": 0.95})
            self.plot.addItem(line)
            self.carrier_lines.append(line)
        self.plot.addItem(pg.InfiniteLine(pos=DETECTOR_BW_HZ, angle=90,
                                          pen=pg.mkPen("g", width=1, style=DASHED)))

        self._freqs = np.fft.rfftfreq(FFT_N, 1 / SAMPLE_RATE)
        self._taper = np.hanning(FFT_N)
        self._taper_gain = self._taper.sum() / 2    # so a unit sine reads 0 dB
        self._air_db, self._message_db, self._listen_db = (
            np.full(len(self._freqs), SPECTRUM_FLOOR_DB) for _ in range(3))

    def update(self, sim):
        for signal, smoothed_db, curve in (
                ("air", self._air_db, self.air_curve),
                ("message", self._message_db, self.message_curve),
                (LISTEN_SIGNAL[sim.listen], self._listen_db, self.listen_curve)):
            new_db = self._spectrum_db(sim.capture.last(signal, FFT_N))
            smoothed_db[:] = SPECTRUM_SMOOTHING * smoothed_db + (1 - SPECTRUM_SMOOTHING) * new_db
            curve.setData(self._freqs, smoothed_db)
        self.plot.setTitle(f"Spectrum: message (cyan), {sim.listen} (yellow), air (grey)<br>"
                           f"<small>shaded: tuner passband &nbsp; green: detector's "
                           f"{DETECTOR_BW_HZ / 1000:g} kHz low-pass cutoff</small>")
        selected = sim.selected_station
        for station, line in zip(sim.stations, self.carrier_lines):
            line.setPos(station.carrier.freq_hz)
            line.setVisible(station.on_air or station is selected)
        fc = selected.carrier.freq_hz
        self.tuned_band.setRegion((fc - MESSAGE_BW_HZ, fc + MESSAGE_BW_HZ))

    def _spectrum_db(self, samples):
        magnitude = np.abs(np.fft.rfft(samples * self._taper)) / self._taper_gain
        return 20 * np.log10(magnitude + 1e-6)


@dataclass(frozen=True)
class ReceiverPlotSpec:
    """What one of the receiver plots shows."""

    title: str
    ylabel: str
    y_range: Tuple[float, float]
    signal: str             # name of the signal in Capture
    theory_curves: int = 1  # how many theory curves there are to compare it with


RECEIVER_PLOTS = (
    ReceiverPlotSpec("① Tuned + AGC = s(t)<br><small>red: ideal envelope ±[1 + m·x]</small>",
                     "s(t)", (-2.6, 2.6), "agc", theory_curves=2),
    ReceiverPlotSpec("② Rectified |s(t)|<br><small>red: its average (2/π)·[1 + m·x]</small>",
                     "|s(t)|", (-0.1, 2.6), "rectified"),
    ReceiverPlotSpec(f"③ Low-passed ({DETECTOR_BW_HZ / 1000:g} kHz) × π/2<br>"
                     "<small>red: ideal envelope 1 + m·x</small>",
                     "envelope", (-0.1, 2.6), "lowpassed"),
    ReceiverPlotSpec("④ DC removed = recovered audio<br><small>red: original m·x(t)</small>",
                     "audio", (-1.6, 1.6), "demod"),
)


class ReceiverPlots:
    """The signal after each receiver stage (yellow) against what theory predicts (red)."""

    def __init__(self, layout):
        theory_pen = pg.mkPen("r", width=2, style=DASHED)
        self.plots = []
        self.curves = []            # per plot: the real signal
        self.theory_curves = []     # per plot: list of theory curves
        for col, spec in enumerate(RECEIVER_PLOTS):
            plot = make_plot(layout, col, spec.title, "Time", "s", spec.ylabel)
            plot.setYRange(*spec.y_range)
            if self.plots:
                plot.setXLink(self.plots[0])
            curve = plot.plot(pen=pg.mkPen("y", width=1))
            curve.setDownsampling(auto=True, method="peak")
            self.plots.append(plot)
            self.curves.append(curve)
            self.theory_curves.append([plot.plot(pen=theory_pen)
                                       for _ in range(spec.theory_curves)])
        self._view_seconds = 0.0

    def set_view(self, seconds):
        self._view_seconds = seconds
        self.plots[0].setXRange(-seconds, 0, padding=0)     # the others are linked to it

    def update(self, sim):
        n = int(self._view_seconds * SAMPLE_RATE)
        t = (np.arange(n) - n) / SAMPLE_RATE
        theory = self._theory(sim, n)
        for spec, curve, theory_curves in zip(RECEIVER_PLOTS, self.curves, self.theory_curves):
            curve.setData(t, sim.capture.last(spec.signal, n))
            for theory_curve, y in zip(theory_curves, theory[spec.signal]):
                theory_curve.setData(t, y)

    @staticmethod
    def _theory(sim, n):
        """Receiver signal -> the curves it should follow, n samples each.

        They come from the transmitter, so each is shifted by the receiver's filter
        delay up to that stage.
        """
        def transmitted(signal, stage):
            return sim.capture.last(signal, n, sim.delays[stage])

        envelope = transmitted("envelope", "agc")
        return {"agc": (envelope, -envelope),
                "rectified": (2 / np.pi * transmitted("envelope", "rectified"),),
                "lowpassed": (transmitted("envelope", "lowpassed"),),
                "demod": (transmitted("scaled_message", "demod"),)}
