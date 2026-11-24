"""AM (amplitude modulation) transmission simulator.

Up to three stations share the air. Each has a recorded voice clip or a WAV file
as its message x(t) and a sine generator as its carrier.

    Transmitter:  s(t) = [1 + m * x(t)] * sin(2*pi*fc*t)
    Air:          all stations, weakened by distance and fading, plus noise
    Receiver:     tuner -> AGC -> envelope detector = rectify -> low-pass -> remove DC

Controls:
    CH 1 / 2 / 3           (or keys 1 - 3) choose the station that the channel
                           controls edit and the receiver is tuned to
  For the chosen channel:
    Up / Down              carrier frequency ±100 Hz
    Shift + Up / Down      carrier frequency ±10 Hz
    PageUp / PageDown      carrier frequency ±1 kHz
    Record / Stop          capture the message from the microphone
    Open WAV…              use a WAV file as the message (or: python main.py a.wav
                           [b.wav [c.wav]] for channels 1 - 3; with no files, the
                           sample recordings go on channels 1 and 2;
                           capture.py records what the PC is playing)
    Clear                  discard the message
    On air                 transmitter on / off (goes on when you give it a message)
    Volume                 speaker level while this channel is chosen (0 - 200 %)
    m slider               modulation index (0 - 1.5; above 1 = overmodulation)
    Distance               1 - 500 km to the receiver
    Skywave                fading from a second path via the ionosphere (day - night)
  For everything:
    Weather                static and lightning (clear - thunderstorm)
    Listen                 what goes to the speakers (any transmitter or receiver
                           stage); the spectrum plot follows this choice
    Receiver view          time span of the receiver-stage plots

The signal processing is in dsp.py.
"""

import os
import sys
import time

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from dsp import (AB_MODE, AB_SOURCES, DETECTOR_BW_HZ, FFT_N, LISTEN_CHANNEL, LISTEN_MODES,
                 MAX_KM, MAX_RECORD_S, MAX_VOLUME, MAX_WAV_S, MESSAGE_BW_HZ, MIN_KM,
                 SAMPLE_RATE, Transmission, load_wav)

# QShortcut lives in QtGui on Qt6 and QtWidgets on Qt5.
QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut

# Where to play. Words that must all appear in the device's "name, host API"
# (see `python -m sounddevice`). None = the Windows default output.
OUTPUT_DEVICE = None

# Messages for channels 1, 2 (and 3) when no WAV files are given on the command line.
SAMPLE_WAVS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
               for name in ("capture_20261123_121213.wav", "capture_20261123_121839.wav")]

# Display
CARRIER_CYCLES = 3      # carrier plot shows this many cycles, in slow motion
CARRIER_POINTS = 1000
SLOWMO = 0.02           # carrier plot speed (real time would alias at 60 fps)
FPS = 60
RECEIVER_VIEWS_MS = [2, 5, 20, 50]   # time spans offered for the receiver-stage plots
START_RECEIVER_VIEW_MS = 5
THREAD_SWITCH_S = 0.0005    # Python's default (5 ms) lets drawing hold up the audio thread
KM_STEPS = 1000         # resolution of the (logarithmic) distance slider


class PipelineDiagram(QtWidgets.QWidget):
    """Clickable block diagram of the transmission chain.

    Each block shows a live mini-waveform of the signal leaving it; clicking a
    block plays that signal (it just picks the matching Listen mode).
    """

    # key, title, subtitle, listen mode, captured channel, y-range of the thumbnail
    # (None = scale to fit, for signals whose level depends on distance)
    STAGES = [
        ("mic", "Source", "original recording", "Original recording", "original", (-1.5, 1.5)),
        ("band", "Band-limit", f"low-pass {MESSAGE_BW_HZ / 1000:g} kHz → x(t)", "Message",
         "message", (-1.1, 1.1)),
        ("mod", "Modulator", "[1 + m·x(t)] · carrier", "AM signal", "am", (-2.4, 2.4)),
        ("air", "Air", "all stations + noise", "On the air", "air", None),
        ("tune", "Tuner", "band-pass around fc", "Tuned", "tuned", None),
        ("agc", "AGC", "", "After AGC", "agc", (-2.4, 2.4)),
        ("rect", "Rectifier", "|s(t)|", "Rectified", "rectified", (-0.1, 2.4)),
        ("lp", "Low-pass", f"{DETECTOR_BW_HZ / 1000:g} kHz → 1 + m·x", "Low-passed",
         "lowpassed", (-0.1, 2.4)),
        ("dc", "DC block", "→ audio m·x(t)", "Demodulated", "demod", (-1.1, 1.1)),
        ("car", "Carrier oscillator", "", "Carrier", "carrier", (-1.2, 1.2)),
    ]
    ROW = ["mic", "band", "mod", "air", "tune", "agc", "rect", "lp", "dc"]  # main chain

    BG = QtGui.QColor(0, 0, 0)
    BOX = QtGui.QColor(30, 30, 30)
    BOX_HOVER = QtGui.QColor(48, 48, 48)
    EDGE = QtGui.QColor(110, 110, 110)
    TEXT = QtGui.QColor(230, 230, 230)
    DIM = QtGui.QColor(150, 150, 150)
    WAVE = QtGui.QColor(0, 220, 220)
    PLAYING = QtGui.QColor(255, 220, 0)
    GROUP = QtGui.QColor(90, 90, 90)

    def __init__(self, tx, on_select):
        super().__init__()
        self.tx = tx
        self.on_select = on_select
        self.view_s = START_RECEIVER_VIEW_MS / 1000
        self.hover = None
        self.rects = {}
        self.setMouseTracking(True)
        self.setMinimumHeight(250)

    # --- geometry ---
    def _layout(self):
        w = self.width()
        margin, gap, top, box_h = 12, 24, 30, 110
        cols = len(self.ROW)
        box_w = (w - 2 * margin - gap * (cols - 1)) / cols
        self.rects = {}
        for i, key in enumerate(self.ROW):
            self.rects[key] = QtCore.QRectF(margin + i * (box_w + gap), top, box_w, box_h)
        mod = self.rects["mod"]
        self.rects["car"] = QtCore.QRectF(mod.left(), mod.bottom() + 34, mod.width(), 70)

    def _stage_at(self, pos):
        for key, rect in self.rects.items():
            if rect.contains(QtCore.QPointF(pos)):
                return key
        return None

    # --- mouse ---
    def mouseMoveEvent(self, event):
        key = self._stage_at(event.position() if hasattr(event, "position") else event.pos())
        if key != self.hover:
            self.hover = key
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if key
                           else QtCore.Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event):
        self.hover = None

    def mousePressEvent(self, event):
        key = self._stage_at(event.position() if hasattr(event, "position") else event.pos())
        if key:
            mode = next(s[3] for s in self.STAGES if s[0] == key)
            self.on_select(mode)

    # --- drawing ---
    def _playing_keys(self):
        """Stages currently audible: (solid, outlined-only) sets."""
        listen = self.tx.listen
        if listen == AB_MODE:
            now = "mic" if self.tx.ab_playing == AB_SOURCES[0] else "dc"
            return {now}, {"mic", "dc"} - {now}
        return {s[0] for s in self.STAGES if s[3] == listen}, set()

    def _arrow(self, p, a, b):
        p.drawLine(a, b)
        d = b - a
        length = max((d.x() ** 2 + d.y() ** 2) ** 0.5, 1e-6)
        ux, uy = d.x() / length, d.y() / length
        head = QtGui.QPolygonF([b,
                                QtCore.QPointF(b.x() - 9 * ux + 5 * uy, b.y() - 9 * uy - 5 * ux),
                                QtCore.QPointF(b.x() - 9 * ux - 5 * uy, b.y() - 9 * uy + 5 * ux)])
        p.setBrush(p.pen().color())
        p.drawPolygon(head)

    def _thumbnail(self, p, rect, channel, yrange):
        """Min/max-per-pixel waveform, so dense signals still look right."""
        n = max(int(self.view_s * SAMPLE_RATE), 2)
        data = self.tx.capture.last(channel, n)
        cols = max(int(rect.width()), 2)
        if yrange is None:
            peak = max(np.abs(data).max() * 1.1, 1e-9)
            yrange = (-peak, peak)
        lo, hi = yrange
        def y_of(v):
            return rect.bottom() - (np.clip(v, lo, hi) - lo) / (hi - lo) * rect.height()
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        if lo < 0 < hi:
            p.drawLine(QtCore.QPointF(rect.left(), y_of(0)), QtCore.QPointF(rect.right(), y_of(0)))
        p.setPen(QtGui.QPen(self.WAVE, 1))
        # One zigzag through every column's minimum and maximum, drawn in a single
        # call: much faster than a line per pixel, and drawing competes with audio.
        starts = np.linspace(0, n, cols, endpoint=False).astype(int)
        ys = np.empty(2 * cols)
        ys[0::2] = y_of(np.minimum.reduceat(data, starts))
        ys[1::2] = y_of(np.maximum.reduceat(data, starts))
        xs = rect.left() + np.repeat(np.arange(cols), 2)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        # Smoothing these dense lines is what costs the time, and it barely shows.
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        p.drawPath(pg.arrayToQPath(xs, ys))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    def paintEvent(self, event):
        self._layout()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self.BG)
        r = self.rects

        # Transmitter / receiver groups.
        group_pen = QtGui.QPen(self.GROUP, 1, QtCore.Qt.PenStyle.DashLine)
        title_font = QtGui.QFont(self.font())
        title_font.setBold(True)
        small_font = QtGui.QFont(self.font())
        small_font.setPointSizeF(self.font().pointSizeF() * 0.85)
        ch = self.tx.channel
        for label, keys in ((f"TRANSMITTER – CHANNEL {ch.number}", ["mic", "band", "mod", "car"]),
                            (f"RECEIVER – TUNED TO CHANNEL {ch.number}",
                             ["tune", "agc", "rect", "lp", "dc"])):
            box = r[keys[0]]
            for k in keys[1:]:
                box = box.united(r[k])
            box = box.adjusted(-7, -22, 7, 7)
            p.setPen(group_pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(box, 8, 8)
            p.setPen(self.DIM)
            p.setFont(small_font)
            p.drawText(box.adjusted(8, 3, 0, 0), QtCore.Qt.AlignmentFlag.AlignLeft, label)

        # Signal-flow arrows.
        p.setPen(QtGui.QPen(self.EDGE, 2))
        for a, b in zip(self.ROW, self.ROW[1:]):
            mid_a = QtCore.QPointF(r[a].right(), r[a].center().y())
            mid_b = QtCore.QPointF(r[b].left() - 1, r[b].center().y())
            self._arrow(p, mid_a, mid_b)
        car, mod = r["car"], r["mod"]
        self._arrow(p, QtCore.QPointF(car.center().x(), car.top()),
                    QtCore.QPointF(mod.center().x(), mod.bottom() + 1))
        # What else is on the air.
        air = r["air"]
        others = QtCore.QRectF(air.left(), car.top(), air.width(), car.height())
        self._arrow(p, QtCore.QPointF(air.center().x(), others.top()),
                    QtCore.QPointF(air.center().x(), air.bottom() + 1))
        p.setPen(QtGui.QPen(self.EDGE, 1, QtCore.Qt.PenStyle.DashLine))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(others, 6, 6)
        lines = [f"CH {c.number}: " + (f"{c.carrier.freq_hz / 1000:g} kHz, {c.distance_km:.3g} km"
                                       if c.on_air else "off air")
                 for c in self.tx.channels if c is not ch]
        lines.append(f"Weather: {self.tx.atmosphere.weather:.0%}")
        p.setPen(self.DIM)
        p.setFont(small_font)
        p.drawText(others.adjusted(8, 4, -4, -4), QtCore.Qt.AlignmentFlag.AlignVCenter,
                   "\n".join(lines))

        # Blocks.
        solid, outlined = self._playing_keys()
        for key, title, subtitle, mode, channel, yrange in self.STAGES:
            rect = r[key]
            if key == "car":
                subtitle = f"fc = {ch.carrier.freq_hz:g} Hz"
            elif key == "agc":
                subtitle = f"gain {self.tx.agc.gain_db:+d} dB"
            if key in solid:
                edge = QtGui.QPen(self.PLAYING, 3)
            elif key in outlined:
                edge = QtGui.QPen(self.PLAYING, 1, QtCore.Qt.PenStyle.DashLine)
            else:
                edge = QtGui.QPen(self.EDGE, 1)
            p.setPen(edge)
            p.setBrush(self.BOX_HOVER if key == self.hover else self.BOX)
            p.drawRoundedRect(rect, 6, 6)

            inner = rect.adjusted(8, 5, -8, -6)
            p.setFont(title_font)
            p.setPen(self.PLAYING if key in solid else self.TEXT)
            p.drawText(inner, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
                       ("▶ " if key in solid else "") + title)
            p.setFont(small_font)
            p.setPen(self.DIM)
            p.drawText(inner.adjusted(0, 19, 0, 0),
                       QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, subtitle)
            wave = QtCore.QRectF(inner.left(), inner.top() + 38, inner.width(),
                                 max(inner.height() - 38, 10))
            self._thumbnail(p, wave, channel, yrange)

        p.setPen(self.DIM)
        p.setFont(small_font)
        p.drawText(QtCore.QRectF(r["rect"].left(), r["car"].top(), r["dc"].right() - r["rect"].left(),
                                 r["car"].height()),
                   QtCore.Qt.AlignmentFlag.AlignCenter,
                   "Click any block to listen to the signal leaving it.\n"
                   f"Thumbnails show the last {self.view_s * 1000:g} ms (Receiver view).")
        p.end()


def make_plot(layout, row, col, title, xlabel, xunits, ylabel):
    plot = layout.addPlot(row=row, col=col, title=title)
    plot.setLabel("bottom", xlabel, units=xunits)
    plot.setLabel("left", ylabel)
    plot.showGrid(x=True, y=True, alpha=0.3)
    return plot


class MainWindow(QtWidgets.QWidget):
    def __init__(self, tx):
        super().__init__()
        self.tx = tx
        self.setWindowTitle("AMSim – AM transmission")
        self.resize(1600, 1150)

        # --- controls: one box for the chosen channel, one for everything else ---
        self.channel_btns = QtWidgets.QButtonGroup(self)
        selector = QtWidgets.QHBoxLayout()
        selector.setSpacing(0)
        for i, ch in enumerate(tx.channels):
            btn = QtWidgets.QPushButton(f"CH {ch.number}")
            btn.setCheckable(True)
            btn.setMinimumHeight(34)
            btn.setStyleSheet("QPushButton:checked { background: #ffdc00; color: black; "
                              "font-weight: bold; }")
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            self.channel_btns.addButton(btn, i)
            selector.addWidget(btn)
        self.channel_btns.button(tx.selected).setChecked(True)
        self.channel_btns.idClicked.connect(self.select_channel)

        self.record_btn = QtWidgets.QPushButton("● Record")
        self.stop_btn = QtWidgets.QPushButton("■ Stop")
        self.open_btn = QtWidgets.QPushButton("Open WAV…")
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.air_box = QtWidgets.QCheckBox("On air")
        self.fc_label = QtWidgets.QLabel()
        self.vol_slider, self.vol_label = self.make_slider(round(MAX_VOLUME * 100), 110, 40)
        self.m_slider, self.m_label = self.make_slider(150, 130, 60)
        self.km_slider, self.km_label = self.make_slider(KM_STEPS, 150, 60)
        self.sky_slider, self.sky_label = self.make_slider(100, 130, 80)
        self.weather_slider, self.weather_label = self.make_slider(100, 150, 120)

        self.listen_box = QtWidgets.QComboBox()
        self.listen_box.addItems(LISTEN_MODES)
        self.listen_box.setCurrentText(tx.listen)
        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems([f"{ms} ms" for ms in RECEIVER_VIEWS_MS])
        self.view_box.setCurrentIndex(RECEIVER_VIEWS_MS.index(START_RECEIVER_VIEW_MS))

        self.status = QtWidgets.QLabel()
        self.readout = QtWidgets.QLabel()
        self.warning = QtWidgets.QLabel()
        self.warning.setStyleSheet("color: #ff6060; font-weight: bold;")
        self.warning.setWordWrap(True)

        for w in (self.record_btn, self.stop_btn, self.open_btn, self.clear_btn, self.air_box,
                  self.listen_box, self.view_box):
            # Keep keyboard focus off the controls so arrow keys stay free.
            w.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.record_btn.clicked.connect(self.on_record)
        self.stop_btn.clicked.connect(self.on_stop)
        self.open_btn.clicked.connect(self.on_open)
        self.clear_btn.clicked.connect(self.on_clear)
        self.air_box.toggled.connect(self.on_air_toggled)
        self.vol_slider.valueChanged.connect(self.on_volume_changed)
        self.m_slider.valueChanged.connect(self.on_m_changed)
        self.km_slider.valueChanged.connect(self.on_km_changed)
        self.sky_slider.valueChanged.connect(self.on_sky_changed)
        self.weather_slider.valueChanged.connect(self.on_weather_changed)
        self.listen_box.currentTextChanged.connect(self.on_listen_changed)
        self.view_box.currentIndexChanged.connect(self.on_view_changed)

        def row(*items):
            """Widgets in a line; a string becomes a label, a number becomes a gap."""
            line = QtWidgets.QHBoxLayout()
            for item in items:
                if isinstance(item, int):
                    line.addSpacing(item)
                else:
                    line.addWidget(QtWidgets.QLabel(item) if isinstance(item, str) else item)
            line.addStretch(1)
            return line

        self.channel_group = QtWidgets.QGroupBox()
        box = QtWidgets.QVBoxLayout(self.channel_group)
        box.addLayout(row(self.record_btn, self.stop_btn, self.open_btn, self.clear_btn, 12,
                          self.air_box, 12, "Volume", self.vol_slider, self.vol_label, 12,
                          self.fc_label))
        box.addLayout(row("Modulation", self.m_slider, self.m_label, 8,
                          "Distance", self.km_slider, self.km_label, 8,
                          "Skywave", self.sky_slider, self.sky_label))
        shared = QtWidgets.QGroupBox("Shared – all channels")
        box = QtWidgets.QVBoxLayout(shared)
        box.addLayout(row("Listen:", self.listen_box, 12, "Receiver view:", self.view_box))
        box.addLayout(row("Weather", self.weather_slider, self.weather_label))
        controls = QtWidgets.QHBoxLayout()
        controls.addLayout(selector)
        controls.addSpacing(8)
        controls.addWidget(self.channel_group, stretch=3)
        controls.addWidget(shared, stretch=2)
        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.status, stretch=1)
        status_row.addWidget(self.readout)

        # --- plots ---
        # Top row: the source signals and the spectrum.
        # Bottom row: the signal as it passes through each receiver stage.
        g = pg.GraphicsLayoutWidget()
        top = g.addLayout(row=0, col=0)
        bottom = g.addLayout(row=1, col=0)

        self.carrier_plot = make_plot(top, 0, 0, "", "Time", "s", "Amplitude")
        self.carrier_plot.setYRange(-1.2, 1.2)
        self.carrier_curve = self.carrier_plot.plot(pen=pg.mkPen("y", width=2))

        self.msg_plot = make_plot(top, 0, 1, "Message x(t): source audio, "
                                  f"low-passed to {MESSAGE_BW_HZ / 1000:g} kHz",
                                  "Time", "s", "x(t)")
        self.msg_curve = self.msg_plot.plot(pen=pg.mkPen("c", width=1))
        self.msg_curve.setDownsampling(auto=True, method="peak")
        self.msg_curve.setClipToView(True)
        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen("r", width=2))
        self.playhead.hide()
        self.msg_plot.addItem(self.playhead)

        self.spec_plot = make_plot(top, 0, 2, "", "Frequency", "Hz", "Level (dB)")
        self.spec_plot.setXRange(0, SAMPLE_RATE / 2, padding=0)
        self.spec_plot.setYRange(-100, 5)
        # The part of the spectrum the tuner lets through, and everything on the air.
        self.tuned_band = pg.LinearRegionItem(movable=False, brush=(255, 220, 0, 30),
                                              pen=pg.mkPen(None))
        self.spec_plot.addItem(self.tuned_band)
        self.spec_air = self.spec_plot.plot(pen=pg.mkPen((150, 150, 150), width=1))
        self.spec_msg = self.spec_plot.plot(pen=pg.mkPen("c", width=1))
        self.spec_sel = self.spec_plot.plot(pen=pg.mkPen("y", width=1))
        self.fc_lines = []
        for ch in tx.channels:
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 255, 255, 80)),
                                   label=f"CH {ch.number}", labelOpts={"position": 0.95})
            self.spec_plot.addItem(line)
            self.fc_lines.append(line)
        self.spec_plot.addItem(pg.InfiniteLine(
            pos=DETECTOR_BW_HZ, angle=90,
            pen=pg.mkPen("g", width=1, style=QtCore.Qt.PenStyle.DashLine)))
        self.freqs = np.fft.rfftfreq(FFT_N, 1 / SAMPLE_RATE)
        self.window = np.hanning(FFT_N)
        self.window_gain = self.window.sum() / 2   # so a unit sine reads 0 dB
        self.spec_msg_db = np.full(len(self.freqs), -120.0)
        self.spec_sel_db = np.full(len(self.freqs), -120.0)
        self.spec_air_db = np.full(len(self.freqs), -120.0)

        # Receiver stages. Each plot shows a stage's output (yellow) and, dashed in
        # red, what that stage *should* produce in theory, so you can compare.
        ref_pen = pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine)
        stage_specs = [
            # (title, ylabel, y-range, channel)
            ("① Tuned + AGC = s(t)<br><small>red: ideal envelope ±[1 + m·x]</small>",
             "s(t)", (-2.6, 2.6), "agc"),
            ("② Rectified |s(t)|<br><small>red: its average (2/π)·[1 + m·x]</small>",
             "|s(t)|", (-0.1, 2.6), "rectified"),
            (f"③ Low-passed ({DETECTOR_BW_HZ / 1000:g} kHz) × π/2<br>"
             "<small>red: ideal envelope 1 + m·x</small>",
             "envelope", (-0.1, 2.6), "lowpassed"),
            ("④ DC removed = recovered audio<br><small>red: original m·x(t)</small>",
             "audio", (-1.6, 1.6), "demod"),
        ]
        self.stages = []
        for col, (title, ylabel, yrange, channel) in enumerate(stage_specs):
            p = make_plot(bottom, 0, col, title, "Time", "s", ylabel)
            p.setYRange(*yrange)
            if col:
                p.setXLink(self.stages[0][0])
            curve = p.plot(pen=pg.mkPen("y", width=1))
            curve.setDownsampling(auto=True, method="peak")
            refs = [p.plot(pen=ref_pen) for _ in range(2 if col == 0 else 1)]
            self.stages.append((p, channel, curve, refs))

        # Clickable block diagram of the whole chain.
        self.diagram = PipelineDiagram(tx, self.listen_box.setCurrentText)
        self.on_view_changed(self.view_box.currentIndex())

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(status_row)
        layout.addWidget(self.warning)
        layout.addWidget(self.diagram)
        layout.addWidget(g, stretch=1)

        # --- carrier frequency shortcuts ---
        for keys, delta in (("Up", 100), ("Down", -100),
                            ("Shift+Up", 10), ("Shift+Down", -10),
                            ("PgUp", 1000), ("PgDown", -1000)):
            sc = QShortcut(QtGui.QKeySequence(keys), self)
            sc.activated.connect(lambda d=delta: self.tx.channel.carrier.nudge(d))
        for i in range(len(tx.channels)):
            sc = QShortcut(QtGui.QKeySequence(str(i + 1)), self)
            sc.activated.connect(lambda i=i: self.select_channel(i))

        self.messages = [""] * len(tx.channels)  # per channel: error / info in the status line
        self.plot_phase = 0.0
        self.last_t = time.perf_counter()
        self.on_weather_changed(self.weather_slider.value())
        self.select_channel(tx.selected)

    @staticmethod
    def make_slider(top, width, label_width):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, top)
        slider.setFixedWidth(width)
        slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        label = QtWidgets.QLabel()
        label.setMinimumWidth(label_width)      # so the row doesn't shift as the text changes
        return slider, label

    @property
    def message(self):
        return self.messages[self.tx.selected]

    @message.setter
    def message(self, text):
        self.messages[self.tx.selected] = text

    # --- control handlers ---
    def select_channel(self, index):
        """Point the channel controls and the receiver at a station."""
        if self.tx.recording:       # the recording belongs to the current channel
            return
        self.tx.selected = index
        ch = self.tx.channel
        self.channel_btns.button(index).setChecked(True)
        self.channel_group.setTitle(f"Channel {ch.number}")
        km_pos = KM_STEPS * np.log(ch.distance_km / MIN_KM) / np.log(MAX_KM / MIN_KM)
        for slider, value in ((self.vol_slider, ch.volume * 100), (self.m_slider, ch.m * 100), (self.km_slider, km_pos),
                              (self.sky_slider, ch.skywave * 100)):
            # Show the channel's settings without writing (rounded) values back to it.
            slider.blockSignals(True)
            slider.setValue(round(value))
            slider.blockSignals(False)
        self.show_channel_labels()
        self.show_sample()
        self.refresh_controls()

    def show_channel_labels(self):
        ch = self.tx.channel
        self.vol_label.setText(f"{ch.volume:.0%}")
        self.m_label.setText(f"m = {ch.m:.2f}")
        km = ch.distance_km
        self.km_label.setText(f"{km:.1f} km" if km < 10 else f"{km:.0f} km")
        self.sky_label.setText("day" if ch.skywave == 0 else
                               "night" if ch.skywave == 1 else f"{ch.skywave:.0%} night")

    def on_record(self):
        try:
            self.tx.channel.looper.start_recording()
        except Exception as exc:  # e.g. no microphone available
            self.message = f"Could not open microphone: {exc}"
            return
        self.message = ""
        self.playhead.hide()
        self.refresh_controls()

    def on_stop(self):
        if self.tx.channel.looper.stop_recording():
            self.message = ""
            self.tx.channel.on_air = True
        else:
            self.message = "Recording was silent – check the mic isn't muted"
        self.show_sample()
        self.refresh_controls()

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open message", "", "WAV files (*.wav);;All files (*)")
        if path:
            self.open_wav(path)

    def open_wav(self, path):
        name = os.path.basename(path)
        try:
            clip, truncated = load_wav(path)
        except Exception as exc:  # missing, unreadable or not a WAV file
            self.message = f"Could not open {name}: {exc}"
            return
        if not self.tx.channel.looper.set_clip(clip):
            self.message = f"{name} is silent or too short"
        else:
            self.message = name
            self.tx.channel.on_air = True
            if truncated:
                self.message += f", first {MAX_WAV_S:g} s only"
        self.show_sample()
        self.refresh_controls()

    def on_clear(self):
        self.tx.channel.looper.clear()
        self.message = ""
        self.show_sample()
        self.refresh_controls()

    def on_m_changed(self, value):
        self.tx.channel.m = value / 100
        self.show_channel_labels()

    def on_volume_changed(self, value):
        self.tx.channel.volume = value / 100
        self.show_channel_labels()

    def on_km_changed(self, value):
        # Logarithmic: equal slider steps multiply the distance equally.
        self.tx.channel.distance_km = MIN_KM * (MAX_KM / MIN_KM) ** (value / KM_STEPS)
        self.show_channel_labels()

    def on_sky_changed(self, value):
        self.tx.channel.skywave = value / 100
        self.show_channel_labels()

    def on_weather_changed(self, value):
        self.tx.atmosphere.weather = value / 100
        names = ("unsettled", "stormy", "thunderstorm")
        name = "clear" if value == 0 else names[min(value // 34, 2)]
        self.weather_label.setText(f"{value}% {name}")

    def on_air_toggled(self, on):
        self.tx.channel.on_air = on

    def on_listen_changed(self, mode):
        self.tx.listen = mode

    def on_view_changed(self, index):
        self.view_s = RECEIVER_VIEWS_MS[index] / 1000
        self.stages[0][0].setXRange(-self.view_s, 0, padding=0)
        self.diagram.view_s = self.view_s

    def refresh_controls(self):
        looper = self.tx.channel.looper
        self.record_btn.setEnabled(not looper.recording)
        self.stop_btn.setEnabled(looper.recording)
        self.open_btn.setEnabled(not looper.recording)
        self.clear_btn.setEnabled(not looper.recording and looper.sample is not None)
        for btn in self.channel_btns.buttons():
            btn.setEnabled(not looper.recording)
        self.air_box.setChecked(self.tx.channel.on_air)

    def show_sample(self):
        sample = self.tx.channel.looper.sample
        if sample is None:
            self.msg_curve.setData([], [])
            self.playhead.hide()
            return
        t = np.arange(len(sample)) / SAMPLE_RATE
        self.msg_curve.setData(t, sample)
        self.msg_plot.setXRange(0, t[-1], padding=0)
        self.msg_plot.setYRange(-1.1, 1.1)
        self.playhead.show()

    # --- per-frame update ---
    def tick(self):
        now = time.perf_counter()
        dt = (now - self.last_t) * SLOWMO
        self.last_t = now
        ch = self.tx.channel
        fc = ch.carrier.freq_hz
        looper = ch.looper

        # Carrier (slow motion so you can see it move).
        self.plot_phase = (self.plot_phase + 2 * np.pi * fc * dt) % (2 * np.pi)
        x = np.linspace(0.0, CARRIER_CYCLES / fc, CARRIER_POINTS)
        self.carrier_curve.setData(x, np.sin(self.plot_phase + 2 * np.pi * fc * x))
        self.carrier_plot.setTitle(
            f"Carrier: fc = {fc:g} Hz   (↑↓ ±100 Hz, Shift ±10 Hz, PgUp/PgDn ±1 kHz)")

        # Message: live waveform while recording, playhead while looping.
        if looper.recording:
            clip = looper.recorded_so_far()
            secs = len(clip) / SAMPLE_RATE
            self.msg_curve.setData(np.arange(len(clip)) / SAMPLE_RATE, clip)
            self.msg_plot.setXRange(0, max(secs, 1.0), padding=0)
            self.msg_plot.setYRange(-1.1, 1.1)
            self.status.setText(f"Recording… {secs:.1f} s (speakers muted)")
            if secs >= MAX_RECORD_S:
                self.on_stop()
        elif looper.sample is not None:
            self.playhead.setPos(looper.pos / SAMPLE_RATE)
            text = "Transmitting" if ch.on_air else "Off air:"
            text += f" {len(looper.sample) / SAMPLE_RATE:.1f} s message"
            if self.message:
                text += f" ({self.message})"
            if self.tx.listen == AB_MODE:
                text += f"   ▶ now playing: {self.tx.ab_playing.upper()}"
            self.status.setText(text)
        else:
            self.status.setText(self.message or "No message – press Record or Open WAV"
                                + (" (carrier only)" if ch.on_air else " (off air)"))

        self.fc_label.setText(f"fc = {fc:g} Hz  (↑↓ keys)")
        snr = self.tx.snr_db
        self.readout.setText(
            ("Signal / noise: station is off air" if snr is None else
             f"Signal / (noise + other stations): {snr:.0f} dB")
            + f"     AGC gain: {self.tx.agc.gain_db:+d} dB")

        # Receiver stages: the last few ms that actually went through the receiver.
        cap = self.tx.capture
        n = int(self.view_s * SAMPLE_RATE)
        t = (np.arange(n) - n) / SAMPLE_RATE
        # The theory curves come from the transmitter, so shift each by the receiver's
        # filter delay up to that stage.
        env = {c: cap.last("envelope", n, d) for c, d in self.tx.delays.items()}
        theory = {"agc": (env["agc"], -env["agc"]),
                  "rectified": (2 / np.pi * env["rectified"],),
                  "lowpassed": (env["lowpassed"],),
                  "demod": (cap.last("mx", n, self.tx.delays["demod"]),)}
        for _, channel, curve, refs in self.stages:
            curve.setData(t, cap.last(channel, n))
            for ref, y in zip(refs, theory[channel]):
                ref.setData(t, y)

        # Spectrum of the message vs whatever you're listening to (smoothed a little).
        listen = self.tx.listen
        for channel, store, curve in (("air", self.spec_air_db, self.spec_air),
                                      ("message", self.spec_msg_db, self.spec_msg),
                                      (LISTEN_CHANNEL[listen], self.spec_sel_db, self.spec_sel)):
            mag = np.abs(np.fft.rfft(cap.last(channel, FFT_N) * self.window)) / self.window_gain
            db = 20 * np.log10(mag + 1e-6)
            store[:] = 0.7 * store + 0.3 * db
            curve.setData(self.freqs, store)
        self.spec_plot.setTitle(f"Spectrum: message (cyan), {listen} (yellow), air (grey)<br>"
                                f"<small>shaded: tuner passband &nbsp; green: detector's "
                                f"{DETECTOR_BW_HZ / 1000:g} kHz low-pass cutoff</small>")
        for c, line in zip(self.tx.channels, self.fc_lines):
            line.setPos(c.carrier.freq_hz)
            line.setVisible(c.on_air or c is ch)
        self.tuned_band.setRegion((fc - MESSAGE_BW_HZ, fc + MESSAGE_BW_HZ))

        self.warning.setText(self.warnings(fc))
        self.diagram.update()

    def warnings(self, fc):
        msgs = []
        ch = self.tx.channel
        if ch.m > 1.0:
            msgs.append(f"Overmodulated (m = {ch.m:.2f} > 1): the envelope crosses "
                        "zero, so the envelope detector distorts.")
        if fc <= MESSAGE_BW_HZ:
            msgs.append(f"Carrier ({fc:g} Hz) is at or below the message bandwidth "
                        f"({MESSAGE_BW_HZ:g} Hz): the lower sideband folds over and the "
                        "receiver can't separate message from carrier.")
        elif 2 * fc - MESSAGE_BW_HZ < 2 * DETECTOR_BW_HZ:
            # After rectification the nearest unwanted copy of the message sits at
            # 2fc - bandwidth; it needs to be well above the detector's cutoff.
            msgs.append("Carrier is close to the message bandwidth: carrier ripple leaks "
                        "through the detector's low-pass filter.")
        for other in self.tx.channels:
            spacing = abs(other.carrier.freq_hz - fc)
            if other is not ch and other.on_air and ch.on_air and spacing < 2 * MESSAGE_BW_HZ:
                msgs.append(f"Channel {other.number} is only {spacing:g} Hz away: stations need "
                            f"{2 * MESSAGE_BW_HZ:g} Hz, so their sidebands overlap and the "
                            "tuner can't separate them.")
        return "   ".join(msgs)


def find_output_device():
    """Resolve OUTPUT_DEVICE to an index, falling back to the default output."""
    if OUTPUT_DEVICE is None:
        return None
    try:
        return sd.query_devices(OUTPUT_DEVICE, kind="output")["index"]
    except ValueError as exc:
        print(f"{exc} – using the default output instead")
        return None


def main():
    sys.setswitchinterval(THREAD_SWITCH_S)
    pg.mkQApp("AMSim")
    tx = Transmission()
    win = MainWindow(tx)
    win.show()
    # The samples are optional: if one has been deleted, its channel just starts empty.
    paths = sys.argv[1:] or SAMPLE_WAVS
    for i, path in enumerate(paths[:len(tx.channels)]):
        if path in SAMPLE_WAVS and not os.path.exists(path):
            continue
        win.select_channel(i)
        win.open_wav(path)
    win.select_channel(0)

    timer = QtCore.QTimer()
    timer.timeout.connect(win.tick)
    timer.start(int(1000 / FPS))

    # Output plays in a background thread for as long as the window is open.
    device = find_output_device()
    win.setWindowTitle(f"AMSim – AM transmission  →  {sd.query_devices(device, kind='output')['name']}")
    with sd.OutputStream(device=device, samplerate=SAMPLE_RATE, channels=1,
                         dtype="float32", callback=tx.audio_out):
        pg.exec()

    for ch in tx.channels:
        if ch.looper.recording:
            ch.looper.stop_recording()


if __name__ == "__main__":
    main()
