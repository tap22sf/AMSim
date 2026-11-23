"""AM (amplitude modulation) transmission simulator.

A recorded voice clip is the message x(t); the sine generator is the carrier.

    Transmitter:  s(t) = [1 + m * x(t)] * sin(2*pi*fc*t)
    Receiver:     envelope detector = rectify -> low-pass -> remove DC

Controls:
    Up / Down              carrier frequency ±100 Hz
    Shift + Up / Down      carrier frequency ±10 Hz
    PageUp / PageDown      carrier frequency ±1 kHz
    Record / Stop          capture the message from the microphone
    Clear                  discard the message
    m slider               modulation index (0 - 1.5; above 1 = overmodulation)
    Listen                 what goes to the speakers (any transmitter or receiver
                           stage); the spectrum plot follows this choice
    Receiver view          time span of the receiver-stage plots
"""

import time

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from scipy import signal as sps

# QShortcut lives in QtGui on Qt6 and QtWidgets on Qt5.
QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut

SAMPLE_RATE = 44100     # audio samples per second (Nyquist = 22.05 kHz)
OUTPUT_GAIN = 0.2       # overall speaker level; keep it modest

# Where to play. Words that must all appear in the device's "name, host API"
# (see `python -m sounddevice`). None = the Windows default output.
OUTPUT_DEVICE = None

# Carrier
START_FC_HZ = 8000.0
MIN_FC_HZ = 10.0
MAX_FC_HZ = 18000.0     # fc + message bandwidth must stay below Nyquist

# Message
MESSAGE_BW_HZ = 3000.0  # voice is low-passed to this (telephone-quality speech)

# Receiver
DETECTOR_BW_HZ = 4000.0 # envelope detector's low-pass cutoff: above the message
                        # band so it doesn't dull the voice a second time
OVERSAMPLE = 8          # the transmitter and detector run at 8 × 44.1 = 352.8 kHz so
                        # the rectifier's carrier harmonics (2fc, 4fc, 6fc …) don't
                        # alias back into the audio band. A real diode is analog and
                        # has no such problem; this makes the simulation behave like it.
MAX_RECORD_S = 30.0
START_M = 0.8           # modulation index

# Display
CARRIER_CYCLES = 3      # carrier plot shows this many cycles, in slow motion
CARRIER_POINTS = 1000
SLOWMO = 0.02           # carrier plot speed (real time would alias at 60 fps)
FFT_N = 8192            # spectrum resolution: 44100 / 8192 ≈ 5.4 Hz per bin
FPS = 60

AB_MODE = "A/B: Original ⇄ Recovered"   # alternates on every pass of the loop
AB_SOURCES = ("Original", "Recovered")
LISTEN_MODES = ["AM signal", "Rectified", "Low-passed", "Demodulated", "Message",
                "Original recording", AB_MODE, "Carrier"]
START_LISTEN = "Original recording"
# Which captured channel each Listen choice corresponds to (for the spectrum).
LISTEN_CHANNEL = {"AM signal": "am", "Rectified": "rectified", "Low-passed": "lowpassed",
                  "Demodulated": "demod", "Message": "message",
                  "Original recording": "original", AB_MODE: "out", "Carrier": "carrier"}
RECEIVER_VIEWS_MS = [2, 5, 20, 50]   # time spans offered for the receiver-stage plots
START_RECEIVER_VIEW_MS = 5


class Carrier:
    """Unit-amplitude sine. Frequency is shared by the GUI and the audio thread."""

    def __init__(self, freq_hz):
        self.freq_hz = freq_hz
        self._phase = 0.0

    def nudge(self, delta_hz):
        self.freq_hz = float(np.clip(self.freq_hz + delta_hz, MIN_FC_HZ, MAX_FC_HZ))

    def render(self, frames, rate=SAMPLE_RATE):
        """Next block at the given sample rate. Phase carries over between blocks."""
        step = 2 * np.pi * self.freq_hz / rate
        block = np.sin(self._phase + step * np.arange(frames))
        self._phase = (self._phase + step * frames) % (2 * np.pi)
        return block


class Looper:
    """Records the message from the microphone and plays it back in a loop.

    The stored clip is band-limited to MESSAGE_BW_HZ and normalised to a peak
    of 1, so that the modulation index m means what the textbook says.
    The unfiltered recording is kept too (same gain), for comparison.
    """

    def __init__(self):
        # (message x(t), original recording, message at OVERSAMPLE× rate) — swapped
        # in as one tuple so the audio thread never sees a half-updated set.
        self._clip = None
        self.pos = 0
        self.lap = 0                # completed passes through the clip
        self.recording = False
        self._chunks = []
        self._in_stream = None

    def start_recording(self):
        self._chunks = []
        self._in_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                         dtype="float32", callback=self._on_input)
        self._in_stream.start()
        self.recording = True

    def _on_input(self, indata, frames, time_info, status):
        self._chunks.append(indata[:, 0].copy())

    def stop_recording(self):
        """Stop capturing. Returns False if the take was silent (and discarded)."""
        if self._in_stream is not None:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None
        self.recording = False

        clip = self.recorded_so_far().astype(np.float64)
        if clip.size < 64:
            return False
        raw = clip - clip.mean()
        lowpass = sps.butter(8, MESSAGE_BW_HZ, fs=SAMPLE_RATE, output="sos")
        filtered = sps.sosfiltfilt(lowpass, raw)
        peak = np.max(np.abs(filtered))
        if peak <= 1e-3:
            return False
        # Same scale factor for both, so the only difference you hear is the filtering.
        sample = (filtered / peak).astype(np.float32)
        original = (raw / peak).astype(np.float32)
        # Precompute the message at the receiver's oversampled rate (it's a loop,
        # so this is done once rather than streamed).
        sample_hi = sps.resample_poly(sample, OVERSAMPLE, 1).astype(np.float32)
        self.pos = 0
        self.lap = 0
        self._clip = (sample, original, sample_hi)
        return True

    @property
    def sample(self):
        """The band-limited message x(t), or None."""
        return None if self._clip is None else self._clip[0]

    def clear(self):
        self._clip = None
        self.pos = 0

    def recorded_so_far(self):
        chunks = list(self._chunks)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def render(self, frames):
        """Next block of (message, original, lap number per sample, message at OVERSAMPLE×).

        The lap number counts complete passes through the clip, so callers can
        tell exactly which sample the loop wrapped on. The oversampled block
        covers the same stretch of time and has frames * OVERSAMPLE samples.
        """
        clip = self._clip
        if clip is None or self.recording:
            silence = np.zeros(frames)
            return silence, silence, np.full(frames, self.lap), np.zeros(frames * OVERSAMPLE)
        sample, original, sample_hi = clip
        n = len(sample)
        ahead = self.pos + np.arange(frames)
        ahead_hi = self.pos * OVERSAMPLE + np.arange(frames * OVERSAMPLE)
        laps = self.lap + ahead // n
        self.lap += (self.pos + frames) // n
        self.pos = (self.pos + frames) % n
        return (sample[ahead % n], original[ahead % n], laps,
                sample_hi[ahead_hi % len(sample_hi)])


class StreamFilter:
    """An SOS filter that keeps its state between blocks (for streaming audio)."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2))

    def __call__(self, block):
        out, self.zi = sps.sosfilt(self.sos, block, zi=self.zi)
        return out


class EnvelopeDetector:
    """Classic AM receiver: full-wave rectify, low-pass, block DC.

    Takes the received signal at OVERSAMPLE × SAMPLE_RATE and returns each
    stage at SAMPLE_RATE. Rectifying and low-passing at the high rate keeps the
    carrier harmonics the rectifier creates (up to many × fc) from aliasing
    into the audio before the low-pass removes them.

    For a full-wave rectified sine the average is (2/pi)·amplitude, so the
    low-passed output is scaled by pi/2 to recover the envelope 1 + m·x(t).
    """

    def __init__(self):
        hi_rate = SAMPLE_RATE * OVERSAMPLE
        # Stage 2: the detector's low-pass. 6th order so the 2fc ripple is
        # well below audibility (≈ -70 dB at 16 kHz for fc = 8 kHz).
        self._lp = StreamFilter(sps.butter(6, DETECTOR_BW_HZ, fs=hi_rate, output="sos"))
        # Only for listening to / plotting stage 1 at 44.1 kHz: anti-alias before
        # dropping samples, otherwise *that* display would be full of aliasing.
        self._rect_aa = StreamFilter(sps.ellip(10, 0.1, 100, 20000, fs=hi_rate, output="sos"))
        # Stage 3: DC block (a real radio uses a series capacitor).
        self._hp = StreamFilter(sps.butter(1, 20.0, btype="highpass",
                                           fs=SAMPLE_RATE, output="sos"))

    def process(self, am_hi):
        """Run one oversampled block through the receiver. Returns every stage at 44.1 kHz."""
        rectified_hi = np.abs(am_hi)                                    # stage 1
        lowpassed = self._lp(rectified_hi)[::OVERSAMPLE] * (np.pi / 2)  # stage 2
        audio = self._hp(lowpassed)                                     # stage 3
        rectified = self._rect_aa(rectified_hi)[::OVERSAMPLE]
        return rectified, lowpassed, audio


class Capture:
    """Ring buffer of every signal the audio thread produced, for the plots to read."""

    CHANNELS = ("carrier", "original", "message", "envelope", "am",
                "rectified", "lowpassed", "demod", "out")

    def __init__(self, size):
        self.size = size
        self.idx = 0
        self.bufs = {name: np.zeros(size) for name in self.CHANNELS}

    def write(self, **blocks):
        n = len(next(iter(blocks.values())))
        where = (self.idx + np.arange(n)) % self.size
        for name, block in blocks.items():
            self.bufs[name][where] = block
        self.idx = (self.idx + n) % self.size

    def last(self, name, n):
        """The most recent n samples of a channel, oldest first."""
        where = (self.idx - n + np.arange(n)) % self.size
        return self.bufs[name][where]


class Transmission:
    """Wires carrier, message, modulator and receiver together for the audio thread."""

    def __init__(self, carrier, looper):
        self.carrier = carrier
        self.looper = looper
        self.detector = EnvelopeDetector()
        self.capture = Capture(FFT_N)
        self.m = START_M
        self.listen = START_LISTEN
        self.ab_playing = AB_SOURCES[0]     # which side of the A/B mode is sounding

    def audio_out(self, outdata, frames, time_info, status):
        # The transmitter runs at OVERSAMPLE × the audio rate so the receiver can
        # rectify without aliasing. Everything we keep is at the audio rate; the
        # AM signal itself is under 22 kHz, so taking every 8th sample is exact.
        c_hi = self.carrier.render(frames * OVERSAMPLE, SAMPLE_RATE * OVERSAMPLE)
        x, original, laps, x_hi = self.looper.render(frames)
        am_hi = (1.0 + self.m * x_hi) * c_hi                # the transmitted signal
        rectified, lowpassed, demod = self.detector.process(am_hi)   # receiver stages
        c = c_hi[::OVERSAMPLE]
        am = am_hi[::OVERSAMPLE]
        envelope = 1.0 + self.m * x

        if self.looper.recording:
            out = np.zeros(frames)              # stay quiet so the mic hears only you
        elif self.listen == AB_MODE:
            # Even laps: the original recording, unfiltered, as captured.
            # Odd laps: the receiver's final output (stage 4, exactly what
            # "Demodulated" plays). Switches on the sample where the loop wraps.
            out = np.where(laps % 2 == 0, original, demod)
            self.ab_playing = AB_SOURCES[int(laps[-1] % 2)]
        else:
            out = {"AM signal": am, "Rectified": rectified, "Low-passed": lowpassed,
                   "Demodulated": demod, "Message": x, "Original recording": original,
                   "Carrier": c}[self.listen]

        self.capture.write(carrier=c, original=original, message=x, envelope=envelope, am=am,
                           rectified=rectified, lowpassed=lowpassed, demod=demod, out=out)
        outdata[:, 0] = np.clip(OUTPUT_GAIN * out, -1.0, 1.0)


class PipelineDiagram(QtWidgets.QWidget):
    """Clickable block diagram of the transmission chain.

    Each block shows a live mini-waveform of the signal leaving it; clicking a
    block plays that signal (it just picks the matching Listen mode).
    """

    # key, title, subtitle, listen mode, captured channel, y-range of the thumbnail
    STAGES = [
        ("mic", "Microphone", "original recording", "Original recording", "original", (-1.5, 1.5)),
        ("band", "Band-limit", f"low-pass {MESSAGE_BW_HZ / 1000:g} kHz → x(t)", "Message",
         "message", (-1.1, 1.1)),
        ("mod", "Modulator", "s(t) = [1 + m·x(t)] · carrier", "AM signal", "am", (-2.4, 2.4)),
        ("rect", "Rectifier", "|s(t)|", "Rectified", "rectified", (-0.1, 2.4)),
        ("lp", "Low-pass", f"{DETECTOR_BW_HZ / 1000:g} kHz, × π/2 → 1 + m·x", "Low-passed",
         "lowpassed", (-0.1, 2.4)),
        ("dc", "DC block", "→ recovered audio m·x(t)", "Demodulated", "demod", (-1.1, 1.1)),
        ("car", "Carrier oscillator", "", "Carrier", "carrier", (-1.2, 1.2)),
    ]
    ROW = ["mic", "band", "mod", "rect", "lp", "dc"]   # left-to-right main chain

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
        margin, gap, top, box_h = 12, 34, 30, 110
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
        edges = np.linspace(0, n, cols + 1).astype(int)
        lo, hi = yrange
        def y_of(v):
            return rect.bottom() - (np.clip(v, lo, hi) - lo) / (hi - lo) * rect.height()
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        if lo < 0 < hi:
            p.drawLine(QtCore.QPointF(rect.left(), y_of(0)), QtCore.QPointF(rect.right(), y_of(0)))
        p.setPen(QtGui.QPen(self.WAVE, 1))
        prev = None
        for i in range(cols):
            seg = data[edges[i]:max(edges[i + 1], edges[i] + 1)]
            x = rect.left() + i
            y1, y2 = y_of(seg.min()), y_of(seg.max())
            p.drawLine(QtCore.QPointF(x, y1), QtCore.QPointF(x, y2))
            if prev is not None:       # join columns so sparse signals read as a line
                p.drawLine(QtCore.QPointF(x - 1, prev), QtCore.QPointF(x, (y1 + y2) / 2))
            prev = (y1 + y2) / 2

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
        for label, keys in (("TRANSMITTER", ["mic", "band", "mod", "car"]),
                            ("RECEIVER", ["rect", "lp", "dc"])):
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
        p.setPen(self.DIM)
        p.setFont(small_font)
        gap_mid = (r["mod"].right() + r["rect"].left()) / 2
        p.drawText(QtCore.QRectF(gap_mid - 40, r["mod"].center().y() - 26, 80, 18),
                   QtCore.Qt.AlignmentFlag.AlignCenter, "s(t)")

        # Blocks.
        solid, outlined = self._playing_keys()
        for key, title, subtitle, mode, channel, yrange in self.STAGES:
            rect = r[key]
            if key == "car":
                subtitle = f"fc = {self.tx.carrier.freq_hz:g} Hz"
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
        self.setWindowTitle("FMSim – AM transmission")
        self.resize(1600, 1150)

        # --- controls row ---
        self.record_btn = QtWidgets.QPushButton("● Record")
        self.stop_btn = QtWidgets.QPushButton("■ Stop")
        self.clear_btn = QtWidgets.QPushButton("Clear")

        self.m_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.m_slider.setRange(0, 150)
        self.m_slider.setValue(int(START_M * 100))
        self.m_slider.setFixedWidth(180)
        self.m_label = QtWidgets.QLabel()

        self.listen_box = QtWidgets.QComboBox()
        self.listen_box.addItems(LISTEN_MODES)
        self.listen_box.setCurrentText(tx.listen)

        self.status = QtWidgets.QLabel()
        self.warning = QtWidgets.QLabel()
        self.warning.setStyleSheet("color: #ff6060; font-weight: bold;")

        for w in (self.record_btn, self.stop_btn, self.clear_btn,
                  self.m_slider, self.listen_box):
            # Keep keyboard focus off the controls so arrow keys stay free.
            w.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        self.record_btn.clicked.connect(self.on_record)
        self.stop_btn.clicked.connect(self.on_stop)
        self.clear_btn.clicked.connect(self.on_clear)
        self.m_slider.valueChanged.connect(self.on_m_changed)
        self.listen_box.currentTextChanged.connect(self.on_listen_changed)

        controls = QtWidgets.QHBoxLayout()
        for w in (self.record_btn, self.stop_btn, self.clear_btn):
            controls.addWidget(w)
        controls.addSpacing(20)
        controls.addWidget(QtWidgets.QLabel("Modulation index"))
        controls.addWidget(self.m_slider)
        controls.addWidget(self.m_label)
        controls.addSpacing(20)
        controls.addWidget(QtWidgets.QLabel("Listen:"))
        controls.addWidget(self.listen_box)
        controls.addSpacing(20)
        controls.addWidget(self.status, stretch=1)

        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems([f"{ms} ms" for ms in RECEIVER_VIEWS_MS])
        self.view_box.setCurrentIndex(RECEIVER_VIEWS_MS.index(START_RECEIVER_VIEW_MS))
        self.view_box.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.view_box.currentIndexChanged.connect(self.on_view_changed)
        controls.insertWidget(controls.count() - 2, QtWidgets.QLabel("Receiver view:"))
        controls.insertWidget(controls.count() - 2, self.view_box)
        controls.insertSpacing(controls.count() - 2, 20)

        # --- plots ---
        # Top row: the source signals and the spectrum.
        # Bottom row: the signal as it passes through each receiver stage.
        g = pg.GraphicsLayoutWidget()
        top = g.addLayout(row=0, col=0)
        bottom = g.addLayout(row=1, col=0)

        self.carrier_plot = make_plot(top, 0, 0, "", "Time", "s", "Amplitude")
        self.carrier_plot.setYRange(-1.2, 1.2)
        self.carrier_curve = self.carrier_plot.plot(pen=pg.mkPen("y", width=2))

        self.msg_plot = make_plot(top, 0, 1, "Message x(t): voice sample, "
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
        self.spec_msg = self.spec_plot.plot(pen=pg.mkPen("c", width=1))
        self.spec_sel = self.spec_plot.plot(pen=pg.mkPen("y", width=1))
        self.fc_line = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 255, 255, 80)))
        self.spec_plot.addItem(self.fc_line)
        self.spec_plot.addItem(pg.InfiniteLine(
            pos=DETECTOR_BW_HZ, angle=90,
            pen=pg.mkPen("g", width=1, style=QtCore.Qt.PenStyle.DashLine)))
        self.freqs = np.fft.rfftfreq(FFT_N, 1 / SAMPLE_RATE)
        self.window = np.hanning(FFT_N)
        self.window_gain = self.window.sum() / 2   # so a unit sine reads 0 dB
        self.spec_msg_db = np.full(len(self.freqs), -120.0)
        self.spec_sel_db = np.full(len(self.freqs), -120.0)

        # Receiver stages. Each plot shows a stage's output (yellow) and, dashed in
        # red, what that stage *should* produce in theory, so you can compare.
        ref_pen = pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine)
        stage_specs = [
            # (title, ylabel, y-range, channel)
            ("① Received s(t)<br><small>red: envelope ±[1 + m·x]</small>",
             "s(t)", (-2.6, 2.6), "am"),
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
        layout.addWidget(self.warning)
        layout.addWidget(self.diagram)
        layout.addWidget(g, stretch=1)

        # --- carrier frequency shortcuts ---
        for keys, delta in (("Up", 100), ("Down", -100),
                            ("Shift+Up", 10), ("Shift+Down", -10),
                            ("PgUp", 1000), ("PgDown", -1000)):
            sc = QShortcut(QtGui.QKeySequence(keys), self)
            sc.activated.connect(lambda d=delta: self.tx.carrier.nudge(d))

        self.message = ""           # error / info text shown when there's no sample
        self.plot_phase = 0.0
        self.last_t = time.perf_counter()
        self.on_m_changed(self.m_slider.value())
        self.refresh_controls()

    # --- control handlers ---
    def on_record(self):
        try:
            self.tx.looper.start_recording()
        except Exception as exc:  # e.g. no microphone available
            self.message = f"Could not open microphone: {exc}"
            return
        self.message = ""
        self.playhead.hide()
        self.refresh_controls()

    def on_stop(self):
        if self.tx.looper.stop_recording():
            self.message = ""
        else:
            self.message = "Recording was silent – check the mic isn't muted"
        self.show_sample()
        self.refresh_controls()

    def on_clear(self):
        self.tx.looper.clear()
        self.msg_curve.setData([], [])
        self.playhead.hide()
        self.refresh_controls()

    def on_m_changed(self, value):
        self.tx.m = value / 100
        self.m_label.setText(f"m = {self.tx.m:.2f}")

    def on_listen_changed(self, mode):
        self.tx.listen = mode

    def on_view_changed(self, index):
        self.view_s = RECEIVER_VIEWS_MS[index] / 1000
        self.stages[0][0].setXRange(-self.view_s, 0, padding=0)
        self.diagram.view_s = self.view_s

    def refresh_controls(self):
        looper = self.tx.looper
        self.record_btn.setEnabled(not looper.recording)
        self.stop_btn.setEnabled(looper.recording)
        self.clear_btn.setEnabled(not looper.recording and looper.sample is not None)

    def show_sample(self):
        sample = self.tx.looper.sample
        if sample is None:
            self.msg_curve.setData([], [])
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
        fc = self.tx.carrier.freq_hz
        looper = self.tx.looper

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
            text = f"Transmitting {len(looper.sample) / SAMPLE_RATE:.1f} s message"
            if self.tx.listen == AB_MODE:
                text += f"   ▶ now playing: {self.tx.ab_playing.upper()}"
            self.status.setText(text)
        else:
            self.status.setText(self.message or "No message – press Record (carrier only)")

        # Receiver stages: the last few ms that actually went through the receiver.
        cap = self.tx.capture
        n = int(self.view_s * SAMPLE_RATE)
        t = (np.arange(n) - n) / SAMPLE_RATE
        env = cap.last("envelope", n)
        theory = {"am": (env, -env),
                  "rectified": (2 / np.pi * env,),
                  "lowpassed": (env,),
                  "demod": (self.tx.m * cap.last("message", n),)}
        for _, channel, curve, refs in self.stages:
            curve.setData(t, cap.last(channel, n))
            for ref, y in zip(refs, theory[channel]):
                ref.setData(t, y)

        # Spectrum of the message vs whatever you're listening to (smoothed a little).
        listen = self.tx.listen
        for channel, store, curve in (("message", self.spec_msg_db, self.spec_msg),
                                      (LISTEN_CHANNEL[listen], self.spec_sel_db, self.spec_sel)):
            mag = np.abs(np.fft.rfft(cap.last(channel, FFT_N) * self.window)) / self.window_gain
            db = 20 * np.log10(mag + 1e-6)
            store[:] = 0.7 * store + 0.3 * db
            curve.setData(self.freqs, store)
        self.spec_plot.setTitle(f"Spectrum: message (cyan) vs {listen} (yellow)<br>"
                                f"<small>white: fc &nbsp; green: detector's "
                                f"{DETECTOR_BW_HZ / 1000:g} kHz low-pass cutoff</small>")
        self.fc_line.setPos(fc)

        self.warning.setText(self.warnings(fc))
        self.diagram.update()

    def warnings(self, fc):
        msgs = []
        if self.tx.m > 1.0:
            msgs.append(f"Overmodulated (m = {self.tx.m:.2f} > 1): the envelope crosses "
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
    pg.mkQApp("FMSim")
    tx = Transmission(Carrier(START_FC_HZ), Looper())
    win = MainWindow(tx)
    win.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(win.tick)
    timer.start(int(1000 / FPS))

    # Output plays in a background thread for as long as the window is open.
    device = find_output_device()
    win.setWindowTitle(f"FMSim – AM transmission  →  {sd.query_devices(device, kind='output')['name']}")
    with sd.OutputStream(device=device, samplerate=SAMPLE_RATE, channels=1,
                         dtype="float32", callback=tx.audio_out):
        pg.exec()

    if tx.looper.recording:
        tx.looper.stop_recording()


if __name__ == "__main__":
    main()
