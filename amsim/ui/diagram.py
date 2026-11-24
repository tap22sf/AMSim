"""The clickable block diagram of the transmission chain."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from amsim.settings import SAMPLE_RATE
from amsim.stages import (AB_MODE, AB_SOURCES, AB_STAGES, MAIN_CHAIN, RECEIVER, STAGE_BY_KEY,
                          STAGES, TRANSMITTER)


class PipelineDiagram(QtWidgets.QWidget):
    """Clickable block diagram of the transmission chain.

    Each block shows a live mini-waveform of the signal leaving it; clicking a
    block plays that signal (it just picks the matching Listen mode).
    """

    BG = QtGui.QColor(0, 0, 0)
    BOX = QtGui.QColor(30, 30, 30)
    BOX_HOVER = QtGui.QColor(48, 48, 48)
    EDGE = QtGui.QColor(110, 110, 110)
    TEXT = QtGui.QColor(230, 230, 230)
    DIM = QtGui.QColor(150, 150, 150)
    WAVE = QtGui.QColor(0, 220, 220)
    PLAYING = QtGui.QColor(255, 220, 0)
    GROUP = QtGui.QColor(90, 90, 90)

    def __init__(self, sim, on_select):
        super().__init__()
        self.sim = sim
        self.on_select = on_select
        self.view_s = 0.0
        self.hover = None
        self.rects = {}
        self.setMouseTracking(True)
        self.setMinimumHeight(250)

    def set_view(self, seconds):
        """How much of each signal the thumbnails show."""
        self.view_s = seconds

    # --- geometry ---
    def _layout(self):
        w = self.width()
        margin, gap, top, box_h = 12, 24, 30, 110
        cols = len(MAIN_CHAIN)
        box_w = (w - 2 * margin - gap * (cols - 1)) / cols
        self.rects = {}
        for i, key in enumerate(MAIN_CHAIN):
            self.rects[key] = QtCore.QRectF(margin + i * (box_w + gap), top, box_w, box_h)
        mod = self.rects["modulator"]
        self.rects["carrier"] = QtCore.QRectF(mod.left(), mod.bottom() + 34, mod.width(), 70)

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
            self.on_select(STAGE_BY_KEY[key].listen_name)

    # --- drawing ---
    def _playing_keys(self):
        """Stages currently audible: (solid, outlined-only) sets."""
        listen = self.sim.listen
        if listen == AB_MODE:
            now = AB_STAGES[AB_SOURCES.index(self.sim.ab_playing)]
            return {now}, set(AB_STAGES) - {now}
        return {stage.key for stage in STAGES if stage.listen_name == listen}, set()

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

    def _thumbnail(self, p, rect, signal, yrange):
        """Min/max-per-pixel waveform, so dense signals still look right."""
        n = max(int(self.view_s * SAMPLE_RATE), 2)
        data = self.sim.capture.last(signal, n)
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
        # Width 0 = one device pixel. On high-DPI screens a width of 1 is two pixels,
        # which takes a much slower drawing path.
        p.setPen(QtGui.QPen(self.WAVE, 0))
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
        station = self.sim.selected_station
        for label, keys in ((f"TRANSMITTER – CHANNEL {station.number}", TRANSMITTER),
                            (f"RECEIVER – TUNED TO CHANNEL {station.number}", RECEIVER)):
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
        for a, b in zip(MAIN_CHAIN, MAIN_CHAIN[1:]):
            mid_a = QtCore.QPointF(r[a].right(), r[a].center().y())
            mid_b = QtCore.QPointF(r[b].left() - 1, r[b].center().y())
            self._arrow(p, mid_a, mid_b)
        car, mod = r["carrier"], r["modulator"]
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
                 for c in self.sim.stations if c is not station]
        lines.append(f"Weather: {self.sim.atmosphere.weather:.0%}")
        p.setPen(self.DIM)
        p.setFont(small_font)
        p.drawText(others.adjusted(8, 4, -4, -4), QtCore.Qt.AlignmentFlag.AlignVCenter,
                   "\n".join(lines))

        # Blocks.
        solid, outlined = self._playing_keys()
        for stage in STAGES:
            key, subtitle, rect = stage.key, stage.subtitle, r[stage.key]
            if key == "carrier":
                subtitle = f"fc = {station.carrier.freq_hz:g} Hz"
            elif key == "agc":
                subtitle = f"gain {self.sim.agc.gain_db:+d} dB"
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
                       ("▶ " if key in solid else "") + stage.title)
            p.setFont(small_font)
            p.setPen(self.DIM)
            p.drawText(inner.adjusted(0, 19, 0, 0),
                       QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop, subtitle)
            wave = QtCore.QRectF(inner.left(), inner.top() + 38, inner.width(),
                                 max(inner.height() - 38, 10))
            self._thumbnail(p, wave, stage.signal, stage.y_range)

        p.setPen(self.DIM)
        p.setFont(small_font)
        p.drawText(QtCore.QRectF(r["rectifier"].left(), car.top(),
                                 r["dc_block"].right() - r["rectifier"].left(), car.height()),
                   QtCore.Qt.AlignmentFlag.AlignCenter,
                   "Click any block to listen to the signal leaving it.\n"
                   f"Thumbnails show the last {self.view_s * 1000:g} ms (Receiver view).")
        p.end()
