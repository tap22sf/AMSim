"""The clickable block diagram of the transmission chain."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from amsim.settings import SAMPLE_RATE
from amsim.stages import (AB_MODE, AB_SOURCES, AB_STAGES, MAIN_CHAIN, RECEIVER, STAGE_BY_KEY,
                          STAGES, TRANSMITTER)

# Colours
BACKGROUND = QtGui.QColor(0, 0, 0)
BLOCK = QtGui.QColor(30, 30, 30)
BLOCK_HOVER = QtGui.QColor(48, 48, 48)
EDGE = QtGui.QColor(110, 110, 110)
TEXT = QtGui.QColor(230, 230, 230)
DIM_TEXT = QtGui.QColor(150, 150, 150)
WAVE = QtGui.QColor(0, 220, 220)
ZERO_LINE = QtGui.QColor(60, 60, 60)
PLAYING = QtGui.QColor(255, 220, 0)
GROUP = QtGui.QColor(90, 90, 90)

# Sizes in pixels
MARGIN = 12                 # left and right of the main chain
TOP = 30                    # above the main chain
BLOCK_GAP = 24
BLOCK_HEIGHT = 110
LOWER_ROW_GAP = 34          # between the main chain and the row below it
LOWER_ROW_HEIGHT = 70
MINIMUM_HEIGHT = 250
SUBTITLE_OFFSET = 19        # below the top of the block's contents
THUMBNAIL_OFFSET = 38
THUMBNAIL_MIN_HEIGHT = 10
ARROW_LENGTH, ARROW_HALF_WIDTH = 9, 5
SMALL_FONT_SCALE = 0.85

ALIGN = QtCore.Qt.AlignmentFlag
DASHED = QtCore.Qt.PenStyle.DashLine
NO_FILL = QtCore.Qt.BrushStyle.NoBrush
ANTIALIASING = QtGui.QPainter.RenderHint.Antialiasing


def top_centre(rect):
    return QtCore.QPointF(rect.center().x(), rect.top())


def below(rect):
    """The point just below the middle of the rectangle's bottom edge."""
    return QtCore.QPointF(rect.center().x(), rect.bottom() + 1)


class PipelineDiagram(QtWidgets.QWidget):
    """Clickable block diagram of the transmission chain.

    Each block shows a live mini-waveform of the signal leaving it; clicking a
    block plays that signal (it calls on_select with the matching Listen mode).
    """

    def __init__(self, sim, on_select):
        super().__init__()
        self.sim = sim
        self.on_select = on_select
        self.view_seconds = 0.0
        self.hovered_key = None
        self.rects = {}             # stage key -> where its block is
        self.setMouseTracking(True)
        self.setMinimumHeight(MINIMUM_HEIGHT)

    def set_view(self, seconds):
        """How much of each signal the thumbnails show."""
        self.view_seconds = seconds

    # --- geometry ---
    def _place_blocks(self):
        count = len(MAIN_CHAIN)
        width = (self.width() - 2 * MARGIN - BLOCK_GAP * (count - 1)) / count
        self.rects = {key: QtCore.QRectF(MARGIN + i * (width + BLOCK_GAP), TOP, width, BLOCK_HEIGHT)
                      for i, key in enumerate(MAIN_CHAIN)}
        self.rects["carrier"] = self._rect_below("modulator")

    def _rect_below(self, key):
        """A place in the lower row, under one of the main blocks."""
        block = self.rects[key]
        return QtCore.QRectF(block.left(), block.bottom() + LOWER_ROW_GAP,
                             block.width(), LOWER_ROW_HEIGHT)

    def _stage_at(self, event):
        # Qt6 has event.position(), Qt5 event.pos().
        position = event.position() if hasattr(event, "position") else event.pos()
        for key, rect in self.rects.items():
            if rect.contains(QtCore.QPointF(position)):
                return key
        return None

    # --- mouse ---
    def mouseMoveEvent(self, event):
        key = self._stage_at(event)
        if key != self.hovered_key:
            self.hovered_key = key
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if key
                           else QtCore.Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event):
        self.hovered_key = None

    def mousePressEvent(self, event):
        key = self._stage_at(event)
        if key:
            self.on_select(STAGE_BY_KEY[key].listen_name)

    # --- drawing ---
    def paintEvent(self, event):
        self._place_blocks()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(ANTIALIASING)
        painter.fillRect(self.rect(), BACKGROUND)
        number = self.sim.selected_station.number
        self._draw_group(painter, f"TRANSMITTER – CHANNEL {number}", TRANSMITTER)
        self._draw_group(painter, f"RECEIVER – TUNED TO CHANNEL {number}", RECEIVER)
        self._draw_arrows(painter)
        self._draw_other_stations(painter)
        playing, waiting = self._audible_keys()
        for stage in STAGES:
            self._draw_block(painter, stage, stage.key in playing, stage.key in waiting)
        self._draw_hint(painter)
        painter.end()

    def _small_font(self):
        font = QtGui.QFont(self.font())
        font.setPointSizeF(self.font().pointSizeF() * SMALL_FONT_SCALE)
        return font

    def _title_font(self):
        font = QtGui.QFont(self.font())
        font.setBold(True)
        return font

    def _draw_small_text(self, painter, rect, alignment, text):
        painter.setFont(self._small_font())
        painter.setPen(DIM_TEXT)
        painter.drawText(rect, alignment, text)

    def _draw_group(self, painter, label, keys):
        """A dashed frame around some blocks."""
        frame = self.rects[keys[0]]
        for key in keys[1:]:
            frame = frame.united(self.rects[key])
        frame = frame.adjusted(-7, -22, 7, 7)
        painter.setPen(QtGui.QPen(GROUP, 1, DASHED))
        painter.setBrush(NO_FILL)
        painter.drawRoundedRect(frame, 8, 8)
        self._draw_small_text(painter, frame.adjusted(8, 3, 0, 0), ALIGN.AlignLeft, label)

    def _draw_arrows(self, painter):
        painter.setPen(QtGui.QPen(EDGE, 2))
        for left_key, right_key in zip(MAIN_CHAIN, MAIN_CHAIN[1:]):
            left, right = self.rects[left_key], self.rects[right_key]
            self._draw_arrow(painter, QtCore.QPointF(left.right(), left.center().y()),
                             QtCore.QPointF(right.left() - 1, right.center().y()))
        for key in ("modulator", "air"):
            self._draw_arrow(painter, top_centre(self._rect_below(key)), below(self.rects[key]))

    @staticmethod
    def _draw_arrow(painter, start, end):
        painter.drawLine(start, end)
        line = end - start
        length = max((line.x() ** 2 + line.y() ** 2) ** 0.5, 1e-6)
        along = line / length
        across = QtCore.QPointF(-along.y(), along.x())
        base = end - along * ARROW_LENGTH
        painter.setBrush(painter.pen().color())
        painter.drawPolygon(QtGui.QPolygonF([end, base - across * ARROW_HALF_WIDTH,
                                             base + across * ARROW_HALF_WIDTH]))

    def _draw_other_stations(self, painter):
        """What else is on the air."""
        rect = self._rect_below("air")
        painter.setPen(QtGui.QPen(EDGE, 1, DASHED))
        painter.setBrush(NO_FILL)
        painter.drawRoundedRect(rect, 6, 6)
        lines = []
        for station in self.sim.stations:
            if station is self.sim.selected_station:
                continue
            state = "off air"
            if station.on_air:
                state = f"{station.carrier.freq_hz / 1000:g} kHz, {station.distance_km:.3g} km"
            lines.append(f"CH {station.number}: {state}")
        lines.append(f"Weather: {self.sim.atmosphere.weather:.0%}")
        self._draw_small_text(painter, rect.adjusted(8, 4, -4, -4), ALIGN.AlignVCenter,
                              "\n".join(lines))

    def _audible_keys(self):
        """Stages you can hear: (playing now, waiting for their turn in A/B mode)."""
        if self.sim.listen == AB_MODE:
            playing = AB_STAGES[AB_SOURCES.index(self.sim.ab_playing)]
            return {playing}, set(AB_STAGES) - {playing}
        return {stage.key for stage in STAGES if stage.listen_name == self.sim.listen}, set()

    def _subtitle(self, stage):
        if stage.key == "carrier":
            return f"fc = {self.sim.selected_station.carrier.freq_hz:g} Hz"
        if stage.key == "agc":
            return f"gain {self.sim.agc.gain_db:+d} dB"
        return stage.subtitle

    def _draw_block(self, painter, stage, playing, waiting):
        rect = self.rects[stage.key]
        if playing:
            edge = QtGui.QPen(PLAYING, 3)
        elif waiting:
            edge = QtGui.QPen(PLAYING, 1, DASHED)
        else:
            edge = QtGui.QPen(EDGE, 1)
        painter.setPen(edge)
        painter.setBrush(BLOCK_HOVER if stage.key == self.hovered_key else BLOCK)
        painter.drawRoundedRect(rect, 6, 6)

        contents = rect.adjusted(8, 5, -8, -6)
        painter.setFont(self._title_font())
        painter.setPen(PLAYING if playing else TEXT)
        painter.drawText(contents, ALIGN.AlignLeft | ALIGN.AlignTop,
                         ("▶ " if playing else "") + stage.title)
        self._draw_small_text(painter, contents.adjusted(0, SUBTITLE_OFFSET, 0, 0),
                              ALIGN.AlignLeft | ALIGN.AlignTop, self._subtitle(stage))
        thumbnail = QtCore.QRectF(contents.left(), contents.top() + THUMBNAIL_OFFSET,
                                  contents.width(),
                                  max(contents.height() - THUMBNAIL_OFFSET, THUMBNAIL_MIN_HEIGHT))
        self._draw_thumbnail(painter, thumbnail, stage)

    def _draw_thumbnail(self, painter, rect, stage):
        """Min/max-per-pixel waveform, so dense signals still look right."""
        n = max(int(self.view_seconds * SAMPLE_RATE), 2)
        samples = self.sim.capture.last(stage.signal, n)
        low, high = stage.y_range or self._range_to_fit(samples)

        def to_y(value):
            return rect.bottom() - (np.clip(value, low, high) - low) / (high - low) * rect.height()

        if low < 0 < high:
            painter.setPen(QtGui.QPen(ZERO_LINE, 1))
            painter.drawLine(QtCore.QPointF(rect.left(), to_y(0)),
                             QtCore.QPointF(rect.right(), to_y(0)))

        # One zigzag through every column's minimum and maximum, drawn in a single
        # call: much faster than a line per pixel, and drawing competes with audio.
        columns = max(int(rect.width()), 2)
        starts = np.linspace(0, n, columns, endpoint=False).astype(int)
        ys = np.empty(2 * columns)
        ys[0::2] = to_y(np.minimum.reduceat(samples, starts))
        ys[1::2] = to_y(np.maximum.reduceat(samples, starts))
        xs = rect.left() + np.repeat(np.arange(columns), 2)

        # Width 0 = one device pixel. On high-DPI screens a width of 1 is two pixels,
        # which takes a much slower drawing path.
        painter.setPen(QtGui.QPen(WAVE, 0))
        painter.setBrush(NO_FILL)
        # Smoothing these dense lines is what costs the time, and it barely shows.
        painter.setRenderHint(ANTIALIASING, False)
        painter.drawPath(pg.arrayToQPath(xs, ys))
        painter.setRenderHint(ANTIALIASING, True)

    @staticmethod
    def _range_to_fit(samples):
        peak = max(np.abs(samples).max() * 1.1, 1e-9)
        return -peak, peak

    def _draw_hint(self, painter):
        left, right = self.rects["rectifier"], self.rects["dc_block"]
        rect = QtCore.QRectF(left.left(), self.rects["carrier"].top(),
                             right.right() - left.left(), LOWER_ROW_HEIGHT)
        self._draw_small_text(painter, rect, ALIGN.AlignCenter,
                              "Click any block to listen to the signal leaving it.\n"
                              f"Thumbnails show the last {self.view_seconds * 1000:g} ms "
                              "(Receiver view).")
