"""The controls at the top of the window."""

import os

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from dsp import load_wav
from amsim.settings import (LAUNCH_RECEIVER_VIEW_MS, MAX_KM, MAX_MODULATION, MAX_RECORD_S,
                            MAX_VOLUME, MAX_WAV_S, MIN_KM, RECEIVER_VIEWS_MS, SAMPLE_RATE)
from amsim.stages import LISTEN_MODES

DISTANCE_SLIDER_STEPS = 1000


def slider_to_km(position):
    """Logarithmic: equal slider steps multiply the distance equally."""
    return MIN_KM * (MAX_KM / MIN_KM) ** (position / DISTANCE_SLIDER_STEPS)


def km_to_slider(km):
    return round(DISTANCE_SLIDER_STEPS * np.log(km / MIN_KM) / np.log(MAX_KM / MIN_KM))


def distance_label(km):
    return f"{km:.1f} km" if km < 10 else f"{km:.0f} km"


def skywave_label(skywave):
    if skywave == 0:
        return "day"
    if skywave == 1:
        return "night"
    return f"{skywave:.0%} night"


def weather_label(percent):
    if percent == 0:
        name = "clear"
    else:
        name = ("unsettled", "stormy", "thunderstorm")[min(percent // 34, 2)]
    return f"{percent}% {name}"


def make_slider(top, width, label_width):
    """A slider from 0 to `top`, and the label that shows its value."""
    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    slider.setRange(0, top)
    slider.setFixedWidth(width)
    label = QtWidgets.QLabel()
    label.setMinimumWidth(label_width)      # so the row doesn't shift as the text changes
    return slider, label


def make_row(*items):
    """Widgets in a line; a string becomes a label, a number becomes a gap."""
    line = QtWidgets.QHBoxLayout()
    for item in items:
        if isinstance(item, int):
            line.addSpacing(item)
        else:
            line.addWidget(QtWidgets.QLabel(item) if isinstance(item, str) else item)
    line.addStretch(1)
    return line


def keep_keyboard_free(*widgets):
    """Keep keyboard focus off the controls, so the arrow keys stay shortcuts."""
    for widget in widgets:
        widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)


class StationSelector(QtWidgets.QWidget):
    """The CH 1 / CH 2 / CH 3 buttons."""

    station_clicked = QtCore.Signal(int)

    def __init__(self, sim):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._buttons = QtWidgets.QButtonGroup(self)
        for index, station in enumerate(sim.stations):
            button = QtWidgets.QPushButton(f"CH {station.number}")
            button.setCheckable(True)
            button.setMinimumHeight(34)
            button.setStyleSheet("QPushButton:checked { background: #ffdc00; color: black; "
                                 "font-weight: bold; }")
            keep_keyboard_free(button)
            self._buttons.addButton(button, index)
            layout.addWidget(button)
        self._buttons.idClicked.connect(self.station_clicked)

    def show_selected(self, index):
        self._buttons.button(index).setChecked(True)


class StationControls(QtWidgets.QGroupBox):
    """Everything that can be set for one station: the selected one."""

    message_changed = QtCore.Signal()           # the station has a different message, or none
    recording_changed = QtCore.Signal(bool)     # recording started (True) or ended (False)

    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self._notes = [""] * len(sim.stations)  # per station: see the note property
        self._create_widgets()
        self._connect_widgets()
        self._lay_out_widgets()

    def _create_widgets(self):
        self.record_button = QtWidgets.QPushButton("● Record")
        self.stop_button = QtWidgets.QPushButton("■ Stop")
        self.open_button = QtWidgets.QPushButton("Open WAV…")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.on_air_box = QtWidgets.QCheckBox("On air")
        self.carrier_label = QtWidgets.QLabel()
        self.volume_slider, self.volume_label = make_slider(round(MAX_VOLUME * 100), 110, 40)
        self.modulation_slider, self.modulation_label = make_slider(
            round(MAX_MODULATION * 100), 130, 60)
        self.distance_slider, self.distance_label = make_slider(DISTANCE_SLIDER_STEPS, 150, 60)
        self.skywave_slider, self.skywave_label = make_slider(100, 130, 80)
        keep_keyboard_free(self.record_button, self.stop_button, self.open_button,
                           self.clear_button, self.on_air_box, self.volume_slider,
                           self.modulation_slider, self.distance_slider, self.skywave_slider)

    def _connect_widgets(self):
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.open_button.clicked.connect(self._choose_wav)
        self.clear_button.clicked.connect(self.clear_message)
        self.on_air_box.toggled.connect(self._on_air_toggled)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.modulation_slider.valueChanged.connect(self._on_modulation_changed)
        self.distance_slider.valueChanged.connect(self._on_distance_changed)
        self.skywave_slider.valueChanged.connect(self._on_skywave_changed)

    def _lay_out_widgets(self):
        rows = QtWidgets.QVBoxLayout(self)
        rows.addLayout(make_row(
            self.record_button, self.stop_button, self.open_button, self.clear_button, 12,
            self.on_air_box, 12,
            "Volume", self.volume_slider, self.volume_label, 12,
            self.carrier_label))
        rows.addLayout(make_row(
            "Modulation", self.modulation_slider, self.modulation_label, 8,
            "Distance", self.distance_slider, self.distance_label, 8,
            "Skywave", self.skywave_slider, self.skywave_label))

    # --- showing the selected station ---
    @property
    def station(self):
        return self.sim.selected_station

    @property
    def note(self):
        """The selected station's file name or error message, for the status line."""
        return self._notes[self.sim.selected_index]

    @note.setter
    def note(self, text):
        self._notes[self.sim.selected_index] = text

    def show_station(self):
        """Call this after another station has been selected."""
        station = self.station
        self.setTitle(f"Channel {station.number}")
        for slider, position in (
                (self.volume_slider, round(station.volume * 100)),
                (self.modulation_slider, round(station.modulation * 100)),
                (self.distance_slider, km_to_slider(station.distance_km)),
                (self.skywave_slider, round(station.skywave * 100))):
            # Show the station's settings without writing (rounded) values back to it.
            slider.blockSignals(True)
            slider.setValue(position)
            slider.blockSignals(False)
        self._show_slider_labels()
        self._show_new_message()

    def update_carrier_label(self):
        self.carrier_label.setText(f"fc = {self.station.carrier.freq_hz:g} Hz  (↑↓ keys)")

    def _show_slider_labels(self):
        station = self.station
        self.volume_label.setText(f"{station.volume:.0%}")
        self.modulation_label.setText(f"m = {station.modulation:.2f}")
        self.distance_label.setText(distance_label(station.distance_km))
        self.skywave_label.setText(skywave_label(station.skywave))

    def _show_new_message(self):
        self.message_changed.emit()
        self._show_available_actions()

    def _show_available_actions(self):
        looper = self.station.looper
        self.record_button.setEnabled(not looper.recording)
        self.stop_button.setEnabled(looper.recording)
        self.open_button.setEnabled(not looper.recording)
        self.clear_button.setEnabled(not looper.recording and looper.sample is not None)
        self.on_air_box.setChecked(self.station.on_air)
        self.recording_changed.emit(looper.recording)

    # --- the station's message ---
    def start_recording(self):
        try:
            self.station.looper.start_recording()
        except Exception as exc:  # e.g. no microphone available
            self.note = f"Could not open microphone: {exc}"
            return
        self.note = ""
        self._show_available_actions()

    def stop_recording(self):
        if self.station.looper.stop_recording():
            self.note = ""
            self.station.on_air = True
        else:
            self.note = "Recording was silent – check the mic isn't muted"
        self._show_new_message()

    def stop_recording_if_full(self):
        looper = self.station.looper
        if looper.recording and len(looper.recorded_so_far()) >= MAX_RECORD_S * SAMPLE_RATE:
            self.stop_recording()

    def _choose_wav(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open message", "", "WAV files (*.wav);;All files (*)")
        if path:
            self.open_wav(path)

    def open_wav(self, path):
        name = os.path.basename(path)
        try:
            clip, truncated = load_wav(path)
        except Exception as exc:  # missing, unreadable or not a WAV file
            self.note = f"Could not open {name}: {exc}"
            return
        if self.station.looper.set_clip(clip):
            self.note = f"{name}, first {MAX_WAV_S:g} s only" if truncated else name
            self.station.on_air = True
        else:
            self.note = f"{name} is silent or too short"
        self._show_new_message()

    def clear_message(self):
        self.station.looper.clear()
        self.note = ""
        self._show_new_message()

    # --- the station's settings ---
    def _on_air_toggled(self, on):
        self.station.on_air = on

    def _on_volume_changed(self, position):
        self.station.volume = position / 100
        self._show_slider_labels()

    def _on_modulation_changed(self, position):
        self.station.modulation = position / 100
        self._show_slider_labels()

    def _on_distance_changed(self, position):
        self.station.distance_km = slider_to_km(position)
        self._show_slider_labels()

    def _on_skywave_changed(self, position):
        self.station.skywave = position / 100
        self._show_slider_labels()


class SharedControls(QtWidgets.QGroupBox):
    """The settings that don't belong to one station."""

    view_changed = QtCore.Signal(float)     # seconds shown in the receiver plots

    def __init__(self, sim):
        super().__init__("Shared – all channels")
        self.sim = sim
        self.listen_box = QtWidgets.QComboBox()
        self.listen_box.addItems(LISTEN_MODES)
        self.listen_box.setCurrentText(sim.listen)
        self.view_box = QtWidgets.QComboBox()
        self.view_box.addItems([f"{ms} ms" for ms in RECEIVER_VIEWS_MS])
        self.view_box.setCurrentIndex(RECEIVER_VIEWS_MS.index(LAUNCH_RECEIVER_VIEW_MS))
        self.weather_slider, self.weather_label = make_slider(100, 150, 120)
        self.weather_slider.setValue(round(sim.atmosphere.weather * 100))
        self._on_weather_changed(self.weather_slider.value())
        keep_keyboard_free(self.listen_box, self.view_box, self.weather_slider)

        self.listen_box.currentTextChanged.connect(self._on_listen_changed)
        self.view_box.currentIndexChanged.connect(self._on_view_changed)
        self.weather_slider.valueChanged.connect(self._on_weather_changed)

        rows = QtWidgets.QVBoxLayout(self)
        rows.addLayout(make_row("Listen:", self.listen_box, 12, "Receiver view:", self.view_box))
        rows.addLayout(make_row("Weather", self.weather_slider, self.weather_label))

    @property
    def view_seconds(self):
        return RECEIVER_VIEWS_MS[self.view_box.currentIndex()] / 1000

    def choose_listen_mode(self, mode):
        self.listen_box.setCurrentText(mode)

    def _on_listen_changed(self, mode):
        self.sim.listen = mode

    def _on_view_changed(self):
        self.view_changed.emit(self.view_seconds)

    def _on_weather_changed(self, percent):
        self.sim.atmosphere.weather = percent / 100
        self.weather_label.setText(weather_label(percent))
