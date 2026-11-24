"""The window: it creates the controls, the diagram and the plots, and connects them."""

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets

from amsim.ui.controls import SharedControls, StationControls, StationSelector
from amsim.ui.diagram import PipelineDiagram
from amsim.ui.plots import CarrierPlot, MessagePlot, ReceiverPlots, SpectrumPlot
from amsim.ui.status import readings_text, status_text, warning_text

# QShortcut lives in QtGui on Qt6 and QtWidgets on Qt5.
QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut

CARRIER_KEYS_HZ = {"Up": 100, "Down": -100,
                   "Shift+Up": 10, "Shift+Down": -10,
                   "PgUp": 1000, "PgDown": -1000}


class MainWindow(QtWidgets.QWidget):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.setWindowTitle("AMSim – AM transmission")
        self.resize(1600, 1150)
        self._create_widgets()
        self._lay_out_widgets()
        self._connect_widgets()
        self._add_shortcuts()
        self._set_view(self.shared_controls.view_seconds)
        self.select_station(sim.selected_index)

    def _create_widgets(self):
        self.selector = StationSelector(self.sim)
        self.station_controls = StationControls(self.sim)
        self.shared_controls = SharedControls(self.sim)

        self.status = QtWidgets.QLabel()
        self.readings = QtWidgets.QLabel()
        self.warning = QtWidgets.QLabel()
        self.warning.setStyleSheet("color: #ff6060; font-weight: bold;")
        self.warning.setWordWrap(True)

        # Clicking a block of the diagram picks the matching entry of the Listen menu.
        self.diagram = PipelineDiagram(self.sim, self.shared_controls.choose_listen_mode)

        # Top row: the source signals and the spectrum.
        # Bottom row: the signal as it passes through each receiver stage.
        self.plot_area = pg.GraphicsLayoutWidget()
        top = self.plot_area.addLayout(row=0, col=0)
        bottom = self.plot_area.addLayout(row=1, col=0)
        self.carrier_plot = CarrierPlot(top, col=0)
        self.message_plot = MessagePlot(top, col=1)
        self.spectrum_plot = SpectrumPlot(top, col=2, stations=self.sim.stations)
        self.receiver_plots = ReceiverPlots(bottom)

    def _lay_out_widgets(self):
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.selector)
        controls.addSpacing(8)
        controls.addWidget(self.station_controls, stretch=3)
        controls.addWidget(self.shared_controls, stretch=2)

        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.status, stretch=1)
        status_row.addWidget(self.readings)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(status_row)
        layout.addWidget(self.warning)
        layout.addWidget(self.diagram)
        layout.addWidget(self.plot_area, stretch=1)

    def _connect_widgets(self):
        self.selector.station_clicked.connect(self.select_station)
        self.station_controls.message_changed.connect(self._show_message)
        self.station_controls.recording_changed.connect(self.selector.setDisabled)
        self.shared_controls.view_changed.connect(self._set_view)

    def _add_shortcuts(self):
        for keys, step_hz in CARRIER_KEYS_HZ.items():
            self._add_shortcut(keys, lambda hz=step_hz: self.sim.selected_station.carrier.nudge(hz))
        for index in range(len(self.sim.stations)):
            self._add_shortcut(str(index + 1), lambda i=index: self.select_station(i))

    def _add_shortcut(self, keys, action):
        QShortcut(QtGui.QKeySequence(keys), self).activated.connect(action)

    # --- actions ---
    def select_station(self, index):
        """Point the station controls and the receiver at a station."""
        if self.sim.recording:       # the recording belongs to the current station
            return
        self.sim.selected_index = index
        self.selector.show_selected(index)
        self.station_controls.show_station()

    def open_wav(self, path):
        """Make a WAV file the selected station's message."""
        self.station_controls.open_wav(path)

    def _show_message(self):
        self.message_plot.show_message(self.sim.selected_station)

    def _set_view(self, seconds):
        self.receiver_plots.set_view(seconds)
        self.diagram.set_view(seconds)

    # --- once per frame ---
    def tick(self):
        self.station_controls.stop_recording_if_full()
        self._update_texts()
        station = self.sim.selected_station
        self.carrier_plot.update(station)
        self.message_plot.update(station)
        self.receiver_plots.update(self.sim)
        self.spectrum_plot.update(self.sim)
        self.diagram.update()

    def _update_texts(self):
        self.station_controls.update_carrier_label()
        self.status.setText(status_text(self.sim, self.station_controls.note))
        self.readings.setText(readings_text(self.sim))
        self.warning.setText(warning_text(self.sim))
