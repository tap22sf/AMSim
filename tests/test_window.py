"""Tests of the controls and the window. They need Qt, but no screen and no audio device."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from amsim.settings import LAUNCH, MAX_KM, MIN_KM
from amsim.simulation import Simulation
from amsim.stages import STAGE_BY_KEY
from amsim.ui.controls import (DISTANCE_SLIDER_STEPS, distance_label, km_to_slider,
                               skywave_label, slider_to_km, weather_label)
from amsim.ui.main_window import MainWindow
from tests.helpers import run, voice


class SliderMaths(unittest.TestCase):
    def test_distance_covers_the_whole_range(self):
        self.assertAlmostEqual(slider_to_km(0), MIN_KM)
        self.assertAlmostEqual(slider_to_km(DISTANCE_SLIDER_STEPS), MAX_KM)

    def test_distance_survives_a_round_trip(self):
        for position in range(DISTANCE_SLIDER_STEPS + 1):
            self.assertEqual(km_to_slider(slider_to_km(position)), position)

    def test_equal_steps_multiply_equally(self):
        self.assertAlmostEqual(slider_to_km(200) / slider_to_km(100),
                               slider_to_km(900) / slider_to_km(800))

    def test_labels(self):
        self.assertEqual([distance_label(km) for km in (1.0, 9.94, 23.4)],
                         ["1.0 km", "9.9 km", "23 km"])
        self.assertEqual([skywave_label(s) for s in (0, 0.3, 1)], ["day", "30% night", "night"])
        self.assertEqual([weather_label(p) for p in (0, 1, 34, 68, 100)],
                         ["0% clear", "1% unsettled", "34% stormy", "68% thunderstorm",
                          "100% thunderstorm"])


class Window(unittest.TestCase):
    def setUp(self):
        self.app = pg.mkQApp()
        self.sim = Simulation(LAUNCH, seed=1)
        self.window = MainWindow(self.sim)
        self.controls = self.window.station_controls

    def tearDown(self):
        self.window.close()

    def test_draws_every_frame_without_errors(self):
        for index, station in enumerate(self.sim.stations):
            station.looper.set_clip(voice(index))
            self.window.select_station(index)
            run(self.sim, 0.1)
            self.window.tick()
            self.assertFalse(self.window.grab().isNull())

    def test_showing_a_station_does_not_change_it(self):
        # Sliders round, so writing their positions back would alter the settings.
        self.sim.stations[1].distance_km = 18.2345
        for index in (1, 0, 1):
            self.window.select_station(index)
        self.assertEqual(self.sim.stations[1].distance_km, 18.2345)

    def test_sliders_change_only_the_selected_station(self):
        self.window.select_station(1)
        self.controls.modulation_slider.setValue(120)
        self.controls.distance_slider.setValue(DISTANCE_SLIDER_STEPS)
        self.assertEqual([s.modulation for s in self.sim.stations], [0.8, 1.2, 0.8])
        self.assertAlmostEqual(self.sim.stations[1].distance_km, MAX_KM)
        self.assertEqual(self.sim.stations[0].distance_km, LAUNCH.stations[0].distance_km)

    def test_shared_controls(self):
        self.window.shared_controls.weather_slider.setValue(75)
        self.assertEqual(self.sim.atmosphere.weather, 0.75)
        self.window.diagram.on_select(STAGE_BY_KEY["tuner"].listen_name)    # as if clicked
        self.assertEqual(self.sim.listen, "Tuned")

    def test_clearing_the_message(self):
        self.sim.selected_station.looper.set_clip(voice(0))
        self.controls.show_station()
        self.assertTrue(self.controls.clear_button.isEnabled())
        self.controls.clear_message()
        self.assertIsNone(self.sim.selected_station.looper.sample)
        self.assertFalse(self.controls.clear_button.isEnabled())

    def test_bad_file_keeps_the_old_message(self):
        self.sim.selected_station.looper.set_clip(voice(0))
        before = self.sim.selected_station.looper.sample
        self.window.open_wav("there is no such file.wav")
        self.assertIn("Could not open", self.controls.note)
        self.assertIs(self.sim.selected_station.looper.sample, before)

    def test_cannot_switch_station_while_recording(self):
        self.sim.selected_station.recorder.recording = True
        self.window.select_station(2)
        self.assertEqual(self.sim.selected_index, 0)

    def test_long_recording_stops_by_itself(self):
        recorder = self.sim.selected_station.recorder
        recorder.recording = True
        recorder._chunks = [np.float32(voice(0, seconds=29.9))]
        self.window.tick()
        self.assertTrue(recorder.recording)
        recorder._chunks.append(np.float32(voice(1, seconds=0.2)))
        self.window.tick()
        self.assertFalse(recorder.recording)
        self.assertIsNotNone(self.sim.selected_station.looper.sample)


if __name__ == "__main__":
    unittest.main()
