"""Tests of the status, reading and warning texts."""

import unittest

from amsim.stages import AB_MODE
from amsim.ui.status import readings_text, status_text, warning_text
from tests.helpers import make, run


class Warnings(unittest.TestCase):
    def test_none_for_a_good_setup(self):
        self.assertEqual(warning_text(make((0, 1, 2))), "")

    def test_overmodulation_starts_above_one(self):
        sim = make()
        sim.selected_station.modulation = 1.0
        self.assertEqual(warning_text(sim), "")
        sim.selected_station.modulation = 1.01
        self.assertIn("Overmodulated (m = 1.01 > 1)", warning_text(sim))

    def test_low_carrier(self):
        sim = make()
        for carrier_hz, expected in ((3000, "at or below the message bandwidth"),
                                     (3010, "carrier ripple leaks"),
                                     (5490, "carrier ripple leaks"),
                                     (5500, "")):
            sim.selected_station.carrier.freq_hz = float(carrier_hz)
            text = warning_text(sim)
            self.assertEqual(text, "") if not expected else self.assertIn(expected, text)

    def test_overlap_needs_both_stations_on_the_air(self):
        sim = make((0, 1))
        first, second = sim.stations[:2]
        second.carrier.freq_hz = first.carrier.freq_hz + 5990
        self.assertIn("Channel 2 is only 5990 Hz away", warning_text(sim))
        second.on_air = False
        self.assertEqual(warning_text(sim), "")
        second.on_air, first.on_air = True, False
        self.assertEqual(warning_text(sim), "")

    def test_neighbours_exactly_far_enough_apart(self):
        sim = make((0, 1))
        sim.stations[1].carrier.freq_hz = sim.stations[0].carrier.freq_hz + 6000
        self.assertEqual(warning_text(sim), "")

    def test_several_at_once(self):
        sim = make((0, 1))
        sim.selected_station.modulation = 1.5
        sim.stations[1].carrier.freq_hz = sim.stations[0].carrier.freq_hz + 1000
        self.assertIn("Overmodulated", warning_text(sim))
        self.assertIn("Channel 2", warning_text(sim))


class Status(unittest.TestCase):
    def test_transmitting(self):
        sim = make()
        self.assertEqual(status_text(sim, ""), "Transmitting 2.0 s message")
        self.assertEqual(status_text(sim, "a.wav"), "Transmitting 2.0 s message (a.wav)")
        sim.selected_station.on_air = False
        self.assertEqual(status_text(sim, ""), "Off air: 2.0 s message")

    def test_no_message(self):
        sim = make(())
        self.assertIn("(carrier only)", status_text(sim, ""))
        sim.selected_station.on_air = False
        self.assertIn("(off air)", status_text(sim, ""))
        self.assertEqual(status_text(sim, "Could not open a.wav"), "Could not open a.wav")

    def test_ab_mode_names_the_side_that_is_playing(self):
        sim = make()
        sim.listen = AB_MODE
        run(sim, 0.1)
        self.assertIn("now playing: ORIGINAL", status_text(sim, ""))
        run(sim, 2.0)       # the message is 2 s long, so this is the second pass
        self.assertIn("now playing: RECOVERED", status_text(sim, ""))

    def test_recording(self):
        sim = make()
        sim.selected_station.recorder.recording = True
        self.assertEqual(status_text(sim, ""), "Recording… 0.0 s (speakers muted)")


class Readings(unittest.TestCase):
    def test_on_and_off_the_air(self):
        sim = make()
        run(sim, 1)
        self.assertRegex(readings_text(sim), r"^Signal / \(noise \+ other stations\): \d+ dB")
        sim.selected_station.on_air = False
        run(sim, 0.1)
        self.assertIn("station is off air", readings_text(sim))
        self.assertRegex(readings_text(sim), r"AGC gain: [+-]\d+ dB$")


if __name__ == "__main__":
    unittest.main()
