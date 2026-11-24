"""Tests that the tables which describe the signals agree with each other."""

import unittest

from amsim.settings import (LAUNCH, LAUNCH_RECEIVER_VIEW_MS, LAUNCH_STATION, MAX_CARRIER_HZ,
                            MAX_KM, MAX_MODULATION, MAX_VOLUME, MIN_CARRIER_HZ, MIN_KM,
                            RECEIVER_VIEWS_MS)
from amsim.simulation import Capture
from amsim.stages import (AB_MODE, AB_STAGES, LISTEN_MODES, LISTEN_SIGNAL, MAIN_CHAIN, RECEIVER,
                          STAGE_BY_KEY, STAGES, TRANSMITTER)


class Stages(unittest.TestCase):
    def test_every_stage_can_be_listened_to(self):
        self.assertEqual(set(LISTEN_MODES), {stage.listen_name for stage in STAGES} | {AB_MODE})

    def test_every_signal_is_captured(self):
        self.assertLessEqual(set(LISTEN_SIGNAL.values()), set(Capture.SIGNALS))

    def test_keys_are_unique_and_groups_use_them(self):
        self.assertEqual(len(STAGE_BY_KEY), len(STAGES))
        for group in (MAIN_CHAIN, TRANSMITTER, RECEIVER, AB_STAGES):
            self.assertLessEqual(set(group), set(STAGE_BY_KEY))


class LaunchSettings(unittest.TestCase):
    def test_within_the_limits_of_the_controls(self):
        for station in LAUNCH.stations:
            self.assertTrue(MIN_CARRIER_HZ <= station.carrier_hz <= MAX_CARRIER_HZ)
            self.assertTrue(0 <= station.modulation <= MAX_MODULATION)
            self.assertTrue(0 <= station.volume <= MAX_VOLUME)
            self.assertTrue(MIN_KM <= station.distance_km <= MAX_KM)
            self.assertTrue(0 <= station.skywave <= 1)
        self.assertTrue(0 <= LAUNCH.weather <= 1)

    def test_choices_exist(self):
        self.assertIn(LAUNCH.listen, LISTEN_MODES)
        self.assertIn(LAUNCH_STATION, range(len(LAUNCH.stations)))
        self.assertIn(LAUNCH_RECEIVER_VIEW_MS, RECEIVER_VIEWS_MS)


if __name__ == "__main__":
    unittest.main()
