"""Constants shared by several modules, and the settings the app starts with.

Constants that only one module needs are at the top of that module.
"""

import os
from dataclasses import dataclass
from typing import Tuple

# --- Audio ---
SAMPLE_RATE = 44100     # audio samples per second (Nyquist = 22.05 kHz)

# --- Limits of the controls ---
MAX_VOLUME = 2.0        # top of each station's volume control (1 = normal)
MAX_MODULATION = 1.5    # above 1 the station is overmodulated
MIN_KM, MAX_KM = 1.0, 500.0
MIN_CARRIER_HZ = 10.0
MAX_CARRIER_HZ = 18000.0    # carrier + message bandwidth must stay below Nyquist

# --- Message ---
MESSAGE_BW_HZ = 3000.0  # voice is low-passed to this (telephone-quality speech)
MAX_RECORD_S = 30.0
MAX_WAV_S = 120.0       # longer WAV files are cut to this (the clip is held in memory)

# --- Receiver ---
DETECTOR_BW_HZ = 4000.0     # envelope detector's low-pass cutoff: above the message
                            # band so it doesn't dull the voice a second time

# --- Plots ---
FFT_N = 8192            # capture length = spectrum resolution: ≈ 5.4 Hz per bin
RECEIVER_VIEWS_MS = (2, 5, 20, 50)   # time spans offered for the receiver-stage plots


@dataclass(frozen=True)
class StationSettings:
    """How one station is set up when the app starts."""

    carrier_hz: float
    modulation: float = 0.8         # modulation index m
    volume: float = 1.0             # speaker level while this station is the chosen one
    distance_km: float = MIN_KM
    skywave: float = 0.0            # 0 = day (ground wave only) … 1 = full night


@dataclass(frozen=True)
class LaunchSettings:
    """Everything the user can set, as it is when the app starts."""

    stations: Tuple[StationSettings, ...]
    listen: str = "Original recording"      # a name from the Listen menu (see stages.py)
    weather: float = 0.0                    # 0 = clear … 1 = thunderstorm


# What the app starts with. Change these to change how AMSim looks at launch.
#
# Each station needs 2 × MESSAGE_BW_HZ = 6 kHz of spectrum, so three of them only
# just fit under Nyquist. With these carriers stations 1 and 2 overlap by 150 Hz.
LAUNCH = LaunchSettings(
    stations=(
        StationSettings(carrier_hz=5900.0, distance_km=6.03, skywave=0.1),
        StationSettings(carrier_hz=11750.0, distance_km=18.2, skywave=0.52),
        StationSettings(carrier_hz=18000.0, distance_km=22.9, skywave=0.3),
    ),
    listen="Demodulated",
    weather=0.2,
)
LAUNCH_STATION = 2          # index of the station tuned in at launch. With WAV files on
                            # the command line it is the first station instead.
LAUNCH_RECEIVER_VIEW_MS = 5

# Messages for the stations when no WAV files are given on the command line.
_REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_WAVS = [os.path.join(_REPOSITORY, name)
               for name in ("capture_20261123_121213.wav", "capture_20261123_121839.wav",
                            "recording_20261124.wav")]

# Where to play. Words that must all appear in the device's "name, host API"
# (see `python -m sounddevice`). None = the system's default output.
OUTPUT_DEVICE = None
