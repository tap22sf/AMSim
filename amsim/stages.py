"""The stages a signal passes through, from the recording to the recovered audio.

This one table drives the block diagram and the Listen menu.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from amsim.settings import DETECTOR_BW_HZ, MESSAGE_BW_HZ


@dataclass(frozen=True)
class Stage:
    """One block of the diagram, and the signal that leaves it."""

    key: str            # short name used in the code
    title: str
    subtitle: str       # the diagram replaces the empty ones with a live reading
    listen_name: str    # what the Listen menu calls this signal
    signal: str         # name of the signal in Capture
    y_range: Optional[Tuple[float, float]]  # of the thumbnail. None = scale to fit, for
                                            # signals whose level depends on distance


STAGES = (
    Stage("source", "Source", "original recording",
          listen_name="Original recording", signal="original", y_range=(-1.5, 1.5)),
    Stage("band_limit", "Band-limit", f"low-pass {MESSAGE_BW_HZ / 1000:g} kHz → x(t)",
          listen_name="Message", signal="message", y_range=(-1.1, 1.1)),
    Stage("modulator", "Modulator", "[1 + m·x(t)] · carrier",
          listen_name="AM signal", signal="am", y_range=(-2.4, 2.4)),
    Stage("air", "Air", "all stations + noise",
          listen_name="On the air", signal="air", y_range=None),
    Stage("tuner", "Tuner", "band-pass around fc",
          listen_name="Tuned", signal="tuned", y_range=None),
    Stage("agc", "AGC", "",
          listen_name="After AGC", signal="agc", y_range=(-2.4, 2.4)),
    Stage("rectifier", "Rectifier", "|s(t)|",
          listen_name="Rectified", signal="rectified", y_range=(-0.1, 2.4)),
    Stage("low_pass", "Low-pass", f"{DETECTOR_BW_HZ / 1000:g} kHz → 1 + m·x",
          listen_name="Low-passed", signal="lowpassed", y_range=(-0.1, 2.4)),
    Stage("dc_block", "DC block", "→ audio m·x(t)",
          listen_name="Demodulated", signal="demod", y_range=(-1.1, 1.1)),
    Stage("carrier", "Carrier oscillator", "",
          listen_name="Carrier", signal="carrier", y_range=(-1.2, 1.2)),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}

# Groups of stage keys, in signal-flow order.
MAIN_CHAIN = ("source", "band_limit", "modulator", "air", "tuner", "agc", "rectifier",
              "low_pass", "dc_block")
TRANSMITTER = ("source", "band_limit", "modulator", "carrier")
RECEIVER = ("tuner", "agc", "rectifier", "low_pass", "dc_block")

# A/B mode alternates between two stages on every pass of the loop.
AB_MODE = "A/B: Original ⇄ Recovered"
AB_STAGES = ("source", "dc_block")
AB_SOURCES = ("Original", "Recovered")      # what the status line calls them

_MENU_ORDER = ("modulator", "air", "tuner", "agc", "rectifier", "low_pass", "dc_block",
               "band_limit", "source", AB_MODE, "carrier")


def _listen_signals():
    """Listen menu entry -> the captured signal it plays, in menu order."""
    signals = {}
    for key in _MENU_ORDER:
        if key == AB_MODE:
            signals[AB_MODE] = "out"
        else:
            signals[STAGE_BY_KEY[key].listen_name] = STAGE_BY_KEY[key].signal
    return signals


LISTEN_SIGNAL = _listen_signals()
LISTEN_MODES = list(LISTEN_SIGNAL)
