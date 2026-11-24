"""The texts above the block diagram: status, readings and warnings."""

from amsim.settings import DETECTOR_BW_HZ, MESSAGE_BW_HZ, SAMPLE_RATE
from amsim.stages import AB_MODE


def status_text(sim, note):
    """What the selected station is doing.

    `note` is the file name or error message that belongs to the station, if any.
    """
    station = sim.selected_station
    if station.recorder.recording:
        seconds = len(station.recorder.recorded_so_far()) / SAMPLE_RATE
        return f"Recording… {seconds:.1f} s (speakers muted)"
    message = station.looper.sample
    if message is None:
        return note or ("No message – press Record or Open WAV"
                        + (" (carrier only)" if station.on_air else " (off air)"))
    text = "Transmitting" if station.on_air else "Off air:"
    text += f" {len(message) / SAMPLE_RATE:.1f} s message"
    if note:
        text += f" ({note})"
    if sim.listen == AB_MODE:
        text += f"   ▶ now playing: {sim.ab_playing.upper()}"
    return text


def readings_text(sim):
    """Signal-to-noise ratio and AGC gain."""
    snr_db = sim.snr_db     # read once: the audio thread may change it at any time
    if snr_db is None:
        signal = "Signal / noise: station is off air"
    else:
        signal = f"Signal / (noise + other stations): {snr_db:.0f} dB"
    return f"{signal}     AGC gain: {sim.agc.gain_db:+d} dB"


def warning_text(sim):
    """Anything about the selected station that will spoil reception, or ""."""
    station = sim.selected_station
    warnings = [overmodulation_warning(station), low_carrier_warning(station)]
    warnings += [overlap_warning(station, other) for other in sim.stations
                 if other is not station]
    return "   ".join(warning for warning in warnings if warning)


def overmodulation_warning(station):
    if station.modulation <= 1.0:
        return ""
    return (f"Overmodulated (m = {station.modulation:.2f} > 1): the envelope crosses "
            "zero, so the envelope detector distorts.")


def low_carrier_warning(station):
    fc = station.carrier.freq_hz
    if fc <= MESSAGE_BW_HZ:
        return (f"Carrier ({fc:g} Hz) is at or below the message bandwidth "
                f"({MESSAGE_BW_HZ:g} Hz): the lower sideband folds over and the "
                "receiver can't separate message from carrier.")
    # After rectification the nearest unwanted copy of the message sits at
    # 2fc - bandwidth; it needs to be well above the detector's cutoff.
    if 2 * fc - MESSAGE_BW_HZ < 2 * DETECTOR_BW_HZ:
        return ("Carrier is close to the message bandwidth: carrier ripple leaks "
                "through the detector's low-pass filter.")
    return ""


def overlap_warning(station, other):
    needed = 2 * MESSAGE_BW_HZ
    spacing = abs(other.carrier.freq_hz - station.carrier.freq_hz)
    if not (station.on_air and other.on_air) or spacing >= needed:
        return ""
    return (f"Channel {other.number} is only {spacing:g} Hz away: stations need "
            f"{needed:g} Hz, so their sidebands overlap and the "
            "tuner can't separate them.")
