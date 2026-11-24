"""AM (amplitude modulation) transmission simulator.

Up to three stations share the air. Each has a recorded voice clip or a WAV file
as its message x(t) and a sine generator as its carrier.

    Transmitter:  s(t) = [1 + m * x(t)] * sin(2*pi*fc*t)
    Air:          all stations, weakened by distance and fading, plus noise
    Receiver:     tuner -> AGC -> envelope detector = rectify -> low-pass -> remove DC

Controls:
    CH 1 / 2 / 3           (or keys 1 - 3) choose the station that the channel
                           controls edit and the receiver is tuned to
  For the chosen channel:
    Up / Down              carrier frequency ±100 Hz
    Shift + Up / Down      carrier frequency ±10 Hz
    PageUp / PageDown      carrier frequency ±1 kHz
    Record / Stop          capture the message from the microphone
    Open WAV…              use a WAV file as the message (or: python main.py a.wav
                           [b.wav [c.wav]] for channels 1 - 3; with no files, the
                           sample recordings go on channels 1 - 3;
                           capture.py records what the PC is playing)
    Clear                  discard the message
    On air                 transmitter on / off (goes on when you give it a message)
    Volume                 speaker level while this channel is chosen (0 - 200 %)
    m slider               modulation index (0 - 1.5; above 1 = overmodulation)
    Distance               1 - 500 km to the receiver
    Skywave                fading from a second path via the ionosphere (day - night)
  For everything:
    Weather                static and lightning (clear - thunderstorm)
    Listen                 what goes to the speakers (any transmitter or receiver
                           stage); the spectrum plot follows this choice
    Receiver view          time span of the receiver-stage plots

The code is in the amsim package: start reading at simulation.py for the signal
processing, or ui/main_window.py for the user interface.
"""

import os
import sys

import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore

from amsim.settings import LAUNCH, LAUNCH_STATION, OUTPUT_DEVICE, SAMPLE_RATE, SAMPLE_WAVS
from amsim.simulation import Simulation
from amsim.ui.main_window import MainWindow

FPS = 60
OUTPUT_LATENCY_S = 0.2  # speaker buffer. The default can be ~12 ms on macOS: too little
                        # when drawing a frame holds up the audio thread
THREAD_SWITCH_S = 0.0005    # Python's default (5 ms) lets drawing hold up the audio thread


def find_output_device():
    """Resolve OUTPUT_DEVICE to an index, falling back to the default output."""
    if OUTPUT_DEVICE is None:
        return None
    try:
        return sd.query_devices(OUTPUT_DEVICE, kind="output")["index"]
    except ValueError as exc:
        print(f"{exc} – using the default output instead")
        return None


def load_messages(window, paths):
    """Give the stations their messages, in order."""
    for index, path in enumerate(paths[:len(window.sim.stations)]):
        # The samples are optional: if one has been deleted, its station just starts empty.
        if path in SAMPLE_WAVS and not os.path.exists(path):
            continue
        window.select_station(index)
        window.open_wav(path)


def run(window):
    """Play and draw until the window is closed."""
    timer = QtCore.QTimer()
    timer.timeout.connect(window.tick)
    timer.start(int(1000 / FPS))

    device = find_output_device()
    device_name = sd.query_devices(device, kind="output")["name"]
    window.setWindowTitle(f"AMSim – AM transmission  →  {device_name}")
    # The output plays in a background thread for as long as the window is open.
    with sd.OutputStream(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                         latency=OUTPUT_LATENCY_S, callback=window.sim.audio_out):
        pg.exec()


def main():
    sys.setswitchinterval(THREAD_SWITCH_S)
    pg.mkQApp("AMSim")
    sim = Simulation(LAUNCH)
    window = MainWindow(sim)
    window.show()

    wav_files = sys.argv[1:]
    load_messages(window, wav_files or SAMPLE_WAVS)
    window.select_station(0 if wav_files else LAUNCH_STATION)
    run(window)

    for station in sim.stations:
        if station.recorder.recording:
            station.recorder.stop()


if __name__ == "__main__":
    main()
