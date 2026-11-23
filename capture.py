"""Record whatever this PC is playing, plus the microphone, to a stereo 16-bit WAV file.

    python capture.py                   record until Ctrl+C, to a time-stamped file
    python capture.py music.wav         record until Ctrl+C
    python capture.py music.wav 30      record for 30 seconds
    python capture.py --no-mic ...      leave the microphone out

Windows only. PC sound comes from WASAPI loopback on the default output device.
The default microphone is mixed into both channels. On speakers the mic also
hears the PC's sound; use headphones to avoid that echo.

Needs:  pip install PyAudioWPatch sounddevice
"""

import argparse
import contextlib
import queue
import sys
import time
import wave

import numpy as np
import pyaudiowpatch as pyaudio
import sounddevice as sd

BLOCK = 1024                # frames per audio callback
NO_AUDIO_TIMEOUT_S = 3.0    # give up if the device goes quiet on us (e.g. unplugged)
SYNC_SLACK_S = 0.1          # mic may drift this far from the PC audio before correction


def to_stereo(block):
    """Fold an (n, channels) float block down to (n, 2).

    Surround channels are in the Windows order FL FR FC LFE BL BR SL SR. The LFE
    is dropped, as in the usual stereo downmix. Sound that is only in the front
    pair (nearly all music and video) passes through unchanged.
    """
    channels = block.shape[1]
    if channels == 1:
        return np.repeat(block, 2, axis=1)
    left, right = block[:, 0].copy(), block[:, 1].copy()
    if channels >= 6:
        left += 0.707 * (block[:, 2] + block[:, 4::2].sum(axis=1))
        right += 0.707 * (block[:, 2] + block[:, 5::2].sum(axis=1))
    return np.column_stack((left, right))


class Mic:
    """The default microphone as a mono stream at the given rate, buffered until read.

    Opened through sounddevice because it can ask Windows to convert the sample
    rate, so the mic needn't run at the same rate as the speakers.
    """

    def __init__(self, rate):
        wasapi = next(api for api in sd.query_hostapis() if "WASAPI" in api["name"])
        index = wasapi["default_input_device"]
        if index < 0:
            raise RuntimeError("no microphone found")
        info = sd.query_devices(index)
        self.name = info["name"]
        self._queue = queue.SimpleQueue()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._stream = sd.InputStream(
            samplerate=rate, device=index, channels=min(info["max_input_channels"], 2),
            dtype="float32", extra_settings=sd.WasapiSettings(auto_convert=True),
            callback=lambda data, *_: self._queue.put(data.mean(axis=1)))

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.close()

    def _fill(self):
        while not self._queue.empty():
            self._buffer = np.concatenate((self._buffer, self._queue.get()))

    def available(self):
        self._fill()
        return len(self._buffer)

    def keep_last(self, frames):
        """Throw away all but the newest frames."""
        self._fill()
        self._buffer = self._buffer[max(len(self._buffer) - frames, 0):]

    def read(self, frames):
        """Take this many frames, padded with silence if there aren't enough."""
        out = np.zeros(frames, dtype=np.float32)
        got = self._buffer[:frames]
        out[:len(got)] = got
        self._buffer = self._buffer[frames:]
        return out


def record(filename, seconds=None, use_mic=True):
    """Record to filename until `seconds` have been captured, or until Ctrl+C."""
    blocks = queue.Queue()

    def on_input(data, frame_count, time_info, status):
        blocks.put(data)
        return None, pyaudio.paContinue

    def play_silence(data, frame_count, time_info, status):
        return bytes(frame_count * speaker["maxOutputChannels"] * 4), pyaudio.paContinue

    with pyaudio.PyAudio() as p:
        speaker = p.get_default_wasapi_device(d_out=True)
        loopback = p.get_wasapi_loopback_analogue_by_dict(speaker)
        channels = loopback["maxInputChannels"]
        rate = int(loopback["defaultSampleRate"])
        limit = None if seconds is None else int(seconds * rate)
        slack = int(SYNC_SLACK_S * rate)
        frames = clipped = 0
        pc_peak = mic_peak = 0.0
        pending = np.zeros((0, 2), dtype=np.float32)    # PC audio waiting for its mic audio

        # Loopback delivers nothing while the PC is silent, which would cut gaps
        # out of the recording. Playing silence keeps the data flowing.
        with (p.open(format=pyaudio.paFloat32, channels=speaker["maxOutputChannels"],
                     rate=rate, output=True, output_device_index=speaker["index"],
                     frames_per_buffer=BLOCK, stream_callback=play_silence),
              p.open(format=pyaudio.paFloat32, channels=channels, rate=rate,
                     input=True, input_device_index=loopback["index"],
                     frames_per_buffer=BLOCK, stream_callback=on_input),
              (Mic(rate) if use_mic else contextlib.nullcontext()) as mic):

            if mic:
                # Start both sources from the same moment, once the mic is running.
                deadline = time.monotonic() + NO_AUDIO_TIMEOUT_S
                while mic.available() == 0:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"no audio arrived from {mic.name}")
                    time.sleep(0.005)
                while not blocks.empty():
                    blocks.get()
                mic.keep_last(0)

            with wave.open(filename, "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(rate)

                source = speaker["name"] + (f" + {mic.name}" if mic else "")
                length = "until Ctrl+C" if seconds is None else f"for {seconds:g} s"
                print(f"Recording {source} ({rate} Hz) {length}...", flush=True)
                try:
                    while limit is None or frames < limit:
                        data = blocks.get(timeout=NO_AUDIO_TIMEOUT_S)
                        block = np.frombuffer(data, dtype=np.float32).reshape(-1, channels)
                        pending = np.concatenate((pending, to_stereo(block)))

                        ready = len(pending)
                        if mic:
                            # The two devices' clocks differ slightly. Wait for the mic
                            # if it is a little behind; past the slack, fill in silence.
                            # If it gets ahead, skip its oldest audio. The PC's audio is
                            # never altered.
                            ready = min(ready, max(mic.available(), len(pending) - slack))
                            mic.keep_last(len(pending) + slack)
                        if limit is not None:
                            ready = min(ready, limit - frames)
                        if ready == 0:
                            continue
                        mix, pending = pending[:ready], pending[ready:]
                        pc_peak = max(pc_peak, float(np.abs(mix).max()))
                        if mic:
                            voice = mic.read(ready)
                            mic_peak = max(mic_peak, float(np.abs(voice).max()))
                            mix = mix + voice[:, np.newaxis]
                        clipped += int((np.abs(mix) > 1.0).any(axis=1).sum())
                        pcm = np.round(np.clip(mix, -1.0, 1.0) * 32767)
                        wav.writeframes(pcm.astype("<i2").tobytes())
                        frames += ready
                except KeyboardInterrupt:
                    pass
                except queue.Empty:
                    print("The output device stopped delivering audio.")

    print(f"Saved {frames / rate:.1f} s to {filename}")
    print(f"Peak level: PC {pc_peak:.0%}" + (f", mic {mic_peak:.0%}" if use_mic else ""))
    if use_mic and mic_peak < 0.001:
        print("The mic was silent - check it isn't muted.")
    if clipped:
        print(f"Too loud in places: {clipped} samples clipped. Turn the mic or the PC down.")


def main():
    parser = argparse.ArgumentParser(
        description="Record what this PC is playing, plus the microphone, to a WAV file.")
    parser.add_argument("filename", nargs="?", default=time.strftime("capture_%Y%m%d_%H%M%S.wav"))
    parser.add_argument("seconds", nargs="?", type=float, help="default: until Ctrl+C")
    parser.add_argument("--no-mic", action="store_true", help="record only what the PC is playing")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("seconds must be positive")
    try:
        record(args.filename, args.seconds, use_mic=not args.no_mic)
    except (RuntimeError, OSError, sd.PortAudioError) as exc:
        sys.exit(f"Could not record: {exc}" +
                 ("" if args.no_mic else "\n(--no-mic records without the microphone)"))


if __name__ == "__main__":
    main()
