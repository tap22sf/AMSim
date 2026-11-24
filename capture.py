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


class PcAudio:
    """What the PC is playing, as stereo blocks (WASAPI loopback of the default output)."""

    def __init__(self, audio):
        self._audio = audio         # a pyaudio.PyAudio
        self._speaker = audio.get_default_wasapi_device(d_out=True)
        self._loopback = audio.get_wasapi_loopback_analogue_by_dict(self._speaker)
        self._channels = self._loopback["maxInputChannels"]
        self._blocks = queue.Queue()
        self._streams = contextlib.ExitStack()
        self.name = self._speaker["name"]
        self.rate = int(self._loopback["defaultSampleRate"])

    def __enter__(self):
        with contextlib.ExitStack() as streams:
            # Loopback delivers nothing while the PC is silent, which would cut gaps
            # out of the recording. Playing silence keeps the data flowing.
            streams.enter_context(self._audio.open(
                format=pyaudio.paFloat32, channels=self._speaker["maxOutputChannels"],
                rate=self.rate, output=True, output_device_index=self._speaker["index"],
                frames_per_buffer=BLOCK, stream_callback=self._play_silence))
            streams.enter_context(self._audio.open(
                format=pyaudio.paFloat32, channels=self._channels, rate=self.rate,
                input=True, input_device_index=self._loopback["index"],
                frames_per_buffer=BLOCK, stream_callback=self._on_input))
            self._streams = streams.pop_all()   # both opened: keep them open
        return self

    def __exit__(self, *exc):
        self._streams.close()

    def _play_silence(self, data, frame_count, time_info, status):
        return bytes(frame_count * self._speaker["maxOutputChannels"] * 4), pyaudio.paContinue

    def _on_input(self, data, frame_count, time_info, status):
        self._blocks.put(data)
        return None, pyaudio.paContinue

    def discard(self):
        """Throw away everything that has arrived so far."""
        while not self._blocks.empty():
            self._blocks.get()

    def read(self):
        """The next stereo block. Raises queue.Empty if the device stops delivering."""
        data = self._blocks.get(timeout=NO_AUDIO_TIMEOUT_S)
        return to_stereo(np.frombuffer(data, dtype=np.float32).reshape(-1, self._channels))


class Mixer:
    """Adds the microphone (if any) to the PC audio, keeping the two in step."""

    def __init__(self, mic, rate):
        self._mic = mic
        self._slack = int(SYNC_SLACK_S * rate)
        self._pending = np.zeros((0, 2), dtype=np.float32)  # PC audio waiting for its mic audio
        self.frames = 0             # mixed so far
        self.clipped = 0
        self.pc_peak = 0.0
        self.mic_peak = 0.0

    def mix(self, block, limit=None):
        """Take in a block of PC audio and return what is ready, at most `limit` frames."""
        self._pending = np.concatenate((self._pending, block))
        ready = self._frames_ready()
        if limit is not None:
            ready = min(ready, limit)
        mix, self._pending = self._pending[:ready], self._pending[ready:]
        if ready == 0:
            return mix
        self.pc_peak = max(self.pc_peak, float(np.abs(mix).max()))
        if self._mic:
            voice = self._mic.read(ready)
            self.mic_peak = max(self.mic_peak, float(np.abs(voice).max()))
            mix = mix + voice[:, np.newaxis]
        self.clipped += int((np.abs(mix) > 1.0).any(axis=1).sum())
        self.frames += ready
        return mix

    def _frames_ready(self):
        waiting = len(self._pending)
        if not self._mic:
            return waiting
        # The two devices' clocks differ slightly. Wait for the mic if it is a little
        # behind; past the slack, fill in silence. If it gets ahead, skip its oldest
        # audio. The PC's audio is never altered.
        ready = min(waiting, max(self._mic.available(), waiting - self._slack))
        self._mic.keep_last(waiting + self._slack)
        return ready


def start_together(pc, mic):
    """Wait until the mic is running, then start both sources from the same moment."""
    deadline = time.monotonic() + NO_AUDIO_TIMEOUT_S
    while mic.available() == 0:
        if time.monotonic() > deadline:
            raise RuntimeError(f"no audio arrived from {mic.name}")
        time.sleep(0.005)
    pc.discard()
    mic.keep_last(0)


def to_pcm(mix):
    """Float samples as the bytes of 16-bit ones."""
    return np.round(np.clip(mix, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def record(filename, seconds=None, use_mic=True):
    """Record to filename until `seconds` have been captured, or until Ctrl+C."""
    with pyaudio.PyAudio() as audio:
        pc = PcAudio(audio)
        with pc, (Mic(pc.rate) if use_mic else contextlib.nullcontext()) as mic:
            if mic:
                start_together(pc, mic)
            mixer = Mixer(mic, pc.rate)
            source = pc.name + (f" + {mic.name}" if mic else "")
            length = "until Ctrl+C" if seconds is None else f"for {seconds:g} s"
            with wave.open(filename, "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(pc.rate)
                print(f"Recording {source} ({pc.rate} Hz) {length}...", flush=True)
                limit = None if seconds is None else int(seconds * pc.rate)
                write_until_stopped(wav, pc, mixer, limit)
    print(f"Saved {mixer.frames / pc.rate:.1f} s to {filename}")
    print_levels(mixer, use_mic)


def write_until_stopped(wav, pc, mixer, limit):
    """Write until `limit` frames (None = no limit), Ctrl+C or the device goes quiet."""
    try:
        while limit is None or mixer.frames < limit:
            remaining = None if limit is None else limit - mixer.frames
            mix = mixer.mix(pc.read(), remaining)
            if len(mix):
                wav.writeframes(to_pcm(mix))
    except KeyboardInterrupt:
        pass
    except queue.Empty:
        print("The output device stopped delivering audio.")


def print_levels(mixer, use_mic):
    print(f"Peak level: PC {mixer.pc_peak:.0%}" + (f", mic {mixer.mic_peak:.0%}" if use_mic else ""))
    if use_mic and mixer.mic_peak < 0.001:
        print("The mic was silent - check it isn't muted.")
    if mixer.clipped:
        print(f"Too loud in places: {mixer.clipped} samples clipped. Turn the mic or the PC down.")


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
