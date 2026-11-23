"""Signal processing for the AM simulator. No GUI in here.

    Channel (×3):  message -> modulator -> path (distance, skywave fading) --+
    Atmosphere:    receiver noise + weather ---------------------------------+-> air
    Receiver:      air -> tuner -> AGC -> rectify -> low-pass -> DC block

Everything runs at SAMPLE_RATE except the rectifier and its low-pass, which run
OVERSAMPLE times faster (see EnvelopeDetector).
"""

import numpy as np
import sounddevice as sd
from scipy import signal as sps
from scipy.io import wavfile

SAMPLE_RATE = 44100     # audio samples per second (Nyquist = 22.05 kHz)
OUTPUT_GAIN = 0.2       # overall speaker level; keep it modest

# Channels. Each needs 2 × MESSAGE_BW_HZ = 6 kHz of spectrum, so three of them
# only just fit under Nyquist: the defaults leave 250 Hz between neighbours.
START_FCS_HZ = (5500.0, 11750.0, 18000.0)
MIN_FC_HZ = 10.0
MAX_FC_HZ = 18000.0     # fc + message bandwidth must stay below Nyquist

# Message
MESSAGE_BW_HZ = 3000.0  # voice is low-passed to this (telephone-quality speech)
MAX_RECORD_S = 30.0
MAX_WAV_S = 120.0       # longer WAV files are cut to this (the clip is held in memory)
START_M = 0.8           # modulation index

# Path: distance
MIN_KM, MAX_KM = 1.0, 500.0
GROUND_LOSS_DB_PER_KM = 0.04    # on top of the 1/d spreading; a simplification of
                                # the real ground-wave curves

# Path: skywave. A second copy arrives via the ionosphere, later and with a
# phase that drifts as the reflecting layer moves.
SKYWAVE_DELAY = round(0.001 * SAMPLE_RATE)  # ≈ 1 ms -> fading notches 1 kHz apart
SKYWAVE_MAX = 0.95              # strength vs the ground wave at "full night"
SKYWAVE_DRIFT_HZ = 0.2          # average phase drift: one fade about every 5 s

# Noise, in the receiver's 6 kHz passband, relative to a carrier 1 km away.
NOISE_FLOOR_DB = -65.0          # the receiver itself, always there
STORM_STATIC_DB = 35.0          # how far a full thunderstorm raises that
STORM_FLASHES_PER_S = 6.0       # lightning at full storm

# Receiver
TUNER_PASS_HZ = 2900.0          # half-width of the tuner's passband …
TUNER_STOP_HZ = 3200.0          # … and where it has fully cut off (≈ -70 dB)
TUNER_LEAK_DB = -50.0           # a real tuner isn't perfect: this much of everything
                                # else on the air gets through anyway
AGC_SECONDS = 0.3
AGC_MAX_GAIN = 2000.0           # 66 dB
AGC_CLIP = 3.0                  # the IF amplifier saturates at 3 × the carrier, so
                                # lightning can't blast the speakers
DETECTOR_BW_HZ = 4000.0 # envelope detector's low-pass cutoff: above the message
                        # band so it doesn't dull the voice a second time
OVERSAMPLE = 8          # the detector runs at 8 × 44.1 = 352.8 kHz so the
                        # rectifier's carrier harmonics (2fc, 4fc, 6fc …) don't
                        # alias back into the audio band. A real diode is analog and
                        # has no such problem; this makes the simulation behave like it.

FFT_N = 8192            # capture length = spectrum resolution: ≈ 5.4 Hz per bin

AB_MODE = "A/B: Original ⇄ Recovered"   # alternates on every pass of the loop
AB_SOURCES = ("Original", "Recovered")
# Listen choice -> the captured signal it plays.
LISTEN_CHANNEL = {"AM signal": "am", "On the air": "air", "Tuned": "tuned", "After AGC": "agc",
                  "Rectified": "rectified", "Low-passed": "lowpassed",
                  "Demodulated": "demod", "Message": "message",
                  "Original recording": "original", AB_MODE: "out", "Carrier": "carrier"}
LISTEN_MODES = list(LISTEN_CHANNEL)
START_LISTEN = "Original recording"


def kaiser_lowpass(pass_hz, stop_hz, stop_db, rate):
    """Linear-phase low-pass. Odd length, so its delay is a whole number of samples."""
    taps, beta = sps.kaiserord(stop_db, (stop_hz - pass_hz) / (rate / 2))
    return sps.firwin(taps | 1, (pass_hz + stop_hz) / 2, window=("kaiser", beta), fs=rate)


def path_gain(km):
    """Amplitude at the receiver relative to the same transmitter 1 km away."""
    loss_db = 20 * np.log10(km / MIN_KM) + GROUND_LOSS_DB_PER_KM * (km - MIN_KM)
    return 10 ** (-loss_db / 20)


class Smooth:
    """Turns a control value into a per-sample ramp, so moving a slider doesn't click."""

    def __init__(self, value):
        self.value = value

    def block(self, target, frames):
        start, self.value = self.value, target
        if start == target:
            return np.full(frames, float(target))
        return start + (target - start) * (np.arange(1, frames + 1) / frames)


class Carrier:
    """Unit-amplitude sine. Frequency is shared by the GUI and the audio thread."""

    def __init__(self, freq_hz):
        self.freq_hz = freq_hz
        self._phase = 0.0

    def nudge(self, delta_hz):
        self.freq_hz = float(np.clip(self.freq_hz + delta_hz, MIN_FC_HZ, MAX_FC_HZ))

    def phases(self, frames):
        """Phase of each sample in the next block. Carries over between blocks."""
        step = 2 * np.pi * self.freq_hz / SAMPLE_RATE
        block = self._phase + step * np.arange(frames)
        self._phase = (self._phase + step * frames) % (2 * np.pi)
        return block


def load_wav(path):
    """Read a WAV file as mono float at SAMPLE_RATE.

    Returns (clip, truncated); at most MAX_WAV_S seconds are kept.
    """
    rate, data = wavfile.read(path)
    keep = int(MAX_WAV_S * rate)
    truncated = len(data) > keep
    data = data[:keep]
    if np.issubdtype(data.dtype, np.integer):
        offset = 128 if data.dtype == np.uint8 else 0       # 8-bit WAVs are unsigned
        data = (data.astype(np.float64) - offset) / 2.0 ** (8 * data.dtype.itemsize - 1)
    else:
        # A NaN or infinity here would stay in the receiver's filters for good.
        data = np.nan_to_num(data.astype(np.float64), posinf=0.0, neginf=0.0)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE and data.size:
        divisor = int(np.gcd(rate, SAMPLE_RATE))
        data = sps.resample_poly(data, SAMPLE_RATE // divisor, rate // divisor)
    return data, truncated


class Looper:
    """Holds a channel's message (recorded or from a file) and plays it back in a loop.

    The stored clip is band-limited to MESSAGE_BW_HZ and normalised to a peak
    of 1, so that the modulation index m means what the textbook says.
    The unfiltered recording is kept too (same gain), for comparison.
    """

    def __init__(self):
        # (message x(t), original recording) — swapped in as one tuple so the
        # audio thread never sees a half-updated set.
        self._clip = None
        self.pos = 0
        self.lap = 0                # completed passes through the clip
        self.recording = False
        self._chunks = []
        self._in_stream = None

    def start_recording(self):
        self._chunks = []
        self._in_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                         dtype="float32", callback=self._on_input)
        self._in_stream.start()
        self.recording = True

    def _on_input(self, indata, frames, time_info, status):
        self._chunks.append(indata[:, 0].copy())

    def stop_recording(self):
        """Stop capturing. Returns False if the take was silent (and discarded)."""
        if self._in_stream is not None:
            self._in_stream.stop()
            self._in_stream.close()
            self._in_stream = None
        self.recording = False
        return self.set_clip(self.recorded_so_far())

    def set_clip(self, clip):
        """Make a mono clip the message. Returns False if it was silent or too short (and ignored)."""
        clip = np.asarray(clip, dtype=np.float64)
        if clip.size < 64:
            return False
        raw = clip - clip.mean()
        # Sharp, because the neighbouring channel starts 250 Hz further up.
        taps = kaiser_lowpass(MESSAGE_BW_HZ - 150, MESSAGE_BW_HZ + 150, 70, SAMPLE_RATE)
        filtered = sps.fftconvolve(raw, taps)[len(taps) // 2:][:len(raw)]  # no delay
        peak = np.max(np.abs(filtered))
        if peak <= 1e-3:
            return False
        # Same scale factor for both, so the only difference you hear is the filtering.
        self.pos = 0
        self.lap = 0
        self._clip = ((filtered / peak).astype(np.float32), (raw / peak).astype(np.float32))
        return True

    @property
    def sample(self):
        """The band-limited message x(t), or None."""
        clip = self._clip
        return None if clip is None else clip[0]

    def clear(self):
        self._clip = None
        self.pos = 0

    def recorded_so_far(self):
        chunks = list(self._chunks)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def render(self, frames):
        """Next block of (message, message SKYWAVE_DELAY earlier, original, lap per sample).

        The lap number counts complete passes through the clip, so callers can
        tell exactly which sample the loop wrapped on.
        """
        clip = self._clip
        if clip is None or self.recording:
            silence = np.zeros(frames)
            return silence, silence, silence, np.full(frames, self.lap)
        sample, original = clip
        n = len(sample)
        ahead = self.pos + np.arange(frames)
        laps = self.lap + ahead // n
        self.lap += (self.pos + frames) // n
        self.pos = (self.pos + frames) % n
        return sample[ahead % n], sample[(ahead - SKYWAVE_DELAY) % n], original[ahead % n], laps


class Channel:
    """One station: its message, its transmitter, and the path to the receiver."""

    def __init__(self, number, freq_hz, rng):
        self.number = number
        self.carrier = Carrier(freq_hz)
        self.looper = Looper()
        self.m = START_M
        self.on_air = False
        self.distance_km = MIN_KM
        self.skywave = 0.0          # 0 = day (ground wave only) … 1 = full night
        self._rng = rng
        self._m = Smooth(START_M)
        self._gain = Smooth(0.0)
        self._sky = Smooth(0.0)
        self._sky_phase = rng.uniform(0, 2 * np.pi)
        self._sky_drift = SKYWAVE_DRIFT_HZ

    def render(self, frames):
        """Next block: dict of the station's signals; "rx" is what reaches the receiver."""
        theta = self.carrier.phases(frames)
        x, x_late, original, laps = self.looper.render(frames)
        m = self._m.block(self.m, frames)
        envelope = 1.0 + m * x
        carrier = np.sin(theta)
        am = envelope * carrier                             # leaves the transmitter

        received = am
        sky = self._sky.block(SKYWAVE_MAX * self.skywave, frames)
        phi = self._drift(frames)
        if sky.any():
            # The late copy. Its carrier phase drifts against the direct one, so
            # the two alternately add up and cancel, at different moments for
            # the carrier and for each part of the sidebands.
            received = am + sky * (1.0 + m * x_late) * np.sin(theta + phi)
        gain = self._gain.block(path_gain(self.distance_km) if self.on_air else 0.0, frames)
        return {"original": original, "message": x, "envelope": envelope, "carrier": carrier,
                "am": am, "laps": laps, "rx": gain * received, "mx": m * x}

    def _drift(self, frames):
        # The drift rate itself wanders slowly, so fades don't come like clockwork.
        dt = frames / SAMPLE_RATE
        pull = min(dt / 4.0, 1.0)
        self._sky_drift += (pull * (SKYWAVE_DRIFT_HZ - self._sky_drift)
                            + 0.08 * np.sqrt(dt) * self._rng.standard_normal())
        step = 2 * np.pi * self._sky_drift / SAMPLE_RATE
        phi = self._sky_phase + step * np.arange(frames)
        self._sky_phase = (self._sky_phase + step * frames) % (2 * np.pi)
        return phi


class Atmosphere:
    """Everything the antenna picks up that isn't a station."""

    def __init__(self, rng):
        self.weather = 0.0          # 0 = clear … 1 = thunderstorm
        self._rng = rng
        self._level = Smooth(self._static_level(0.0))
        self._tail = np.zeros(0)    # lightning that continues into later blocks

    @staticmethod
    def _static_level(weather):
        """RMS of white noise whose share inside the 6 kHz passband hits the target."""
        in_band_db = NOISE_FLOOR_DB + STORM_STATIC_DB * weather
        in_band_power = 0.5 * 10 ** (in_band_db / 10)       # 0.5 = power of a unit carrier
        return np.sqrt(in_band_power * (SAMPLE_RATE / 2) / (2 * MESSAGE_BW_HZ))

    def render(self, frames):
        w = self.weather
        out = self._level.block(self._static_level(w), frames) * self._rng.standard_normal(frames)
        for _ in range(self._rng.poisson(STORM_FLASHES_PER_S * w ** 2 * frames / SAMPLE_RATE)):
            self._add_flash(int(self._rng.integers(frames)), w)
        n = min(frames, len(self._tail))
        out[:n] += self._tail[:n]
        self._tail = self._tail[n:]
        return out

    def _add_flash(self, at, w):
        """One flash = a few strokes, each a burst of noise that dies away."""
        rng = self._rng
        peak_db = -42 + 34 * w + 8 * rng.standard_normal()
        for _ in range(rng.integers(1, 5)):
            decay = rng.uniform(0.003, 0.03) * SAMPLE_RATE
            length = int(6 * decay)
            burst = (10 ** (min(peak_db, 6) / 20) * np.exp(-np.arange(length) / decay)
                     * rng.standard_normal(length))
            if at + length > len(self._tail):
                self._tail = np.pad(self._tail, (0, at + length - len(self._tail)))
            self._tail[at:at + length] += burst
            at += int(rng.uniform(0.03, 0.09) * SAMPLE_RATE)
            peak_db -= rng.uniform(0, 6)


class StreamFilter:
    """An SOS filter that keeps its state between blocks (for streaming audio)."""

    def __init__(self, sos):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2))

    def __call__(self, block):
        out, self.zi = sps.sosfilt(self.sos, block, zi=self.zi)
        return out


class Tuner:
    """Linear-phase band-pass around the selected carrier; rejects other stations.

    Handles several signals at once (one per row). Retuning cross-fades from
    the old filter to the new one over a block.
    """

    def __init__(self, freq_hz, rows):
        self._lowpass = kaiser_lowpass(TUNER_PASS_HZ, TUNER_STOP_HZ, 70, SAMPLE_RATE)
        self.delay = len(self._lowpass) // 2
        self._history = np.zeros((rows, len(self._lowpass) - 1))
        self._freq = freq_hz
        self._taps = self._design(freq_hz)

    def _design(self, freq_hz):
        n = np.arange(len(self._lowpass)) - self.delay
        taps = self._lowpass * 2 * np.cos(2 * np.pi * freq_hz / SAMPLE_RATE * n)
        taps[self.delay] += 10 ** (TUNER_LEAK_DB / 20)
        return taps

    def process(self, block, freq_hz):
        x = np.concatenate([self._history, block], axis=1)
        frames = block.shape[1]
        self._history = x[:, frames:]
        out = self._filter(x)
        if freq_hz != self._freq:
            self._freq = freq_hz
            self._taps = self._design(freq_hz)
            out = out + (self._filter(x) - out) * (np.arange(1, frames + 1) / frames)
        return out

    def _filter(self, x):
        # Not mode="valid": with blocks shorter than the filter that is ambiguous
        # about which input is the longer one.
        full = sps.fftconvolve(x, self._taps[None, :], axes=1)
        return full[:, len(self._taps) - 1:x.shape[1]]


class AGC:
    """Automatic gain control: holds the average carrier level at 1.

    This is why a distant station on a real radio sounds noisy rather than quiet.
    """

    def __init__(self):
        self._level = 0.0
        self.gain = 1.0
        self.jump()

    @property
    def gain_db(self):
        """Current gain, to the nearest whole dB (an int, so never "-0")."""
        return round(20 * float(np.log10(self.gain)))

    def jump(self):
        """Adapt quickly for a moment (after switching stations)."""
        self._fast_left = int(0.08 * SAMPLE_RATE)

    def process(self, block):
        # The average of a rectified sine is 2/pi times its amplitude.
        rect = np.abs(block) * (np.pi / 2)
        fast = min(self._fast_left, len(rect))
        self._fast_left -= fast
        level = np.concatenate([self._follow(rect[:fast], 0.01),
                                self._follow(rect[fast:], AGC_SECONDS)])
        gain = 1.0 / np.maximum(level, 1.0 / AGC_MAX_GAIN)
        self.gain = float(gain[-1])
        return np.clip(block * gain, -AGC_CLIP, AGC_CLIP)

    def _follow(self, rect, seconds):
        if not len(rect):
            return rect
        a = np.exp(-1.0 / (seconds * SAMPLE_RATE))
        out, _ = sps.lfilter([1 - a], [1, -a], rect, zi=[a * self._level])
        self._level = float(out[-1])
        return out


class Upsampler:
    """Streaming OVERSAMPLE× interpolation."""

    def __init__(self):
        # Keep everything up to the highest sideband, remove the copies that
        # start at the mirror image of it (44.1 kHz minus that).
        top = MAX_FC_HZ + TUNER_STOP_HZ
        taps = kaiser_lowpass(top, SAMPLE_RATE - top, 80, SAMPLE_RATE * OVERSAMPLE)
        # Pad so the delay is a whole number of 44.1 kHz samples.
        half = -(-(len(taps) // 2) // OVERSAMPLE) * OVERSAMPLE
        self._taps = np.pad(taps, half - len(taps) // 2) * OVERSAMPLE
        self.delay = half // OVERSAMPLE
        self._history = np.zeros(2 * self.delay)

    def process(self, block):
        x = np.concatenate([self._history, block])
        self._history = x[len(block):]
        start = len(self._taps) - 1
        return sps.upfirdn(self._taps, x, up=OVERSAMPLE)[start:start + len(block) * OVERSAMPLE]


class EnvelopeDetector:
    """Classic AM detector: full-wave rectify, low-pass, block DC.

    Rectifying and low-passing at OVERSAMPLE × SAMPLE_RATE keeps the carrier
    harmonics the rectifier creates (up to many × fc) from aliasing into the
    audio before the low-pass removes them.

    For a full-wave rectified sine the average is (2/pi)·amplitude, so the
    low-passed output is scaled by pi/2 to recover the envelope 1 + m·x(t).
    """

    def __init__(self):
        hi_rate = SAMPLE_RATE * OVERSAMPLE
        self._up = Upsampler()
        # Stage 2: the detector's low-pass. 8th order so the 2fc ripple is well
        # below audibility even for the lowest channel (≈ -70 dB at 11 kHz).
        self._lp = StreamFilter(sps.butter(8, DETECTOR_BW_HZ, fs=hi_rate, output="sos"))
        # Only for listening to / plotting stage 1 at 44.1 kHz: anti-alias before
        # dropping samples, otherwise *that* display would be full of aliasing.
        self._rect_aa = StreamFilter(sps.ellip(10, 0.1, 100, 20000, fs=hi_rate, output="sos"))
        # Stage 3: DC block (a real radio uses a series capacitor).
        self._hp = StreamFilter(sps.butter(1, 20.0, btype="highpass",
                                           fs=SAMPLE_RATE, output="sos"))
        # Delays (in 44.1 kHz samples), so the plots can line theory up with practice.
        self.rect_delay = self._up.delay + self._delay_of(self._rect_aa.sos, hi_rate)
        self.lp_delay = self._up.delay + self._delay_of(self._lp.sos, hi_rate)

    @staticmethod
    def _delay_of(sos, rate):
        f = 500.0       # low enough to be in every passband; delay is flat there
        w = np.exp(-2j * np.pi * f / rate * np.arange(3))
        h = np.prod((sos[:, :3] @ w) / (sos[:, 3:] @ w))
        return round(-np.angle(h) / (2 * np.pi * f) * SAMPLE_RATE)

    def process(self, am):
        """Run one block through the detector. Returns every stage at 44.1 kHz."""
        rectified_hi = np.abs(self._up.process(am))                     # stage 1
        lowpassed = self._lp(rectified_hi)[::OVERSAMPLE] * (np.pi / 2)  # stage 2
        audio = self._hp(lowpassed)                                     # stage 3
        rectified = self._rect_aa(rectified_hi)[::OVERSAMPLE]
        return rectified, lowpassed, audio


class Capture:
    """Ring buffer of every signal the audio thread produced, for the plots to read."""

    CHANNELS = ("carrier", "original", "message", "envelope", "mx", "am", "air", "tuned",
                "agc", "rectified", "lowpassed", "demod", "out")

    def __init__(self, size):
        self.size = size
        self.idx = 0
        self.bufs = {name: np.zeros(size) for name in self.CHANNELS}

    def write(self, **blocks):
        n = len(next(iter(blocks.values())))
        where = (self.idx + np.arange(n)) % self.size
        for name, block in blocks.items():
            self.bufs[name][where] = block
        self.idx = (self.idx + n) % self.size

    def last(self, name, n, delay=0):
        """n samples of a signal, oldest first, ending `delay` samples ago."""
        where = (self.idx - delay - n + np.arange(n)) % self.size
        return self.bufs[name][where]


class Transmission:
    """Wires the stations, the air and the receiver together for the audio thread."""

    def __init__(self, seed=None):
        rng = np.random.default_rng(seed)
        self.channels = [Channel(i + 1, fc, rng) for i, fc in enumerate(START_FCS_HZ)]
        self.channels[0].on_air = True
        self.atmosphere = Atmosphere(rng)
        self.tuner = Tuner(START_FCS_HZ[0], rows=2)
        self.agc = AGC()
        self.detector = EnvelopeDetector()
        self.capture = Capture(FFT_N)
        self.selected = 0                   # channel being edited and received
        self.listen = START_LISTEN
        self.ab_playing = AB_SOURCES[0]     # which side of the A/B mode is sounding
        self.snr_db = None                  # wanted station vs everything else, after the tuner
        self._tuned_to = 0
        self._powers = None
        # How far each receiver stage lags the transmitter, for the plots.
        self.delays = {"agc": self.tuner.delay,
                       "rectified": self.tuner.delay + self.detector.rect_delay,
                       "lowpassed": self.tuner.delay + self.detector.lp_delay,
                       "demod": self.tuner.delay + self.detector.lp_delay}

    @property
    def channel(self):
        return self.channels[self.selected]

    @property
    def recording(self):
        return any(ch.looper.recording for ch in self.channels)

    def process(self, frames):
        """Simulate one block; returns what goes to the speakers (before gain)."""
        selected = self.selected            # the GUI may change these at any time
        listen = self.listen
        if selected != self._tuned_to:
            self._tuned_to = selected
            self._powers = None
            self.agc.jump()

        signals = [ch.render(frames) for ch in self.channels]
        sel = signals[selected]
        wanted = sel["rx"]
        unwanted = self.atmosphere.render(frames)
        for i, s in enumerate(signals):
            if i != selected:
                unwanted = unwanted + s["rx"]

        # Wanted and unwanted go through the tuner separately only so the SNR can
        # be measured; the receiver itself uses their sum.
        parts = self.tuner.process(np.stack([wanted, unwanted]),
                                   self.channels[selected].carrier.freq_hz)
        tuned = parts.sum(axis=0)
        self._measure_snr(parts, self.channels[selected].on_air)
        agc = self.agc.process(tuned)
        rectified, lowpassed, demod = self.detector.process(agc)

        blocks = {name: sel[name] for name in ("carrier", "original", "message", "envelope",
                                               "mx", "am")}
        blocks.update(air=wanted + unwanted, tuned=tuned, agc=agc, rectified=rectified,
                      lowpassed=lowpassed, demod=demod)
        if self.recording:
            out = np.zeros(frames)              # stay quiet so the mic hears only you
        elif listen == AB_MODE:
            # Even laps: the original recording, unfiltered, as captured.
            # Odd laps: the receiver's final output (exactly what "Demodulated"
            # plays). Switches on the sample where the loop wraps.
            laps = sel["laps"]
            out = np.where(laps % 2 == 0, sel["original"], demod)
            self.ab_playing = AB_SOURCES[int(laps[-1] % 2)]
        else:
            out = blocks[LISTEN_CHANNEL[listen]]
        self.capture.write(out=out, **blocks)
        return out

    def _measure_snr(self, parts, on_air):
        power = (parts ** 2).mean(axis=1)
        if self._powers is None:
            self._powers = power
        keep = np.exp(-parts.shape[1] / SAMPLE_RATE / 0.5)
        self._powers = keep * self._powers + (1 - keep) * power
        w, u = self._powers
        self.snr_db = 10 * np.log10(max(w, 1e-20) / max(u, 1e-20)) if on_air else None

    def audio_out(self, outdata, frames, time_info, status):
        outdata[:, 0] = np.clip(OUTPUT_GAIN * self.process(frames), -1.0, 1.0)
