# Copyright 2024 Bill Sommers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Retro floppy-drive synthesis engine (numpy, 48 kHz, 8-bit crush).

Each "instrument" emulates one of the audible parts of a floppy drive:

    motor   - the spinning drive motor: a low 50 Hz square hum with a
              higher mechanical whine.  Used for bass / drone lines.
    step    - stepper-motor ticks: a noise burst plus a woody thunk,
              the classic "tick-tick-tick-tick" of a floppy seeking.
    head    - the read/write head: a square-wave blip with a slight low
              pass.  This is the main melodic voice (sounds like a 1982
              sound card playing the drive's own audio).
    beep    - a brighter sine+harmonic blip for counters / arpeggio.
    whir    - a filtered noise sweep for "drive spinning up/down" SFX.

The final mix is pushed through ``retro_fx`` which soft-clips, adds a
little hiss and quantises to 8-bit levels for that unmistakable 1980s
sound-card texture.
"""

from __future__ import annotations

import numpy as np

try:  # scipy makes the low-pass a vectorised lfilter (fast); fall back to a loop
    from scipy.signal import lfilter as _lfilter
except ImportError:  # pragma: no cover
    _lfilter = None

__all__ = [
    "SAMPLE_RATE",
    "INSTRUMENTS",
    "midi_from_notation",
    "frequency_from_notation",
    "FloppySynth",
]

#: Rendering rate. High sample rate = smooth waveforms; the *retro* feel
#: comes from 8-bit quantisation + square waves, not from low-rate audio.
SAMPLE_RATE = 48_000

#: Recognised instrument names (validated by the song parser as well).
#: ``motor`` (== ``m35``) and ``m525`` are the 3.5"/5.25" drive motors;
#: ``step`` (== ``step35``) / ``step525`` are the matching stepper ticks.
INSTRUMENTS = frozenset({"motor", "m35", "m525",
                         "step", "step35", "step525",
                         "head", "beep", "whir"})

#: Voices that sound musical with >1 note at once (polyphonic chords).
CHORD_VOICES = frozenset({"head", "beep"})

_NOTE_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class UnknownInstrumentError(ValueError):
    """Raised when the song file names an instrument we cannot render."""


def midi_from_notation(notation: str) -> int:
    """Parse a note name like ``C4`` / ``A#5`` / ``Bb3`` into a MIDI number.

    C4 == MIDI 60 (middle C) -> frequency 261.63 Hz.
    """
    text = notation.strip().upper()
    if text in ("REST", "R", "-"):
        raise ValueError("not a pitch")
    letter = text[0]
    if letter not in _NOTE_SEMITONES:
        raise ValueError(f"unknown note letter: {notation!r}")
    # Flats survive .upper() as 'B' (e.g. Bb5 -> BB5), sharps stay '#'.
    accidental = text[1] if len(text) > 1 and text[1] in "#B" else ""
    # e.g. "C4", "A#5" -> octave is the digit(s) after the accidental
    rest = text[2:] if accidental else text[1:]
    try:
        octave = int(rest)
    except ValueError as exc:
        raise ValueError(f"bad note name: {notation!r}") from exc
    if not 0 <= octave <= 7:
        raise ValueError(f"octave out of range: {notation!r}")
    semis = _NOTE_SEMITONES[letter]
    if accidental == "#":
        semis += 1
    elif accidental == "B":
        semis = (semis - 1) % 12
    return 12 * (octave + 1) + semis


def frequency_from_notation(notation: str) -> float:
    """Convert a note name (e.g. ``G4``) to Hz. REST is not allowed here."""
    midi = midi_from_notation(notation)
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


# --------------------------------------------------------------------------
# Primitive waveform helpers
# --------------------------------------------------------------------------

def _square(freq: float, n: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    return np.sign(np.sin(2.0 * np.pi * freq * t + phase))


def _saw(freq: float, n: int) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    return 2.0 * ((freq * t) % 1.0) - 1.0


def _env_attack_release(n: int, attack: float, release: float) -> np.ndarray:
    """Linear attack / exponential-ish release envelope, values in [0, 1]."""
    a = np.minimum(1.0, np.arange(n) / max(1, int(attack * SAMPLE_RATE)))
    rel_samples = max(1, int(release * SAMPLE_RATE))
    n_rel = min(n, rel_samples)
    r = np.ones(n)
    if n_rel:
        r[-n_rel:] = np.linspace(1.0, 0.0, n_rel, endpoint=False)
    return a * r


def _lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    """One-pole low-pass -- cheap, gives the muddled sound-card warmth."""
    if x.size == 0:
        return x
    if _lfilter is not None:
        rc = 1.0 / (2.0 * np.pi * cutoff)
        dt = 1.0 / SAMPLE_RATE
        coeff = dt / (rc + dt)
        # First-order IIR: y[n] = coeff*x[n] + (1 - coeff)*y[n-1]
        return _lfilter([coeff], [1.0, -(1.0 - coeff)], x)
    # Pure-python fallback (slow but dependency-free).
    rc = 1.0 / (2.0 * np.pi * cutoff)
    coeff = (1.0 / SAMPLE_RATE) / (rc + (1.0 / SAMPLE_RATE))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + coeff * (x[i] - y[i - 1])
    return y


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

class FloppySynth:
    """Renders individual floppy-drive instrument sounds to float arrays."""

    def __init__(self, rate: int = SAMPLE_RATE) -> None:
        self.rate = rate

    # -- the five voices ----------------------------------------------------

    def motor(self, duration: float, freq: float, big: bool = False) -> np.ndarray:
        """Spinning drive motor: low square hum + mechanical whine.

        ``freq`` is the musical pitch of the note on the line; we derive a
        low motor fundamental (freq/8) so the melody reads as a growly bass.

        Two drive sizes are modelled so a song can mix them freely:

        * 3.5"  (``big=False``) - brisk small motor: higher whine, ~8 Hz
          speed wobble, snappy release.
        * 5.25" (``big=True``)  - lazy big box: even deeper fundamental,
          softer whine, ~5 Hz wobble, more low-end body, longer release.
        """
        n = int(duration * self.rate)
        if big:
            fundamental = max(freq / 16.0, 16.0)   # slower, lazier motor
            whine = fundamental * 2.5              # very soft belt whine
            wobble_hz = 4.5                        # heavy rotor ripple
            release = 0.14
            amp = 0.98
        else:
            fundamental = max(freq / 8.0, 20.0)    # ~motor coil frequency
            whine = fundamental * 4.5              # belt/pulley whine
            wobble_hz = 8.0
            release = 0.08
            amp = 0.9
        body = 0.6 * _square(fundamental, n)
        body += 0.25 * _square(fundamental * 2.0, n)
        body += (0.30 if big else 0.15) * _saw(whine, n)
        if big:  # extra low 'body' of the big box
            body += 0.2 * _saw(fundamental * 0.75, n)
        # Slow amplitude wobble = motor speed ripple (the "wub-wub-wub")
        t = np.arange(n) / self.rate
        wobble = 0.85 + 0.15 * np.sin(2.0 * np.pi * wobble_hz * t)
        env = _env_attack_release(n, attack=0.010, release=release)
        return body * wobble * env * amp

    def step(self, duration: float, velocity: float = 0.8,
             box: bool = False) -> np.ndarray:
        """Stepper-motor tick: the iconic floppy "tick".

        ``box=True`` models a 5.25" drive: the big lid resonates, so the
        thunk is deeper, the click duller and longer.  3.5" stays crisp.
        """
        n = max(int(0.055 * self.rate), 1)
        idx = np.arange(n)
        rng = np.random.default_rng(1337)             # fixed seed = reproducible
        if box:
            dec_t, dec_c, dec_p = 0.006, 0.004, 0.005     # long rings
            thunk = np.exp(-idx / (dec_t * self.rate)) * _saw(60.0, n)
            click = (rng.standard_normal(n) * 0.4) * np.exp(-idx / (dec_c * self.rate))
            ping = _square(1100.0, n) * np.exp(-idx / (dec_p * self.rate))
            lid = np.sin(2.0 * np.pi * 95.0 * idx / self.rate) * np.exp(-idx / (0.012 * self.rate))
            sample = 1.0 * thunk + 0.7 * click + 0.25 * ping + 0.6 * lid
        else:
            thunk = np.exp(-idx / (0.004 * self.rate)) * _saw(95.0, n)
            click = (rng.standard_normal(n) * 0.5) * np.exp(-idx / (0.0025 * self.rate))
            ping = _square(1900.0, n) * np.exp(-idx / (0.003 * self.rate))
            sample = 0.9 * thunk + 0.8 * click + 0.35 * ping
        return sample * velocity * 0.8

    def head(self, duration: float, freq: float) -> np.ndarray:
        """Read/write head blip: square wave, low-passed -- the lead voice.

        Square + low-pass is exactly the 1980s chip-synth lead patch.
        """
        n = int(duration * self.rate)
        f = max(freq, 250.0)                          # keep it floppy, not flute
        wave = _square(f, n) + 0.5 * _square(f * 2.0, n)
        wave = _lowpass(wave, cutoff=min(9500.0, f * 6.0))
        env = _env_attack_release(n, attack=0.004, release=0.045)
        return wave * env * 0.8

    def beep(self, duration: float, freq: float) -> np.ndarray:
        """Brighter counter-melody voice: sine + 2nd harmonic."""
        n = int(duration * self.rate)
        f = max(freq, 300.0)
        t = np.arange(n) / self.rate
        wave = np.sin(2.0 * np.pi * f * t) + 0.4 * np.sin(4.0 * np.pi * f * t)
        env = _env_attack_release(n, attack=0.004, release=0.05)
        return wave * env * 0.6

    def whir(self, duration: float, up: bool = True) -> np.ndarray:
        """Drive spin-up/spin-down: filtered noise sweep + pitch slide."""
        n = int(duration * self.rate)
        t = np.arange(n) / self.rate
        f0, f1 = (120.0, 640.0) if up else (640.0, 120.0)
        freq_slide = f0 + (f1 - f0) * (t / max(duration, 1e-6)) ** 1.5
        wave = np.sin(2.0 * np.pi * np.cumsum(freq_slide) / self.rate)
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(n) * 0.25
        env = _env_attack_release(n, attack=0.02, release=0.09)
        return _lowpass(wave + noise, cutoff=4000.0) * env * 0.7

    # -- generic dispatch ----------------------------------------------------

    def render(self, name: str, duration: float, note: str | None = None,
               velocity: float = 0.8, up: bool = True) -> np.ndarray:
        """Render one instrument event to a float array in [-1, 1]."""
        instr = name.lower()
        if instr not in INSTRUMENTS:
            raise UnknownInstrumentError(name)
        freq = frequency_from_notation(note) if note else None
        pitch = freq if freq else 130.8            # default C3
        if instr in ("motor", "m35"):              # 3.5" drive motor
            return self.motor(duration, pitch, big=False)
        if instr == "m525":                        # 5.25" drive motor
            return self.motor(duration, pitch, big=True)
        if instr == "step" or instr == "step35":
            return self.step(duration, velocity, box=False)
        if instr == "step525":
            return self.step(duration, velocity, box=True)
        if instr == "head":
            return self.head(duration, freq if freq else 440.0)
        if instr == "beep":
            return self.beep(duration, freq if freq else 880.0)
        if instr == "whir":
            return self.whir(duration, up=up) * velocity
        raise UnknownInstrumentError(name)  # pragma: no cover


# --------------------------------------------------------------------------
# Retro post-processing
# --------------------------------------------------------------------------

def retro_fx(mix: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Turn a clean float mix into "1985 PC speaker" texture.

    1. soft-clip with tanh (sound-card style saturation),
    2. add faint tape/driver hiss,
    3. quantise to 8-bit levels,
    4. gentle overall low-pass (~14 kHz) so the top end is soft.
    """
    x = np.tanh(1.5 * mix)                          # 1. saturation
    rng = np.random.default_rng(42)
    x += rng.standard_normal(x.size) * 0.008        # 2. hiss
    x = np.round(x * 127.0) / 127.0                 # 3. 8-bit crush
    x = _lowpass(x, cutoff=14_000)                  # 4. softened highs
    return x


def normalise(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Peak-normalise a mix so it just about fills the bit depth."""
    top = float(np.max(np.abs(x))) if x.size else 0.0
    if top <= 1e-9:
        return x
    return x * (peak / top)
