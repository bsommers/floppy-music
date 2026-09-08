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

"""Song file parsing for floppytrack.

Song files (extension ``.ftk``) are plain text::

    TEMPO 112                 # beats per minute (a beat = one quarter note)
    # comments and blank lines are ignored
    # relative events (played one after another):
    #   INSTRUMENT  BEATS  NOTE [VELOCITY 0.0-1.0]      |  ... NOTE rest ...
    # absolute events (layer voices, e.g. bass under the melody):
    #   INSTRUMENT  AT START_BEAT  BEATS  NOTE [VELOCITY]
    head 0.5 D4
    head 0.5 G4
    step 1 rest
    motor 1 AT 8 4 C3 0.9

Instruments : motor, m35, m525, step, step35, step525, head, beep, whir
              (motor == m35 == 3.5" drive;  m525 == 5.25" drive;
               step == step35;  step525 == 5.25" stepper tick)
BEATS       : note length in beats (fractions allowed, e.g. 0.5 = eighth)
NOTE        : musical pitch like D4 / A#5 / Bb3, or ``rest``
CHORDS      : 2+ pitches on one line = polyphony (head/beep only), e.g.
              ``head AT 8 2 D4 A4 F#5 0.9``
VELOCITY    : optional 0.0-1.0 loudness (default 0.8)

Relative events advance a cursor; ``AT`` events place the note at an
absolute beat position without moving the cursor.  Use ``AT`` to layer
several instruments on the same timeline (melody + bass + ticks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .synth import CHORD_VOICES, INSTRUMENTS, midi_from_notation

__all__ = ["Note", "Song", "load_song"]


@dataclass
class Note:
    """One event in the song: an instrument sounding at a musical instant.

    Attributes:
        instrument: one of ``motor m35 m525 step step35 step525 head beep whir``.
        start_beat: when it starts, in beats from the top of the song.
        beats: how long it lasts, in beats.
        pitch: primary note name (e.g. ``D4``) or ``None`` for ``rest``.
        chord: extra simultaneous pitches (polyphony); first is ``pitch``.
        velocity: loudness 0.0-1.0.
    """

    instrument: str
    start_beat: float
    beats: float
    pitch: str | None
    velocity: float = 0.8
    #: absolute start beat as written with ``AT`` (``None`` = relative)
    absolute: float | None = None
    #: additional simultaneous pitches (polyphony); ``()`` = monophonic
    chord: tuple[str, ...] = ()


@dataclass
class Song:
    """A parsed song: tempo plus a list of note events (in start order)."""

    tempo: float = 120.0
    notes: list[Note] = field(default_factory=list)

    @property
    def beat_seconds(self) -> float:
        """Duration of one beat in seconds (from BPM)."""
        return 60.0 / self.tempo

    @property
    def duration_seconds(self) -> float:
        """Total song length in seconds, including a little tail for releases."""
        if not self.notes:
            return 0.0
        last = max(n.start_beat + n.beats for n in self.notes)
        return last * self.beat_seconds + 0.5      # 0.5 s for release tails

    @property
    def end_beat(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.start_beat + n.beats for n in self.notes)


class SongFormatError(ValueError):
    """Raised for anything we do not understand in a .ftk file."""


# D4, C, F#5, Bb3 ... (case-insensitive).  Deliberately NOT a bare digit.
_NOTE_TOKEN = re.compile(r"[A-Ga-g][#b]?[0-6]")

# Comment start: '#' at line start or preceded by whitespace.
# (IMPORTANT: an in-word '#' is a musical sharp -- e.g. F#5.)
_COMMENT = re.compile(r"^(?=#)|(?<=\s)#")


def _parse_line(line: str, lineno: int, song: Song) -> None:
    """Parse one non-comment line of a song file into ``song``."""
    tokens = line.split()
    if not tokens:
        return
    if tokens[0].upper() == "TEMPO":
        if len(tokens) != 2:
            raise SongFormatError(f"line {lineno}: TEMPO needs one number")
        try:
            song.tempo = float(tokens[1])
        except ValueError as exc:
            raise SongFormatError(f"line {lineno}: bad tempo {tokens[1]!r}") from exc
        if song.tempo <= 0:
            raise SongFormatError(f"line {lineno}: tempo must be > 0")
        return

    instr = tokens[0].lower()
    if instr not in INSTRUMENTS:
        raise SongFormatError(
            f"line {lineno}: unknown instrument {tokens[0]!r} "
            f"(choose from {sorted(INSTRUMENTS)})"
        )
    if len(tokens) < 2:
        raise SongFormatError(
            f"line {lineno}: need at least BEATS, got: {line!r}"
        )
    i = 1
    absolute: float | None = None
    # Optional absolute placement: INSTRUMENT AT <beat> BEATS NOTE [VEL]
    if tokens[i].upper() == "AT":
        try:
            absolute = float(tokens[i + 1])
        except ValueError as exc:
            raise SongFormatError(
                f"line {lineno}: bad AT position {tokens[i+1]!r}"
            ) from exc
        i += 2
        if absolute < 0:
            raise SongFormatError(f"line {lineno}: AT beat must be >= 0")
        if len(tokens) < i + 2:
            raise SongFormatError(
                f"line {lineno}: after AT need BEATS NOTE, got: {line!r}"
            )
    try:
        beats = float(tokens[i])
    except ValueError as exc:
        raise SongFormatError(f"line {lineno}: bad beat length {tokens[i]!r}") from exc
    if beats <= 0:
        raise SongFormatError(f"line {lineno}: beats must be > 0")
    i += 1

    # -- the pitch / chord tokens --------------------------------------
    # One pitch = monophonic.  2+ pitches in a row = a polyphonic chord
    # (head/beep voices only).  Anything else at this position must be
    # the optional trailing velocity.
    if i >= len(tokens):
        raise SongFormatError(f"line {lineno}: missing NOTE after BEATS: {line!r}")

    pitches: list[str] = []
    while i < len(tokens) and _NOTE_TOKEN.fullmatch(tokens[i]):
        try:
            midi_from_notation(tokens[i])
        except ValueError as exc:
            raise SongFormatError(f"line {lineno}: {exc}") from exc
        pitches.append(tokens[i].upper())
        i += 1

    if not pitches:
        if tokens[i].lower() in ("rest", "r", "-"):
            i += 1
        else:
            raise SongFormatError(
                f"line {lineno}: bad note {tokens[i]!r} "
                f"(expected D4, A#5, Bb3, a chord, or 'rest')"
            )

    if len(pitches) > 1 and instr not in CHORD_VOICES:
        raise SongFormatError(
            f"line {lineno}: chords are only supported on {sorted(CHORD_VOICES)}, "
            f"not on {instr!r}"
        )

    velocity = 0.8
    if i < len(tokens):
        try:
            velocity = float(tokens[i])
        except ValueError as exc:
            raise SongFormatError(
                f"line {lineno}: bad velocity {tokens[i]!r}"
            ) from exc
        if not 0.0 < velocity <= 1.0:
            raise SongFormatError(f"line {lineno}: velocity must be in (0, 1]")
        i += 1
    if i < len(tokens):
        raise SongFormatError(f"line {lineno}: unexpected extra tokens: {tokens[i:]!r}")

    pitch = pitches[0] if pitches else None
    chord = tuple(pitches[1:])

    song.notes.append(
        Note(instrument=instr, start_beat=0.0, beats=beats,
             pitch=pitch, velocity=velocity, absolute=absolute,
             chord=chord)
    )


def load_song(path: str) -> Song:
    """Read and validate a ``.ftk`` song file into a :class:`Song`."""
    song = Song()
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            # Strip inline comments.  Careful: '#' inside a note name
            # is a sharp (F#5) -- only '#' at a token boundary comments.
            m = _COMMENT.search(raw)
            line = (raw[: m.start()] if m else raw).strip()
            if line:
                _parse_line(line, lineno, song)

    if not song.notes:
        raise SongFormatError(f"{path}: song has no notes")

    # Assign start beats: relative notes flow back-to-back; AT notes use
    # their absolute position (for layered bass / percussion voices).
    cursor = 0.0
    for note in song.notes:
        if note.absolute is not None:
            note.start_beat = note.absolute
        else:
            note.start_beat = cursor             # relative: follow the cursor
            cursor += note.beats
    return song
