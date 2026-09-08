#!/usr/bin/env python3
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

"""floppytrack -- make music from 3.5"/5.25" floppy drive sounds.

Usage:
    python3 main.py songs/star_wars_theme.ftk
    python3 main.py songs/star_wars_theme.ftk --no-mp3 --no-play
    python3 main.py --list songs/star_wars_theme.ftk

Reads a song description from a ``.ftk`` file, synthesises the floppy-drive
performance at 48 kHz, crushes it to a retro 8-bit texture, and writes
``output/<name>.wav`` + ``output/<name>.mp3``.  Optionally plays it back
through the Linux audio system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from floppytrack.render import render_song
from floppytrack.songspec import Song, SongFormatError, load_song
from floppytrack.synth import INSTRUMENTS, SAMPLE_RATE

#: Project root = the directory containing this file.
ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "output"
DEFAULT_SONGS = ROOT / "songs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesise floppy-drive chiptune music (.ftk files).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("song", nargs="?",
                        type=Path,
                        help="input song file (.ftk)")
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT,
                        help="directory for .wav/.mp3 output")
    parser.add_argument("--no-mp3", action="store_true",
                        help="skip MP3 encoding (wav only)")
    parser.add_argument("-n", "--no-play", action="store_true",
                        help="do not play the result after rendering")
    parser.add_argument("--list", action="store_true",
                        help="print the song as note events and exit")
    return parser


def print_song(song: Song) -> None:
    """Human-readable dump of a parsed song (useful for checking edits)."""
    n_chords = sum(1 for n in song.notes if n.chord)
    print(f"tempo: {song.tempo} bpm  |  {len(song.notes)} events  |  "
          f"{song.duration_seconds:.2f}s at {SAMPLE_RATE} Hz"
          + (f"  |  {n_chords} polyphonic chord(s)" if n_chords else ""))
    print(f"{'beat':>8}  {'inst':<6} {'beats':>5}  {'pitches':<14} vel")
    for n in song.notes:
        pitches = n.pitch or "rest"
        if n.chord:
            pitches = ",".join([n.pitch or "?"] + list(n.chord))
        print(f"{n.start_beat:8.2f}  {n.instrument:<6} {n.beats:>5.2f}  "
              f"{pitches:<14} {n.velocity:.2f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Default to the bundled Star Wars track when no file is given.
    song_path = args.song
    if song_path is None:
        song_path = DEFAULT_SONGS / "star_wars_theme.ftk"
        if not song_path.exists():
            print("error: no --song given and the default song is missing",
                  file=sys.stderr)
            return 2
    if not song_path.exists():
        print(f"error: song file not found: {song_path}", file=sys.stderr)
        return 2

    try:
        song = load_song(str(song_path))
    except SongFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.list:
        print_song(song)
        return 0

    print(f"synthesising {song_path.name} "
          f"({len(song.notes)} events, {song.tempo} bpm, "
          f"{SAMPLE_RATE} Hz, 8-bit retro crush)...")
    result = render_song(song, args.out_dir, stem=song_path.stem,
                         make_mp3=not args.no_mp3)
    print(f"wrote {result.wav}")
    if result.mp3:
        print(f"wrote {result.mp3}")

    if not args.no_play:
        print("playing...")
        try:
            from floppytrack.render import play_wav
            play_wav(result.wav)
        except RuntimeError as exc:
            print(f"(could not play: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
