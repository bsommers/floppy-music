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

"""Render a Song to .wav / .mp3 files and play it on Linux.

Pipeline:
    song events  ->  sum into one float buffer (48 kHz)
                   ->  retro_fx (soft-clip, hiss, 8-bit crush)
                   ->  normalise
                   ->  16-bit PCM .wav  (written as int16; the audio is
                        8-bit-crushed so it *sounds* like 8-bit audio)
                   ->  optional .mp3 via ffmpeg or lame
                   ->  optional playback (ffplay / aplay / paplay / mpv)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from .songspec import Song
from .synth import FloppySynth, normalise, retro_fx, SAMPLE_RATE

__all__ = ["render_song", "synthesize", "encode_mp3", "play_wav", "AudioResult"]


class AudioResult:
    """Where the rendered audio ended up."""

    def __init__(self, wav: Path, mp3: Path | None) -> None:
        self.wav = wav
        self.mp3 = mp3


def synthesize(song: Song) -> np.ndarray:
    """Synthesise a song to float samples in [-1, 1] (48 kHz, retro-crushed).

    Used by both :func:`render_song` (to file) and the GUI visualizer
    (for real-time streaming playback with volume / mute / pause).
    """
    synth = FloppySynth()
    total = int(song.duration_seconds * SAMPLE_RATE)
    mix = np.zeros(total, dtype=np.float64)

    # Render and place every note (overlap is fine -- voices sum).
    for note in song.notes:
        duration = max(note.beats * song.beat_seconds * 1.05, 0.02)
        try:
            if note.chord:
                # Polyphonic: render each pitch of the chord at once and
                # sum them, scaling so total loudness stays sensible.
                voices = ([note.pitch] if note.pitch else []) + list(note.chord)
                gain = 1.0 / (len(voices) ** 0.5)
                frag = None
                for voice in voices:
                    a = synth.render(
                        note.instrument,
                        duration=duration,
                        note=voice,
                        velocity=note.velocity,
                    ) * gain
                    frag = a if frag is None else frag + a
            else:
                frag = synth.render(
                    note.instrument,
                    duration=duration,
                    note=note.pitch,
                    velocity=note.velocity,
                )
        except Exception as exc:  # give the user a musical line number
            labels = ([note.pitch] if note.pitch else []) + list(note.chord)
            raise RuntimeError(
                f"failed to render {note.instrument} "
                f"(beat {note.start_beat}, {labels}): {exc}"
            ) from exc
        start = int(note.start_beat * song.beat_seconds * SAMPLE_RATE)
        stop = min(start + frag.size, total)
        if start < total:
            mix[start:stop] += frag[: stop - start]

    # Retro post-processing + normalise to full-scale 8-bit levels.
    mix = retro_fx(mix, SAMPLE_RATE)
    return normalise(mix)


def render_song(song: Song, out_dir: Path, stem: str,
                make_mp3: bool = True) -> AudioResult:
    """Synthesise ``song`` and write ``<stem>.wav`` (and mp3) to ``out_dir``.

    Args:
        song: the parsed song.
        out_dir: directory for output files (created if missing).
        stem: file name stem, e.g. ``star_wars_theme``.
        make_mp3: also encode an .mp3 copy (needs ffmpeg or lame).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mix = synthesize(song)

    # Write 16-bit PCM wav (8-bit-crushed content, big headroom intact).
    wav_path = out_dir / f"{stem}.wav"
    _write_wav(wav_path, mix)

    # 5. Optional mp3 copy.
    mp3_path: Path | None = None
    if make_mp3:
        mp3_path = out_dir / f"{stem}.mp3"
        encode_mp3(wav_path, mp3_path)

    return AudioResult(wav=wav_path, mp3=mp3_path)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    """Write float samples in [-1, 1] as a 16-bit mono PCM wave file."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)                       # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def encode_mp3(wav: Path, mp3: Path) -> None:
    """Encode wav -> mp3 using ffmpeg or lame (320 kbps, CBR)."""
    ffmpeg = shutil.which("ffmpeg")
    lame = shutil.which("lame")
    if not (ffmpeg or lame):
        raise RuntimeError("no MP3 encoder found: install ffmpeg or lame")
    if ffmpeg:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
               "-ar", str(SAMPLE_RATE), "-b:a", "320k", str(mp3)]
    else:  # plain lame
        cmd = ["lame", "--silent", "--aqua", str(wav), str(mp3)]
    subprocess.run(cmd, check=True, capture_output=True)


# Player order = best routing first.  Bare ALSA ``aplay`` is a last resort:
# on this machine ~/.asoundrc sends it to a USB-microphone card, so any
# Pulse-based player (paplay / ffplay / mpv) is preferred.
_PLAYERS = ("paplay", "ffplay", "mpv", "aplay")


def _audio_env() -> dict:
    """Point Pulse users at the desktop session socket (/run/user/1000)."""
    env = dict(os.environ)
    for d in ("/run/user/1000",):
        if os.path.isdir(d):
            env.setdefault("XDG_RUNTIME_DIR", d)
            env.setdefault("PULSE_SERVER", f"unix:{d}/pulse/native")
    return env


def play_wav(path: Path) -> None:
    """Play a wav file through the system audio (first player found wins)."""
    env = _audio_env()
    for player in _PLAYERS:
        exe = shutil.which(player)
        if not exe:
            continue
        if player == "ffplay":
            cmd = [exe, "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
        elif player == "mpv":
            cmd = [exe, "--really-quiet", str(path)]
        elif player == "aplay":
            cmd = [exe, "-q", str(path)]
        else:  # paplay
            cmd = [exe, str(path)]
        try:
            subprocess.run(cmd, check=True, env=env)
            return
        except (subprocess.CalledProcessError, OSError):
            continue  # try the next player
    raise RuntimeError(
        "no working Linux audio player found "
        "(tried paplay, ffplay, mpv, aplay)"
    )
