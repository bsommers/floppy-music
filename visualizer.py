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

"""floppytrack visualizer -- a modern floppy-drive music player & EQ.

Run with plain Python (uses the *built-in* Tk toolkit, nothing to install):

    python3 visualizer.py                # loads songs/ directory
    python3 visualizer.py songs/foo.ftk  # start with a specific song

Features:
    * live animated graphic equalizer (FFT spectrum of the playing audio)
    * play / pause / stop transport
    * volume slider + mute / unmute button
    * clean dark theme in blues, greys and black

Playback streams 48 kHz synthesized audio to ``aplay`` in a background
thread, which lets us apply volume / mute / pause in real time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from tkinter import messagebox

from floppytrack.render import synthesize
from floppytrack.songspec import Song
from floppytrack.synth import SAMPLE_RATE

# --------------------------------------------------------------------------
# Palette -- dark, modern, blues / greys / blacks
# --------------------------------------------------------------------------
C_BG        = "#0A0E14"   # app background          (near-black, blue tint)
C_PANEL     = "#10161F"   # panels / buttons        (dark grey-blue)
C_PANEL_HI  = "#1C2634"   # hover / active panels
C_TRACK     = "#151C27"   # equalizer bar tracks    (subtle grey)
C_BAR       = "#2563EB"   # equalizer bars          (blue)
C_BAR_CAP   = "#60A5FA"   # bar top caps            (light blue)
C_TEXT      = "#E6EAF2"   # primary text            (off-white)
C_TEXT_DIM  = "#8A97A9"   # secondary text          (cool grey)
C_ACCENT    = "#1D4ED8"   # primary action buttons  (strong blue)
C_ACCENT_HI = "#2E5FE8"
C_BORDER    = "#1E2836"   # hairline borders

FONT        = ("DejaVu Sans", 10)
FONT_TITLE  = ("DejaVu Sans", 15, "bold")
FONT_SMALL  = ("DejaVu Sans", 9)

N_BARS      = 32          # equalizer bands
FFT_SIZE    = 2048        # samples per FFT frame


# --------------------------------------------------------------------------
# Audio engine: real-time streaming player with volume/mute/pause
# --------------------------------------------------------------------------

# Raw-PCM stdin sinks, tried in order until one that works is found.
def _audio_env() -> dict:
    """Env that makes aplay/paplay talk to the DESKTOP's PipeWire session.

    Plain ALSA on this machine is hijacked by ~/.asoundrc (default = a USB
    microphone card), and the Pulse socket lives under /run/user/1000.
    """
    env = dict(os.environ)
    for d in ("/run/user/1000",):
        if os.path.isdir(d):
            env.setdefault("XDG_RUNTIME_DIR", d)
            env.setdefault("PULSE_SERVER", f"unix:{d}/pulse/native")
    return env


RAW_SINKS = (
    ("paplay", ["paplay", "--raw", "--rate", str(SAMPLE_RATE),
                "--format", "s16le", "--channels", "1",
                "--latency-msec", "100", "--process-time-msec", "20"]),
    ("aplay-pulse", ["aplay", "-D", "pulse", "-q", "--rate", str(SAMPLE_RATE),
                     "-f", "S16_LE", "-t", "raw", "-"]),
)


def _open_sink():
    """Try each raw-stdin audio sink; return a working Popen or None."""
    env = _audio_env()
    for name, cmd in RAW_SINKS:
        if not shutil.which(cmd[0]):
            continue
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            # Probe: a rejected stream (bad device / params) breaks the pipe.
            proc.stdin.write(np.zeros(256, dtype=np.int16).tobytes())
            proc.stdin.flush()
            return proc
        except (OSError, BrokenPipeError, FileNotFoundError):
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    pass
            continue
    return None


class PlaybackEngine:
    """Streams one synthesized song to a raw-PCM audio sink (48 kHz).

    The feeder thread always writes frames at the correct wall-clock
    rate (so the audio stream never stalls).  What it writes depends on
    state:

        * playing        -> the music (scaled by volume, or silent if muted)
        * paused/stopped -> digital silence (position frozen when paused)
    """

    CHUNK = 512                                   # samples written per tick

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

        self._samples: np.ndarray = np.zeros(0, dtype=np.float64)
        self._pos = 0                             # float playhead
        self.playing = False
        self.muted = False
        self.volume = 0.75                        # 0.0 - 1.0
        self.finished = False
        self._audio_lost = False                  # sink kept dying

        # Test-tone state: TEST briefly swaps in a 1 s arpeggio buffer;
        # the real song's samples/position are stashed and restored when
        # the tone ends (or STOP is pressed).
        self._test_active = False
        self._main_samples: np.ndarray | None = None
        self._main_pos = 0.0
        self._had_song = False

    # -- song loading -------------------------------------------------------

    def load(self, song: Song) -> None:
        """(Re-)synthesize a song and reset the transport to stopped."""
        with self._lock:
            # Stop the old feeder before replacing the buffer.
            self._running = False
        self._join_thread()
        self._samples = synthesize(song)
        with self._lock:
            self._pos = 0
            self.playing = False
            self.finished = False
            self._audio_lost = False
        self._start_thread()

    # -- transport controls -------------------------------------------------

    def play(self) -> None:
        with self._lock:
            if self.finished:
                self._pos = 0
                self.finished = False
            self.playing = True
            self.muted = False              # PLAY == "I want to hear this"
            lost = getattr(self, "_audio_lost", False)
            if lost:
                self._audio_lost = False
        th = getattr(self, "_thread", None)
        # Revive the feeder whenever it ever died: after reaching END OF
        # SONG it breaks out of the loop; the next PLAY must respawn it or
        # we stream pure air while the playhead still advances.
        if lost or th is None or not th.is_alive():
            self._start_thread()

    def load_samples(self, samples: np.ndarray) -> None:
        """Load raw samples (used by the test tone) and reset transport."""
        with self._lock:
            self._running = False
        self._join_thread()
        self._samples = samples
        with self._lock:
            self._pos = 0
            self.playing = False
            self.finished = False
            self._audio_lost = False
        self._start_thread()

    def test_tone(self) -> None:
        """Play a short unmistakable arpeggio through the live sink."""
        sr = SAMPLE_RATE
        def tone(f: float, d: float) -> np.ndarray:
            k = int(d * sr)
            x = np.arange(k) / sr
            return np.sign(np.sin(2 * np.pi * f * x)) * 0.4 * np.exp(-2.0 * x)
        sig = np.concatenate([tone(440, .25), tone(554.37, .25),
                              tone(659.25, .25), tone(880, .6)])
        pad = max(0, sr - sig.size)                # exactly 1 s, never negative
        if pad:
            sig = np.concatenate([sig, np.zeros(pad)])
        with self._lock:
            self._test_active = True
            self._had_song = bool(len(self._samples))
            self._main_samples = self._samples
            self._main_pos = self._pos
        self.load_samples(sig)
        self.play()

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def stop(self) -> None:
        with self._lock:
            self.playing = False
            self._pos = 0
            if self._test_active:                 # bail out of the tone test
                if self._had_song and self._main_samples is not None:
                    self._samples = self._main_samples
                    self._main_samples = None
                else:
                    self._samples = np.zeros(0)
                self._test_active = False
                self.finished = True

    def set_volume(self, value: float) -> None:
        with self._lock:
            self.volume = max(0.0, min(1.0, value))
            if self.volume > 0.0:
                self.muted = False                 # moving the fader unmutes

    def toggle_mute(self) -> bool:
        """Toggle mute; returns the new mute state."""
        with self._lock:
            self.muted = not self.muted
            return self.muted

    # -- spectrum for the visualizer ----------------------------------------

    def spectrum(self, n_bands: int = N_BARS) -> np.ndarray:
        """Mag nitude spectrum of the current audio window, 0..1 per band."""
        with self._lock:
            samples = self._samples
            pos = int(self._pos)
        if len(samples) == 0:
            return np.zeros(n_bands)
        if pos >= len(samples) or samples[pos] == 0.0:
            # still: fall back to a short window so paused bars don't snap
            pass

        end = min(pos + FFT_SIZE, len(samples))
        seg = np.asarray(samples[max(0, pos):end], dtype=np.float64).copy()
        if seg.size < 4:
            return np.zeros(n_bands)
        win = np.hanning(seg.size)
        spec = np.abs(np.fft.rfft(seg * win))
        # Map bands across ~80 Hz ... 9 kHz (log-spaced) with a little
        # averaging of adjacent bins so the bars track musical energy.
        freqs = np.geomspace(80.0, 9000.0, n_bands)
        bins = freqs * seg.size / SAMPLE_RATE
        vals = []
        top = float(spec.max()) or 1.0
        for b in bins:
            i0 = int(b)
            i1 = min(i0 + 3, len(spec))          # average 2-3 adjacent bins
            v = (float(spec[i0:i1].mean()) / top) if i1 > i0 else 0.0
            vals.append(max(0.0, min(1.0, v)))
        return np.asarray(vals)

    def position_fraction(self) -> float:
        with self._lock:
            n = len(self._samples)
            return 0.0 if n == 0 else min(1.0, self._pos / n)

    # -- feeder thread ------------------------------------------------------

    def _start_thread(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _join_thread(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
            finally:
                self._proc = None

    def _run(self) -> None:
        """Feed 16-bit PCM to the chosen audio sink (correct wall-clock rate).

        If the sink dies mid-stream (dropped audio device, aplay crash,
        ...), keep trying to reopen it until ~2 s pass or playback stops,
        so a transient glitch doesn't kill the whole song.
        """
        chunk_sec = self.CHUNK / SAMPLE_RATE
        zero = np.zeros(self.CHUNK, dtype=np.int16)
        proc = _open_sink()
        if proc is None:
            self._audio_lost = True
            return
        self._proc = proc
        try:
            while self._running:
                with self._lock:
                    if self._test_active and self._pos >= len(self._samples):
                        # TEST buffer done -> restore the real song.
                        if self._had_song and self._main_samples is not None:
                            self._samples = self._main_samples
                            self._pos = self._main_pos
                            self._main_samples = None
                        else:
                            self.playing = False
                            self.finished = True
                        self._test_active = False
                    playing = self.playing
                    muted = self.muted
                    volume = self.volume
                    pos = int(self._pos)
                    total = len(self._samples)
                    if self._pos >= total:
                        self.finished = True
                        playing = False
                    advance = playing        # pause freezes the playhead

                if playing:
                    n = min(self.CHUNK, total - pos)
                    chunk = np.zeros(self.CHUNK)
                    chunk[:n] = self._samples[pos:pos + n]
                    payload = (zero if (muted or volume == 0.0)
                               else np.clip(chunk * volume * 32767.0,
                                            -32768, 32767).astype(np.int16))
                else:
                    payload = zero

                try:
                    proc.stdin.write(bytes(payload))
                except (BrokenPipeError, OSError):
                    self._kill(proc)
                    # Sink died -- retry re-opening for up to 2 s so a
                    # transient glitch doesn't kill the whole song.
                    deadline = time.time() + 2.0
                    proc = None
                    while self._running and time.time() < deadline:
                        time.sleep(0.15)
                        proc = _open_sink()
                        if proc is not None:
                            break
                    if proc is None:
                        self._audio_lost = True
                        break
                    self._proc = proc
                    continue     # we lost a bit of clock; just resume

                if advance:
                    with self._lock:
                        self._pos += n
                time.sleep(chunk_sec)
        finally:
            self._kill(proc)
            self._proc = None

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
        except subprocess.SubprocessError:
            try:
                proc.kill()
            except Exception:
                pass


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class FloppyVisualizer(tk.Tk):
    """The main application window: equalizer canvas + transport controls."""

    def __init__(self, songs_dir: Path, default_song: Path | None = None) -> None:
        super().__init__()
        self.title("Floppytrack - floppy drive music")
        self.configure(bg=C_BG)
        self.geometry("900x520")
        self.minsize(700, 420)

        self.engine = PlaybackEngine()
        self.songs_dir = songs_dir
        self._bar_smooth = np.zeros(N_BARS)
        self._bar_released = np.zeros(N_BARS)

        self._build_header()
        self._build_eq_canvas()
        self._build_controls()

        self._load_song(default_song)
        self.bind("<Configure>", lambda _e: self._draw_eq())
        self.after(33, self._tick)                 # ~30 fps animation

    # ---- UI construction ----------------------------------------------------

    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=C_BG)
        bar.pack(side="top", fill="x", padx=24, pady=(18, 4))

        tk.Label(bar, text="FLOPPYTRACK", font=FONT_TITLE,
                 fg=C_TEXT, bg=C_BG).pack(side="left")
        self.status_lbl = tk.Label(bar, text="READY", font=FONT_SMALL,
                                   fg=C_TEXT_DIM, bg=C_BG)
        self.status_lbl.pack(side="left", padx=14, pady=(8, 0))
        self.now_lbl = tk.Label(bar, text="", font=FONT_SMALL,
                                fg=C_BAR_CAP, bg=C_BG)
        self.now_lbl.pack(side="right", pady=(8, 0))

        # Hairline divider
        tk.Frame(self, bg=C_BORDER, height=1).pack(fill="x",
                                                   padx=24, pady=(10, 0))

    def _build_eq_canvas(self) -> None:
        self.canvas = tk.Canvas(self, bg=C_PANEL, highlightthickness=1,
                                highlightbackground=C_BORDER)
        self.canvas.pack(fill="both", expand=True, padx=24, pady=18)

    def _build_controls(self) -> None:
        panel = tk.Frame(self, bg=C_PANEL, highlightthickness=1,
                         highlightbackground=C_BORDER)
        panel.pack(fill="x", padx=24, pady=(0, 20))

        pad = dict(padx=8, pady=16)

        # Song chooser
        tk.Label(panel, text="SONG", font=FONT_SMALL, fg=C_TEXT_DIM,
                 bg=C_PANEL).grid(row=0, column=0, sticky="w", **pad)
        self.song_var = tk.StringVar()
        self.song_combo = tk.OptionMenu(panel, self.song_var, "*")
        self.song_combo["menu"].configure(bg=C_PANEL, fg=C_TEXT,
                                          activebackground=C_BAR,
                                          activeforeground=C_TEXT)
        self.song_combo.configure(
            bg=C_BG, fg=C_TEXT, activebackground=C_PANEL_HI,
            activeforeground=C_TEXT, font=FONT, relief="flat",
            padx=18, pady=6, highlightthickness=1,
            highlightbackground=C_BORDER,
        )
        self.song_combo.grid(row=0, column=1, **pad)

        # Play / pause
        self.play_btn = self._make_button(panel, " PLAY ", self._on_play,
                                          bg=C_ACCENT, active=C_ACCENT_HI,
                                          fg=C_TEXT, big=True)
        self.play_btn.grid(row=0, column=2, **pad)

        # Stop
        self.stop_btn = self._make_button(panel, " STOP", self._on_stop,
                                          bg=C_PANEL_HI, active=C_BG)
        self.stop_btn.grid(row=0, column=3, **pad)

        # Mute
        self.mute_lbl = "MUTE"
        self.mute_btn = self._make_button(panel, self.mute_lbl, self._on_mute,
                                          bg=C_PANEL_HI, active=C_BG)
        self.mute_btn.grid(row=0, column=4, **pad)

        # One-shot test tone (diagnoses silent playback instantly)
        self.test_btn = self._make_button(panel, " TEST", self._on_test,
                                         bg=C_PANEL_HI, active=C_BG)
        self.test_btn.grid(row=0, column=5, **pad)

        # Volume
        tk.Label(panel, text="VOL", font=FONT_SMALL, fg=C_TEXT_DIM,
                 bg=C_PANEL).grid(row=0, column=6, sticky="w", **pad)
        self.vol_var = tk.DoubleVar(value=75.0)
        self.vol_slider = tk.Scale(
            panel, from_=0, to=100, orient="horizontal", variable=self.vol_var,
            command=self._on_volume, length=150, bg=C_PANEL, fg=C_TEXT,
            troughcolor=C_BAR, highlightthickness=0, sliderrelief="flat",
            activebackground=C_BAR_CAP, showvalue=False,
            sliderlength=18, width=8,
        )
        self.vol_slider.grid(row=0, column=7, **pad)
        self.vol_lbl = tk.Label(panel, text="75%", font=FONT_SMALL,
                                fg=C_TEXT, bg=C_PANEL, width=5)
        self.vol_lbl.grid(row=0, column=8, **pad)

    def _make_button(self, parent: tk.Misc, text: str, cmd,
                     bg: str, active: str, fg: str = C_TEXT,
                     big: bool = False) -> tk.Button:
        btn = tk.Button(parent, text=text, command=cmd, font=FONT,
                        bg=bg, fg=fg, activebackground=active,
                        activeforeground=fg, relief="flat",
                        padx=18 if big else 14, pady=8,
                        highlightthickness=1, highlightbackground=C_BORDER,
                        cursor="hand2", bd=0)
        return btn

    # ---- equalizer rendering -------------------------------------------------

    def _draw_eq(self) -> None:
        c = self.canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50:
            return
        c.delete("all")

        margin_x, margin_top, margin_bot = 12, 18, 18
        iw = w - 2 * margin_x
        ih = h - margin_top - margin_bot
        gap = max(2, iw // 400)
        bw = max(4, iw // N_BARS - gap)

        for i in range(N_BARS):
            x0 = margin_x + i * (bw + gap)
            value = self._bar_smooth[i]
            # Dark track behind each bar
            c.create_rectangle(x0, margin_top, x0 + bw, margin_top + ih,
                               fill=C_TRACK, outline="", width=1)
            bar_h = max(3, int(value * ih))
            y1 = margin_top + ih
            y0 = y1 - bar_h
            c.create_rectangle(x0, y0, x0 + bw, y1,
                               fill=C_BAR, outline="")
            # Bright cap on the leading edge
            cap = min(4, bar_h)
            c.create_rectangle(x0, y0, x0 + bw, y0 + cap,
                               fill=C_BAR_CAP, outline="")

    # ---- animation / state ticks ----------------------------------------------

    def _tick(self) -> None:
        target = self.engine.spectrum(N_BARS)
        # Fast attack, slow release makes the bars feel alive.
        self._bar_smooth = np.maximum(target, 0.82 * self._bar_smooth)
        self._draw_eq()

        s = "PLAYING" if self.engine.playing else (
            "PAUSED" if self.engine.position_fraction() > 0.001
            else ("MUTED-PLAYING" if not self.engine.finished else "READY"))
        if self.engine.finished and not self.engine.playing:
            s = "END OF SONG"
        if self.engine.muted:
            s += "  (MUTED)"
        if getattr(self.engine, "_audio_lost", False):
            s = "AUDIO DEVICE LOST -- press PLAY to retry"
            self.status_lbl.configure(text=s, fg=C_BAR_CAP)
            self.after(33, self._tick)
            return
        self.status_lbl.configure(text=s)
        self.after(33, self._tick)

    # ---- event handlers --------------------------------------------------------

    def _list_songs(self) -> list[Path]:
        if not self.songs_dir.is_dir():
            return []
        return sorted(self.songs_dir.glob("*.ftk"))

    def _load_song(self, path: Path | None) -> None:
        from floppytrack.songspec import load_song
        songs = self._list_songs()
        names = [p.name for p in songs]
        menu = self.song_combo["menu"]
        menu.delete(0, "end")
        for name in names:
            menu.add_command(label=name,
                             command=lambda n=name: self._choose(n))
        if names:
            self.song_var.set(names[0])
            self.now_lbl.configure(text=names[0])
        pick = path if path and path.exists() else (songs[0] if songs else None)
        if pick is None:
            messagebox.showwarning(
                "No songs", "No .ftk song files were found in the songs/ "
                f"directory: {self.songs_dir}")
            return
        from floppytrack.songspec import SongFormatError
        try:
            song = load_song(str(pick))
        except (SongFormatError, ValueError) as exc:
            messagebox.showerror("Bad song file", str(exc))
            return
        self.engine.load(song)
        self.status_lbl.configure(text="READY")

    def _choose(self, name: str) -> None:
        self._load_song(self.songs_dir / name)

    def _on_test(self) -> None:
        """Fire the engine's test tone through the live sink."""
        self.engine.test_tone()
        self.status_lbl.configure(text="TEST TONE", fg=C_ACCENT)

    def _on_play(self) -> None:
        if self.engine.playing:
            self.engine.pause()
            self.play_btn.configure(text=" PLAY ", bg=C_ACCENT)
        else:
            self.engine.play()
            self.play_btn.configure(text=" PAUSE", bg=C_ACCENT_HI)

    def _on_stop(self) -> None:
        self.engine.stop()
        self.play_btn.configure(text=" PLAY ")
        self._bar_smooth = np.zeros(N_BARS)

    def _on_mute(self) -> None:
        muted = self.engine.toggle_mute()
        self.mute_btn.configure(
            text=" UNMUTE" if muted else " MUTE",
            bg=C_BAR if muted else C_PANEL_HI,
        )
        self.vol_lbl.configure(fg=C_TEXT_DIM if muted else C_TEXT)

    def _on_volume(self, _value: object) -> None:
        pct = int(round(self.vol_var.get()))
        self.engine.set_volume(pct / 100.0)
        self.vol_lbl.configure(text=f"{pct}%")
        # Moving the fader unmutes (see PlaybackEngine.set_volume)
        if self.engine.muted:
            return
        if self.mute_btn.cget("text") == "  UNMUTE":
            self.mute_btn.configure(text="  MUTE", bg=C_PANEL_HI)
        self.vol_lbl.configure(fg=C_TEXT)


# --------------------------------------------------------------------------

def main() -> None:
    import sys

    root = Path(__file__).resolve().parent
    default = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    app = FloppyVisualizer(songs_dir=root / "songs", default_song=default)

    def _cleanup(*_a) -> None:
        app.engine._running = False
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", _cleanup)
    app.mainloop()


if __name__ == "__main__":
    main()
