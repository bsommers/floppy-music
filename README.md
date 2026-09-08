# floppytrack — floppy drive synth

Music made of 3.5" / 5.25" floppy drive sounds.  Synthesised in Python
(numpy), rendered at **48 kHz** and crushed to an **8-bit** retro texture,
written to `.wav` + `.mp3`, played through the Linux audio stack.

## Voices

| instrument | sound                                        | use                |
|------------|----------------------------------------------|--------------------|
| `head`     | square read/write head blip, low-passed      | **lead melody**    |
| `motor`/`m35` | 3.5" drive: brisk buzzy hum + high whine + 8 Hz wobble | bass / drone |
| `m525`     | 5.25" drive: deeper, lazier, boxier motor    | heavy bass — **mix with m35 in the same song** |
| `step`     | woody stepper tick (noise + thunk)           | percussion         |
| `beep`     | bright sine + 2nd-harmonic blip              | counter melody     |
| `whir`     | filtered noise sweep                         | spin-up / spin-down|

## Install

```bash
pip install numpy scipy     # scipy optional (faster low-pass)
# mp3 needs one of:
sudo apt install ffmpeg     # or: lame
```

## Run

```bash
python3 main.py                        # compose + render + play the Star Wars theme
python3 main.py songs/star_wars_theme.ftk --no-play
python3 main.py songs/my_song.ftk -o out/ --no-mp3
python3 main.py --list songs/my_song.ftk   # check a parsed song
```

Output lands in `output/`: `star_wars_theme.wav` (16-bit PCM carrier,
8-bit-crushed content) and `star_wars_theme.mp3`.

## Song file format (`.ftk`)

```
TEMPO 108                 # beats per minute (beat = quarter note)

# relative event: played when the cursor reaches it
head   0.5  D5  0.9       # instrument, beats, note, optional velocity 0.0-1.0
step   1   rest
motor  4   C2  0.9

# absolute event: layered voice at an exact beat (does not move the cursor)
motor  AT 20  4  C2  0.9
head   AT 20  1  D5

# POLYPHONY: a head/beep line may carry several notes -> they blip at once
# (the 8-bit way to write chords), e.g. a D-minor-ish stab:
beep   AT 0  2  D4 A4 D5  0.4
beep   1  C4 E4 G4        # relative chords work too
```

Notes: `A0`…`B7` with `#` / `b` accidentals (`C4` = middle C).
`velocity` scales loudness.  `whir` can take `up` implicitly (default up).
Chords accept 2-6 simultaneous notes on `head`/`beep` only.
A note's `#` is NOT a comment (comments start at a token boundary only).

Layout:

```
main.py               # CLI entry point
floppytrack/
  synth.py            # instrument synthesis + retro 8-bit FX
  songspec.py         # .ftk parser (tempos, relative + AT-absolute events)
  render.py           # mix -> wav/mp3 -> Linux playback
songs/
  star_wars_theme.ftk # Star Wars theme (main theme + Leia theme)
output/               # rendered .wav / .mp3 files
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
