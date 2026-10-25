# Background capture tests

All tests for the background-window feature live here. Use the `transformers5`
environment, from the repository root.

## Interactive Chess Titans test

```powershell
conda activate transformers5
python tests/background_capture/chess_titans.py
```

This opens Chess Titans automatically (or uses its already-open window), then
waits for Enter. Prepare the board, cover the game with another window without
minimizing it, and return to the terminal to press Enter. The terminal itself
can cover the game. The test takes ten raw screenshots one second apart while
leaving focus alone, and prints their paths. Chess Titans stays open afterward.

Images and `results.json` go to `outputs/background_capture/chess_<timestamp>/`.
Inspect the PNGs: they should show the board, including its covered portions,
with no pixels from the covering window. Foreground/minimized attempts are
skipped and capture failures are recorded. A successful capture is not an
automatic assertion that Chess Titans rendered correctly; inspect the images.

For a different installation or a longer run:

```powershell
python tests/background_capture/chess_titans.py --exe "D:\Games\Chess\chess.exe" --count 30 --interval 1
```

Run it twice to check partial and full coverage. To check freshness, first make
a move, cover the game while the computer is thinking, then start capture and
look for the computer's move in subsequent screenshots.

## Automated regression tests

Offline regression tests (no desktop interaction):

```powershell
conda run -n transformers5 python -m pytest tests/background_capture -q
```

Live regression tests create and close their own target and covering windows.
They temporarily activate the cover window. They require an unlocked Windows
desktop; avoid switching windows during the run.

```powershell
conda run -n transformers5 python -m pip install -r tests/background_capture/requirements.txt
conda run -n transformers5 python -m pytest tests/background_capture --live-background-capture -q
```

The live matrix covers GDI (Tk canvas) and GPU-rendered OpenGL (SDL/pygame):

| Check | Assertion |
| --- | --- |
| Partial coverage | Every captured client pixel matches the target |
| Full coverage | Every captured client pixel matches the target |
| Freshness | Changing content while covered appears in the next capture |
| Positive control | MSS sees the magenta cover, proving actual occlusion |
| Focus and position | Cover stays foreground; target does not move |
| Minimized target | No frame; target remains minimized |

OpenGL cases skip if the optional test dependency `pygame` is absent. The normal
test run skips all live cases unless `--live-background-capture` is supplied.
Test windows run in a separate process so their UI event loop stays responsive
during PrintWindow calls. Captures go to pytest's temporary directory.

On this development machine, existing PrintWindow passed the GDI and OpenGL
coverage/freshness checks, while MSS captured the cover as expected. This is
evidence for these renderers, not a guarantee that every game implements
PrintWindow correctly. Test the actual target game before unattended use.
