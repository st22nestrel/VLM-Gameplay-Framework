# Background window input

Use `windowmessage` to send keyboard, mouse movement, held buttons and
`click_at` directly to a Windows game window, including when it is covered.
The sender never activates the target, moves the desktop cursor, or falls back
to SendInput. The game must support legacy Win32 input messages and continue
processing them while inactive. This is not a universal background input device.

## Chess Titans

Chess Titans ignores the mouse message's coordinates and reads `GetCursorPos`
inside its mouse input translator. Ordinary posted clicks therefore do nothing
even when the game is foregrounded. Changing timing or using SendMessage alone
does not fix this.

`windowmessage` recognizes `ChessWindowClass` and uses a temporary Frida hook
inside that process. During each synchronous mouse dispatch, `GetCursorPos`
on the recipient's UI thread returns the virtual screen position. Outside that
dispatch it returns the real cursor position. Keyboard messages remain posted
so the game's accelerator loop still handles F2. The hook unloads when the
session stops; it changes no game files. Other window classes do not use hooks.

Install the optional dependency in the environment that runs the controller:

```powershell
python -m pip install -r requirements-windowmessage.txt
python tests/background_capture/chess_moves.py --input-method windowmessage --allow-unfocused
```

The dependency is already installed in this machine's `transformers5`
environment. Start on the recorded fresh board, or add `--new-game` to reset it.
Leave the game restored; it can be covered by another window.

For a fully covered replay with foreground/cursor assertions:

```powershell
python tests/background_capture/chess_moves_covered.py --new-game
```

This was verified against actual Chess Titans: all three White moves and
opponent replies passed with an owned covering window retaining foreground
focus and the desktop cursor unchanged. The hook is specific to cursor reads;
it does not add Raw Input, DirectInput or device-state emulation.
See [Frida's instrumentation API](https://frida.re/docs/javascript-api/).

## General usage

```powershell
conda activate transformers5
python -m src.control step --title "Your Game" --input-method windowmessage --allow-unfocused --method background --cursor-mode raw --actions-file click.json
python -m src.control step --title "Your Game" --input-method windowmessage --allow-unfocused --method background --cursor-mode raw --action move_forward --duration-ms 500
```

For the first command, save `[{"name":"click_at","params":{"x":100,"y":150}}]`
in `click.json` (an actions file avoids PowerShell's version-dependent handling
of quotes in native command arguments).

Coordinates are integer pixels relative to the target's client area, with its
top-left corner at `(0, 0)`. Mouse movement is absolute within this area; it
does not produce relative Raw Input for FPS camera control. Plain button actions
use the last virtual mouse position, initially the client center. `click_at`
posts movement, left-down and left-up at the requested position. If left is
already held, it only moves the virtual cursor, preserving the hold.

Focus policy remains independent: supply `--allow-unfocused` to turn off the
framework's default focus requirement. For the model gameplay/discovery loops,
use the same flags or configure:

```yaml
actions:
  input_method: windowmessage
  require_focus: false
capture:
  method: background
```

Suites inherit these settings; set `game.focus: false` to avoid suite launch
activation. Capture support is separate from input support. `background` capture
uses PrintWindow, which some renderers do not support. `cursor-mode: raw` avoids
drawing the real desktop cursor onto a screenshot of the virtual cursor's target.
Covered windows are supported; minimized windows must be restored.

The Python API is:

```python
from src.control.session import GameSession

game = GameSession(input_method="windowmessage", require_focus=False,
                   method="background", cursor_mode="raw")
try:
    game.attach(title="Your Game")
    game.step(actions=[{"name": "click_at", "params": {"x": 100, "y": 150}}])
finally:
    game.stop()
```

## Routing and state

Mouse hit testing searches only within the target window's child hierarchy;
overlapping desktop windows do not intercept it. Invisible, disabled and
transparent children are skipped. A held-button drag remains on its original
recipient until all buttons are released, even when crossing another child.
Coordinates are translated to that child's client area for every event.
Mouse message flags include all held buttons plus Shift/Ctrl.

Keyboard events use the target thread's focused window if it belongs to the
target. If it has no such focus, the last mouse recipient is used, then the root
window. This does not set actual keyboard focus. A key release goes to the same
recipient as its press. The backend encodes extended navigation keys, generic
modifier virtual keys, repeat/transition bits, and Alt/F10 system-key messages.
It sends key transitions, not timed OS auto-repeat or explicit text entry.

Cleanup tracks successful presses inside the backend, so partially executed
actions can still be released. It attempts every release even when one fails.
Invalid windows, minimized targets, changed attachments, unsupported keys, and
PostMessage errors are surfaced; they never trigger a foreground-input fallback.
A successful post means queued delivery, not confirmation that the game acted.

## Evaluation of the original `postmessage` experiment

The legacy backend is preserved for comparison. Source inspection found:

| Behavior | Original `postmessage` | New `windowmessage` |
| --- | --- | --- |
| `click_at` | SetCursorPos + SendInput, affects the real desktop | Window-directed move/down/up |
| Plain buttons after movement | Always use client center | Use last virtual position |
| Mouse drag flags | Movement has zero flags; button-up clears all flags | Preserve other buttons and Shift/Ctrl |
| Child windows | Always send to the top-level HWND | Route within the target hierarchy |
| Keyboard scan flags | Extended navigation marker encoded as part of the scan byte | Separate scan byte and extended-key bit |
| Alt/F10 and modifiers | Ordinary key messages and left-specific modifier VKs | System-key messages and generic modifier VKs |
| Coordinate packing | Y word is not masked | Validate signed 16-bit range and mask both words |
| Interrupted click | No local finally around button-up | Attempt release in finally; retain state for cleanup |

The opt-in live tests compare the old and new senders against a covered,
separate-process native Win32 receiver. They verify the legacy center reset and
missing drag state, and check new click/drag/key delivery, child coordinates,
and unchanged desktop cursor/foreground window. The legacy `click_at` defect
is established by source inspection and mocked tests; the live comparison does
not exercise its desktop click. These receiver tests do not establish support
for any particular game engine.
An additional live CLI test sends a click and saves a background screenshot
through the complete control-session path.

```powershell
conda run -n transformers5 python -m pytest tests/test_window_message.py tests/background_input -q --live-background-input
```

## Compatibility limits

Posting keyboard messages does not establish real device state. Games polling
GetAsyncKeyState, GetKeyboardState, DirectInput or Raw Input can ignore these
events; applications that inspect real cursor position or require activation
can also reject them. Native controls that derive text/modifiers from keyboard
state may not handle posted chords like hardware input. Games that pause when
inactive must expose a setting or integration that permits background play.
Changing PostMessage to SendMessage does not create device state either.

For those games, a game/engine-specific integration (for example an input API
inside an owned game) is required. Ordinary windowmessage delivery does not
attach input queues, fake activation, or modify input-device settings. The
Chess Titans path injects the temporary cursor hook described above.
Windows UIPI also restricts posting to processes at a higher integrity level;
run the controller and target with matching permissions.

Win32 references:
[keyboard input](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-keyboard-input),
[mouse flags and coordinates](https://learn.microsoft.com/en-us/windows/win32/inputdev/wm-mousemove),
[Raw Input](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-raw-input),
[PostMessage and UIPI](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-postmessagew).
