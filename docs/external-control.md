# External game control CLI

Use the framework's capture and input systems from a shell or another agent.
This entry point imports no inference backends and makes no model/API requests.
Run commands from the repository root in the `transformers5` conda environment.

## Quick start (PowerShell)

```powershell
conda activate transformers5
python -m src.control list-windows
python -m src.control actions
python -m src.control observe --title "Your Game"
python -m src.control step --title "Your Game" --action move_forward --duration-ms 250 --dry-run
python -m src.control step --title "Your Game" --action move_forward --duration-ms 250
```

`--title` must match exactly one visible window. Use `--hwnd 0x12345` from
`list-windows` when titles are ambiguous. Each one-shot command attaches afresh;
no daemon or session file is needed. `observe` and `step` bring the target to
the foreground by default. `--no-focus` leaves focus alone. Both commands require
the target to be focused, including with `--no-focus` and `--dry-run`. `--dry-run`
sends no input, changes no focus, and skips action/settle waits, but still captures
a real screenshot. Focus is checked before and after capture; a failed check
returns a `focus_lost` error without saving or returning the screenshot. These
checks cannot detect a brief focus change entirely between checks.

Combine actions with repeated `--action` flags:

```powershell
python -m src.control step --title "Your Game" --action move_forward --action sprint --duration-ms 500
```

`actions` describes canonical names, bindings, and required parameters.
The default [starter configuration](../configs/actions/control-actions.yaml)
includes WASD, Shift, Ctrl, jump, interact, reload, menu keys, mouse buttons,
aiming, and clicking. Change bindings for your game with `--action-config`:

```powershell
python -m src.control actions --action-config configs/actions/default-actions_chess.yaml
```

## Coordinates and structured actions

Use a JSON array for parameterized actions. An action is either its exact
canonical name or `{"name": "...", "params": {...}}`. Unknown names, unexpected
parameters, wrong types, and coordinates outside the client area are rejected.
Up to 32 actions can be combined in a step. Actions in one step form a single
tick: holds, taps, mouse movement, then clicks, using the existing executor's
ordering. Use separate steps for a sequence such as click, wait, click.

JSON files avoid Windows shell quoting differences:

```powershell
'[{"name":"click_at","params":{"x":320,"y":240}}]' | Set-Content -Encoding UTF8 click.json
python -m src.control step --title "Your Game" --actions-file click.json --duration-ms 0
```

`--actions` also accepts an inline JSON array when your shell preserves quotes.
An empty array (`[]`) waits for `duration_ms` and returns a new screenshot.

Coordinates refer to **client-area pixels**, with `(0,0)` at the top left of
the screenshot, excluding title bars and window borders. Images are saved at
their native capture resolution. `click_at` positions the pointer and left
clicks. `aim` uses the framework's existing FPS behavior: its target pixel is
converted to a relative mouse delta from the client center. Sensitivity and
camera response remain game dependent.

## Step timing and cleanup

`--duration-ms` is 0–10000 (default 250). Holds remain down for this interval
after the initial tick has executed. One-shot taps retain the executor's
existing short press duration. Inputs are released in `finally`, including
taps interrupted between key-down and key-up. Focus is checked during the hold
interval about every 20ms; loss of focus aborts the step and releases inputs.
`--settle-ms` is 0–5000 (default 100), measured after release and before capture.
Each step finishes with all its inputs released, ready for the caller to think.

The game continues running between calls. This first CLI version does not
freeze or slow it. A screenshot and focus check do not guarantee that a game
accepted injected input; inspect the resulting frame to assess progress.

## JSON responses and artifacts

Each command writes one JSON response to stdout. Framework messages go to
stderr. Successful responses use `{"ok":true,"result":{...}}`; failures use
`{"ok":false,"error":{"code":"...","message":"..."}}`.

`observe` returns an absolute `screenshot` path, raw/cursor variant paths,
image dimensions, capture timestamp, client/window rectangles, target identity,
foreground status, and a `blank` flag. `step` returns the same observation
under `result.observation`, together with the actions, plan, execution summary,
timing, and `inputs_released`. Open the screenshot with an image-viewing tool
before deciding the next action. Blank images are flagged because a dark scene
and a failed capture can look alike.

Artifacts go to `outputs/control/<session-id>/` by default. Set `--out` to
collect frames in a chosen directory. Filenames are unique across calls.
Use `--cursor-mode raw|cursor|both` (default `both`) and
`--method auto|dxcam|win32|mss` (default `auto`). Auto tries DXCam, Windows
PrintWindow/BitBlt, then MSS, skipping blank results where possible. `win32`
uses the existing PrintWindow-to-BitBlt fallback. Desktop-based methods capture
overlapping windows too, so keep the game visible.

Exit codes: `0` success, `2` invalid arguments/actions, `1` other failures,
`130` Ctrl+C. If capture fails or loses focus after a step, the error includes the completed
step summary and `actions_may_have_executed`; observe again before retrying.

## Persistent JSON-lines mode

```powershell
python -m src.control serve --out outputs/control/session
```

Write one request per stdin line; each produces exactly one response line.
There is no startup response. An optional `id` is echoed unchanged.

```jsonl
{"id":1,"op":"list-windows"}
{"id":2,"op":"attach","title":"Your Game"}
{"id":3,"op":"actions"}
{"id":4,"op":"observe"}
{"id":5,"op":"step","actions":["move_forward","sprint"],"duration_ms":400,"settle_ms":100}
{"id":6,"op":"step","actions":[{"name":"click_at","params":{"x":320,"y":240}}],"duration_ms":0}
{"id":7,"op":"stop"}
```

`serve --title "Your Game"` attaches at startup instead. `attach` accepts
either `title` or `hwnd`, and can switch targets while retaining the CLI's
action/capture settings. A failed attach preserves the previous target.
Malformed requests return errors and the loop continues. `stop`, EOF, and
Ctrl+C clean up the session; they do not close the game. `stop` exits the loop.
The stream process returns 0 on normal shutdown even if individual requests
failed: inspect each response's `ok` field.

The reusable `src.control.session.GameSession` provides `attach`, `actions`,
`observe`, `step`, and `stop` for a future MCP wrapper. Commands are synchronous
and should be serialized per session.
