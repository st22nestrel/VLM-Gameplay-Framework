# Configuration

Run commands from the repository root: YAML file paths and output paths are
relative to the current working directory. The CLI loads a root `.env` if present,
without replacing values already in the process environment.

`configs/runtime/gameplay.yaml` is the default standalone Qwen3-VL chess example
and the shared base required by the experiments suite. Suite defaults override
the runtime base, then each run's overrides take precedence.

| Section | Settings |
| --- | --- |
| `target_window` | `hwnd` or `title_substring`; gameplay CLI selects via `--hwnd` or `--title` |
| `capture` | `method`: auto/dxcam/printwindow/bitblt/mss; after-action or background capture; cursor overlay; settle delay |
| `actions` | `config_path` to action definitions and mouse coordinate conversion behavior |
| `gameplay` | backend, model, prompts, token limit, output directory, delay, pause key, finish strategy |
| `gameplay.model_settings` | sampling, thinking, coordinate normalization, LoRA, and generation schedules |
| `gameplay.message_history` | none, all, last_n, or compact history; historical images and action context |
| `gameplay.state_context` | repeated-action and low-image-change recovery hints |
| `gameplay.visual_context` | off, always_last_2, or on_stuck previous/current composite |
| `gameplay.progress_tracking` | fixed-region image-change measurement and optional debug crops |
| `gameplay.openai` / `gameplay.llama_cpp` | backend-specific server and credential settings |

The chess prompts request exactly `click at <x> <y>` in a 0-1000 coordinate space.
`normalize_coordinates_from_0_1000: true` converts those coordinates to pixels in
the captured window. The action file supports click-at, mouse motion, and mouse
buttons. To control another application, provide its task prompt and action YAML;
the executor also supports keyboard actions. Parameterized actions are parsed
before separator-based non-parameterized commands.

Use `capture-window --title "Chess Titans" --count 1` to inspect capture output,
or `parse-text "click at 500 500" --action-config <path>` to inspect parsing.
`capture.method: auto` tries DXCam if installed, then Win32 capture, then MSS.
Foreground input is the default; window visibility, focus, and permissions affect
whether a particular application accepts capture/input.

The optional freeze feature is retained but disabled by default. It needs a
separate `xspeedhack` installation and permissions to attach to the target.

To inspect control bindings in a settings screen, use:

```powershell
python -m src.cli.main discover-actions --config configs/runtime/discovery.yaml --title "Chess Titans" --prompt-type universal --max-ticks 10
```

Discovery needs a screen with readable controls/keybindings; it does not infer
undisplayed bindings. Its generated action config is saved with the run artifacts.
