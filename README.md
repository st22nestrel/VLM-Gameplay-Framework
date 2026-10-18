# VLM Gameplay Framework & cli tools

Run a vision-language model against a Windows application: capture its window,
ask the model for an action, and execute keyboard/mouse input. The examples target
Chess Titans; change the prompts and action definitions to control another game
or application. Compatibility depends on the application's capture and input behavior.

Capture targets the game window's client area. Keep it visible and focused:
desktop capture methods can include overlapping windows. The control CLI checks
focus before and after capture and rejects screenshots on focus loss. The model
loop guards input, but still sends model requests when unfocused.
The CLI does not restrict a coding agent's access to its other tools.

Includes a game control CLI for external coding agents and shell scripts, local
Transformers inference (including LoRA adapters), OpenAI-compatible servers and
providers, a managed llama.cpp server, action discovery, repeated live gameplay
suites, analysis, and a browser-based results viewer. Session recording,
input playback, replay experiments, and private test data are excluded.

## Setup

Use Windows and Python 3.12. From the repository root:

```powershell
conda create -n transformers5 python=3.12
conda activate transformers5
python -m pip install -r requirements-transformers.txt
```

If `transformers5` already exists, activate it and use its installed dependencies.
For CUDA inference, install the PyTorch build appropriate to your GPU before the
requirements file; see [PyTorch installation](https://pytorch.org/get-started/locally/).
For Google, vLLM, or llama.cpp only, `requirements.txt` is sufficient on the agent
machine; the model server has its own installation. DXCam capture is optional
(`python -m pip install dxcam`); other capture methods work without it.

Public Hugging Face models need no API key. For a provider or gated model, copy
[.env.example](.env.example) to `.env` and replace the relevant dummy values.
Do not copy a private worktree's `.env`, outputs, datasets, or model weights.

## Game control CLI

**The CLI is meant to be used by (coding) agents.** Just tell your agent to use cli tools present in this repo to play game X until certain condition met (like destination reached, win etc.) and observe agent playing (you should not do anything else as game window must be in focus to be correctly captured).

Use `python -m src.control` to capture screenshots and send timed keyboard/mouse
actions from a shell or external coding agent, without loading a model or making
API requests. Launch your game, then run:

```powershell
python -m src.control list-windows
python -m src.control actions
python -m src.control observe --title "Your Game"
python -m src.control step --title "Your Game" --action move_forward --duration-ms 250
```

Commands return JSON with screenshot paths; each step releases its inputs before
capturing the result. `--title` must match one visible window. Use `--dry-run` to
capture without sending input, and `--action-config` to customize bindings.
For structured clicks and persistent JSON-lines control via `serve`, see
[External Game Control CLI](docs/external-control.md).

## First run with a model

Launch Chess Titans yourself, then run these commands from the repository root:

```powershell
python -m src.cli.main list-windows
python -m src.cli.main parse-text "click at 500 500"
python -m src.cli.main gameplay --title "Chess Titans" --dry-run --no-wait --max-ticks 5
python -m src.cli.main gameplay --title "Chess Titans" --no-wait --max-ticks 10
```

`--dry-run` loads the model and captures the window but does not execute actions.
The last command sends real input. Keep the target in the foreground; F1 toggles
pause, and Ctrl+C stops the loop and releases held input. The default configuration
is `configs/runtime/gameplay.yaml`, using Qwen3-VL 2B and normalized click coordinates.

## Included configurations

| Runtime file in `configs/runtime/` | Purpose |
| --- | --- |
| `gameplay.yaml` | Qwen3-VL 2B local chess; shared experiments base |
| `discovery.yaml` | Optional Qwen3-VL controls/keybindings discovery |
| `gameplay_gemma4_google.yaml` | Gemma 4 through Google AI Studio |
| `gameplay_qwen35_9b_vllm.yaml` | Quantized Qwen3.5 9B on an existing vLLM server |
| `gameplay_qwen35_9b_llama_cpp.yaml` | Quantized Qwen3.5 9B GGUF via llama.cpp |

Select another runtime with `gameplay --config <path> --title "Chess Titans"`.
See [backend setup](docs/backends.md) and [configuration](docs/configuration.md).

## Gameplay experiments and viewer

The local Qwen3-VL evaluation suite is
`configs/eval/gameplay_suite_qwen3vl_2b_chess_experiments.yaml`. It has 14 conditions
with two repeats (28 games), including fine-tuned runs that require a separately
provided LoRA adapter at `data/models/adapters/qwen3-vl-2b-chess-lora/` (gitignored).
**The adapter is not published yet: fine-tuned runs and the full suite do not work
on a fresh public checkout.** Base-model runs work without it. Once published on
Hugging Face, the adapter can be downloaded into this folder or referenced by its
repository ID. See [chess experiments](docs/chess_experiments.md) for the
conditions, prerequisites, and commands to run only the public base model.

For Google-hosted Gemma 4, use
`configs/eval/gameplay_suite_gemma4_google_chess_experiments.yaml`. It compares
26B A4B and 31B across the same seven conditions, with two repeats and 30 model
turns per game. It captures/probes progress every second and starts model requests
at least two seconds apart. See [Gemma comparison](docs/gemma4_chess_experiments.md).

```powershell
python -m src.cli.main gameplay-suite --config configs/eval/gameplay_suite_qwen3vl_2b_chess_experiments.yaml --run qwen3vl_2b_base_explicit_fixed --validate-only
python -m src.cli.main view-gameplay-suite --suite-dir outputs/gameplay-suites/<timestamp>
```

The viewer also accepts `--run-dir outputs/gameplay-runs/<timestamp>` for individual
runs. Generated logs and screenshots stay under ignored `outputs/`.

The supported CLI commands are listed by `python -m src.cli.main --help`.
The source lives in `src/actions`, `src/backends`, `src/capture`, `src/loop`,
`src/discovery`, `src/suites`, and `src/window_session.py`.

Maintainers: [explicit publishing workflow](docs/publishing.md).
