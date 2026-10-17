# Qwen3-VL 2B chess experiments

Suite: `configs/eval/gameplay_suite_qwen3vl_2b_chess_experiments.yaml`.
The suite retains the full suite's experiment settings, with a clearer local
adapter path. Every condition runs on both the base model and a separately
provided LoRA adapter. Two repeats of seven conditions and two models give
**28 games, up to 1,400 VLM turns/clicks**. This is a screening budget;
raise `execution.repeats` to 3 or more when confirming promising conditions.
Fifty ticks is a short interaction test, not a full-game chess-strength estimate.

## What each condition tells us

Run names have `qwen3vl_2b_base_` or `qwen3vl_2b_finetuned_` followed by:

| Suffix | Change | Compare against | Why test it? |
| --- | --- | --- | --- |
| `baseline` | Original prompts and original three-option sampling cycle | Other conditions | Preserve a reference for each model. |
| `explicit_mixed` | White, two-click move instructions, selected-piece handling, square centers, explicit 0–1000 coordinates | `baseline` | Separate interface/prompt confusion from lack of chess ability. This tests the prompt changes together. |
| `explicit_fixed` | Same explicit prompts; fixed temperature 1.0, top_p 0.95, top_k 20 | `explicit_mixed` | Check whether switching between precise and creative decoding disrupts successive clicks. This is the control for subsequent experiments. |
| `visual_pair` | Labeled previous/current image on every request after the first | `explicit_fixed` | Give direct evidence of selection highlights and whether the last click changed the board, without replaying old assistant answers. |
| `stuck_hint` | Chess-specific recovery text after three consecutive low-image-change checks | `explicit_fixed` | Test whether a small, conditional intervention can break loops without supplying past replies. |
| `stuck_visual` | Same recovery hint plus a previous/current pair only while stuck | `stuck_hint` | Isolate whether visual evidence improves recovery beyond the hint alone. Usually keeps the simpler current-image input. |
| `exploratory_fixed` | Fixed temperature 1.2, top_p 0.98, top_k 40 | `explicit_fixed` | Test whether moderate exploration breaks repetitive coordinate choices; inspect legality and format errors as well as progress. |

All conditions use `message_history.mode: none`. The suite avoids replaying the
long screenshot histories and action lists that already performed poorly.
`visual_context` is a separate mechanism: one composite image replaces the
current image; it does not add a second historical chat turn.

The original option selection is **sequential**, not random: standard, precise,
creative, repeat. Values inside the precise/creative ranges are sampled. Its
`seed: 42` controls that parameter sampler, not PyTorch token sampling or the
opponent. `same_per_suite` also reuses that parameter sequence across repeats.
Treat repeat results as noisy live trials, not matched deterministic games.

## Running

Chess Titans must be installed at `game.executable`, or that setting must be
updated to your installation. The suite expects the chess LoRA adapter in
`data/models/adapters/qwen3-vl-2b-chess-lora/`. The entire `data/` directory is
gitignored; model files are never included in Git commits. Neither the game nor
adapter weights are distributed with the repository.

**The fine-tuned adapter is not published yet. Fine-tuned runs and full-suite
validation will fail on a fresh public checkout until the adapter is provided.**
The maintainer's local worktree has a copy in the folder above. Once the adapter
is published on Hugging Face, users will be able to download its files into this
folder, or replace the single `&finetuned_adapter` value with the published
Hugging Face repository ID. Every fine-tuned condition reuses that YAML anchor.
See [adapter setup](backends.md#chess-lora-adapter) for the expected files.

Users without an adapter can validate and run just the public base model:

```powershell
conda run -n transformers5 python -m src.cli.main gameplay-suite --config configs/eval/gameplay_suite_qwen3vl_2b_chess_experiments.yaml --run qwen3vl_2b_base_explicit_fixed --validate-only
```

Remove `--validate-only` to run this selected condition. It still runs twice for
up to 50 ticks each; this is not a ten-tick smoke run.

Run commands from the repository root in the required environment:

```powershell
conda run -n transformers5 python -m src.cli.main gameplay-suite --config configs/eval/gameplay_suite_qwen3vl_2b_chess_experiments.yaml --validate-only
```

Remove `--validate-only` to execute the full suite. `--dry-run` still launches the
game and runs inference, but sends no input; it cannot measure gameplay progress.

For a smaller first pass, compare the fixed control with always-on visual pairs
on both models (eight games including the two repeats):

```powershell
conda run -n transformers5 python -m src.cli.main gameplay-suite --config configs/eval/gameplay_suite_qwen3vl_2b_chess_experiments.yaml --run qwen3vl_2b_base_explicit_fixed,qwen3vl_2b_finetuned_explicit_fixed,qwen3vl_2b_base_visual_pair,qwen3vl_2b_finetuned_visual_pair
```

Use the same `--run` option to select any other comparison. Each selected entry
still runs `execution.repeats` times. Keep White's side, game difficulty, window
size, and starting state consistent across trials. Restarting the executable does
not itself guarantee a new board if Chess Titans restores saved games.

## How to adjust after the first results

| Observation | Next adjustment |
| --- | --- |
| Explicit prompts help | Keep them as the next baseline. To identify the useful part, remove either the two-click guidance or coordinate guidance in a separate follow-up. |
| Explicit prompts hurt the LoRA | Try the winning visual/recovery setting with the original short prompts; training may favor that input style. |
| Fixed standard beats mixed | Keep `generation_params.strategy: "off"`. Next try a fixed temperature around 0.7 with top_p/top_k unchanged, to separate temperature from distribution truncation. |
| Exploration helps but clicks become erratic | Lower temperature from 1.2 toward 1.0 first; then test top_p/top_k separately. Avoid changing all three again at once. |
| Exploration fails to break loops | Inspect input state and click placement before increasing randomness. If useful, test temperature 1.3 as a separate follow-up. |
| Visual comparison helps but is slow | Lower `visual_context.panel_width` from 768 to 640, then 512; alternatively prefer `stuck_visual` if recovery is the only benefit. Change the anchored `visual_pair` mapping to update all four visual runs. |
| Pieces/highlights are unclear | Increase panel_width toward 800, the size of recent 800x496 captures. Raising it beyond the source width cannot recover detail absent from the capture. Inspect the actual saved model image. |
| Pair clicks are systematically shifted | Inspect the coordinate convention first. The model must report coordinates relative to the current panel's screenshot content, excluding its header. See below. |
| Recovery triggers during valid moves | Raise `screen_change_repeat_threshold` from 3 to 4–5 or lower `screen_change_threshold` from 0.003 toward 0.001. Lowering the difference threshold makes the detector less eager. |
| Recovery never triggers in obvious loops | Raise `screen_change_threshold` toward 0.005 or lower the consecutive-check count from 3 to 2, after inspecting actual logged scores. |
| Frames show animation, an opponent still thinking, or stale highlights | Test `capture.post_action_settle_ms: 750` or `1000` as a separate comparison against the same condition. Keep timing constant across compared runs. |

`state_context` compares the whole image as a 32x32 grayscale thumbnail, rounded
to three decimals. It does not understand chess progress, and small legal changes
may look static. Its action repetition path compares canonical action names, not
coordinates, so all `click at` actions look repeated. The recovery conditions set
`repeat_threshold: 999` (above the 50-tick budget), using the consecutive low-image-
change path instead. Keep that value above `max_ticks` if extending trials.
The default movement-game hint is replaced with chess instructions. `on_stuck`
uses the same detector; it can work even with hint text disabled.

## Checking the comparison images and results

The runtime constructs two side-by-side panels, each 768 pixels wide, with a
28-pixel label strip above them. The first request has only the current image.
The previous panel is the screenshot from the previous VLM request, before that
turn's click; the current panel is captured after the action and settle interval.
It can still contain an animation or an incomplete computer reply.

**Coordinate handling is an experiment limitation.** The runtime scales model
coordinates to the original game frame, with no automatic unprojection from the
composite. The custom `prompt_note` therefore explicitly asks for 0–1000 positions
within RIGHT/CURRENT's screenshot content, excluding labels. The geometry is
consistent if the model follows that instruction; its compliance needs checking
in actual trials. If it reports positions in the full composite, a prompt-only
setup will misclick. A follow-up could add explicit coordinate remapping or use
two separate image parts if this proves unreliable.

Keep `debug.dump_resolved_gameplay_config: true` and `log_vlm_io: true`. Inspect:

- `resolved_gameplay.yaml`: effective raw config after merging; check the adapter,
  generation strategy, history mode, and visual mode.
- `tick_*_visual_context.png`: actual composite supplied to the model.
- `vlm_io.json`: actual prompts, responses, and screenshot paths.
- `gameplay_log.yaml`: `visual_context.applied`, `state_context.applied`,
  `screen_change_score_raw`, `low_screen_change_count`, model-space `model_actions`,
  pixel-space `parsed_actions`, and `progress_tracking` per tick.

After a suite finishes:

```powershell
conda run -n transformers5 python -m src.cli.main analyze-gameplay-suite --suite-dir outputs/gameplay-suites/<timestamp>
conda run -n transformers5 python -m src.cli.main view-gameplay-suite --suite-dir outputs/gameplay-suites/<timestamp>
```

Compare both repeats per model, prioritizing visible completed legal moves,
repeated-square loops, invalid responses, and latency. The analyzer's
`total_progress_count` is a proxy derived from the existing fixed region
`left=710, top=38, width=42, height=64`, not a move counter or chess-quality score.
Check its saved debug crops and ensure the region still covers the intended
indicator at your window size. A fully parseable stream of clicks can make no
useful chess progress. Extend the tick budget and repeat count only for promising
conditions once the interaction mechanics work.
