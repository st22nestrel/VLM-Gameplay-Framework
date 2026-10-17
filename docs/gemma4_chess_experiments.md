# Gemma 4 chess comparison through Google

`configs/eval/gameplay_suite_gemma4_google_chess_experiments.yaml` copies the
Qwen3-VL screening conditions and compares `gemma-4-26b-a4b-it` with
`gemma-4-31b-it`. Both are listed in
[Google's hosted Gemma guide](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api).
The backend uses Google's OpenAI-compatible endpoint and `GEMINI_API_KEY` from
the environment/root `.env`.

The suite retains baseline, explicit mixed sampling, explicit fixed sampling,
always-on visual pairs, stuck hints, stuck-triggered visual pairs, and exploratory
sampling. Each condition uses identical settings for both models, with thinking
disabled. `top_k` is omitted because this backend forwards temperature and top_p.
The original Qwen suite remains a separate configuration.

Seven conditions x two models x two repeats produce 28 games, each capped at
30 model turns: at most 840 model requests. Capture and progress checks run every
1 second in a background thread, including while awaiting API responses. Model
request starts are at least 2 seconds apart (at most 30/minute); slow responses
reduce the rate. SDK retries are disabled, so API errors are logged for the turn
and subsequent attempts follow the loop's cadence. This spacing does not
guarantee compliance with token or daily quotas for a particular Google project.

Validate without launching a game, loading credentials, or making API calls:

```powershell
conda run -n transformers5 python -m src.cli.main gameplay-suite --config configs/eval/gameplay_suite_gemma4_google_chess_experiments.yaml --validate-only
```

Run the full comparison:

```powershell
conda run -n transformers5 python -m src.cli.main gameplay-suite --config configs/eval/gameplay_suite_gemma4_google_chess_experiments.yaml
```

To compare only the explicit fixed control (four games / at most 120 requests):

```powershell
conda run -n transformers5 python -m src.cli.main gameplay-suite --config configs/eval/gameplay_suite_gemma4_google_chess_experiments.yaml --run gemma4_26b_explicit_fixed,gemma4_31b_explicit_fixed
```

`--dry-run` still calls the API, but sends no game input. Keep the starting board,
White's side, difficulty, and window size consistent. The progress region remains
`left=710, top=38, width=42, height=64`; inspect its crops at the actual window size.
Progress events detect the hourglass indicator, rather than chess strength.
Signals shorter than the one-second capture interval can still be missed.
The earlier Qwen suite checked progress only on model turns; account for this
sampling difference when comparing its proxy counts against the Gemma results.

Logs retain every background progress sample in `progress_samples`, plus
`total_progress_count`, `frame_cadence_ms`, `inference_interval_ms`, and each turn's
`inference_started_at`. A sample may occur during an API response or after the
final click, so the total can exceed the last tick's pre-inference snapshot.
Use the existing analyze/view commands with the resulting suite directory.

Visual-pair conditions retain the source suite's coordinate convention: output
coordinates must be relative to RIGHT/CURRENT's screenshot, excluding its header.
No automatic composite-coordinate remapping is added. Stuck detection still uses
consecutive model-turn image checks, while progress uses every captured frame.

Offline regression checks:

```powershell
conda run -n transformers5 python -m unittest discover -s tests -v
```
