# VLM backends

All backends use the same action parser, executor, gameplay loop, and viewer.
`gameplay.backend` selects `transformers`, `openai`, or `llama_cpp_server`.
Model support and available generation controls depend on the backend/provider.

## Transformers

Install `requirements-transformers.txt`, then use `configs/runtime/gameplay.yaml`.
Weights use the shared Hugging Face cache outside the repository. Local inference
retains Qwen and Gemma model handling, thinking controls, generation parameter
schedules, and PEFT/LoRA loading.

For your own adapter, set `gameplay.model_settings.lora_adapter_path` to its path.
`lora_base_model_source: config` uses `gameplay.model`; `adapter` uses adapter
metadata. The adapter and base weights must be provided separately.

### Chess LoRA adapter

The experiments suite uses `data/models/adapters/qwen3-vl-2b-chess-lora/` for the local
adapter. This folder, like all of `data/`, is gitignored. Keep the complete
adapter download together, including `adapter_config.json`,
`adapter_model.safetensors`, and any supplied tokenizer, processor, and chat
template files.

The maintainer's local copy is available in that folder. **There is currently no
public Hugging Face upload, so fresh checkouts cannot run fine-tuned conditions.**
Until publication, select base-model runs as described in
[chess experiments](chess_experiments.md). After publication, either download the
adapter into the folder above or set the suite's `&finetuned_adapter` value to its
published Hugging Face repository ID. Keep `lora_base_model_source: config` to
preserve the suite's configured base model. No upload or automatic download is
configured yet.

## Google AI Studio

Use `configs/runtime/gameplay_gemma4_google.yaml` with `GEMINI_API_KEY` in `.env`.
The `openai` backend here means the protocol/client, and the requests go to Google.
This example disables chat-template extras and uses Google's thinking-level
setting. See [Google's Gemma API guide](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api).

```powershell
python -m src.cli.main gameplay --config configs/runtime/gameplay_gemma4_google.yaml --title "Chess Titans" --max-ticks 10
```

## vLLM with quantized Qwen3.5 9B

Run vLLM separately on a supported server environment. The Windows agent connects
to its OpenAI-compatible HTTP endpoint; installing vLLM in the agent environment
is unnecessary. For example, on the server:

```bash
vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit --host 127.0.0.1 --port 8000 --max-model-len 8192
```

Then use `configs/runtime/gameplay_qwen35_9b_vllm.yaml` on the agent. Set
`gameplay.openai.base_url` to the reachable server address and `gameplay.model` to
the model name the server exposes. `VLLM_API_KEY=EMPTY` is a dummy client value for
an unauthenticated local server; use the actual configured key when authentication
is enabled. Quantization support depends on the GPU and vLLM build. See the
[quantized model card](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit) and
[vLLM Qwen3.5 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md).

## llama.cpp

Put a vision-capable `llama-server` executable on PATH, or set an absolute
`gameplay.llama_cpp.server_path` in `gameplay_qwen35_9b_llama_cpp.yaml`. The example
downloads the configured GGUF model and vision projector through Hugging Face and
manages the server lifecycle. Set `auto_start: false` to connect to a server you
already manage. Local paths can replace `hf_repo_id`/`hf_filename` and the projector
download. The [GGUF model card](https://huggingface.co/w-ahmad/Qwen3.5-9B-GGUF-MoQ)
describes the quantization.

## Other compatible providers

Keep `backend: openai`, select your provider's model and `base_url`, and name its
credential variable in `api_key_env`. The optional `api_key_id_env` plus
`api_key_secret_env` pair supports Modal proxy tokens. Both values are joined as
`id:secret`. `.env.example` documents the supplied names; custom names need matching
private `.env` entries. No credentials are bundled.
