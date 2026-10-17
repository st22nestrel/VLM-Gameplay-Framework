"""Backend construction and lifecycle helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from src.backends.base import VLMBackend
from src.loop.gameplay_config import GameplayRuntimeConfig


def create_backend_from_gameplay_config(config: GameplayRuntimeConfig) -> VLMBackend:
    """Create a VLM backend from normalized gameplay runtime config."""
    backend_name = config.backend_name
    model_settings = config.model_settings

    if backend_name == "transformers":
        from src.backends.transformers_backend import TransformersBackend

        return TransformersBackend(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            enable_thinking=model_settings.enable_thinking,
            thinking_budget=model_settings.thinking_budget,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            top_k=model_settings.top_k,
            lora_adapter_path=model_settings.lora_adapter_path,
            lora_base_model_source=model_settings.lora_base_model_source,
        )

    if backend_name == "openai":
        from src.backends.openai_backend import OpenAIBackend

        openai_cfg = config.openai_config
        return OpenAIBackend(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            api_key_env=openai_cfg.get("api_key_env", "OPENAI_API_KEY"),
            api_key_id_env=openai_cfg.get("api_key_id_env"),
            api_key_secret_env=openai_cfg.get("api_key_secret_env"),
            base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
            timeout_sec=int(openai_cfg.get("timeout_sec", 60)),
            enable_thinking=model_settings.enable_thinking,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            reasoning_effort=openai_cfg.get("reasoning_effort"),
            send_chat_template_kwargs=bool(
                openai_cfg.get("send_chat_template_kwargs", True)
            ),
            google_thinking_level=openai_cfg.get("google_thinking_level"),
            max_retries=int(openai_cfg.get("max_retries", 2)),
        )

    if backend_name == "llama_cpp_server":
        from src.backends.llama_cpp_server_backend import LlamaCppServerBackend

        llama_cfg = config.llama_cpp_config
        return LlamaCppServerBackend(
            model_id=config.model_id,
            max_new_tokens=config.max_new_tokens,
            enable_thinking=model_settings.enable_thinking,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            top_k=model_settings.top_k,
            base_url=llama_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
            auto_start=llama_cfg.get("auto_start", True),
            server_path=llama_cfg.get("server_path", "llama-server"),
            host=llama_cfg.get("host", "127.0.0.1"),
            port=int(llama_cfg.get("port", 8080)),
            startup_timeout_sec=int(llama_cfg.get("startup_timeout_sec", 300)),
            request_timeout_sec=int(llama_cfg.get("request_timeout_sec", 300)),
            hf_repo_id=llama_cfg.get("hf_repo_id"),
            hf_variant=llama_cfg.get("hf_variant"),
            hf_filename=llama_cfg.get("hf_filename"),
            hf_mmproj_filename=llama_cfg.get("hf_mmproj_filename"),
            local_model_path=llama_cfg.get("local_model_path"),
            local_mmproj_path=llama_cfg.get("local_mmproj_path"),
            n_gpu_layers=llama_cfg.get("n_gpu_layers", -1),
            ctx_size=int(llama_cfg.get("ctx_size", 8192)),
            parallel=int(llama_cfg.get("parallel", 1)),
            threads=llama_cfg.get("threads"),
            flash_attn=llama_cfg.get("flash_attn", True),
            jinja=llama_cfg.get("jinja", True),
            alias=llama_cfg.get("alias"),
            hf_token_env=llama_cfg.get("hf_token_env", "HF_TOKEN"),
        )

    raise ValueError(f"Unknown backend: {backend_name!r}")


@contextmanager
def loaded_backend(
    backend: VLMBackend,
    *,
    announce: bool = False,
) -> Iterator[VLMBackend]:
    """Load a backend for the duration of the context, then unload it."""
    loaded_here = False
    if not backend.is_loaded:
        if announce:
            print(f"Loading model {backend.model_name}...")
        backend.load_model()
        loaded_here = True
        if announce:
            print("Model loaded.")

    try:
        yield backend
    finally:
        if loaded_here and backend.is_loaded:
            backend.unload_model()
