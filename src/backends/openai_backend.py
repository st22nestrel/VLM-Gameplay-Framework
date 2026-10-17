"""OpenAI API-backed multimodal VLM backend.

This backend sends chat completion requests with image + text content to the
OpenAI API using the OpenAI Python library. It mirrors the llama.cpp server
backend's multimodal request format so the gameplay/discovery loops can stay
unchanged.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Optional

from openai import OpenAI
from PIL import Image

from src.backends.base import VLMBackend, resolve_message_images

logger = logging.getLogger(__name__)


class OpenAIBackend(VLMBackend):
    """VLM backend that calls the OpenAI API using the OpenAI Python library.

    Parameters
    ----------
    model_id : str
        OpenAI model name (e.g. "gpt-4.1-mini").
    max_new_tokens : int
        Maximum tokens to generate per request.
    api_key_env : str
        Name of the environment variable containing the OpenAI API key.
    api_key_id_env, api_key_secret_env : Optional[str]
        Names of environment variables holding a token id/secret pair that
        should be combined as ``"{id}.{secret}"`` to form the bearer token
        (e.g. Modal proxy auth tokens). When both are set, they take
        precedence over ``api_key_env``.
    base_url : str
        Base URL for the OpenAI API (default ``https://api.openai.com/v1``).
    timeout_sec : int
        HTTP request timeout in seconds.
    enable_thinking : bool
        Kept for parity with other backends; currently only logged.
    send_chat_template_kwargs : bool
        Whether to send extra_body.chat_template_kwargs.enable_thinking with
        each request. Some OpenAI-compatible endpoints (e.g. Google's Gemini
        API) reject unknown fields with a 400 error; set this to False for
        those endpoints.
    temperature, top_p : Optional[float]
        Sampling parameters forwarded to the API when provided.
    reasoning_effort : Optional[str]
        Reasoning effort level forwarded to the API when provided (e.g.
        "low", "medium", "high" -- accepted values depend on the model).
    google_thinking_level : Optional[str]
        Thinking level forwarded as extra_body.google.thinking_config.thinking_level.
        This is the Gemini-API-specific knob for controlling/disabling
        "thinking" on Gemma models served via Google AI Studio (e.g.
        "MINIMAL" to effectively disable thinking). Unlike
        `chat_template_kwargs.enable_thinking` or `reasoning_effort`, Gemma
        models accept this field without a 400 error. Leave unset for
        backends/models that don't support it.
    max_retries : int
        SDK retries after connection/rate-limit errors (default 2). Set to 0
        when the gameplay loop must pace every request attempt itself.
    """

    def __init__(
        self,
        model_id: str = "gpt-4.1-mini",
        max_new_tokens: int = 64,
        api_key_env: str = "OPENAI_API_KEY",
        api_key_id_env: Optional[str] = None,
        api_key_secret_env: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: int = 60,
        enable_thinking: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        send_chat_template_kwargs: bool = True,
        google_thinking_level: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._api_key_env = api_key_env
        self._api_key_id_env = api_key_id_env
        self._api_key_secret_env = api_key_secret_env
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._enable_thinking = enable_thinking
        self._temperature = temperature
        self._top_p = top_p
        # Some OpenAI-compatible endpoints (e.g. Google's Gemini API) reject
        # unknown fields like extra_body.chat_template_kwargs with a 400
        # error. Set this to False for such endpoints.
        self._send_chat_template_kwargs = send_chat_template_kwargs
        self._reasoning_effort = reasoning_effort
        self._google_thinking_level = google_thinking_level
        self._max_retries = max_retries

        self._client: Optional[OpenAI] = None
        self._loaded = False

        if self._enable_thinking:
            logger.warning(
                "enable_thinking is set for OpenAI backend, but thinking-mode "
                "formatting is not explicitly handled; responses may include "
                "reasoning text."
            )

    # -- properties ---------------------------------------------------------

    @property
    def backend_name(self) -> str:  # pragma: no cover - simple property
        return "openai"

    @property
    def model_name(self) -> str:  # pragma: no cover - simple property
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- lifecycle ----------------------------------------------------------

    def load_model(self) -> None:
        if self.is_loaded:
            logger.warning("Model already loaded, skipping")
            return

        if self._api_key_id_env and self._api_key_secret_env:
            token_id = os.getenv(self._api_key_id_env)
            token_secret = os.getenv(self._api_key_secret_env)
            if not token_id or not token_secret:
                raise RuntimeError(
                    f"Environment variables {self._api_key_id_env!r} and "
                    f"{self._api_key_secret_env!r} must both be set to build "
                    "the bearer token."
                )
            api_key = f"{token_id}.{token_secret}"
        else:
            api_key = os.getenv(self._api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Environment variable {self._api_key_env!r} is not set. "
                    "Set it to your OpenAI API key."
                )

        self._client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=float(self._timeout_sec),
            max_retries=self._max_retries,
        )
        self._loaded = True
        logger.info(
            "OpenAI backend ready for model %s (base_url=%s)",
            self._model_id,
            self._base_url,
        )

    def unload_model(self) -> None:
        self._loaded = False
        self._client = None

    # -- inference ----------------------------------------------------------

    @property
    def supported_generation_params(self) -> frozenset[str]:
        return frozenset({"temperature", "top_p", "reasoning_effort"})

    @property
    def static_generation_params(self) -> dict[str, float | int]:
        return {
            name: value
            for name, value in (
                ("temperature", self._temperature),
                ("top_p", self._top_p),
                ("reasoning_effort", self._reasoning_effort),
            )
            if value is not None
        }

    def infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> str:
        image_url = self._image_to_data_url(image)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        return self._run_chat_completion(messages, generation_params)

    def infer_messages(
        self,
        image: Image.Image,
        messages: list[dict[str, Any]],
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> str:
        """Run inference with caller-provided chat messages.

        Image parts may contain historical image paths or PIL images. A part
        without an image value refers to the current screenshot.
        """
        resolved_messages = self._resolve_image_placeholders(messages, image)
        return self._run_chat_completion(resolved_messages, generation_params)

    def _resolve_image_placeholders(
        self,
        messages: list[dict[str, Any]],
        image: Image.Image,
    ) -> list[dict[str, Any]]:
        messages, _ = resolve_message_images(messages, image)
        resolved: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                resolved.append(message)
                continue

            new_content: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    new_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_to_data_url(part["image"])},
                        }
                    )
                else:
                    new_content.append(part)
            resolved.append({**message, "content": new_content})
        return resolved

    def _run_chat_completion(
        self,
        messages: list[dict[str, Any]],
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> str:
        if not self.is_loaded or self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build kwargs for the API call
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": self._max_new_tokens,
        }

        kwargs.update(self.resolve_generation_params(generation_params).effective)

        # Optional thinking-mode control for backends that support chat templates.
        # This mirrors the Gemma-4 `enable_thinking` flag used in the
        # Transformers backend, and passes it through to the upstream API as
        # extra_body.chat_template_kwargs.enable_thinking.
        extra_body: dict[str, Any] = {}
        if self._send_chat_template_kwargs:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self._enable_thinking,
            }

        # Gemini-API-specific thinking control (used for Gemma models, which
        # reject chat_template_kwargs / reasoning_effort but accept this).
        if self._google_thinking_level is not None:
            extra_body["extra_body"] = {
                "google": {
                    "thinking_config": {
                        "thinking_level": self._google_thinking_level,
                    }
                }
            }

        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self._client.chat.completions.create(**kwargs)
        return self._extract_text_response(response).strip()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_text_response(response: Any) -> str:
        """Extract text content from OpenAI chat completion response."""
        if not response.choices:
            raise RuntimeError(f"Unexpected OpenAI response: no choices returned")

        message = response.choices[0].message
        content = message.content

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content) if content else ""

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
