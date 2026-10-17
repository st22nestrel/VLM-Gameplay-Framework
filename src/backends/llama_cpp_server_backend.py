"""llama.cpp server-backed multimodal VLM backend.

Supports GGUF Gemma-4 models served through `llama-server`'s OpenAI-compatible
HTTP API. The backend can either:

1. Connect to an already running `llama-server` instance, or
2. Auto-start `llama-server` after resolving model artifacts from local paths
   or the Hugging Face Hub.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from src.backends.base import VLMBackend

logger = logging.getLogger(__name__)


class LlamaCppServerBackend(VLMBackend):
    """VLM backend backed by a local or auto-started llama.cpp server."""

    def __init__(
        self,
        model_id: str = "ggml-org/gemma-4-E4B-it-GGUF",
        max_new_tokens: int = 64,
        enable_thinking: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        base_url: str = "http://127.0.0.1:8080/v1",
        auto_start: bool = True,
        server_path: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8080,
        startup_timeout_sec: int = 300,
        request_timeout_sec: int = 300,
        hf_repo_id: Optional[str] = None,
        hf_variant: Optional[str] = None,
        hf_filename: Optional[str] = None,
        hf_mmproj_filename: Optional[str] = None,
        local_model_path: Optional[str] = None,
        local_mmproj_path: Optional[str] = None,
        n_gpu_layers: Optional[int] = -1,
        ctx_size: int = 8192,
        parallel: int = 1,
        threads: Optional[int] = None,
        flash_attn: bool | str = True,
        jinja: bool = True,
        alias: Optional[str] = None,
        hf_token_env: str = "HF_TOKEN",
    ) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._enable_thinking = enable_thinking
        self._temperature = temperature if temperature is not None else 1.0
        self._top_p = top_p if top_p is not None else 0.95
        self._top_k = top_k if top_k is not None else 64

        self._base_url = self._normalize_base_url(base_url)
        self._auto_start = auto_start
        self._server_path = server_path
        self._host = host
        self._port = port
        self._startup_timeout_sec = startup_timeout_sec
        self._request_timeout_sec = request_timeout_sec

        self._hf_repo_id = hf_repo_id
        self._hf_variant = hf_variant
        self._hf_filename = hf_filename
        self._hf_mmproj_filename = hf_mmproj_filename
        self._local_model_path = local_model_path
        self._local_mmproj_path = local_mmproj_path

        self._n_gpu_layers = n_gpu_layers
        self._ctx_size = ctx_size
        self._parallel = parallel
        self._threads = threads
        self._flash_attn = flash_attn
        self._jinja = jinja
        self._alias = alias
        self._hf_token_env = hf_token_env

        self._loaded = False
        self._server_process: Optional[subprocess.Popen[str]] = None
        self._server_log_path: Optional[Path] = None
        self._server_log_handle = None
        self._server_help_text: Optional[str] = None

        if self._enable_thinking:
            logger.warning(
                "Gemma-4 thinking mode is not explicitly supported by the llama.cpp "
                "server backend; responses may include reasoning text."
            )

    @property
    def backend_name(self) -> str:
        return "llama_cpp_server"

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        if self.is_loaded:
            logger.warning("Model already loaded, skipping")
            return

        if self._auto_start:
            command = self._build_server_command()
            self._start_server(command)

        self._wait_until_ready()
        self._loaded = True
        logger.info("llama.cpp server backend ready for model %s", self._model_id)

    def unload_model(self) -> None:
        self._loaded = False

        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait(timeout=5)
            self._server_process = None

        if self._server_log_handle is not None:
            self._server_log_handle.close()
            self._server_log_handle = None

    @property
    def supported_generation_params(self) -> frozenset[str]:
        return frozenset({"temperature", "top_p", "top_k"})

    @property
    def static_generation_params(self) -> dict[str, float | int]:
        return {
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
        }

    def infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_url = self._image_to_data_url(image)
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "max_tokens": self._max_new_tokens,
        }
        payload.update(self.resolve_generation_params(generation_params).effective)

        response = self._request_json(
            "POST",
            f"{self._base_url}/chat/completions",
            payload,
        )
        return self._extract_text_response(response).strip()

    def _build_server_command(self) -> list[str]:
        binary = shutil.which(self._server_path) or self._server_path
        if shutil.which(binary) is None and not Path(binary).exists():
            raise FileNotFoundError(
                "llama-server binary not found. Install llama.cpp and ensure "
                f"`{self._server_path}` is on PATH, or set llama_cpp.server_path "
                "to the full path of llama-server.exe."
            )
        command = [binary]

        if self._hf_repo_id and not self._hf_filename and not self._local_model_path:
            spec = self._hf_repo_id
            if self._hf_variant:
                spec = f"{spec}:{self._hf_variant}"
            command.extend(["-hf", spec])
        else:
            model_path, mmproj_path = self._resolve_model_artifacts()
            command.extend(["-m", model_path])
            if mmproj_path:
                command.extend(["--mmproj", mmproj_path])

        command.extend([
            "--host", self._host,
            "--port", str(self._port),
            "--ctx-size", str(self._ctx_size),
            "--parallel", str(self._parallel),
        ])

        if self._n_gpu_layers is not None:
            command.extend(["--n-gpu-layers", str(self._n_gpu_layers)])
        if self._threads is not None:
            command.extend(["-t", str(self._threads)])
        command.extend(self._flash_attn_args(binary))
        if self._jinja:
            command.append("--jinja")
        command.extend(self._reasoning_args(binary))
        if self._alias:
            command.extend(["--alias", self._alias])

        return command

    def _start_server(self, command: list[str]) -> None:
        log_handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="llama-server-",
            suffix=".log",
            delete=False,
        )
        self._server_log_handle = log_handle
        self._server_log_path = Path(log_handle.name)

        logger.info("Starting llama-server: %s", " ".join(command))
        self._server_process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _flash_attn_args(self, binary: str) -> list[str]:
        mode = self._normalize_flash_attn_mode(self._flash_attn)
        if mode is None:
            return []

        if self._supports_flash_attn_value(binary):
            return ["--flash-attn", mode]

        # Older llama.cpp builds exposed flash attention as a plain flag.
        return ["--flash-attn"] if mode != "off" else []

    def _resolve_model_artifacts(self) -> tuple[str, Optional[str]]:
        model_path = self._resolve_model_file()
        mmproj_path = self._resolve_mmproj_file()
        return model_path, mmproj_path

    def _resolve_model_file(self) -> str:
        if self._local_model_path:
            model_path = Path(self._local_model_path)
            if model_path.exists():
                return str(model_path)
            if not self._hf_repo_id or not self._hf_filename:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            logger.warning(
                "Model file not found at %s; downloading %s from %s to the "
                "Hugging Face cache instead",
                model_path,
                self._hf_filename,
                self._hf_repo_id,
            )
            return self._download_hf_file(self._hf_filename, "model")

        if not self._hf_repo_id or not self._hf_filename:
            raise ValueError(
                "llama_cpp_server backend requires either local_model_path or "
                "hf_repo_id + hf_filename when auto_start is enabled."
            )

        return self._download_hf_file(self._hf_filename, "model")

    def _resolve_mmproj_file(self) -> Optional[str]:
        if self._local_mmproj_path:
            mmproj_path = Path(self._local_mmproj_path)
            if mmproj_path.exists():
                return str(mmproj_path)
            if not self._hf_repo_id or not self._hf_mmproj_filename:
                raise FileNotFoundError(f"mmproj file not found: {mmproj_path}")
            logger.warning(
                "mmproj file not found at %s; downloading %s from %s to the "
                "Hugging Face cache instead",
                mmproj_path,
                self._hf_mmproj_filename,
                self._hf_repo_id,
            )
            return self._download_hf_file(self._hf_mmproj_filename, "mmproj")

        if self._hf_mmproj_filename:
            return self._download_hf_file(self._hf_mmproj_filename, "mmproj")

        return None

    def _download_hf_file(self, filename: str, description: str) -> str:
        try:
            from huggingface_hub import constants, hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for GGUF auto-download. "
                "Install dependencies from requirements.txt."
            ) from exc

        token = (
            os.environ.get(self._hf_token_env)
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        logger.info(
            "Resolving %s artifact %s from %s in Hugging Face cache %s",
            description,
            filename,
            self._hf_repo_id,
            constants.HF_HUB_CACHE,
        )
        return hf_hub_download(
            repo_id=self._hf_repo_id,
            filename=filename,
            token=token,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self._startup_timeout_sec
        last_error: Optional[Exception] = None

        while time.time() < deadline:
            if self._server_process is not None and self._server_process.poll() is not None:
                tail = self._read_server_log_tail()
                raise RuntimeError(
                    "llama-server exited before becoming ready.\n"
                    f"Recent log output:\n{tail}"
                )

            try:
                response = self._request_json(
                    "GET",
                    f"{self._base_url}/models",
                    timeout_sec=min(30, self._request_timeout_sec),
                )
                if response.get("data"):
                    return
            except Exception as exc:  # pragma: no cover - exercised via polling
                last_error = exc

            time.sleep(1.0)

        raise RuntimeError(
            "Timed out waiting for llama-server to become ready."
            + (f" Last error: {last_error}" if last_error else "")
        )

    def _supports_flash_attn_value(self, binary: str) -> bool:
        help_text = self._get_server_help_text(binary)
        return "--flash-attn [on|off|auto]" in help_text

    def _reasoning_args(self, binary: str) -> list[str]:
        if not self._supports_reasoning_flag(binary):
            return []

        mode = "on" if self._enable_thinking else "off"
        return ["--reasoning", mode]

    def _supports_reasoning_flag(self, binary: str) -> bool:
        help_text = self._get_server_help_text(binary)
        return "--reasoning [on|off|auto]" in help_text

    def _get_server_help_text(self, binary: str) -> str:
        if self._server_help_text is not None:
            return self._server_help_text

        try:
            result = subprocess.run(
                [binary, "-h"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except OSError as exc:
            logger.debug("Failed to query llama-server help for %s: %s", binary, exc)
            self._server_help_text = ""
            return self._server_help_text

        self._server_help_text = (result.stdout or "") + (result.stderr or "")
        return self._server_help_text

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Optional[dict[str, Any]] = None,
        timeout_sec: Optional[int] = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        timeout = timeout_sec if timeout_sec is not None else self._request_timeout_sec
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from llama-server: {body}") from exc

    def _extract_text_response(self, response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected llama-server response: {response}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(content)

    def _read_server_log_tail(self, max_chars: int = 4000) -> str:
        if self._server_log_path is None or not self._server_log_path.exists():
            return "(no log output captured)"
        text = self._server_log_path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:] if len(text) > max_chars else text

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    @staticmethod
    def _normalize_flash_attn_mode(value: bool | str) -> Optional[str]:
        if isinstance(value, bool):
            return "on" if value else None

        mode = str(value).strip().lower()
        if mode in {"", "false", "off", "0", "none"}:
            return None
        if mode in {"true", "on", "1", "yes"}:
            return "on"
        if mode == "auto":
            return "auto"

        raise ValueError(
            "flash_attn must be one of: true, false, 'on', 'off', or 'auto'."
        )

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
