"""Transformers-based VLM backend (reference implementation).

Uses Hugging Face Transformers with AutoProcessor and AutoModelForImageTextToText.
Default target model: Qwen/Qwen3-VL-2B-Instruct.
Also supports Google Gemma-4 models (e.g., google/gemma-4-E4B-it).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from PIL import Image

from src.backends.base import VLMBackend, resolve_message_images

logger = logging.getLogger(__name__)

_QWEN_IM_END_TOKEN_ID = 151645
_QWEN_THINK_END_TOKEN_ID = 151668
_QWEN_THINKING_EARLY_STOP_TEXT = (
    "\n\n Considering the limited time by the user, I have to give the "
    "solution based on the thinking directly now.\n</think>\n\n"
)
_QWEN_THINKING_LOG_MAX_CHARS = 4000


class TransformersBackend(VLMBackend):
    """VLM backend using Hugging Face Transformers.

    Parameters
    ----------
    model_id : str
        Model identifier on Hugging Face Hub.
    device_map : str
        Device mapping strategy (default: 'auto').
    max_new_tokens : int
        Maximum tokens to generate per inference call.
    torch_dtype : str
        Tensor dtype: 'auto' uses float16 on CUDA, float32 on CPU.
    enable_thinking : bool
        Enable thinking mode for models whose chat template supports it
        (Gemma-4, Qwen3/Qwen3.5 hybrid thinking models).
    thinking_budget : int | None
        For Qwen models, cap the first thinking pass to this many tokens, then
        append Qwen's early-stopping prompt and generate the final answer.
    temperature : float
        Sampling temperature (default: 1.0 for Gemma-4, None for others).
    top_p : float
        Nucleus sampling parameter (default: 0.95 for Gemma-4, None for others).
    top_k : int
        Top-k sampling parameter (default: 64 for Gemma-4, None for others).
    lora_adapter_path : str | None
        Optional PEFT/LoRA adapter path. When set, the adapter metadata is used
        to select the base model if it declares one.
    lora_base_model_source : {"adapter", "config"}
        Whether LoRA loading should use the base model declared by the adapter
        metadata or the model_id supplied by the runtime config.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        device_map: str = "auto",
        max_new_tokens: int = 64,
        torch_dtype: str = "auto",
        enable_thinking: bool = False,
        thinking_budget: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        lora_adapter_path: Optional[str] = None,
        lora_base_model_source: Optional[str] = "adapter",
    ) -> None:
        self._model_id = model_id
        self._resolved_model_id = model_id
        self._device_map = device_map
        self._max_new_tokens = max_new_tokens
        self._torch_dtype_str = torch_dtype
        self._enable_thinking = enable_thinking
        self._thinking_budget = self._normalize_thinking_budget(thinking_budget)
        self._lora_adapter_path = (
            str(lora_adapter_path).strip() if lora_adapter_path else None
        )
        self._lora_base_model_source = self._normalize_lora_base_model_source(
            lora_base_model_source
        )
        self._qwen_template_kwarg_warning_logged = False
        self._model = None
        self._processor = None

        self._set_model_family_flags(model_id)

        # Set recommended sampling parameters for Gemma-4
        # Source: https://huggingface.co/google/gemma-4-E4B-it
        if self._is_gemma4:
            self._temperature = temperature if temperature is not None else 1.0
            self._top_p = top_p if top_p is not None else 0.95
            self._top_k = top_k if top_k is not None else 64
        else:
            self._temperature = temperature
            self._top_p = top_p
            self._top_k = top_k

        if self._thinking_budget is not None and not self._is_qwen:
            logger.warning(
                "thinking_budget is currently implemented only for Qwen models; "
                "ignoring it for %s",
                self._model_id,
            )
            self._thinking_budget = None
        if (
            self._thinking_budget is not None
            and not self._enable_thinking
            and not self._is_qwen_thinking_model
        ):
            logger.warning(
                "thinking_budget=%d is set while enable_thinking=False; "
                "budget generation will still be used, but the model template "
                "may disable thinking.",
                self._thinking_budget,
            )

    def _set_model_family_flags(self, model_id: str) -> None:
        model_id_lower = model_id.lower()
        # Detect known model families with special chat-template behavior.
        self._is_gemma4 = "gemma-4" in model_id_lower or "gemma4" in model_id_lower
        self._is_qwen = "qwen" in model_id_lower
        self._is_qwen_thinking_model = "thinking" in model_id_lower and self._is_qwen

    def _normalize_lora_base_model_source(self, value: Optional[str]) -> str:
        if value is None:
            return "adapter"
        normalized = str(value).strip().lower()
        if normalized not in {"adapter", "config"}:
            raise ValueError(
                "lora_base_model_source must be 'adapter' or 'config'"
            )
        return normalized

    def _normalize_thinking_budget(self, thinking_budget: Optional[int]) -> Optional[int]:
        if thinking_budget is None:
            return None

        try:
            budget = int(thinking_budget)
        except (TypeError, ValueError) as exc:
            raise ValueError("thinking_budget must be an integer or null") from exc
        if budget <= 0:
            raise ValueError("thinking_budget must be positive when set")
        if budget < self._max_new_tokens:
            logger.warning(
                "thinking_budget=%d is lower than max_new_tokens=%d; clamping "
                "thinking_budget to max_new_tokens",
                budget,
                self._max_new_tokens,
            )
            budget = self._max_new_tokens
        return budget

    @property
    def backend_name(self) -> str:
        return "transformers"

    @property
    def model_name(self) -> str:
        return self._resolved_model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def load_model(self) -> None:
        if self.is_loaded:
            logger.warning("Model already loaded, skipping")
            return

        import torch
        from transformers import AutoProcessor

        if self._torch_dtype_str == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, self._torch_dtype_str)

        base_model_id = self._model_id
        if self._lora_adapter_path is not None:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(self._lora_adapter_path)
            base_model_id = self._base_model_id_from_lora_config(peft_config)
            logger.info(
                "Loading LoRA adapter from %s with base model %s (dtype=%s, device_map=%s)...",
                self._lora_adapter_path, base_model_id, dtype, self._device_map,
            )
        else:
            logger.info(
                "Loading model %s (dtype=%s, device_map=%s)...",
                self._model_id, dtype, self._device_map,
            )
        t0 = time.time()
        self._resolved_model_id = base_model_id
        self._set_model_family_flags(base_model_id)

        # Load processor from adapter path if available, otherwise from the base.
        processor_path = self._lora_adapter_path if self._lora_adapter_path is not None else self._model_id
        try:
            self._processor = AutoProcessor.from_pretrained(processor_path)
        except (OSError, ValueError):
            if self._lora_adapter_path is None:
                raise
            logger.warning(
                "Could not load processor from LoRA adapter %s; falling back to base model %s",
                self._lora_adapter_path,
                base_model_id,
            )
            self._processor = AutoProcessor.from_pretrained(base_model_id)

        # Use appropriate model class based on model type
        # Gemma-4 uses AutoModelForCausalLM for text-only or AutoModelForMultimodalLM for multimodal
        # Source: https://huggingface.co/google/gemma-4-E4B-it#getting-started
        if self._is_gemma4:
            try:
                from transformers import AutoModelForMultimodalLM
                logger.info("Loading Gemma-4 model with AutoModelForMultimodalLM")
                self._model = AutoModelForMultimodalLM.from_pretrained(
                    base_model_id,
                    dtype=dtype,
                    device_map=self._device_map,
                )
            except ImportError:
                # Fallback for older transformers versions
                from transformers import AutoModelForCausalLM
                logger.warning("AutoModelForMultimodalLM not available, using AutoModelForCausalLM")
                self._model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    dtype=dtype,
                    device_map=self._device_map,
                )
        else:
            # Qwen and other VLM models use AutoModelForImageTextToText
            from transformers import AutoModelForImageTextToText
            self._model = AutoModelForImageTextToText.from_pretrained(
                base_model_id,
                dtype=dtype,
                device_map=self._device_map,
                attn_implementation="flash_attention_2"
            )

        # Load LoRA adapter if specified
        if self._lora_adapter_path is not None:
            from peft import PeftModel
            logger.info("Loading LoRA adapter from %s...", self._lora_adapter_path)
            self._model = PeftModel.from_pretrained(
                self._model,
                self._lora_adapter_path,
            )
            set_adapter = getattr(self._model, "set_adapter", None)
            if callable(set_adapter):
                set_adapter("default")
            self._model.eval()
            logger.info(
                "LoRA adapter loaded successfully (active_adapter=%s)",
                self._active_adapter_name(),
            )

        elapsed = time.time() - t0
        logger.info("Model loaded in %.1fs", elapsed)

    def _base_model_id_from_lora_config(self, peft_config) -> str:
        adapter_base_model_id = getattr(peft_config, "base_model_name_or_path", None)
        if self._lora_base_model_source == "config":
            if adapter_base_model_id and adapter_base_model_id != self._model_id:
                logger.warning(
                    "LoRA adapter %s was trained from %s, but "
                    "lora_base_model_source=config; loading requested model %s.",
                    self._lora_adapter_path,
                    adapter_base_model_id,
                    self._model_id,
                )
            return self._model_id

        if not adapter_base_model_id:
            return self._model_id
        if adapter_base_model_id != self._model_id:
            logger.warning(
                "LoRA adapter %s was trained from %s, but config requested %s; "
                "loading the adapter base model instead.",
                self._lora_adapter_path,
                adapter_base_model_id,
                self._model_id,
            )
        return adapter_base_model_id

    def _active_adapter_name(self) -> str:
        if self._model is None:
            return "none"
        for attr_name in ("active_adapter", "active_adapters"):
            active = getattr(self._model, attr_name, None)
            try:
                value = active() if callable(active) else active
            except Exception:
                continue
            if value is not None:
                return str(value)
        return "unknown"

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Model unloaded")

    @property
    def supported_generation_params(self) -> frozenset[str]:
        return frozenset({"temperature", "top_p", "top_k"})

    @property
    def static_generation_params(self) -> dict[str, float | int]:
        return {
            name: value
            for name, value in (
                ("temperature", self._temperature),
                ("top_p", self._top_p),
                ("top_k", self._top_k),
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        return self.infer_messages(
            image,
            messages,
            generation_params=generation_params,
        )

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
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        inputs = self._build_inputs(image, messages)

        inputs = inputs.to(self._model.device)

        gen_kwargs = self._generation_kwargs(generation_params)

        # Generate
        with torch.inference_mode():
            if self._use_qwen_thinking_budget():
                output_ids = self._generate_with_qwen_thinking_budget(
                    inputs,
                    gen_kwargs,
                )
            else:
                output_ids = self._model.generate(**inputs, **gen_kwargs)

        return self._decode_response(output_ids, inputs).strip()

    def _build_inputs(
        self,
        image: Image.Image,
        messages: list[dict[str, Any]],
    ):
        resolved_messages, images = resolve_message_images(messages, image)
        # Apply chat template with Gemma-4 specific parameters.
        # Source: https://huggingface.co/google/gemma-4-E4B-it#getting-started
        if self._is_gemma4:
            return self._processor.apply_chat_template(
                resolved_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=self._enable_thinking,
            )

        # Qwen3/Qwen3.5 templates accept enable_thinking in current
        # Transformers. Keep a fallback for older processors and thinking-only
        # checkpoints whose template does not expose the kwarg.
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._is_qwen and self._chat_template_supports_enable_thinking():
            template_kwargs["enable_thinking"] = self._enable_thinking

        try:
            text = self._processor.apply_chat_template(messages, **template_kwargs)
        except TypeError as exc:
            if "enable_thinking" not in template_kwargs:
                raise
            if not self._qwen_template_kwarg_warning_logged:
                logger.warning(
                    "Processor chat template for %s does not accept "
                    "enable_thinking; retrying without it: %s",
                    self._model_id,
                    exc,
                )
                self._qwen_template_kwarg_warning_logged = True
            template_kwargs.pop("enable_thinking", None)
            text = self._processor.apply_chat_template(messages, **template_kwargs)

        return self._processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        )

    def _chat_template_supports_enable_thinking(self) -> bool:
        chat_template = getattr(self._processor, "chat_template", None)
        if isinstance(chat_template, dict):
            chat_template = chat_template.get("default", "")
        if chat_template is None:
            return False
        return "enable_thinking" in str(chat_template)

    def _generation_kwargs(
        self,
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> dict[str, Any]:
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self._max_new_tokens}
        resolved = self.resolve_generation_params(generation_params).effective

        # Add sampling parameters if specified.
        # Gemma-4 recommended: temperature=1.0, top_p=0.95, top_k=64.
        # Qwen recommends sampling for thinking mode; greedy decoding can
        # encourage repetition loops.
        if resolved:
            gen_kwargs["do_sample"] = resolved.get("temperature") != 0
        gen_kwargs.update(resolved)
        return gen_kwargs

    def _use_qwen_thinking_budget(self) -> bool:
        return self._is_qwen and self._thinking_budget is not None

    def _generate_with_qwen_thinking_budget(
        self,
        inputs,
        gen_kwargs: dict[str, Any],
    ):
        """Apply Qwen's documented two-pass thinking-budget workaround.

        The first pass is capped by thinking_budget. If the model has not ended
        the response or closed the thinking block, append Qwen's early-stop text
        and run a second pass for the configured answer token budget.
        """
        import torch

        input_len = inputs["input_ids"].shape[-1]
        budget_kwargs = dict(gen_kwargs)
        budget_kwargs["max_new_tokens"] = self._thinking_budget
        generated_ids = self._model.generate(**inputs, **budget_kwargs)
        output_ids = generated_ids[0][input_len:].tolist()

        im_end_id = self._token_id("<|im_end|>", _QWEN_IM_END_TOKEN_ID)
        think_end_id = self._token_id("</think>", _QWEN_THINK_END_TOKEN_ID)
        if im_end_id in output_ids:
            return generated_ids

        if think_end_id not in output_ids:
            self._log_qwen_truncated_thinking(output_ids)
            early_stopping_ids = self._tokenize_for_continuation(
                _QWEN_THINKING_EARLY_STOP_TEXT
            )
            input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
        else:
            input_ids = generated_ids

        second_inputs = self._build_qwen_continuation_inputs(inputs, input_ids)
        second_kwargs = dict(gen_kwargs)
        second_kwargs["max_new_tokens"] = self._max_new_tokens
        return self._model.generate(**second_inputs, **second_kwargs)

    def _log_qwen_truncated_thinking(self, output_ids: list[int]) -> None:
        text = self._decode_qwen_token_ids_for_log(output_ids)
        if len(text) > _QWEN_THINKING_LOG_MAX_CHARS:
            omitted = len(text) - _QWEN_THINKING_LOG_MAX_CHARS
            text = (
                text[:_QWEN_THINKING_LOG_MAX_CHARS]
                + f"\n... [truncated log output; {omitted} chars omitted]"
            )
        if not text:
            text = "(empty)"

        logger.info(
            "Qwen thinking budget reached after %d generated token(s); "
            "forcing </think>\nTruncated thinking output:\n%s",
            len(output_ids),
            text,
        )

    def _decode_qwen_token_ids_for_log(self, output_ids: list[int]) -> str:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        try:
            return tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        except Exception as exc:
            logger.warning("Failed to decode Qwen thinking output for log: %s", exc)
            return repr(output_ids)

    def _build_qwen_continuation_inputs(self, inputs, input_ids):
        import torch

        original_len = inputs["input_ids"].shape[-1]
        new_len = input_ids.shape[-1]
        pad_len = new_len - original_len

        second_inputs = dict(inputs)
        if pad_len > 0:
            for key, value in list(second_inputs.items()):
                if key in {"input_ids", "attention_mask"}:
                    continue
                if not torch.is_tensor(value):
                    continue
                if value.ndim < 2 or value.shape[-1] != original_len:
                    continue

                pad_shape = list(value.shape)
                pad_shape[-1] = pad_len
                pad = torch.zeros(
                    pad_shape,
                    dtype=value.dtype,
                    device=input_ids.device,
                )
                second_inputs[key] = torch.cat(
                    [value.to(input_ids.device), pad],
                    dim=-1,
                )

        second_inputs["input_ids"] = input_ids
        second_inputs["attention_mask"] = torch.ones_like(
            input_ids,
            dtype=torch.int64,
            device=input_ids.device,
        )
        return second_inputs

    def _tokenize_for_continuation(self, text: str):
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        encoded = tokenizer(
            [text],
            return_tensors="pt",
            return_attention_mask=False,
        )
        return encoded.input_ids.to(self._model.device)

    def _token_id(self, token: str, default: int) -> int:
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            return default
        token_id = convert(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
        return default

    def _decode_response(self, output_ids, inputs) -> str:
        # Decode response
        if self._is_gemma4:
            # For Gemma-4, decode and parse response to extract thinking/final answer
            # Source: https://huggingface.co/google/gemma-4-E4B-it#getting-started
            input_len = inputs["input_ids"].shape[-1]
            response_raw = self._processor.decode(
                output_ids[0][input_len:], skip_special_tokens=False
            )
            # Use parse_response to handle thinking output
            # Returns dict with 'thinking' and 'text' keys
            try:
                parsed = self._processor.parse_response(response_raw)
                # Extract final text, ignoring thinking output for gameplay
                if isinstance(parsed, dict):
                    response = parsed.get("text", response_raw)
                else:
                    response = str(parsed)
            except Exception as e:
                logger.warning("Failed to parse Gemma-4 response: %s, using raw output", e)
                response = response_raw
        elif self._is_qwen:
            response = self._decode_qwen_response(output_ids, inputs)
        else:
            # For other models, decode only generated tokens
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            response = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            # Strip thinking tags if present (Qwen3 models may output <think>...</think>)
            response = _strip_thinking_tags(response)

        return response

    def _decode_qwen_response(self, output_ids, inputs) -> str:
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, input_len:]
        generated_token_ids = generated_ids[0].tolist()
        think_end_id = self._token_id("</think>", _QWEN_THINK_END_TOKEN_ID)

        if think_end_id in generated_token_ids:
            index = len(generated_token_ids) - generated_token_ids[::-1].index(
                think_end_id
            )
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            return tokenizer.decode(
                generated_token_ids[index:],
                skip_special_tokens=True,
            )

        response = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return _strip_thinking_tags(response)


def _strip_thinking_tags(text: str) -> str:
    """Remove Qwen/Gemma-style thinking blocks from model output."""
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    lower = text.lower()
    close_idx = lower.rfind("</think>")
    if close_idx != -1:
        text = text[close_idx + len("</think>") :]

    lower = text.lower()
    open_idx = lower.find("<think>")
    if open_idx != -1:
        text = text[:open_idx]

    text = re.sub(r"<\|(?:im_end|endoftext)\|>", "", text)
    return text.strip()
