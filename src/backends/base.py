"""Base interface for VLM backends."""

from __future__ import annotations

import abc
import dataclasses
import logging
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GenerationParamsResolution:
    """Generation parameters a backend will send and ignore."""

    effective: dict[str, float | int]
    ignored: dict[str, float | int]


class VLMBackend(abc.ABC):
    """Abstract base class for VLM inference backends.

    All backends must implement load_model, unload_model, and infer.
    """

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend (e.g. 'transformers', 'lmdeploy')."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model identifier (e.g. 'Qwen/Qwen3-VL-2B-Instruct')."""
        ...

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded and ready for inference."""
        ...

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load the model into memory. Must be called before infer()."""
        ...

    @abc.abstractmethod
    def unload_model(self) -> None:
        """Release model resources."""
        ...

    @property
    def supported_generation_params(self) -> frozenset[str]:
        """Per-inference generation parameter names supported by this backend."""
        return frozenset()

    @property
    def static_generation_params(self) -> dict[str, float | int]:
        """Configured parameters sent when no per-inference override is supplied."""
        return {}

    def resolve_generation_params(
        self,
        generation_params: Optional[dict[str, Any]] = None,
    ) -> GenerationParamsResolution:
        """Merge supported per-inference overrides with backend static settings."""
        effective = dict(self.static_generation_params)
        ignored: dict[str, float | int] = {}
        for name, value in (generation_params or {}).items():
            if name in self.supported_generation_params:
                effective[name] = value
            else:
                ignored[name] = value

        if ignored:
            warned = getattr(self, "_ignored_generation_params_warned", set())
            for name in sorted(ignored):
                if name not in warned:
                    logger.warning(
                        "Backend %s does not support generation parameter %s; "
                        "it will be ignored",
                        self.backend_name,
                        name,
                    )
                    warned.add(name)
            self._ignored_generation_params_warned = warned

        return GenerationParamsResolution(effective=effective, ignored=ignored)

    @abc.abstractmethod
    def infer(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> str:
        """Run inference on an image with the given prompts.

        Parameters
        ----------
        image : PIL.Image.Image
            The game screenshot.
        system_prompt : str
            System-level instructions (stable across turns).
        user_prompt : str
            Per-turn user prompt.
        generation_params : dict, optional
            Per-inference overrides for supported generation parameters.

        Returns
        -------
        str
            The model's text response.
        """
        ...


def resolve_message_images(
    messages: list[dict[str, Any]], current_image: Image.Image,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Bind image parts to pixels in message order without mutating the caller.

    An image part with no ``image`` value refers to the current screenshot;
    historical parts carry a saved image path (or an already loaded PIL image).
    """
    resolved = []
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            resolved.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                source = part.get("image")
                if source is None:
                    image = current_image
                elif isinstance(source, Image.Image):
                    image = source
                else:
                    with Image.open(source) as saved:
                        image = saved.convert("RGB")
                images.append(image)
                parts.append({**part, "image": image})
            else:
                parts.append(part)
        resolved.append({**message, "content": parts})
    return resolved, images
