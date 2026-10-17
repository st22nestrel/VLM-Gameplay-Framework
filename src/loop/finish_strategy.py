"""Game-finished detection and handling strategies.

Two strategies:
  - ask_model: ask the VLM to explain why the game appears finished, then stop.
  - ask_user:  pause for user confirmation; user can provide a correction prompt
               to continue, or confirm to stop.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
from typing import Optional

from PIL import Image

from src.backends.base import VLMBackend
from src.actions.schema import ParsedAction

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FinishResult:
    """Result of a finish strategy check."""

    action: str  # "stop" or "continue"
    model_explanation: Optional[str] = None
    user_correction: Optional[str] = None


class FinishDetector:
    """Detects when the VLM thinks the game is finished.

    Triggers when the VLM response contains a finish keyword AND no valid
    actions were parsed from it.
    """

    def __init__(self, keywords: list[str]) -> None:
        self._keywords = [kw.lower() for kw in keywords]

    def is_finished(self, response: str, parsed_actions: list[ParsedAction]) -> bool:
        if parsed_actions:
            return False
        lower = response.lower()
        return any(kw in lower for kw in self._keywords)


class FinishStrategy(abc.ABC):
    """Base class for game-finished handling strategies."""

    @abc.abstractmethod
    def handle(
        self,
        response: str,
        image: Image.Image,
        backend: VLMBackend,
        system_prompt: str,
    ) -> FinishResult:
        """Handle a detected finish signal.

        Parameters
        ----------
        response : str
            The VLM response that triggered the finish detection.
        image : PIL.Image.Image
            The latest game screenshot.
        backend : VLMBackend
            The backend for follow-up inference (ask_model strategy).
        system_prompt : str
            Current system prompt.

        Returns
        -------
        FinishResult
            Whether to stop or continue, with optional context.
        """
        ...


class AskModelStrategy(FinishStrategy):
    """Ask the model to explain why the game appears finished, then stop."""

    def handle(self, response, image, backend, system_prompt):
        logger.info("Finish detected. Asking model for explanation...")
        explanation_prompt = (
            "The game appears to be finished. "
            "Please explain why the game ended and what happened."
        )
        try:
            explanation = backend.infer(image, system_prompt, explanation_prompt)
        except Exception as e:
            logger.error("Failed to get model explanation: %s", e)
            explanation = f"(error: {e})"

        logger.info("Model explanation: %s", explanation)
        return FinishResult(action="stop", model_explanation=explanation)


class AskUserStrategy(FinishStrategy):
    """Pause for user confirmation before stopping."""

    def handle(self, response, image, backend, system_prompt):
        print("\n" + "=" * 60)
        print("GAME FINISH DETECTED")
        print(f"VLM response: {response!r}")
        print("=" * 60)
        print("\nOptions:")
        print("  [Enter]  - confirm finished, stop the loop")
        print("  [text]   - provide correction prompt, continue playing")
        user_input = input("\n> ").strip()

        if not user_input:
            return FinishResult(action="stop")

        logger.info("User correction: %s", user_input)
        return FinishResult(action="continue", user_correction=user_input)


def create_finish_strategy(name: str) -> FinishStrategy:
    """Factory for finish strategies."""
    strategies = {
        "ask_model": AskModelStrategy,
        "ask_user": AskUserStrategy,
    }
    cls = strategies.get(name)
    if cls is None:
        logger.warning("Unknown finish strategy %r, defaulting to ask_user", name)
        cls = AskUserStrategy
    return cls()
