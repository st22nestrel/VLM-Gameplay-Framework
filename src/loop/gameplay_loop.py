"""Live VLM gameplay loop: capture -> infer -> parse -> execute -> repeat.

By default, the next frame is captured synchronously after the previous VLM
action executes. A continuous background strategy remains available for games
where real-time sampling is more important than strict action/frame ordering.
"""

from __future__ import annotations

import datetime
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageChops, ImageDraw, ImageStat
import yaml

from src.window_session import WindowSession
from src.capture import windows_graphics_capture, fallback_capture, dxcam_capture
from src.capture.cursor_overlay import save_frame_screenshots
from src.capture.frame import CapturedFrame
from src.actions.schema import ParsedAction
from src.actions.parser import parse_vlm_text
from src.actions.state import ActionState, TickPlan
from src.actions.executor import ActionExecutor
from src.backends.base import VLMBackend
from src.loop.gameplay_config import GameplayRuntimeConfig, load_gameplay_config
from src.loop.generation_params import (
    GenerationParamsSampler,
    GenerationParamsSelection,
    GenerationParamsUsageTracker,
)
from src.loop.coord_utils import normalize_xy_from_0_1000
from src.loop.frame_queue import FrameQueue
from src.loop.finish_strategy import (
    FinishDetector,
    FinishResult,
    create_finish_strategy,
)
from src.loop.hotkey import PauseController
from src.loop.speedhack import FreezeController

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
You are playing a first-person shooter game. Your goal is to complete the level objectives.
Available actions: move forward, move backward, move left, move right, shoot, aim x y
Respond with ONLY the action(s) to take. Use "and" to combine actions.
Examples: "move forward", "move forward and left", "shoot", "aim 100 200"\
"""

_DEFAULT_USER_PROMPT = (
    "Look at the current game screenshot. What action(s) should you take next?"
)

_DEFAULT_FINISH_KEYWORDS = [
    "game over",
    "level complete",
    "mission accomplished",
    "finished",
    "completed",
    "objective complete",
]

_DEFAULT_HISTORY_USER_TEMPLATE = (
    "Previous turn {tick} prompt:\n{user_prompt}"
)
_DEFAULT_ACTION_HISTORY_USER_TEMPLATE = "Previous actions ({action_order}): {actions}"
_DEFAULT_HISTORY_ASSISTANT_TEMPLATE = "{response}"
_DEFAULT_COMPACT_USER_PROMPT = "Summary of earlier gameplay turns:"
_DEFAULT_COMPACT_ASSISTANT_TEMPLATE = """\
You are playing the game and need to continue from the current screenshot.
Earlier completed turns:
{turns}\
"""
_DEFAULT_COMPACT_TURN_TEMPLATE = (
    "Tick {tick}: prompt={user_prompt!r}; response={response!r}; "
    "parsed_actions={parsed_actions}."
)
_DEFAULT_STATE_CONTEXT_TEMPLATE = (
    "{stuck_reason} {stuck_hint}"
)
_DEFAULT_STUCK_HINT = (
    "Prefer drop, move_backward, side movement or any action that differs from "
    "previous action."
)
_DEFAULT_STUCK_REASON_ACTION = (
    "Previous action was repeated {repeated_action_count} times and the screen "
    "did not change much."
)
_DEFAULT_STUCK_REASON_SCREEN = (
    "The screen did not change much for {low_screen_change_count} checks in a row."
)
_DEFAULT_VISUAL_CONTEXT_NOTE = (
    "Visual context: the attached image has two labeled panels. LEFT/PREVIOUS "
    "is the screenshot from the previous VLM request; RIGHT/CURRENT is the "
    "latest screenshot for this prompt. Choose the next action for RIGHT/CURRENT."
)


@dataclass(frozen=True)
class MessageHistoryConfig:
    """Configuration for adding previous gameplay turns to model messages."""

    mode: str = "none"
    last_n: int = 5
    image_last_n: Optional[int] = None
    action_order: str = "oldest_first"
    history_user_template: str = _DEFAULT_HISTORY_USER_TEMPLATE
    history_assistant_template: str = _DEFAULT_HISTORY_ASSISTANT_TEMPLATE
    compact_max_turns: int = 5
    compact_as: str = "user_assistant"
    compact_user_prompt: str = _DEFAULT_COMPACT_USER_PROMPT
    compact_assistant_template: str = _DEFAULT_COMPACT_ASSISTANT_TEMPLATE
    compact_turn_template: str = _DEFAULT_COMPACT_TURN_TEMPLATE

    @classmethod
    def from_config(cls, cfg: Any) -> "MessageHistoryConfig":
        if not isinstance(cfg, dict):
            return cls()

        mode = str(cfg.get("mode", "none")).strip().lower()
        mode_aliases = {
            "": "none",
            "off": "none",
            "false": "none",
            "disabled": "none",
            "no_history": "none",
            "current": "none",
            "last": "last_n",
            "last_n_turns": "last_n",
            "compact_context": "compact",
            "compaction": "compact",
        }
        mode = mode_aliases.get(mode, mode)
        if mode not in {"none", "all", "last_n", "compact", "actions"}:
            logger.warning("Unknown message_history.mode=%r; using 'none'", mode)
            mode = "none"

        compaction_cfg = cfg.get("compaction", {})
        if not isinstance(compaction_cfg, dict):
            compaction_cfg = {}

        last_n = _positive_int(cfg.get("last_n", cfg.get("max_turns", 5)), 5)
        action_order = str(cfg.get("action_order", "oldest_first")).strip().lower()
        if action_order not in {"oldest_first", "newest_first"}:
            logger.warning(
                "Unknown message_history.action_order=%r; using 'oldest_first'",
                action_order,
            )
            action_order = "oldest_first"
        compact_max_turns = _positive_int(
            compaction_cfg.get(
                "max_turns",
                cfg.get("compact_after", cfg.get("max_uncompacted_turns", 5)),
            ),
            5,
        )

        compact_as = str(
            compaction_cfg.get(
                "message_shape",
                compaction_cfg.get("as", cfg.get("compact_as", "user_assistant")),
            )
        ).strip().lower()
        compact_shape_aliases = {
            "pair": "user_assistant",
            "user-assistant": "user_assistant",
            "user_assistant_pair": "user_assistant",
            "assistant": "assistant_only",
            "assistant-only": "assistant_only",
        }
        compact_as = compact_shape_aliases.get(compact_as, compact_as)
        if compact_as not in {"user_assistant", "assistant_only"}:
            logger.warning(
                "Unknown message_history.compaction.message_shape=%r; "
                "using 'user_assistant'",
                compact_as,
            )
            compact_as = "user_assistant"

        image_last_n = cfg.get("image_last_n")
        if image_last_n is not None:
            try:
                if isinstance(image_last_n, bool) or int(image_last_n) != float(image_last_n):
                    raise ValueError
                image_last_n = int(image_last_n)
                if image_last_n < 0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                logger.warning(
                    "Invalid message_history.image_last_n=%r; following message history",
                    image_last_n,
                )
                image_last_n = None

        return cls(
            mode=mode,
            last_n=last_n,
            image_last_n=image_last_n,
            action_order=action_order,
            history_user_template=str(
                cfg.get(
                    "history_user_template",
                    _DEFAULT_ACTION_HISTORY_USER_TEMPLATE
                    if mode == "actions" else _DEFAULT_HISTORY_USER_TEMPLATE,
                )
            ),
            history_assistant_template=str(
                cfg.get(
                    "history_assistant_template",
                    _DEFAULT_HISTORY_ASSISTANT_TEMPLATE,
                )
            ),
            compact_max_turns=compact_max_turns,
            compact_as=compact_as,
            compact_user_prompt=str(
                compaction_cfg.get(
                    "user_prompt",
                    cfg.get("compact_user_prompt", _DEFAULT_COMPACT_USER_PROMPT),
                )
            ),
            compact_assistant_template=str(
                compaction_cfg.get(
                    "assistant_template",
                    cfg.get(
                        "compact_assistant_template",
                        _DEFAULT_COMPACT_ASSISTANT_TEMPLATE,
                    ),
                )
            ),
            compact_turn_template=str(
                compaction_cfg.get(
                    "turn_template",
                    cfg.get("compact_turn_template", _DEFAULT_COMPACT_TURN_TEMPLATE),
                )
            ),
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "last_n": self.last_n,
            "image_last_n": self.image_last_n,
            "action_order": self.action_order,
            "history_user_template": self.history_user_template,
            "compact_max_turns": self.compact_max_turns,
            "compact_as": self.compact_as,
        }


@dataclass(frozen=True)
class StateContextConfig:
    """Configuration for adding cheap runtime state to the current prompt."""

    enabled: bool = False
    repeat_threshold: int = 2
    screen_change_repeat_threshold: int = 2
    screen_change_threshold: float = 0.02
    stuck_hint: str = _DEFAULT_STUCK_HINT
    template: str = _DEFAULT_STATE_CONTEXT_TEMPLATE

    @classmethod
    def from_config(cls, cfg: Any) -> "StateContextConfig":
        if not isinstance(cfg, dict):
            return cls()

        return cls(
            enabled=_bool_value(cfg.get("enabled", False), False),
            repeat_threshold=_positive_int(
                cfg.get("repeat_threshold", cfg.get("repeated_action_threshold", 2)),
                2,
            ),
            screen_change_repeat_threshold=_positive_int(
                cfg.get(
                    "screen_change_repeat_threshold",
                    cfg.get("low_screen_change_repeat_threshold", 2),
                ),
                2,
            ),
            screen_change_threshold=_float_value(
                cfg.get("screen_change_threshold", 0.02),
                0.02,
            ),
            stuck_hint=str(cfg.get("stuck_hint", _DEFAULT_STUCK_HINT)).strip()
            or _DEFAULT_STUCK_HINT,
            template=str(cfg.get("template", _DEFAULT_STATE_CONTEXT_TEMPLATE)),
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "repeat_threshold": self.repeat_threshold,
            "screen_change_repeat_threshold": self.screen_change_repeat_threshold,
            "screen_change_threshold": self.screen_change_threshold,
        }


@dataclass(frozen=True)
class VisualContextConfig:
    """Configuration for sending a previous/current combined screenshot."""

    mode: str = "off"
    panel_width: int = 384
    label_height: int = 28
    prompt_note: str = _DEFAULT_VISUAL_CONTEXT_NOTE

    @classmethod
    def from_config(cls, cfg: Any) -> "VisualContextConfig":
        if not isinstance(cfg, dict):
            return cls()

        mode = str(cfg.get("mode", "off")).strip().lower()
        aliases = {
            "": "off",
            "disabled": "off",
            "false": "off",
            "none": "off",
            "always": "always_last_2",
            "last_2": "always_last_2",
            "last_two": "always_last_2",
            "always_last_two": "always_last_2",
            "stuck": "on_stuck",
            "when_stuck": "on_stuck",
            "stuck_only": "on_stuck",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"off", "always_last_2", "on_stuck"}:
            logger.warning("Unknown visual_context.mode=%r; using 'off'", mode)
            mode = "off"

        return cls(
            mode=mode,
            panel_width=_positive_int(cfg.get("panel_width", 384), 384),
            label_height=_positive_int(cfg.get("label_height", 28), 28),
            prompt_note=str(cfg.get("prompt_note", _DEFAULT_VISUAL_CONTEXT_NOTE)).strip()
            or _DEFAULT_VISUAL_CONTEXT_NOTE,
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "panel_width": self.panel_width,
            "label_height": self.label_height,
        }


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _float_value(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)


def _duration_ms_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _format_template(
    template: str,
    values: dict[str, Any],
    fallback: Optional[str] = None,
    warning_context: str = "message history",
) -> str:
    try:
        return template.format(**values)
    except Exception as exc:
        logger.warning("Failed to format %s template: %s", warning_context, exc)
        if fallback is not None:
            try:
                return fallback.format(**values)
            except Exception:
                return fallback
        return str(values)


# -- capture helper -----------------------------------------------------------


def _do_capture(session: WindowSession, method: str) -> Optional[CapturedFrame]:
    """Perform a single capture using the specified method.

    Args:
        session: WindowSession for the target window
        method: Capture method to use ("dxcam", "mss", "printwindow", "bitblt", "auto")
    """
    if method == "dxcam":
        return dxcam_capture.capture_frame(session)
    if method == "mss":
        return fallback_capture.capture_frame(session)
    if method in ("printwindow", "bitblt"):
        return windows_graphics_capture.capture_frame(session)
    # auto fallback chain
    if dxcam_capture.is_available():
        frame = dxcam_capture.capture_frame(session)
        if frame is not None:
            return frame
    frame = windows_graphics_capture.capture_frame(session)
    if frame is not None:
        return frame
    return fallback_capture.capture_frame(session)


# -- gameplay loop ------------------------------------------------------------


class GameplayLoop:
    """Live VLM gameplay loop.

    Parameters
    ----------
    session : WindowSession
        Attached target window.
    config : GameplayRuntimeConfig
        Normalized runtime config loaded by the caller.
    backend : VLMBackend
        Loaded or loadable VLM backend owned by the caller.
    output_dir : Path | None
        Override output directory. If None, auto-generated under gameplay.output_dir.
    dry_run : bool
        If True, parse and plan actions but don't execute them.
    no_wait : bool
        Skip the wait-for-ready prompt.
    max_ticks : int | None
        Stop after this many ticks (None = unlimited, until finish or Ctrl-C).
    delay_ms : int | None
        Delay before action execution starts (ms). None means use config, default 3000.
    generation_params_seed : int | None
        Effective sampler seed supplied by a suite run. None uses the config seed.
    propagate_keyboard_interrupt : bool
        If True, re-raise KeyboardInterrupt after cleanup and log writing.
    """

    def __init__(
        self,
        session: WindowSession,
        config: GameplayRuntimeConfig,
        backend: VLMBackend,
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        no_wait: bool = False,
        max_ticks: Optional[int] = None,
        delay_ms: Optional[int] = None,
        generation_params_seed: Optional[int] = None,
        propagate_keyboard_interrupt: bool = False,
    ) -> None:
        self.session = session
        self.config = config
        self.raw_config = config.raw_config

        self._backend_name = config.backend_name
        self._model_id = config.model_id
        self._max_new_tokens = config.max_new_tokens

        model_settings = config.model_settings
        self._enable_thinking = model_settings.enable_thinking
        self._thinking_budget = model_settings.thinking_budget
        self._temperature = model_settings.temperature
        self._top_p = model_settings.top_p
        self._top_k = model_settings.top_k
        self._generation_params_config = model_settings.generation_params
        self._generation_params_sampler = GenerationParamsSampler(
            self._generation_params_config,
            effective_seed=generation_params_seed,
        )
        self._generation_params_usage = GenerationParamsUsageTracker(
            self._generation_params_config,
            effective_seed=self._generation_params_sampler.effective_seed,
        )
        self._normalize_coordinates_from_0_1000 = (
            model_settings.normalize_coordinates_from_0_1000
        )
        self._lora_adapter_path = model_settings.lora_adapter_path
        self._lora_base_model_source = model_settings.lora_base_model_source
        self._llama_cpp_cfg = config.llama_cpp_config
        self._openai_cfg = config.openai_config

        # Prompts
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt
        self._message_history_cfg = config.message_history
        self._state_context_cfg = config.state_context
        self._visual_context_cfg = config.visual_context
        self._progress_cfg = config.progress_tracking
        self._log_vlm_io = config.log_vlm_io
        self._action_hold_duration_ms = config.action_hold_duration_ms
        self._action_hold_release_at: Optional[float] = None
        self._history_backend_warning_logged = False

        # Freeze-while-waiting-on-VLM config
        self._freeze_cfg = config.freeze
        self._freeze_controller: Optional[FreezeController] = None

        # Capture config
        self.capture_method = config.capture_method
        self.capture_strategy = config.capture_strategy
        self.post_action_settle_ms = config.post_action_settle_ms
        self.cursor_mode = config.cursor_mode
        self.vlm_image_mode = config.vlm_image_mode
        self.frame_cadence_ms = config.frame_cadence_ms
        self.inference_interval_ms = config.inference_interval_ms
        self._last_inference_started_at: Optional[float] = None

        # Finish config
        self.finish_strategy_name = config.finish_strategy_name
        self.finish_detector = FinishDetector(config.finish_keywords)
        self.finish_strategy = create_finish_strategy(self.finish_strategy_name)

        # Action config
        self.action_config = config.action_config
        self.disable_center_delta_conversion = config.disable_center_delta_conversion

        # Pause/resume hotkey
        self.pause_key: Optional[str] = config.pause_key
        self._pause_controller: Optional[PauseController] = None

        # Runtime options
        self.dry_run = dry_run
        self.no_wait = no_wait
        self.max_ticks = max_ticks
        self.propagate_keyboard_interrupt = propagate_keyboard_interrupt
        self.interrupted = False

        # Delay: CLI > config > default 3000
        if delay_ms is not None:
            self.delay_ms = delay_ms
        else:
            self.delay_ms = config.delay_ms

        self._backend = backend

        # Output directory
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = config.output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frame queue and capture thread
        self._frame_queue = FrameQueue()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._frames_captured = 0
        self._post_action_settle_until: Optional[float] = None
        self._frame_not_before: Optional[float] = None

        # Tick log
        self._tick_log: list[dict] = []
        self._vlm_io_log: list[dict[str, Any]] = []
        self._last_inference_error: Optional[str] = None
        self._last_state_context_image: Optional[Image.Image] = None

        # Turn-progress tracking (fixed-region change detection)
        self._progress_reference_crop: Optional[Image.Image] = None
        self._last_progress_crop: Optional[Image.Image] = None
        self._progress_region_active: bool = False
        self._progress_count: int = 0
        self._progress_lock = threading.Lock()
        self._progress_samples: list[dict[str, Any]] = []
        self._progress_samples_consumed = 0

    # -- capture thread -------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background capture thread: captures frames at configured cadence."""
        interval = self.frame_cadence_ms / 1000.0
        next_capture = time.monotonic()
        while not self._stop_event.is_set():
            try:
                frame = _do_capture(self.session, self.capture_method)
                if frame is not None:
                    self._frames_captured += 1
                    self._sample_background_progress(frame)
                    self._frame_queue.put(frame)
                else:
                    logger.debug("Capture returned None")
            except Exception as e:
                logger.error("Capture error: %s", e)
            next_capture = max(next_capture + interval, time.monotonic())
            self._stop_event.wait(max(0.0, next_capture - time.monotonic()))

    def _start_capture_thread(self) -> None:
        if self.capture_strategy != "background":
            return
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="capture-thread"
        )
        self._capture_thread.start()
        logger.info("Capture thread started (cadence=%dms)", self.frame_cadence_ms)

    def _stop_capture_thread(self) -> None:
        if self._capture_thread is None:
            return
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        logger.info("Capture thread stopped")

    # -- wait for ready -------------------------------------------------------

    def wait_for_ready(self) -> None:
        print("\n" + "=" * 60)
        print("GAMEPLAY LOOP READY")
        print(f"  Backend:         {self._backend.backend_name}")
        print(f"  Model:           {self._backend.model_name}")
        print(f"  Frame cadence:   {self.frame_cadence_ms}ms")
        print(f"  Min request interval: {self.inference_interval_ms}ms")
        print(f"  Capture strategy: {self.capture_strategy}")
        if self.capture_strategy == "after_action":
            print(f"  Action settle:   {self.post_action_settle_ms}ms")
        print(f"  Finish strategy: {self.finish_strategy_name}")
        print(f"  Message history: {self._message_history_cfg.mode}")
        if self._state_context_cfg.enabled:
            print("  State context:   enabled")
        if self._visual_context_cfg.mode != "off":
            print(f"  Visual context:  {self._visual_context_cfg.mode}")
        if self._action_hold_duration_ms is not None:
            print(f"  Action hold:     {self._action_hold_duration_ms}ms")
        if self._freeze_cfg.enabled:
            print("  Freeze on VLM:   enabled")
        print(f"  Screenshots:     save={self.cursor_mode}, vlm={self.vlm_image_mode}")
        print(f"  Target:          {self.session.target.short_str()}")
        print(f"  Output:          {self.output_dir}")
        if self.dry_run:
            print("  Dry run:         YES (no input will be sent)")
        if self.max_ticks:
            print(f"  Max ticks:       {self.max_ticks}")
        print(f"  Delay:           {self.delay_ms}ms")
        if self.pause_key:
            print(f"  Pause/resume:    {self.pause_key.upper()}")
        print("=" * 60)
        print("\nSwitch to the game and prepare the level.")
        user_input = input(
            "Press ENTER when ready to begin, or type delay in ms + ENTER: "
        ).strip()
        if user_input:
            try:
                self.delay_ms = int(user_input)
                print(f"  Delay overridden to {self.delay_ms}ms")
            except ValueError:
                print(f"  Ignoring invalid delay value: {user_input!r}")
        print()

    # -- message history ------------------------------------------------------

    @staticmethod
    def _action_signature(record: dict[str, Any]) -> str:
        parsed_actions_raw = record.get("parsed_actions") or []
        names: list[str] = []
        for item in parsed_actions_raw:
            if isinstance(item, dict):
                name = str(item.get("canonical_name", "")).strip()
            else:
                name = str(item).strip()
            if name:
                names.append(name)
        return "+".join(names) if names else "none"

    def _recent_action_repetition(self) -> tuple[str, int]:
        if not self._tick_log:
            return "none", 0

        last_action = self._action_signature(self._tick_log[-1])
        if last_action == "none":
            return "none", 0

        repeated = 0
        for record in reversed(self._tick_log):
            if self._action_signature(record) != last_action:
                break
            repeated += 1
        return last_action, repeated

    @staticmethod
    def _screen_change_score(
        previous_image: Optional[Image.Image],
        current_image: Image.Image,
    ) -> Optional[float]:
        if previous_image is None:
            return None

        try:
            size = (32, 32)
            previous_small = previous_image.convert("L").resize(size)
            current_small = current_image.convert("L").resize(size)
            diff = ImageChops.difference(previous_small, current_small)
            mean_delta = ImageStat.Stat(diff).mean[0] / 255.0
        except Exception as exc:
            logger.debug("Could not compute screen change score: %s", exc)
            return None

        return round(mean_delta, 3)

    # -- turn progress tracking ------------------------------------------------

    def _progress_region_crop(self, image: Image.Image) -> Optional[Image.Image]:
        region = self._progress_cfg.region
        if region is None:
            return None
        left, top, width, height = region
        right = left + width
        bottom = top + height
        if left < 0 or top < 0 or right > image.width or bottom > image.height:
            logger.warning(
                "progress_tracking.region %s is out of bounds for image size "
                "(%d, %d); skipping progress check for this tick",
                region,
                image.width,
                image.height,
            )
            return None
        return image.crop((left, top, right, bottom))

    @staticmethod
    def _region_change_score(
        previous_crop: Optional[Image.Image],
        current_crop: Optional[Image.Image],
    ) -> Optional[float]:
        if previous_crop is None or current_crop is None:
            return None

        try:
            previous_gray = previous_crop.convert("L")
            current_gray = current_crop.convert("L")
            if previous_gray.size != current_gray.size:
                current_gray = current_gray.resize(previous_gray.size)
            diff = ImageChops.difference(previous_gray, current_gray)
            mean_delta = ImageStat.Stat(diff).mean[0] / 255.0
        except Exception as exc:
            logger.debug("Could not compute progress region change score: %s", exc)
            return None

        return round(mean_delta, 4)

    def _progress_tracking_values(
        self, tick_idx: int, current_image: Image.Image, *, debug_prefix: str = "tick"
    ) -> dict[str, Any]:
        """Detect turn progress from a fixed screen region (e.g. a spinner).

        Reference mode compares every crop with the first valid crop, so an
        animated indicator remains one event until it returns to its initial
        state. Legacy any-rising-edge mode compares consecutive crops.
        """
        cfg = self._progress_cfg
        meta: dict[str, Any] = {
            "enabled": cfg.enabled,
            "region_changed": False,
            "change_score": None,
            "progress_event": False,
            "progress_count": self._progress_count,
        }
        if not cfg.enabled or cfg.region is None:
            return meta

        current_crop = self._progress_region_crop(current_image)

        if cfg.save_debug_crops and current_crop is not None:
            debug_path = self.output_dir / f"{debug_prefix}_{tick_idx:04d}_progress_tracking.png"
            try:
                current_crop.save(debug_path)
            except Exception as exc:
                logger.debug("Could not save progress tracking debug crop: %s", exc)

        if cfg.mode == "reference":
            if self._progress_reference_crop is None and current_crop is not None:
                self._progress_reference_crop = current_crop.copy()
                score = None
            else:
                score = self._region_change_score(
                    self._progress_reference_crop,
                    current_crop,
                )
        else:
            score = self._region_change_score(self._last_progress_crop, current_crop)

        changed = score is not None and score >= cfg.change_threshold
        progress_event = changed and not self._progress_region_active

        if progress_event:
            self._progress_count += 1
        if score is not None or cfg.mode == "any-rising-edge":
            self._progress_region_active = changed
        self._last_progress_crop = current_crop

        meta.update(
            {
                "region_changed": changed,
                "change_score": score,
                "progress_event": progress_event,
                "progress_count": self._progress_count,
            }
        )
        return meta

    def _sample_background_progress(self, frame: CapturedFrame) -> None:
        """Measure every captured raw frame, including while inference blocks."""
        if not self._progress_cfg.enabled:
            return
        with self._progress_lock:
            sample_idx = len(self._progress_samples)
            meta = self._progress_tracking_values(
                sample_idx, frame.image, debug_prefix="capture",
            )
            self._progress_samples.append({
                "sample": sample_idx,
                "timestamp": frame.timestamp,
                **meta,
            })

    def _progress_for_tick(self, tick_idx: int, image: Image.Image) -> dict[str, Any]:
        if self.capture_strategy != "background" or not self._progress_cfg.enabled:
            return self._progress_tracking_values(tick_idx, image)
        # The producer owns the detector. Do not remeasure an older queued frame
        # or count the same spinner transition a second time on the main thread.
        with self._progress_lock:
            samples = self._progress_samples[self._progress_samples_consumed:]
            meta = dict(self._progress_samples[-1])
            self._progress_samples_consumed = len(self._progress_samples)
            meta["progress_event"] = any(s["progress_event"] for s in samples)
            meta["progress_events_since_tick"] = sum(s["progress_event"] for s in samples)
            meta["samples_since_tick"] = len(samples)
        # Preserve the viewer's per-tick debug crop without touching the detector.
        if self._progress_cfg.save_debug_crops:
            crop = self._progress_region_crop(image)
            if crop is not None:
                crop.save(self.output_dir / f"tick_{tick_idx:04d}_progress_tracking.png")
        return meta

    def _state_context_values(self, current_image: Image.Image) -> dict[str, Any]:
        last_action, repeated_count = self._recent_action_repetition()
        score = self._screen_change_score(
            self._last_state_context_image,
            current_image,
        )
        low_screen_change = (
            score is not None
            and score <= self._state_context_cfg.screen_change_threshold
        )
        previous_state = (
            self._tick_log[-1].get("state_context", {})
            if self._tick_log
            else {}
        )
        previous_low_count = (
            int(previous_state.get("low_screen_change_count", 0))
            if isinstance(previous_state, dict)
            else 0
        )
        low_screen_change_count = (
            previous_low_count + 1
            if low_screen_change
            else 0
        )
        action_stuck = (
            last_action != "none"
            and repeated_count >= self._state_context_cfg.repeat_threshold
            and low_screen_change
        )
        screen_stuck = (
            low_screen_change_count
            >= self._state_context_cfg.screen_change_repeat_threshold
        )
        stuck_detected = action_stuck or screen_stuck
        stuck_hint = "none"
        stuck_reason = "none"
        if stuck_detected:
            stuck_hint = self._state_context_cfg.stuck_hint
            if action_stuck:
                stuck_reason = _format_template(
                    _DEFAULT_STUCK_REASON_ACTION,
                    {"repeated_action_count": repeated_count},
                    _DEFAULT_STUCK_REASON_ACTION,
                    warning_context="state context stuck reason",
                )
            else:
                stuck_reason = _format_template(
                    _DEFAULT_STUCK_REASON_SCREEN,
                    {"low_screen_change_count": low_screen_change_count},
                    _DEFAULT_STUCK_REASON_SCREEN,
                    warning_context="state context stuck reason",
                )

        return {
            "stuck_detected": stuck_detected,
            "action_stuck": action_stuck,
            "screen_stuck": screen_stuck,
            "last_action": last_action,
            "repeated_action_count": repeated_count,
            "low_screen_change": low_screen_change,
            "low_screen_change_count": low_screen_change_count,
            "screen_change_score": "unknown" if score is None else f"{score:.3f}",
            "screen_change_score_raw": score,
            "stuck_reason": stuck_reason,
            "stuck_hint": stuck_hint,
        }

    def _current_prompt_with_state_context(
        self,
        base_prompt: str,
        current_image: Image.Image,
    ) -> tuple[str, dict[str, Any]]:
        values = self._state_context_values(current_image)
        prompt_applied = self._state_context_cfg.enabled and values["stuck_detected"]
        meta = {
            "enabled": self._state_context_cfg.enabled,
            "applied": prompt_applied,
            **values,
        }
        if not self._state_context_cfg.enabled:
            return base_prompt, meta

        if not prompt_applied:
            return base_prompt, meta

        note = _format_template(
            self._state_context_cfg.template,
            values,
            _DEFAULT_STATE_CONTEXT_TEMPLATE,
            warning_context="state context",
        )
        return f"{base_prompt.strip()}\n{note}", meta

    @staticmethod
    def _append_prompt_line(prompt: str, note: str) -> str:
        note = note.strip()
        if not note:
            return prompt
        return f"{prompt.strip()}\n{note}"

    @staticmethod
    def _resize_visual_panel(image: Image.Image, panel_width: int) -> Image.Image:
        width = max(1, int(panel_width))
        source = image.convert("RGB")
        if source.width <= 0 or source.height <= 0:
            return source.resize((width, width))
        height = max(1, round(source.height * (width / source.width)))
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        return source.resize((width, height), resampling)

    def _compose_visual_context_image(
        self,
        previous_image: Image.Image,
        current_image: Image.Image,
    ) -> Image.Image:
        cfg = self._visual_context_cfg
        previous_panel = self._resize_visual_panel(previous_image, cfg.panel_width)
        current_panel = self._resize_visual_panel(current_image, cfg.panel_width)
        panel_height = max(previous_panel.height, current_panel.height)
        label_height = max(1, cfg.label_height)
        width = previous_panel.width + current_panel.width
        height = label_height + panel_height

        canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), "LEFT / PREVIOUS", fill=(255, 255, 255))
        draw.text((previous_panel.width + 8, 7), "RIGHT / CURRENT", fill=(255, 255, 255))
        draw.line(
            [(previous_panel.width, 0), (previous_panel.width, height)],
            fill=(255, 255, 255),
            width=2,
        )
        canvas.paste(previous_panel, (0, label_height))
        canvas.paste(current_panel, (previous_panel.width, label_height))
        return canvas

    def _visual_context_for_tick(
        self,
        tick_idx: int,
        current_image: Image.Image,
        current_screenshot_path: Optional[str],
        state_context_meta: dict[str, Any],
    ) -> tuple[Image.Image, Optional[str], dict[str, Any], Optional[str]]:
        cfg = self._visual_context_cfg
        meta: dict[str, Any] = {
            "enabled": cfg.mode != "off",
            "mode": cfg.mode,
            "applied": False,
        }
        if cfg.mode == "off":
            return current_image, current_screenshot_path, meta, None

        if self._last_state_context_image is None:
            meta["reason"] = "no_previous_image"
            return current_image, current_screenshot_path, meta, None

        if cfg.mode == "on_stuck" and not state_context_meta.get("stuck_detected", False):
            meta["reason"] = "not_stuck"
            return current_image, current_screenshot_path, meta, None

        combined = self._compose_visual_context_image(
            self._last_state_context_image,
            current_image,
        )
        combined_path = self.output_dir / f"tick_{tick_idx:04d}_visual_context.png"
        combined.save(combined_path)

        previous_path = None
        if self._tick_log:
            previous_path = self._tick_log[-1].get("screenshot")

        meta.update({
            "applied": True,
            "reason": "always" if cfg.mode == "always_last_2" else "stuck",
            "previous_screenshot": previous_path,
            "current_screenshot": current_screenshot_path,
            "vlm_screenshot": str(combined_path),
        })
        return combined, str(combined_path), meta, cfg.prompt_note

    def _backend_supports_message_history(self) -> bool:
        return callable(getattr(self._backend, "infer_messages", None))

    def _current_prompt_with_action_history(
        self, current_user_prompt: str,
    ) -> tuple[str, dict[str, Any]]:
        """Append a bounded action list without replaying messages or images."""
        cfg = self._message_history_cfg
        if cfg.mode != "actions":
            return current_user_prompt, {}

        # Collect newest first so the limit counts actions, including multi-action turns.
        actions: list[str] = []
        for record in reversed(self._tick_log):
            if record.get("skipped") or record.get("inference_error"):
                continue
            # Model-space coordinates avoid mixing pixels with normalized model output.
            parsed = record.get("model_actions", record.get("parsed_actions")) or []
            for action in reversed(parsed):
                name = str(action["canonical_name"])
                params = action.get("params") or {}
                definition = self.action_config.get_action(name)
                param_names = (
                    [param.name for param in definition.parameters]
                    if definition is not None else list(params)
                )
                values = [str(params[key]) for key in param_names if key in params]
                actions.append(" ".join([name.replace("_", " "), *values]))
                if len(actions) >= cfg.last_n:
                    break
            if len(actions) >= cfg.last_n:
                break

        if cfg.action_order == "oldest_first":
            actions.reverse()
        meta = {
            "actions_sent": len(actions),
            "action_order": cfg.action_order,
            "last_n": cfg.last_n,
            "prior_turns_available": len(self._tick_log),
            "raw_turns_sent": 0,
            "compacted_turns": 0,
            "historical_images_sent": 0,
            "messages_sent": 2,
            "message_roles": ["system", "user"],
        }
        if actions:
            history_text = _format_template(
                cfg.history_user_template,
                {
                    "actions": ", ".join(actions),
                    "action_order": cfg.action_order.replace("_", " "),
                },
                _DEFAULT_ACTION_HISTORY_USER_TEMPLATE,
            )
            current_user_prompt = self._append_prompt_line(
                current_user_prompt,
                history_text,
            )
        return current_user_prompt, meta

    def _record_user_prompt(self, record: dict[str, Any]) -> str:
        prompt = record.get("user_prompt_actual", record.get("user_prompt"))
        if prompt in (None, "(default)"):
            return self.user_prompt
        return str(prompt)

    def _history_turn_values(self, record: dict[str, Any]) -> dict[str, Any]:
        parsed_actions_raw = record.get("parsed_actions") or []
        parsed_names = []
        for item in parsed_actions_raw:
            if isinstance(item, dict):
                parsed_names.append(str(item.get("canonical_name", "")))
            else:
                parsed_names.append(str(item))
        parsed_actions = ", ".join(name for name in parsed_names if name) or "(none)"

        return {
            "tick": record.get("tick", "?"),
            "user_prompt": self._record_user_prompt(record),
            "response": str(record.get("vlm_response", "")),
            "parsed_actions": parsed_actions,
            "screenshot": str(record.get("vlm_screenshot") or record.get("screenshot") or ""),
            "skipped": bool(record.get("skipped", False)),
            "inference_error": str(record.get("inference_error") or ""),
            "last_action": str(
                (record.get("state_context") or {}).get("last_action", "none")
            ),
            "repeated_action_count": str(
                (record.get("state_context") or {}).get(
                    "repeated_action_count",
                    0,
                )
            ),
            "screen_change_score": str(
                (record.get("state_context") or {}).get(
                    "screen_change_score",
                    "unknown",
                )
            ),
            "stuck_hint": str(
                (record.get("state_context") or {}).get("stuck_hint", "none")
            ),
        }

    @staticmethod
    def _text_user_message(text: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def _raw_history_messages(self, records: list[dict]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        image_limit = self._message_history_cfg.image_last_n
        image_start = 0 if image_limit is None else max(0, len(records) - image_limit)
        for index, record in enumerate(records):
            values = self._history_turn_values(record)
            user_text = _format_template(
                self._message_history_cfg.history_user_template,
                values,
                _DEFAULT_HISTORY_USER_TEMPLATE,
            )
            assistant_text = _format_template(
                self._message_history_cfg.history_assistant_template,
                values,
                _DEFAULT_HISTORY_ASSISTANT_TEMPLATE,
            )
            user_message = self._text_user_message(user_text)
            if index >= image_start and values["screenshot"]:
                user_message["content"].insert(
                    0, {"type": "image", "image": values["screenshot"]}
                )
            messages.append(user_message)
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def _compact_history_messages(
        self,
        records: list[dict],
    ) -> list[dict[str, Any]]:
        if not records:
            return []

        turn_lines = [
            _format_template(
                self._message_history_cfg.compact_turn_template,
                self._history_turn_values(record),
                _DEFAULT_COMPACT_TURN_TEMPLATE,
            )
            for record in records
        ]
        summary = _format_template(
            self._message_history_cfg.compact_assistant_template,
            {"turns": "\n".join(turn_lines)},
            _DEFAULT_COMPACT_ASSISTANT_TEMPLATE,
        )

        if self._message_history_cfg.compact_as == "assistant_only":
            return [{"role": "assistant", "content": summary}]

        return [
            self._text_user_message(self._message_history_cfg.compact_user_prompt),
            {"role": "assistant", "content": summary},
        ]

    def _history_messages_for_completed_turns(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cfg = self._message_history_cfg
        turns = list(self._tick_log)
        meta: dict[str, Any] = {
            "mode": cfg.mode,
            "prior_turns_available": len(turns),
            "raw_turns_sent": 0,
            "compacted_turns": 0,
            "compact_as": cfg.compact_as,
        }

        if cfg.mode == "all":
            selected = turns
            meta["raw_turns_sent"] = len(selected)
            return self._raw_history_messages(selected), meta

        if cfg.mode == "last_n":
            selected = turns[-cfg.last_n :]
            meta["raw_turns_sent"] = len(selected)
            return self._raw_history_messages(selected), meta

        if cfg.mode == "compact":
            compact_records: list[dict]
            raw_records: list[dict]
            if len(turns) <= cfg.compact_max_turns:
                compact_records = []
                raw_records = turns
            else:
                compact_records = turns[: -cfg.compact_max_turns]
                raw_records = turns[-cfg.compact_max_turns :]

            meta["raw_turns_sent"] = len(raw_records)
            meta["compacted_turns"] = len(compact_records)
            messages = self._compact_history_messages(compact_records)
            messages.extend(self._raw_history_messages(raw_records))
            return messages, meta

        return [], meta

    def _build_inference_messages(
        self,
        current_user_prompt: str,
    ) -> tuple[Optional[list[dict[str, Any]]], dict[str, Any]]:
        cfg = self._message_history_cfg
        meta: dict[str, Any] = {
            "mode": cfg.mode,
            "enabled": cfg.mode != "none",
            "backend_message_history_supported": self._backend_supports_message_history(),
            "messages_sent": 0,
            "message_roles": [],
        }

        if cfg.mode in {"none", "actions"}:
            return None, meta

        if not meta["backend_message_history_supported"]:
            if not self._history_backend_warning_logged:
                logger.warning(
                    "message_history.mode=%s requested, but backend %s does not "
                    "implement infer_messages(); sending current turn only",
                    cfg.mode,
                    self._backend.backend_name,
                )
                self._history_backend_warning_logged = True
            return None, meta

        history_messages, history_meta = self._history_messages_for_completed_turns()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *history_messages,
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": current_user_prompt},
                ],
            },
        ]
        meta.update(history_meta)
        meta["image_last_n"] = cfg.image_last_n
        meta["historical_images_sent"] = sum(
            part.get("type") == "image"
            for message in history_messages
            for part in message.get("content", [])
            if isinstance(part, dict)
        )
        meta["messages_sent"] = len(messages)
        meta["message_roles"] = [str(message.get("role", "")) for message in messages]
        return messages, meta

    @staticmethod
    def _serialize_message_content(content: Any) -> Any:
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"type": "text", "text": str(part)})
                    continue

                part_type = str(part.get("type", "text"))
                if part_type == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif part_type == "image":
                    parts.append({
                        "type": "image",
                        "image": str(part.get("image", "current_vlm_image")),
                    })
                else:
                    safe_part = {"type": part_type}
                    for key, value in part.items():
                        if key == "type":
                            continue
                        safe_part[str(key)] = value if isinstance(
                            value,
                            (str, int, float, bool, type(None)),
                        ) else str(value)
                    parts.append(safe_part)
            return parts

        if content is None:
            return ""
        return str(content)

    def _serialize_vlm_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "role": str(message.get("role", "")),
                "content": self._serialize_message_content(message.get("content")),
            }
            for index, message in enumerate(messages)
        ]

    @staticmethod
    def _effective_vlm_messages(
        system_prompt: str,
        user_prompt: str,
        messages: Optional[list[dict[str, Any]]],
    ) -> tuple[str, list[dict[str, Any]]]:
        if messages is not None:
            return "infer_messages", messages

        return "infer", [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

    def _vlm_io_record(
        self,
        tick_idx: int,
        timestamp: float,
        image_path: Optional[str],
        system_prompt: str,
        user_prompt: str,
        messages: Optional[list[dict[str, Any]]],
        response: str,
        latency_ms: float,
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> dict[str, Any]:
        api, effective_messages = self._effective_vlm_messages(
            system_prompt,
            user_prompt,
            messages,
        )
        record = {
            "tick": tick_idx,
            "timestamp": timestamp,
            "api": api,
            "image": image_path,
            "messages": self._serialize_vlm_messages(effective_messages),
            "response": response,
            "latency_ms": latency_ms,
        }
        if generation_params:
            record["generation_params"] = generation_params
        return record

    # -- interruptible inference ----------------------------------------------

    def _infer_interruptible(
        self,
        image,
        system_prompt: str,
        user_prompt: str,
        messages: Optional[list[dict[str, Any]]] = None,
        poll_callback: Optional[Callable[[], None]] = None,
        generation_params: Optional[dict[str, float | int]] = None,
    ) -> str:
        """Run VLM inference in a daemon thread so Ctrl+C works.

        model.generate() is a C/CUDA call that blocks Python signal handling.
        By running it in a background thread the main thread can poll with
        short joins and respond to KeyboardInterrupt immediately.
        """
        self._last_inference_error = None
        result: list[Optional[str]] = [None]
        error: list[Optional[Exception]] = [None]

        def _worker() -> None:
            self._last_inference_started_at = time.monotonic()
            try:
                if messages is not None:
                    if generation_params is None:
                        result[0] = self._backend.infer_messages(image, messages)
                    else:
                        result[0] = self._backend.infer_messages(
                            image,
                            messages,
                            generation_params=generation_params,
                        )
                else:
                    if generation_params is None:
                        result[0] = self._backend.infer(
                            image,
                            system_prompt,
                            user_prompt,
                        )
                    else:
                        result[0] = self._backend.infer(
                            image,
                            system_prompt,
                            user_prompt,
                            generation_params=generation_params,
                        )
            except Exception as exc:
                error[0] = exc

        t = threading.Thread(target=_worker, daemon=True, name="infer-thread")
        t.start()

        # Poll with short timeout — KeyboardInterrupt is raised here
        while t.is_alive():
            t.join(timeout=0.1)
            if poll_callback is not None:
                poll_callback()

        if error[0] is not None:
            self._last_inference_error = repr(error[0])
            logger.error("Inference error: %s", error[0])
            return ""
        return result[0] or ""

    def _normalize_parsed_coordinates(
        self,
        parsed: list[ParsedAction],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Normalize any (x, y) params from 0–1000 space into pixel coords.

        This is used for models that emit coordinates in a fixed 0–1000 range
        regardless of the actual image resolution (e.g. some Qwen / Gemma
        variants), when enabled via model_settings.normalize_coordinates_from_0_1000.
        """
        if frame_width <= 0 or frame_height <= 0:
            return

        for action in parsed:
            params = getattr(action, "params", None)
            if not isinstance(params, dict):
                continue
            if "x" not in params or "y" not in params:
                continue

            x_raw = params["x"]
            y_raw = params["y"]
            x_px, y_px = normalize_xy_from_0_1000(
                x_raw,
                y_raw,
                frame_width,
                frame_height,
            )
            params["x"] = x_px
            params["y"] = y_px

    # -- action hold duration -------------------------------------------------

    def _timed_action_hold_enabled(self) -> bool:
        return self._action_hold_duration_ms is not None

    def _schedule_action_release_if_needed(self, state: ActionState) -> dict[str, Any]:
        if not self._timed_action_hold_enabled():
            self._action_hold_release_at = None
            return {
                "mode": "until_next_action",
                "duration_ms": None,
                "scheduled": False,
            }

        held = list(state.held_inputs)
        if not held:
            self._action_hold_release_at = None
            return {
                "mode": "timed",
                "duration_ms": self._action_hold_duration_ms,
                "scheduled": False,
            }

        duration_sec = self._action_hold_duration_ms / 1000.0
        self._action_hold_release_at = time.monotonic() + duration_sec
        return {
            "mode": "timed",
            "duration_ms": self._action_hold_duration_ms,
            "scheduled": True,
            "held_actions": held,
        }

    def _release_actions_if_due(
        self,
        state: ActionState,
        executor: ActionExecutor,
        *,
        force: bool = False,
    ) -> list[str]:
        if self._action_hold_release_at is None:
            return []
        if not force and time.monotonic() < self._action_hold_release_at:
            return []

        released = state.release_all()
        self._action_hold_release_at = None
        if released:
            executor.release_all(released)
            released_names = [h.canonical_name for h in released]
            logger.info("Timed action release: %s", released_names)
            return released_names
        return []

    def _schedule_post_action_settle(
        self,
        plan: TickPlan,
        execution_summary: dict[str, Any],
    ) -> None:
        """Require the next model frame to follow the executed action and settle."""
        if (
            execution_summary.get("skipped", False)
            or execution_summary.get("dry_run", self.dry_run)
        ):
            self._post_action_settle_until = None
            return

        has_action = any(
            (
                plan.to_release,
                plan.to_press,
                plan.to_tap,
                plan.mouse_moves,
                plan.click_ats,
            )
        )
        if not has_action:
            self._post_action_settle_until = None
            return

        self._post_action_settle_until = (
            time.monotonic() + self.post_action_settle_ms / 1000.0
        )
        self._frame_not_before = time.time() + self.post_action_settle_ms / 1000.0

    def _wait_for_post_action_settle(
        self,
        state: ActionState,
        executor: ActionExecutor,
    ) -> None:
        """Wait for the post-action deadline while servicing timed releases."""
        settle_until = self._post_action_settle_until
        if settle_until is None:
            return

        while True:
            released = self._release_actions_if_due(state, executor)
            now = time.monotonic()
            if released:
                # A timed key/button release is itself a visible action. Give
                # the game the full settle interval after that state change.
                settle_until = max(
                    settle_until,
                    now + self.post_action_settle_ms / 1000.0,
                )
                self._frame_not_before = time.time() + self.post_action_settle_ms / 1000.0
            remaining = settle_until - now
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))

        self._post_action_settle_until = None

    def _get_frame_with_action_release(
        self,
        timeout: float,
        state: ActionState,
        executor: ActionExecutor,
    ) -> Optional[CapturedFrame]:
        if not self._timed_action_hold_enabled():
            return self._frame_queue.get(timeout=timeout)

        deadline = time.monotonic() + timeout
        while True:
            self._release_actions_if_due(state, executor)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            frame = self._frame_queue.get(timeout=min(0.1, remaining))
            if frame is not None:
                return frame

    def _capture_synchronously(
        self,
        timeout: float,
        state: ActionState,
        executor: ActionExecutor,
    ) -> Optional[CapturedFrame]:
        """Capture one fresh frame, retrying transient backend misses."""
        if self._post_action_settle_until is not None:
            self._wait_for_post_action_settle(state, executor)
        deadline = time.monotonic() + timeout
        while True:
            self._release_actions_if_due(state, executor)
            try:
                frame = _do_capture(self.session, self.capture_method)
            except Exception as exc:
                logger.error("Capture error: %s", exc)
                frame = None
            if frame is not None:
                self._frames_captured += 1
                return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.05, remaining))

    def _next_frame(
        self,
        timeout: float,
        state: ActionState,
        executor: ActionExecutor,
    ) -> Optional[CapturedFrame]:
        if self.capture_strategy == "background":
            self._wait_for_post_action_settle(state, executor)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                frame = self._get_frame_with_action_release(remaining, state, executor)
                if frame is None or self._frame_not_before is None:
                    return frame
                if frame.timestamp >= self._frame_not_before:
                    return frame
        return self._capture_synchronously(timeout, state, executor)

    def _wait_for_inference_slot(
        self, state: ActionState, executor: ActionExecutor,
    ) -> bool:
        """Throttle request starts, without blocking the capture producer."""
        if self._last_inference_started_at is None:
            return True
        deadline = self._last_inference_started_at + self.inference_interval_ms / 1000.0
        while True:
            self._release_actions_if_due(state, executor)
            if self._pause_controller and self._pause_controller.is_paused:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.05, remaining))

    def _wait_for_final_progress_sample(
        self, state: ActionState, executor: ActionExecutor,
    ) -> None:
        """Observe the final click once before stopping the capture producer."""
        if (
            self.capture_strategy != "background"
            or not self._progress_cfg.enabled
            or self._frame_not_before is None
        ):
            return
        self._wait_for_post_action_settle(state, executor)
        deadline = time.monotonic() + 2 * self.frame_cadence_ms / 1000.0
        while time.monotonic() < deadline:
            self._release_actions_if_due(state, executor)
            with self._progress_lock:
                if (
                    self._progress_samples
                    and self._progress_samples[-1]["timestamp"] >= self._frame_not_before
                ):
                    return
            time.sleep(0.05)

    # -- single tick ----------------------------------------------------------

    def _run_tick(
        self,
        tick_idx: int,
        frame: CapturedFrame,
        state: ActionState,
        executor: ActionExecutor,
        user_prompt_override: Optional[str] = None,
    ) -> tuple[dict, Optional[FinishResult]]:
        """Execute a single gameplay tick.

        Returns (tick_record, finish_result_or_None).
        """
        # Save screenshot
        screenshot_path = self.output_dir / f"tick_{tick_idx:04d}.png"
        screenshot_result = save_frame_screenshots(
            frame,
            screenshot_path,
            cursor_mode=self.cursor_mode,
            selected_mode=self.vlm_image_mode,
        )
        current_image = screenshot_result.selected_image
        current_screenshot_path = (
            str(screenshot_result.selected_path)
            if screenshot_result.selected_path is not None
            else None
        )

        progress_meta = self._progress_for_tick(tick_idx, current_image)

        base_user_prompt = user_prompt_override or self.user_prompt
        current_user_prompt, state_context_meta = self._current_prompt_with_state_context(
            base_user_prompt,
            current_image,
        )
        inference_image, vlm_screenshot_path, visual_context_meta, visual_note = (
            self._visual_context_for_tick(
                tick_idx,
                current_image,
                current_screenshot_path,
                state_context_meta,
            )
        )
        if visual_note is not None:
            current_user_prompt = self._append_prompt_line(
                current_user_prompt,
                visual_note,
            )
        # Retain the exact model input, including cursor/visual-context variants.
        if (
            vlm_screenshot_path is None
            and self._message_history_cfg.mode not in {"none", "actions"}
        ):
            retained_path = self.output_dir / f"tick_{tick_idx:04d}_vlm.png"
            inference_image.save(retained_path)
            vlm_screenshot_path = str(retained_path)
        current_user_prompt, action_history_meta = self._current_prompt_with_action_history(
            current_user_prompt
        )
        inference_messages, message_history_meta = self._build_inference_messages(
            current_user_prompt
        )
        message_history_meta.update(action_history_meta)
        generation_selection: Optional[GenerationParamsSelection] = None
        generation_overrides: Optional[dict[str, float | int]] = None
        if self._generation_params_config.enabled:
            generation_selection = self._generation_params_sampler.select()
            generation_overrides = generation_selection.requested
        generation_resolution = self._backend.resolve_generation_params(
            generation_overrides
        )
        generation_usage_key = self._generation_params_usage.record(
            generation_selection,
            generation_resolution.effective,
        )

        # Freeze the game (if enabled) while we wait for the VLM response,
        # so the screenshot we just took stays representative of the state
        # the model is reasoning about.
        if self._freeze_controller is not None:
            self._freeze_controller.freeze()

        # Run inference in a thread so Ctrl+C stays responsive during
        # long-running model.generate() calls (C extension blocks signals).
        t0 = time.time()
        try:
            response = self._infer_interruptible(
                inference_image,
                self.system_prompt,
                current_user_prompt,
                messages=inference_messages,
                poll_callback=lambda: self._release_actions_if_due(state, executor),
                generation_params=generation_overrides,
            )
        finally:
            # Unfreeze before acting on the response so the chosen action
            # actually takes effect in the (now-running) game.
            if self._freeze_controller is not None:
                self._freeze_controller.unfreeze()
        inference_error = self._last_inference_error
        inference_ms = round((time.time() - t0) * 1000, 1)
        if self._log_vlm_io:
            self._vlm_io_log.append(
                self._vlm_io_record(
                    tick_idx,
                    frame.timestamp,
                    vlm_screenshot_path,
                    self.system_prompt,
                    current_user_prompt,
                    inference_messages,
                    response,
                    inference_ms,
                    (
                        generation_resolution.effective
                        if generation_selection is not None
                        else None
                    ),
                )
            )

        # Parse actions
        parsed = parse_vlm_text(response, self.action_config)
        model_actions = [
            {"canonical_name": action.canonical_name, "params": dict(action.params)}
            for action in parsed
        ]

        # Check for finish
        finish_result: Optional[FinishResult] = None
        if self.finish_detector.is_finished(response, parsed):
            finish_result = self.finish_strategy.handle(
                response, inference_image, self._backend, self.system_prompt
            )

        # Optionally normalize coordinates emitted in 0–1000 space
        if self._normalize_coordinates_from_0_1000:
            self._normalize_parsed_coordinates(parsed, frame.width, frame.height)

        # Execute actions
        plan = state.compute_tick(parsed)
        summary = executor.execute_tick(plan)
        action_hold_meta = self._schedule_action_release_if_needed(state)
        self._schedule_post_action_settle(plan, summary)

        # Build record
        record: dict = {
            "tick": tick_idx,
            "timestamp": frame.timestamp,
            "screenshot": current_screenshot_path,
            "vlm_image_mode": screenshot_result.selected_mode,
            "vlm_screenshot": vlm_screenshot_path,
            "screenshot_raw": (
                str(screenshot_result.raw_path)
                if screenshot_result.raw_path is not None
                else None
            ),
            "screenshot_cursor": (
                str(screenshot_result.cursor_path)
                if screenshot_result.cursor_path is not None
                else None
            ),
            "user_prompt": (
                base_user_prompt
                if base_user_prompt != self.user_prompt
                else "(default)"
            ),
            "user_prompt_actual": current_user_prompt,
            "state_context": state_context_meta,
            "visual_context": visual_context_meta,
            "progress_tracking": progress_meta,
            "vlm_response": response,
            "message_history": message_history_meta,
            "model_actions": model_actions,
            "parsed_actions": [
                {"canonical_name": a.canonical_name, "params": a.params}
                for a in parsed
            ],
            "plan": {
                "released": [h.canonical_name for h in plan.to_release],
                "pressed": [h.canonical_name for h in plan.to_press],
                "tapped": [h.canonical_name for h in plan.to_tap],
                "mouse_moves": plan.mouse_moves,
            },
            "action_hold": action_hold_meta,
            "skipped": summary.get("skipped", False),
            "inference_error": inference_error,
            "inference_started_at": t0,
            "inference_latency_ms": inference_ms,
            "frames_dropped": self._frame_queue.total_dropped,
        }

        if finish_result is not None:
            record["finish"] = {
                "action": finish_result.action,
                "model_explanation": finish_result.model_explanation,
                "user_correction": finish_result.user_correction,
            }

        if generation_selection is not None:
            generation_meta: dict[str, Any] = {
                "key": generation_usage_key,
                "source": generation_selection.source,
                "requested": generation_selection.requested,
                "effective": generation_resolution.effective,
                "ignored": generation_resolution.ignored,
            }
            if generation_selection.selection_index is not None:
                generation_meta["selection_index"] = (
                    generation_selection.selection_index
                )
            record["generation_params"] = generation_meta

        if screenshot_result.cursor_metadata is not None:
            record["cursor_overlay"] = screenshot_result.cursor_metadata.to_dict()

        self._last_state_context_image = current_image.copy()

        return record, finish_result

    # -- main run -------------------------------------------------------------

    def run(self) -> Path:
        """Execute the gameplay loop. Returns path to the YAML log file."""
        if not self._backend.is_loaded:
            raise RuntimeError(
                "Backend must be loaded before GameplayLoop.run(); "
                "call backend.load_model() in the caller."
            )

        # Wait for user
        if not self.no_wait:
            self.wait_for_ready()
        else:
            print(
                f"Running gameplay loop "
                f"(max_ticks={self.max_ticks}, dry_run={self.dry_run})..."
            )

        # Pre-execution delay (skip in dry-run)
        if not self.dry_run and self.delay_ms > 0:
            delay_sec = self.delay_ms / 1000.0
            print(f"Starting in {delay_sec:.1f}s — switch to game window!")
            time.sleep(delay_sec)

        # Attach the freeze controller (if enabled) once the target process
        # is up and focused.
        if self._freeze_cfg.enabled:
            self._freeze_controller = FreezeController(
                pid=self.session.target.pid,
                arch=self._freeze_cfg.arch,
                freeze_speed=self._freeze_cfg.freeze_speed,
                normal_speed=self._freeze_cfg.normal_speed,
            )
            if not self._freeze_controller.attach():
                self._freeze_controller = None

        # State and executor
        state = ActionState(self.action_config)
        executor = ActionExecutor(
            self.session,
            dry_run=self.dry_run,
            disable_center_delta_conversion=self.disable_center_delta_conversion,
        )

        # Start pause/resume hotkey listener
        if self.pause_key:
            try:
                self._pause_controller = PauseController(self.pause_key)
                if self._pause_controller.start():
                    print(
                        f"Pause/resume hotkey registered: "
                        f"{self.pause_key.upper()}"
                    )
                else:
                    logger.warning(
                        "Could not register pause hotkey (%s) — "
                        "pause/resume disabled",
                        self.pause_key,
                    )
                    self._pause_controller = None
            except ValueError as e:
                logger.warning("Pause hotkey disabled: %s", e)
                self._pause_controller = None

        # The background strategy starts a producer here.  The default
        # after-action strategy captures inline, so no frame can be produced
        # during VLM inference or before the prior action executes.
        self._start_capture_thread()

        tick_idx = 0
        user_prompt_override: Optional[str] = None
        was_paused = False

        try:
            while True:
                if self.max_ticks is not None and tick_idx >= self.max_ticks:
                    logger.info("Max ticks reached (%d), stopping", self.max_ticks)
                    self._wait_for_final_progress_sample(state, executor)
                    break

                # Handle pause/resume transitions
                if self._pause_controller and self._pause_controller.is_paused:
                    if not was_paused:
                        # Entering pause: release held keys, stop capture
                        released = state.release_all()
                        self._action_hold_release_at = None
                        if released:
                            executor.release_all(released)
                            logger.info(
                                "Pause: released %s",
                                [h.canonical_name for h in released],
                            )
                        self._stop_capture_thread()
                        self._frame_queue.clear()
                        print(
                            f"\n  ** PAUSED — press "
                            f"{self.pause_key.upper()} to resume **"
                        )
                        was_paused = True
                    time.sleep(0.1)
                    continue

                if was_paused:
                    # Resuming: restart capture
                    print("  ** RESUMED **\n")
                    self._frame_queue.clear()
                    self._start_capture_thread()
                    was_paused = False

                # Wait before selecting the image, so rate limiting does not
                # leave us sending a screenshot taken before the prior action.
                if not self._wait_for_inference_slot(state, executor):
                    continue

                # Get latest frame (wait up to 2x cadence)
                timeout = (self.frame_cadence_ms / 1000.0) * 2
                frame = self._next_frame(
                    timeout,
                    state,
                    executor,
                )
                if frame is None:
                    logger.warning(
                        "No frame available after %.1fs timeout", timeout
                    )
                    continue

                record, finish_result = self._run_tick(
                    tick_idx, frame, state, executor, user_prompt_override
                )
                self._tick_log.append(record)
                user_prompt_override = None

                # Print progress
                parsed_names = ", ".join(
                    a["canonical_name"] for a in record["parsed_actions"]
                ) or "(none)"
                status = (
                    "ERROR"
                    if record.get("inference_error")
                    else "SKIPPED"
                    if record["skipped"]
                    else "OK"
                )
                print(
                    f"  Tick {tick_idx + 1}: {record['vlm_response']!r} "
                    f"-> [{parsed_names}]  "
                    f"[{record['inference_latency_ms']}ms] [{status}]"
                )
                if record.get("inference_error"):
                    print(f"    Inference error: {record['inference_error']}")
                progress_tracking = record.get("progress_tracking") or {}
                if progress_tracking.get("progress_event"):
                    print(
                        f"\033[92m    ⏳ Progress detected "
                        f"(total: {progress_tracking.get('progress_count')})\033[0m"
                    )

                # Handle finish
                if finish_result is not None:
                    if finish_result.action == "stop":
                        logger.info("Gameplay loop stopped by finish strategy")
                        break
                    elif finish_result.user_correction:
                        user_prompt_override = finish_result.user_correction

                tick_idx += 1

        except KeyboardInterrupt:
            self.interrupted = True
            print("\nInterrupted by user.")
        finally:
            if self._pause_controller:
                self._pause_controller.stop()
                self._pause_controller = None
            self._stop_capture_thread()
            released = state.release_all()
            self._action_hold_release_at = None
            self._post_action_settle_until = None
            if released:
                executor.release_all(released)
                logger.info(
                    "Cleanup: released %s",
                    [h.canonical_name for h in released],
                )
            if self._freeze_controller is not None:
                self._freeze_controller.close()
                self._freeze_controller = None

        log_path = self._write_log()
        print(f"\nGameplay loop complete. {len(self._tick_log)} ticks. Log: {log_path}")
        if self.interrupted and self.propagate_keyboard_interrupt:
            raise KeyboardInterrupt
        return log_path

    def _write_log(self) -> Path:
        """Write the structured YAML log."""
        vlm_io_path = self._write_vlm_io_log()
        log_data = {
            "run_id": self.run_id,
            "backend": self._backend.backend_name,
            "model": self._backend.model_name,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "message_history": self._message_history_cfg.to_log_dict(),
            "state_context": self._state_context_cfg.to_log_dict(),
            "visual_context": self._visual_context_cfg.to_log_dict(),
            "progress_tracking": self._progress_cfg.to_log_dict(),
            "total_progress_count": self._progress_count,
            "progress_samples": self._progress_samples,
            "log_vlm_io": self._log_vlm_io,
            "vlm_io_log": str(vlm_io_path) if vlm_io_path is not None else None,
            "action_hold_duration_ms": self._action_hold_duration_ms,
            "freeze": self._freeze_cfg.to_log_dict(),
            "frame_cadence_ms": self.frame_cadence_ms,
            "inference_interval_ms": self.inference_interval_ms,
            "capture_strategy": self.capture_strategy,
            "post_action_settle_ms": self.post_action_settle_ms,
            "max_new_tokens": self._max_new_tokens,
            "enable_thinking": self._enable_thinking,
            "thinking_budget": self._thinking_budget,
            "finish_strategy": self.finish_strategy_name,
            "capture_method": self.capture_method,
            "cursor_mode": self.cursor_mode,
            "vlm_image_mode": self.vlm_image_mode,
            "pause_key": self.pause_key,
            "dry_run": self.dry_run,
            "max_ticks": self.max_ticks,
            "target_window": self.session.target.short_str(),
            "target_hwnd": f"{self.session.target.hwnd:#010x}",
            "total_ticks": len(self._tick_log),
            "total_frames_captured": self._frames_captured,
            "total_frames_dropped": self._frame_queue.total_dropped,
            "generation_params_usage": self._generation_params_usage.summary(),
            "ticks": self._tick_log,
        }

        log_path = self.output_dir / "gameplay_log.yaml"
        with open(log_path, "w") as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False)

        return log_path

    def _write_vlm_io_log(self) -> Optional[Path]:
        if not self._log_vlm_io:
            return None

        log_path = self.output_dir / "vlm_io.json"
        data = {
            "run_id": self.run_id,
            "backend": self._backend.backend_name,
            "model": self._backend.model_name,
            "items": self._vlm_io_log,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

        return log_path
