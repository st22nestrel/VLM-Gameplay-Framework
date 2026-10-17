"""Gameplay runtime configuration loading and normalization."""

from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from src.actions.schema import ActionConfig, load_action_config
from src.capture.cursor_overlay import (
    cursor_mode_from_config,
    vlm_image_mode_from_config,
)
from src.loop.generation_params import GenerationParamsConfig

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
_DEFAULT_STATE_CONTEXT_TEMPLATE = "{stuck_reason} {stuck_hint}"
_DEFAULT_STUCK_HINT = (
    "Prefer drop, move_backward, side movement or any action that differs from "
    "previous action."
)
_DEFAULT_VISUAL_CONTEXT_NOTE = (
    "Visual context: the attached image has two labeled panels. LEFT/PREVIOUS "
    "is the screenshot from the previous VLM request; RIGHT/CURRENT is the "
    "latest screenshot for this prompt. Choose the next action for RIGHT/CURRENT."
)


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class ProgressTrackingConfig:
    """Configuration for detecting turn progress via a fixed screen region.

    Watches a fixed pixel region of the screenshot (e.g. a spinning
    "waiting for move" hourglass icon). By default, the first valid crop is
    used as the reference state and a transition away from that state counts
    as one progress event. The legacy ``any-rising-edge`` mode compares
    consecutive ticks instead.
    """

    enabled: bool = False
    region: Optional[tuple[int, int, int, int]] = None  # (left, top, width, height)
    change_threshold: float = 0.02
    save_debug_crops: bool = False
    mode: str = "reference"

    @classmethod
    def from_config(cls, cfg: Any) -> "ProgressTrackingConfig":
        if not isinstance(cfg, dict):
            return cls()

        region_cfg = cfg.get("region")
        region: Optional[tuple[int, int, int, int]] = None
        if isinstance(region_cfg, dict):
            try:
                region = (
                    int(region_cfg["left"]),
                    int(region_cfg["top"]),
                    int(region_cfg["width"]),
                    int(region_cfg["height"]),
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Invalid progress_tracking.region=%r; disabling progress tracking",
                    region_cfg,
                )
                region = None

        enabled = _bool_value(cfg.get("enabled", False), False)
        if enabled and region is None:
            logger.warning(
                "progress_tracking.enabled=true but region is missing/invalid; "
                "disabling progress tracking"
            )
            enabled = False

        mode = str(cfg.get("mode", "reference")).strip().lower()
        if mode not in {"reference", "any-rising-edge"}:
            logger.warning(
                "Invalid progress_tracking.mode=%r; using 'reference'",
                cfg.get("mode"),
            )
            mode = "reference"

        return cls(
            enabled=enabled,
            region=region,
            mode=mode,
            change_threshold=_float_value(cfg.get("change_threshold", 0.02), 0.02),
            save_debug_crops=_bool_value(cfg.get("save_debug_crops", False), False),
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "region": (
                {
                    "left": self.region[0],
                    "top": self.region[1],
                    "width": self.region[2],
                    "height": self.region[3],
                }
                if self.region is not None
                else None
            ),
            "mode": self.mode,
            "change_threshold": self.change_threshold,
            "save_debug_crops": self.save_debug_crops,
        }


@dataclasses.dataclass(frozen=True)
class FreezeConfig:
    """Configuration for freezing the game while waiting on the VLM.

    Uses the xspeedhack library (DLL-injection based) to stop the target
    process's clock right after a screenshot is taken, and restores normal
    speed once the VLM response is parsed and the action is about to run.
    """

    enabled: bool = False
    arch: str = "x64"
    freeze_speed: float = 0.0
    normal_speed: float = 1.0

    @classmethod
    def from_config(cls, cfg: Any) -> "FreezeConfig":
        if not isinstance(cfg, dict):
            return cls()

        arch = str(cfg.get("arch", "x64")).strip().lower()
        if arch not in {"x86", "x64"}:
            logger.warning("Unknown freeze.arch=%r; using 'x64'", arch)
            arch = "x64"

        return cls(
            enabled=_bool_value(cfg.get("enabled", False), False),
            arch=arch,
            freeze_speed=_float_value(cfg.get("freeze_speed", 0.0), 0.0),
            normal_speed=_float_value(cfg.get("normal_speed", 1.0), 1.0),
        )

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "arch": self.arch,
            "freeze_speed": self.freeze_speed,
            "normal_speed": self.normal_speed,
        }


@dataclasses.dataclass(frozen=True)
class ModelSettingsConfig:
    """Backend-agnostic model settings used by live gameplay."""

    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    generation_params: GenerationParamsConfig = dataclasses.field(
        default_factory=GenerationParamsConfig
    )
    normalize_coordinates_from_0_1000: bool = False
    lora_adapter_path: Optional[str] = None
    lora_base_model_source: str = "adapter"

    @classmethod
    def from_gameplay_config(
        cls,
        gameplay_cfg: dict[str, Any],
        *,
        lora_base_model_source_override: Optional[str] = None,
    ) -> "ModelSettingsConfig":
        model_settings = gameplay_cfg.get("model_settings", {})
        if not isinstance(model_settings, dict):
            model_settings = {}
        gemma4_cfg = gameplay_cfg.get("gemma4", {})
        if not isinstance(gemma4_cfg, dict):
            gemma4_cfg = {}

        def _ms_get(key: str, default: Any = None) -> Any:
            return model_settings.get(key, gemma4_cfg.get(key, default))

        return cls(
            enable_thinking=_bool_value(_ms_get("enable_thinking", False), False),
            thinking_budget=_ms_get("thinking_budget"),
            temperature=_optional_float(_ms_get("temperature")),
            top_p=_optional_float(_ms_get("top_p")),
            top_k=_optional_int(_ms_get("top_k")),
            generation_params=GenerationParamsConfig.from_config(
                _ms_get("generation_params")
            ),
            normalize_coordinates_from_0_1000=_bool_value(
                _ms_get("normalize_coordinates_from_0_1000", False),
                False,
            ),
            lora_adapter_path=_optional_str(_ms_get("lora_adapter_path")),
            lora_base_model_source=(
                lora_base_model_source_override
                if lora_base_model_source_override is not None
                else str(_ms_get("lora_base_model_source", "adapter"))
            ),
        )


@dataclasses.dataclass(frozen=True)
class GameplayRuntimeConfig:
    """Normalized gameplay runtime settings consumed by GameplayLoop."""

    raw_config: dict[str, Any]
    backend_name: str
    model_id: str
    max_new_tokens: int
    model_settings: ModelSettingsConfig
    llama_cpp_config: dict[str, Any]
    openai_config: dict[str, Any]
    system_prompt: str
    user_prompt: str
    message_history: MessageHistoryConfig
    state_context: StateContextConfig
    visual_context: VisualContextConfig
    progress_tracking: ProgressTrackingConfig
    log_vlm_io: bool
    action_hold_duration_ms: Optional[int]
    freeze: FreezeConfig
    capture_method: str
    capture_strategy: str
    post_action_settle_ms: int
    cursor_mode: str
    vlm_image_mode: str
    frame_cadence_ms: int
    inference_interval_ms: int
    finish_strategy_name: str
    finish_keywords: list[str]
    action_config_path: str
    action_config: ActionConfig
    disable_center_delta_conversion: bool
    pause_key: Optional[str]
    delay_ms: int
    output_dir: Path

    @classmethod
    def from_raw(
        cls,
        raw: dict[str, Any],
        *,
        lora_base_model_source_override: Optional[str] = None,
    ) -> "GameplayRuntimeConfig":
        if not isinstance(raw, dict):
            raise ValueError("Gameplay config must be a mapping")

        gp = raw.get("gameplay", {})
        if not isinstance(gp, dict):
            gp = {}
        capture_cfg = raw.get("capture", {})
        if not isinstance(capture_cfg, dict):
            capture_cfg = {}
        actions_cfg = raw.get("actions", {})
        if not isinstance(actions_cfg, dict):
            actions_cfg = {}

        action_config_path = str(
            actions_cfg.get("config_path", "configs/actions/default-actions_chess.yaml")
        )
        action_config = load_action_config(action_config_path)
        model_settings = ModelSettingsConfig.from_gameplay_config(
            gp,
            lora_base_model_source_override=lora_base_model_source_override,
        )

        finish_keywords = gp.get("finish_keywords", _DEFAULT_FINISH_KEYWORDS)
        if not isinstance(finish_keywords, list):
            finish_keywords = list(_DEFAULT_FINISH_KEYWORDS)

        pause_key = gp.get("pause_key", "f1")
        if pause_key is not None:
            pause_key = str(pause_key)

        return cls(
            raw_config=copy.deepcopy(raw),
            backend_name=str(gp.get("backend", "transformers")),
            model_id=str(gp.get("model", "Qwen/Qwen3-VL-2B-Instruct")),
            max_new_tokens=int(gp.get("max_new_tokens", 64)),
            model_settings=model_settings,
            llama_cpp_config=copy.deepcopy(gp.get("llama_cpp", {})) or {},
            openai_config=copy.deepcopy(gp.get("openai", {})) or {},
            system_prompt=str(
                gp.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
            ).strip(),
            user_prompt=str(gp.get("user_prompt", _DEFAULT_USER_PROMPT)).strip(),
            message_history=MessageHistoryConfig.from_config(
                gp.get("message_history", {})
            ),
            state_context=StateContextConfig.from_config(gp.get("state_context", {})),
            visual_context=VisualContextConfig.from_config(gp.get("visual_context", {})),
            progress_tracking=ProgressTrackingConfig.from_config(
                gp.get("progress_tracking", {})
            ),
            log_vlm_io=_bool_value(
                gp.get("log_vlm_io", gp.get("log_vlm_inputs", True)),
                True,
            ),
            action_hold_duration_ms=_duration_ms_value(
                gp.get("action_hold_duration_ms", gp.get("action_duration_ms")),
            ),
            freeze=FreezeConfig.from_config(gp.get("freeze", {})),
            capture_method=str(capture_cfg.get("method", "auto")),
            capture_strategy=_capture_strategy_value(
                capture_cfg.get(
                    "strategy",
                    gp.get("capture_strategy", "after_action"),
                )
            ),
            post_action_settle_ms=_nonnegative_int_value(
                capture_cfg.get(
                    "post_action_settle_ms",
                    gp.get("post_action_settle_ms", 250),
                ),
                250,
            ),
            cursor_mode=cursor_mode_from_config(capture_cfg, default="both"),
            vlm_image_mode=vlm_image_mode_from_config(
                capture_cfg,
                default="cursor",
            ),
            frame_cadence_ms=max(1, int(gp.get("frame_cadence_ms", 1000))),
            inference_interval_ms=_nonnegative_int_value(
                gp.get("inference_interval_ms", 0), 0,
            ),
            finish_strategy_name=str(gp.get("finish_strategy", "ask_user")),
            finish_keywords=[str(item) for item in finish_keywords],
            action_config_path=action_config_path,
            action_config=action_config,
            disable_center_delta_conversion=bool(
                actions_cfg.get("disable_center_delta_conversion", False)
            ),
            pause_key=pause_key,
            delay_ms=int(gp["delay_ms"]) if gp.get("delay_ms") is not None else 3000,
            output_dir=Path(gp.get("output_dir", "outputs/gameplay-runs")),
        )


def load_gameplay_config(path: str | Path) -> dict[str, Any]:
    """Load runtime gameplay YAML as a raw mapping."""
    with open(Path(path), "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge_config(
    base: dict[str, Any],
    overrides: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Deep-merge dictionaries; lists, scalars, and nulls replace outright."""
    merged = copy.deepcopy(base)
    if not overrides:
        return merged
    if not isinstance(overrides, dict):
        raise ValueError("Config overrides must be a mapping")
    _deep_merge_in_place(merged, overrides)
    return merged


def _deep_merge_in_place(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge_in_place(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_value(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, parsed)


def _nonnegative_int_value(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


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


def _capture_strategy_value(value: Any) -> str:
    """Normalize the gameplay frame acquisition strategy."""
    text = str(value or "after_action").strip().lower().replace("-", "_")
    aliases = {
        "post_action": "after_action",
        "synchronous": "after_action",
        "continuous": "background",
        "realtime": "background",
    }
    strategy = aliases.get(text, text)
    if strategy not in {"after_action", "background"}:
        logger.warning(
            "Unknown capture.strategy=%r; using 'after_action'", value
        )
        return "after_action"
    return strategy
