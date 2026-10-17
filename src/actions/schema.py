"""Action schema: canonical actions, execution definitions, and config loading."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# ── data classes ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ExecutionDef:
    """How a single action is physically executed."""

    type: str  # hold_key | press_key | mouse_button | mouse_move | click_at
    key: Optional[str] = None  # virtual-key name for keyboard actions
    keys: tuple[str, ...] = ()  # multiple virtual-key names for chorded actions
    button: Optional[str] = None  # left | right | middle for mouse_button
    hold: bool = False  # stays active until next tick replaces it


@dataclasses.dataclass(frozen=True)
class ParameterDef:
    """A parameter expected by an action (e.g. x, y for aim)."""

    name: str
    type: str = "int"
    required: bool = True


@dataclasses.dataclass(frozen=True)
class ActionDef:
    """Definition of one canonical action loaded from YAML config."""

    canonical_name: str
    synonyms: tuple[str, ...]
    execution: ExecutionDef
    parameters: tuple[ParameterDef, ...] = ()

    @property
    def has_parameters(self) -> bool:
        return len(self.parameters) > 0


@dataclasses.dataclass(frozen=True)
class ParsedAction:
    """A single action parsed from VLM text output."""

    canonical_name: str
    params: dict[str, int | float] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        if self.params:
            p = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"ParsedAction({self.canonical_name!r}, {p})"
        return f"ParsedAction({self.canonical_name!r})"


# ── config loading ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class ActionConfig:
    """Full action configuration loaded from YAML."""

    actions: dict[str, ActionDef]  # canonical_name -> ActionDef
    synonym_map: dict[str, str]  # normalized synonym -> canonical_name
    separators: list[str]

    def get_action(self, canonical_name: str) -> Optional[ActionDef]:
        return self.actions.get(canonical_name)

    def resolve_synonym(self, text: str) -> Optional[str]:
        """Return canonical name for a synonym, or None if not recognized."""
        return self.synonym_map.get(text.lower().strip())

    def canonical_names(self) -> list[str]:
        return list(self.actions.keys())


def load_action_config(path: str | Path) -> ActionConfig:
    """Load an ActionConfig from a YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    separators = raw.get("separators", [" and ", ";", ",", "\n"])
    actions: dict[str, ActionDef] = {}
    synonym_map: dict[str, str] = {}

    for entry in raw.get("actions", []):
        name = entry["canonical_name"]
        exec_raw = entry["execution"]
        execution = ExecutionDef(
            type=exec_raw["type"],
            key=exec_raw.get("key"),
            keys=tuple(str(k).lower().strip() for k in exec_raw.get("keys", [])),
            button=exec_raw.get("button"),
            hold=exec_raw.get("hold", False),
        )
        params = tuple(
            ParameterDef(
                name=p["name"],
                type=p.get("type", "int"),
                required=p.get("required", True),
            )
            for p in entry.get("parameters", [])
        )
        synonyms = tuple(s.lower().strip() for s in entry.get("synonyms", []))

        action_def = ActionDef(
            canonical_name=name,
            synonyms=synonyms,
            execution=execution,
            parameters=params,
        )
        actions[name] = action_def

        # Map each synonym to the canonical name
        for syn in synonyms:
            if syn in synonym_map:
                logger.warning(
                    "Duplicate synonym %r: already mapped to %r, now overwriting with %r",
                    syn, synonym_map[syn], name,
                )
            synonym_map[syn] = name

        # Also map the canonical name itself (normalized)
        synonym_map[name.lower().replace("_", " ")] = name
        synonym_map[name.lower()] = name

    logger.debug(
        "Loaded %d actions with %d synonyms from %s",
        len(actions), len(synonym_map), path,
    )
    return ActionConfig(actions=actions, synonym_map=synonym_map, separators=separators)
