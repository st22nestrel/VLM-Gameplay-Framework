"""Action state tracker: manages held keys and mouse buttons across ticks."""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from src.actions.schema import ActionConfig, ActionDef, ParsedAction

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class HeldInput:
    """An input (key or mouse button) that is currently held down."""

    canonical_name: str  # which action is holding this
    input_type: str  # "key" or "mouse_button"
    key: Optional[str] = None
    button: Optional[str] = None


@dataclasses.dataclass
class TickPlan:
    """What the executor should do for one tick.

    Computed by comparing the previous held state with the new actions.
    """

    to_release: list[HeldInput]  # inputs to release (key-up / button-up)
    to_press: list[HeldInput]  # inputs to press and hold (key-down / button-down)
    mouse_moves: list[dict[str, int]]  # list of {x, y} relative deltas
    click_ats: list[dict[str, int]]  # list of {x, y} absolute click positions
    to_tap: list[HeldInput] = dataclasses.field(default_factory=list)


class ActionState:
    """Tracks which keys/buttons are held and computes tick transitions."""

    def __init__(self, config: ActionConfig) -> None:
        self._config = config
        self._held: dict[str, HeldInput] = {}  # physical input id -> HeldInput

    @property
    def held_inputs(self) -> dict[str, HeldInput]:
        """Currently held inputs (read-only view)."""
        view: dict[str, HeldInput] = {}
        for physical_id, held in self._held.items():
            if held.canonical_name not in view:
                view[held.canonical_name] = held
            else:
                view[f"{held.canonical_name}:{physical_id}"] = held
        return view

    def compute_tick(self, actions: list[ParsedAction]) -> TickPlan:
        """Given the new set of parsed actions, compute what to release/press.

        Hold-type actions persist until replaced. Non-hold actions (like aim)
        execute once and don't persist.
        """
        new_holds: dict[str, HeldInput] = {}
        to_tap: list[HeldInput] = []
        mouse_moves: list[dict[str, int]] = []
        click_ats: list[dict[str, int]] = []

        for action in actions:
            action_def = self._config.get_action(action.canonical_name)
            if action_def is None:
                logger.warning("Unknown action %r in state tracker", action.canonical_name)
                continue

            ex = action_def.execution

            if ex.type == "mouse_move":
                # Mouse moves are one-shot, not held
                if "x" in action.params and "y" in action.params:
                    mouse_moves.append({"x": action.params["x"], "y": action.params["y"]})
                continue

            if ex.type == "click_at":
                # Click_at is one-shot, not held
                if "x" in action.params and "y" in action.params:
                    click_ats.append({"x": action.params["x"], "y": action.params["y"]})
                continue

            if not ex.hold:
                # Non-hold key/button actions execute once and don't persist.
                to_tap.extend(_tap_inputs_for_action(action.canonical_name, action_def))
                continue

            # Hold-type action: track each physical input. This supports key
            # chords such as Shift+W without double-pressing shared modifiers.
            for physical_id, held in _held_inputs_for_action(
                action.canonical_name,
                action_def,
            ):
                new_holds[physical_id] = held

        # Compute diff: what to release and what to press
        to_release = []
        to_press = []

        # Release inputs that are no longer in the new set
        for physical_id, held in self._held.items():
            if physical_id not in new_holds:
                to_release.append(held)

        # Press inputs that are new (not already held)
        for physical_id, held in new_holds.items():
            if physical_id not in self._held:
                to_press.append(held)

        # If a one-shot tap targets an input that is also meant to remain held
        # this tick, skip the tap so it cannot key-up the held action.
        to_tap = [
            held for held in to_tap
            if _physical_id_for_held(held) not in new_holds
        ]

        # Update state
        self._held = new_holds

        plan = TickPlan(
            to_release=to_release,
            to_press=to_press,
            mouse_moves=mouse_moves,
            click_ats=click_ats,
            to_tap=to_tap,
        )

        logger.debug(
            "Tick plan: release=%d, press=%d, tap=%d, mouse_moves=%d, click_ats=%d, held=%d",
            len(to_release), len(to_press), len(to_tap),
            len(mouse_moves), len(click_ats), len(self._held),
        )
        return plan

    def release_all(self) -> list[HeldInput]:
        """Release all currently held inputs. Returns what was released."""
        released = list(self._held.values())
        self._held.clear()
        return released


def _held_inputs_for_action(
    canonical_name: str,
    action_def: ActionDef,
) -> list[tuple[str, HeldInput]]:
    ex = action_def.execution

    if ex.type in {"hold_key", "hold_keys"}:
        keys = list(ex.keys)
        if ex.key:
            keys.append(ex.key)
        result = []
        for key in keys:
            normalized = key.lower().strip()
            held = HeldInput(
                canonical_name=canonical_name,
                input_type="key",
                key=normalized,
            )
            result.append((f"key:{normalized}", held))
        return result

    if ex.type == "mouse_button" and ex.button:
        button = ex.button.lower().strip()
        held = HeldInput(
            canonical_name=canonical_name,
            input_type="mouse_button",
            button=button,
        )
        return [(f"mouse:{button}", held)]

    return []


def _tap_inputs_for_action(
    canonical_name: str,
    action_def: ActionDef,
) -> list[HeldInput]:
    ex = action_def.execution

    if ex.type in {"press_key", "press_keys"}:
        keys = list(ex.keys)
        if ex.key:
            keys.append(ex.key)
        return [
            HeldInput(
                canonical_name=canonical_name,
                input_type="key",
                key=key.lower().strip(),
            )
            for key in keys
        ]

    if ex.type == "mouse_button" and ex.button:
        return [
            HeldInput(
                canonical_name=canonical_name,
                input_type="mouse_button",
                button=ex.button.lower().strip(),
            )
        ]

    return []


def _physical_id_for_held(held: HeldInput) -> str:
    if held.input_type == "key" and held.key:
        return f"key:{held.key.lower().strip()}"
    if held.input_type == "mouse_button" and held.button:
        return f"mouse:{held.button.lower().strip()}"
    return f"{held.input_type}:{held.canonical_name}"
