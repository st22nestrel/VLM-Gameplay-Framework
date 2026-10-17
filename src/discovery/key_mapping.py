"""English keyboard reverse lookup: maps key name strings to ExecutionDef objects."""

from __future__ import annotations

import logging
from typing import Optional

from src.actions.schema import ExecutionDef

logger = logging.getLogger(__name__)

# ── alias normalization ───────────────────────────────────────────────────────

_ALIASES: dict[str, str] = {
    "spacebar": "space",
    "space bar": "space",
    "lshift": "shift",
    "left shift": "shift",
    "rshift": "shift",
    "right shift": "shift",
    "lctrl": "ctrl",
    "left ctrl": "ctrl",
    "rctrl": "ctrl",
    "right ctrl": "ctrl",
    "lalt": "alt",
    "left alt": "alt",
    "ralt": "alt",
    "right alt": "alt",
    "return": "enter",
    "escape": "esc",
    "del": "delete",
    "ins": "insert",
    "pgup": "page_up",
    "page up": "page_up",
    "pageup": "page_up",
    "pgdn": "page_down",
    "page down": "page_down",
    "pagedown": "page_down",
    "pagedn": "page_down",
    "up arrow": "up",
    "down arrow": "down",
    "left arrow": "left_arrow",
    "right arrow": "right_arrow",
    "arrow up": "up",
    "arrow down": "down",
    "arrow left": "left_arrow",
    "arrow right": "right_arrow",
    "backquote": "backtick",
    "grave": "backtick",
    "tilde": "backtick",
    "`": "backtick",
    "minus": "-",
    "dash": "-",
    "hyphen": "-",
    "equals": "=",
    "equal": "=",
    "plus": "=",
    "comma": ",",
    "period": ".",
    "dot": ".",
    "slash": "/",
    "forwardslash": "/",
    "forward slash": "/",
    "backslash": "\\",
    "semicolon": ";",
    "apostrophe": "'",
    "quote": "'",
    "open bracket": "[",
    "close bracket": "]",
    "left bracket": "[",
    "right bracket": "]",
    # Mouse aliases
    "mouse1": "left_click",
    "mouse 1": "left_click",
    "lmb": "left_click",
    "left mouse": "left_click",
    "left mouse button": "left_click",
    "left click": "left_click",
    "mouse2": "right_click",
    "mouse 2": "right_click",
    "rmb": "right_click",
    "right mouse": "right_click",
    "right mouse button": "right_click",
    "right click": "right_click",
    "mouse3": "middle_click",
    "mouse 3": "middle_click",
    "mmb": "middle_click",
    "middle mouse": "middle_click",
    "middle mouse button": "middle_click",
    "middle click": "middle_click",
}

# ── keyboard key → ExecutionDef ───────────────────────────────────────────────

# All keys that map to press_key/hold_key ExecutionDef (key name used in executor)
_KEYBOARD_KEYS: set[str] = {
    # Letters
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    # Digits
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # F-keys
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    # Navigation
    "up", "down", "left_arrow", "right_arrow",
    "home", "end", "page_up", "page_down", "insert", "delete",
    # Modifiers & common
    "space", "enter", "esc", "tab", "shift", "ctrl", "alt",
    "backspace", "caps_lock",
    # Punctuation
    "backtick", "-", "=", "[", "]", "\\", ";", "'", ",", ".", "/",
}

# Mouse buttons → ExecutionDef
_MOUSE_BUTTONS: dict[str, str] = {
    "left_click": "left",
    "right_click": "right",
    "middle_click": "middle",
}

# ── hold heuristic ────────────────────────────────────────────────────────────

_HOLD_KEYWORDS = {"move", "walk", "run", "strafe", "crouch", "sprint", "prone"}


def _should_hold(action_name: str) -> bool:
    """Heuristic: movement-like actions get hold=True."""
    name_lower = action_name.lower()
    return any(kw in name_lower for kw in _HOLD_KEYWORDS)


# ── public API ────────────────────────────────────────────────────────────────


def resolve_key(
    key_name: str,
    action_name: str = "",
) -> Optional[ExecutionDef]:
    """Look up a key name and return the corresponding ExecutionDef.

    Args:
        key_name: English key name (e.g. "w", "space", "left click").
        action_name: The action this key is bound to. Used for hold heuristic.

    Returns:
        ExecutionDef or None if the key name is unrecognized.
    """
    normalized = key_name.lower().strip()

    # Apply aliases
    if normalized in _ALIASES:
        normalized = _ALIASES[normalized]

    # Check mouse buttons
    if normalized in _MOUSE_BUTTONS:
        return ExecutionDef(
            type="mouse_button",
            button=_MOUSE_BUTTONS[normalized],
            hold=_should_hold(action_name),
        )

    # Check keyboard keys
    if normalized in _KEYBOARD_KEYS:
        hold = _should_hold(action_name)
        return ExecutionDef(
            type="hold_key" if hold else "press_key",
            key=normalized,
            hold=hold,
        )

    logger.warning("Unrecognized key name: %r", key_name)
    return None
