"""Discovery response parser: extracts navigation commands and key mappings from VLM text."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── data classes ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class NavigationAction:
    """A navigation command extracted from VLM response."""

    type: str  # "click" | "scroll"
    x: int
    y: int


@dataclasses.dataclass(frozen=True)
class DiscoveredMapping:
    """An action-to-key mapping extracted from VLM response."""

    action_name: str
    key_name: str
    raw_line: str


@dataclasses.dataclass(frozen=True)
class DiscoveryParseResult:
    """Result of parsing a single VLM discovery response."""

    navigation_actions: list[NavigationAction]
    discovered_mappings: list[DiscoveredMapping]
    is_exit: bool
    raw_text: str


# ── regex patterns ────────────────────────────────────────────────────────────

# click 500 300  /  click(500, 300)  /  click 500, 300
_RE_CLICK = re.compile(
    r"^\s*click\s*\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?\s*$", re.IGNORECASE
)

# scroll 0 -3  /  scroll(0, -3)
_RE_SCROLL = re.compile(
    r"^\s*scroll\s*\(?\s*(-?\d+)\s*[,\s]\s*(-?\d+)\s*\)?\s*$", re.IGNORECASE
)

# action name : key   /   action name - key   /   action name = key
_RE_MAPPING = re.compile(
    r"^\s*(.+?)\s*(?:[:=\-])\s+(.+?)\s*$"
)


# ── public API ────────────────────────────────────────────────────────────────


def parse_discovery_response(
    text: str,
    exit_keywords: Optional[list[str]] = None,
) -> DiscoveryParseResult:
    """Parse a VLM discovery response into navigation actions and/or mappings.

    Args:
        text: Raw VLM response text.
        exit_keywords: List of phrases that signal discovery is complete.
            Defaults to common exit phrases if not provided.

    Returns:
        DiscoveryParseResult with extracted actions, mappings, and exit flag.
    """
    if exit_keywords is None:
        exit_keywords = [
            "no more mappings",
            "discovery complete",
            "all mappings found",
            "no additional mappings",
        ]

    text_stripped = text.strip()
    if not text_stripped:
        return DiscoveryParseResult(
            navigation_actions=[],
            discovered_mappings=[],
            is_exit=False,
            raw_text=text,
        )

    # Exit detection
    text_lower = text_stripped.lower()
    is_exit = any(kw.lower() in text_lower for kw in exit_keywords)

    # Try JSON parsing first
    mappings_from_json = _try_parse_json(text_stripped)
    if mappings_from_json is not None:
        return DiscoveryParseResult(
            navigation_actions=[],
            discovered_mappings=mappings_from_json,
            is_exit=is_exit,
            raw_text=text,
        )

    # Parse line by line
    nav_actions: list[NavigationAction] = []
    mappings: list[DiscoveredMapping] = []

    for line in text_stripped.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try navigation: click
        m = _RE_CLICK.match(line)
        if m:
            nav_actions.append(NavigationAction("click", int(m.group(1)), int(m.group(2))))
            continue

        # Try navigation: scroll
        m = _RE_SCROLL.match(line)
        if m:
            nav_actions.append(NavigationAction("scroll", int(m.group(1)), int(m.group(2))))
            continue

        # Skip lines that are just exit keywords
        if any(kw.lower() in line.lower() for kw in exit_keywords):
            continue

        # Try mapping with separator (: - =)
        m = _RE_MAPPING.match(line)
        if m:
            action_name = m.group(1).strip()
            key_name = m.group(2).strip()
            # Reject if action_name looks like a navigation command
            if action_name.lower() in ("click", "scroll"):
                continue
            mappings.append(DiscoveredMapping(action_name, key_name, line))
            continue

        # Fallback: last word is key, rest is action name
        parts = line.split()
        if len(parts) >= 2:
            key_name = parts[-1]
            action_name = " ".join(parts[:-1])
            # Only use fallback if key_name looks like a single key (short)
            if len(key_name) <= 12:
                mappings.append(DiscoveredMapping(action_name, key_name, line))

    return DiscoveryParseResult(
        navigation_actions=nav_actions,
        discovered_mappings=mappings,
        is_exit=is_exit,
        raw_text=text,
    )


# ── internal helpers ──────────────────────────────────────────────────────────


def _try_parse_json(text: str) -> Optional[list[DiscoveredMapping]]:
    """Try to parse text as JSON mapping. Returns None if not JSON."""
    first_char = text.lstrip()[:1]
    if first_char not in ("{", "["):
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    mappings: list[DiscoveredMapping] = []

    if isinstance(data, dict):
        for action_name, key_name in data.items():
            mappings.append(DiscoveredMapping(
                action_name=str(action_name),
                key_name=str(key_name),
                raw_line=f"{action_name}: {key_name}",
            ))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for action_name, key_name in item.items():
                    mappings.append(DiscoveredMapping(
                        action_name=str(action_name),
                        key_name=str(key_name),
                        raw_line=f"{action_name}: {key_name}",
                    ))

    return mappings if mappings else None
