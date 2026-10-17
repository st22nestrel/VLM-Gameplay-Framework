"""Tolerant parser: turns inconsistent VLM text into canonical ParsedActions."""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.actions.schema import ActionConfig, ParsedAction

logger = logging.getLogger(__name__)

# Patterns for extracting coordinates after an action name.
# Order matters — try most specific first.
_COORD_PATTERNS = [
    re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"),  # "(100, 200)"
    re.compile(r"(\d+)\s*,\s*(\d+)"),  # "100,200" or "100, 200"
    re.compile(r"(\d+)\s+(\d+)"),  # "100 200"
]
_COORD_PREFIX_PATTERNS = [
    re.compile(r"^[\s:=-]*\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
    re.compile(r"^[\s:=-]*(\d+)\s*,\s*(\d+)"),
    re.compile(r"^[\s:=-]*(\d+)\s+(\d+)"),
]

# Regex to match parameterized actions (action name + coordinates) in raw text.
# Built dynamically per config in _extract_parameterized_actions.
_STRIP_CHARS = ".!?:\"'`*#"


def parse_vlm_text(text: str, config: ActionConfig) -> list[ParsedAction]:
    """Parse free-form VLM output text into a list of canonical ParsedActions.

    The parser:
    1. Pre-extracts parameterized actions (e.g. aim with coordinates) before splitting.
    2. Splits remaining text on configured separators.
    3. Resolves each fragment to a canonical action via synonym lookup.
    4. Returns unrecognized fragments as warnings, not errors.
    """
    if not text or not text.strip():
        logger.warning("Empty VLM text, no actions parsed")
        return []

    text = text.strip()

    # Phase 1: extract parameterized actions (aim x y) from raw text
    # before separator splitting can break coordinate patterns.
    param_actions, remaining = _extract_parameterized_actions(text, config)

    # Phase 2: split remaining text on separators and parse fragments
    results: list[ParsedAction] = list(param_actions)

    if remaining.strip():
        fragments = _split_text(remaining, config.separators)
        for frag in fragments:
            parsed = _parse_fragment(frag, config)
            if parsed is not None:
                results.append(parsed)
            else:
                logger.warning("Unrecognized action fragment: %r", frag)

    return results


def _extract_parameterized_actions(
    text: str, config: ActionConfig
) -> tuple[list[ParsedAction], str]:
    """Find and extract parameterized actions from raw text.

    Returns (extracted_actions, remaining_text_with_extractions_removed).
    This prevents coordinate commas/spaces from being misinterpreted as separators.
    """
    extracted: list[ParsedAction] = []

    for action_def in config.actions.values():
        if not action_def.has_parameters:
            continue

        all_names = list(action_def.synonyms) + [
            action_def.canonical_name.replace("_", " "),
            action_def.canonical_name,
        ]
        # Sort longest-first so "aim at" matches before "aim"
        all_names.sort(key=len, reverse=True)

        for name in all_names:
            match = _find_parameterized_match(text, name)
            if match is None:
                continue

            idx, name_end, coord_end, params = match
            after_name = text[name_end:]
            if not params:
                continue

            extracted.append(ParsedAction(
                canonical_name=action_def.canonical_name,
                params=params,
            ))

            # Remove the matched region from text so it doesn't get re-parsed.
            # Find how far the coordinate match extends.
            full_match_end = name_end + coord_end
            text = text[:idx] + text[full_match_end:]
            break  # re-scan from scratch for this action type

    return extracted, text


def _find_parameterized_match(
    text: str,
    action_name: str,
) -> Optional[tuple[int, int, int, dict[str, int]]]:
    """Find an action name immediately followed by coordinates.

    This intentionally does not search the whole suffix for coordinates. Verbose
    model output often contains words such as "look" or "move" early in a
    sentence and a coordinate later in the explanation; treating that as one
    action produces unsafe false positives.
    """
    name_pattern = re.compile(
        rf"(?<!\w){re.escape(action_name)}(?!\w)",
        flags=re.IGNORECASE,
    )
    for match in name_pattern.finditer(text):
        after_name = text[match.end():]
        params, coord_end = _extract_coordinates_at_start(after_name)
        if params:
            return match.start(), match.end(), coord_end, params
    return None


def _split_text(text: str, separators: list[str]) -> list[str]:
    """Split VLM text on configured separators, preserving order."""
    sorted_seps = sorted(separators, key=len, reverse=True)
    escaped = [re.escape(s) for s in sorted_seps]
    pattern = "|".join(escaped)
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [_normalize(p) for p in parts if p.strip()]


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation."""
    text = text.lower().strip()
    text = text.strip(_STRIP_CHARS)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_fragment(fragment: str, config: ActionConfig) -> Optional[ParsedAction]:
    """Try to resolve a single text fragment into a ParsedAction."""
    # Direct synonym lookup on the full fragment
    canonical = config.resolve_synonym(fragment)
    if canonical is not None:
        return _action_without_params(canonical, config)

    # Try progressively shorter prefix matching
    # e.g. "move forward slowly" -> try "move forward slowly", then "move forward"
    words = fragment.split()
    for end in range(len(words), 0, -1):
        prefix = " ".join(words[:end])
        canonical = config.resolve_synonym(prefix)
        if canonical is not None:
            return _action_without_params(canonical, config)

    return None


def _action_without_params(
    canonical_name: str,
    config: ActionConfig,
) -> Optional[ParsedAction]:
    action_def = config.get_action(canonical_name)
    if action_def is not None and action_def.has_parameters:
        return None
    return ParsedAction(canonical_name=canonical_name)


def _extract_coordinates(text: str) -> dict[str, int]:
    """Extract x, y coordinates from text. Returns empty dict if not found."""
    text = text.strip()
    if not text:
        return {}
    for pattern in _COORD_PATTERNS:
        m = pattern.search(text)
        if m:
            return {"x": int(m.group(1)), "y": int(m.group(2))}
    return {}


def _extract_coordinates_at_start(text: str) -> tuple[dict[str, int], int]:
    """Extract x, y coordinates only if they appear right after an action name."""
    if not text:
        return {}, 0
    for pattern in _COORD_PREFIX_PATTERNS:
        m = pattern.match(text)
        if m:
            return {"x": int(m.group(1)), "y": int(m.group(2))}, m.end()
    return {}, 0
