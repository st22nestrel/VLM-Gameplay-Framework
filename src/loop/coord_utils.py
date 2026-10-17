"""Coordinate utility helpers for gameplay and discovery loops.

These helpers are intentionally small and dependency-free so they can be
reused across different loops without importing heavy modules.
"""

from __future__ import annotations


def normalize_xy_from_0_1000(
    x: int | float,
    y: int | float,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Scale ``(x, y)`` from a 0–1000 coordinate space into pixel coords.

    Many VLMs (e.g. some Qwen / Gemma variants) emit coordinates in a fixed
    [0, 1000] range regardless of the actual image resolution. This helper
    converts those normalized coordinates into pixel positions for the given
    ``width``/``height``.

    The result is clamped to the target bounds. If ``width``/``height`` are
    invalid or the inputs cannot be parsed as numbers, the original values are
    returned as best-effort integers.
    """
    if width <= 0 or height <= 0:
        return int(x), int(y)

    try:
        x_norm = float(x)
        y_norm = float(y)
    except (TypeError, ValueError):
        return int(x), int(y)

    # Clamp to the expected [0, 1000] range.
    x_norm = max(0.0, min(1000.0, x_norm))
    y_norm = max(0.0, min(1000.0, y_norm))

    # Use width-1/height-1 so 1000 maps to the last pixel index.
    px = int(round((x_norm / 1000.0) * max(width - 1, 0)))
    py = int(round((y_norm / 1000.0) * max(height - 1, 0)))
    return px, py
