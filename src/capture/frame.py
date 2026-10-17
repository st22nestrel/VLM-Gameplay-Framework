"""Canonical frame object shared across capture backends."""

from __future__ import annotations

import dataclasses
import ctypes
import ctypes.wintypes
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image


def capture_cursor_position() -> Optional[tuple[int, int]]:
    """Snapshot the cursor position for a framebuffer capture.

    Keeping this value on the frame prevents screenshot persistence from
    pairing old pixels with a cursor position sampled much later.
    """
    if sys.platform != "win32":
        return None
    point = ctypes.wintypes.POINT()
    try:
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return None
    except Exception:
        return None
    return int(point.x), int(point.y)


@dataclasses.dataclass
class CapturedFrame:
    """A single captured frame with metadata."""

    image: Image.Image
    timestamp: float  # time.time()
    hwnd: int
    window_rect: tuple[int, int, int, int]  # (left, top, right, bottom)
    client_rect: tuple[int, int, int, int]  # client area in screen coords
    capture_method: str  # e.g. "printwindow", "bitblt", "mss"
    cursor_position: Optional[tuple[int, int]] = None
    saved_path: Optional[Path] = None

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def save(self, path: Path) -> Path:
        """Save image to disk and record the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(str(path))
        self.saved_path = path
        return path

    def is_blank(self, threshold: int = 10) -> bool:
        """Heuristic: detect all-black or near-black captures (common failure mode)."""
        gray = self.image.convert("L")
        extrema = gray.getextrema()
        return extrema[1] < threshold

    def to_metadata(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "hwnd": f"{self.hwnd:#010x}",
            "window_rect": list(self.window_rect),
            "client_rect": list(self.client_rect),
            "capture_method": self.capture_method,
            "cursor_position": (
                list(self.cursor_position) if self.cursor_position is not None else None
            ),
            "width": self.width,
            "height": self.height,
            "saved_path": str(self.saved_path) if self.saved_path else None,
        }
