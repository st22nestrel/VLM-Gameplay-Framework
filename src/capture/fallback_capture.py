"""Fallback capture using mss (screen-region based on window rect).

This captures the screen region where the window is located. It works
reliably for windowed and borderless-windowed games but will include
overlapping windows if any are present.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import mss
from PIL import Image

from src.capture.frame import CapturedFrame, capture_cursor_position
from src.window_session import WindowSession

logger = logging.getLogger(__name__)

# Module-level mss instance (reused for performance)
_sct: Optional[mss.mss] = None


def _get_sct() -> mss.mss:
    global _sct
    if _sct is None:
        _sct = mss.mss()
    return _sct


def capture_frame(session: WindowSession) -> Optional[CapturedFrame]:
    """Capture the screen region corresponding to the target window's client area."""
    if not session.is_valid():
        logger.error("Target window %#010x no longer exists", session.target.hwnd)
        return None

    for warning in session.check_health():
        logger.warning(warning)

    client_rect = session.get_client_rect()
    window_rect = session.get_rect()
    cl, ct, cr, cb = client_rect
    width = cr - cl
    height = cb - ct

    if width <= 0 or height <= 0:
        logger.error("Invalid client area dimensions: %dx%d", width, height)
        return None

    monitor = {"left": cl, "top": ct, "width": width, "height": height}

    try:
        sct = _get_sct()
        sct_img = sct.grab(monitor)
        ts = time.time()
        cursor_position = capture_cursor_position()
        img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
    except Exception as exc:
        logger.error("mss capture failed: %s", exc)
        return None

    return CapturedFrame(
        image=img,
        timestamp=ts,
        hwnd=session.target.hwnd,
        window_rect=window_rect,
        client_rect=client_rect,
        capture_method="mss",
        cursor_position=cursor_position,
    )
