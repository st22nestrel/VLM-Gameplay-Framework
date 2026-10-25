"""Occlusion-safe screenshots using the existing PrintWindow implementation.

Never activates/restores the target and never falls back to desktop pixels.
Applications must continue rendering while unfocused and support PrintWindow.
"""

from __future__ import annotations

import logging
import sys

from src.capture import windows_graphics_capture
from src.capture.frame import CapturedFrame
from src.window_session import WindowSession

logger = logging.getLogger(__name__)


def capture_frame(session: WindowSession) -> CapturedFrame | None:
    if sys.platform != "win32":
        logger.warning("Background capture is Windows-only")
        return None
    try:
        if not session.is_valid() or session.is_minimized():
            logger.warning("Background capture requires a valid, non-minimized window")
            return None
        frame = windows_graphics_capture.capture_frame(session, allow_desktop_fallback=False)
        # is_valid() can reattach the session to a different HWND.
        if not session.is_valid() or session.is_minimized():
            return None
        if frame is not None and frame.hwnd != session.target.hwnd:
            return None
        return frame
    except Exception as exc:
        # The window may close between validation and querying its geometry.
        logger.warning("Background capture failed: %s", exc)
        return None
