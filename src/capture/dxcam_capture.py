"""DXCam capture backend using DXGI Desktop Duplication.

Uses the GPU output buffer directly, bypassing GDI entirely.  This is the
most reliable method for capturing DirectX 11/12 games — including
fullscreen titles running under Windows Fullscreen Optimizations (FSO).

Multi-monitor aware: detects which monitor the target window is on and
creates a DXCam camera for that output, translating screen coordinates
to monitor-relative coordinates automatically.

Limitations:
  - Requires ``dxcam`` (``pip install dxcam``).
  - Only works on Windows with a DXGI-capable GPU.
  - True exclusive fullscreen with FSO disabled cannot be captured by any
    non-injection method — the user should use Borderless Fullscreen or
    leave FSO enabled.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

from PIL import Image

from src.capture.frame import CapturedFrame, capture_cursor_position
from src.window_session import WindowSession

logger = logging.getLogger(__name__)

# Cache: output_idx -> DXCamera.  One camera per monitor.
_cameras: dict[int, object] = {}
_dxcam_import_failed = False


def _get_dxcam():
    """Import dxcam or return None."""
    global _dxcam_import_failed
    if _dxcam_import_failed:
        return None
    try:
        import dxcam
        return dxcam
    except Exception as exc:
        logger.warning("dxcam not available (install with `pip install dxcam`): %s", exc)
        _dxcam_import_failed = True
        return None


def _get_camera(output_idx: int = 0):
    """Return a DXCamera for the given monitor index, creating if needed."""
    if output_idx in _cameras:
        return _cameras[output_idx]
    dx = _get_dxcam()
    if dx is None:
        return None
    try:
        cam = dx.create(output_idx=output_idx, output_color="RGB")
        _cameras[output_idx] = cam
        logger.debug("DXCam camera created for output %d", output_idx)
        return cam
    except Exception as exc:
        logger.warning("DXCam camera creation failed for output %d: %s", output_idx, exc)
        return None


def _get_monitors() -> list[tuple[int, int, int, int]]:
    """Return list of monitor rects in virtual-screen coords, ordered by index."""
    try:
        import win32api
        return [m[2] for m in win32api.EnumDisplayMonitors()]
    except Exception:
        return []


def _find_monitor_for_rect(
    client_rect: tuple[int, int, int, int],
) -> tuple[int, tuple[int, int, int, int]]:
    """Return (output_idx, monitor_rect) for the monitor containing the rect center.

    Falls back to output 0 if detection fails.
    """
    monitors = _get_monitors()
    if not monitors:
        return 0, (0, 0, 1920, 1080)

    cx = (client_rect[0] + client_rect[2]) // 2
    cy = (client_rect[1] + client_rect[3]) // 2

    for idx, (ml, mt, mr, mb) in enumerate(monitors):
        if ml <= cx < mr and mt <= cy < mb:
            return idx, (ml, mt, mr, mb)

    # Fallback: primary monitor (index 0)
    return 0, monitors[0]


def is_available() -> bool:
    """Return True if dxcam can be imported."""
    return _get_dxcam() is not None


def capture_frame(session: WindowSession) -> Optional[CapturedFrame]:
    """Capture a single frame via DXGI Desktop Duplication, cropped to client area."""
    if sys.platform != "win32":
        logger.warning("dxcam_capture is Windows-only")
        return None

    if _get_dxcam() is None:
        return None

    if not session.is_valid():
        logger.error("Target window %#010x no longer exists", session.target.hwnd)
        return None

    client_rect = session.get_client_rect()
    window_rect = session.get_rect()
    cl, ct, cr, cb = client_rect
    width = cr - cl
    height = cb - ct

    if width <= 0 or height <= 0:
        logger.error("Invalid client area dimensions: %dx%d", width, height)
        return None

    # Determine which monitor and translate to monitor-relative coords
    output_idx, (ml, mt, mr, mb) = _find_monitor_for_rect(client_rect)

    cam = _get_camera(output_idx)
    if cam is None:
        return None

    # Clamp client rect to monitor bounds and translate to monitor-relative
    rel_left = max(cl - ml, 0)
    rel_top = max(ct - mt, 0)
    rel_right = min(cr - ml, mr - ml)
    rel_bottom = min(cb - mt, mb - mt)

    if rel_right <= rel_left or rel_bottom <= rel_top:
        logger.error("Client area does not overlap monitor %d", output_idx)
        return None

    try:
        region = (rel_left, rel_top, rel_right, rel_bottom)
        frame_array = cam.grab(region=region)
        ts = time.time()
        cursor_position = capture_cursor_position()

        if frame_array is None:
            # grab() returns None when the screen hasn't updated since last call
            logger.debug("DXCam grab returned None (no new frame)")
            return None

        img = Image.fromarray(frame_array)
    except Exception as exc:
        logger.error("DXCam capture failed: %s", exc)
        return None

    return CapturedFrame(
        image=img,
        timestamp=ts,
        hwnd=session.target.hwnd,
        window_rect=window_rect,
        client_rect=client_rect,
        capture_method="dxcam",
        cursor_position=cursor_position,
    )
