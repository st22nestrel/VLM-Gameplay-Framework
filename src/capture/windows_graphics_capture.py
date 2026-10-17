"""Primary Windows capture using win32 APIs (PrintWindow / BitBlt).

Uses per-window capture via PrintWindow with PW_RENDERFULLCONTENT flag,
which works for most windowed/borderless-windowed applications including
games using DirectX/OpenGL. Falls back to BitBlt from the window DC if
PrintWindow produces a blank image.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from typing import Optional

from PIL import Image

from src.capture.frame import CapturedFrame, capture_cursor_position
from src.window_session import WindowSession

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    import win32gui
    import win32ui
    import win32con

# PrintWindow flag: render full content including DX surfaces (Win 8.1+)
PW_RENDERFULLCONTENT = 0x00000002


def capture_frame(session: WindowSession) -> Optional[CapturedFrame]:
    """Capture a single frame from the session's target window.

    Tries PrintWindow first (with PW_RENDERFULLCONTENT), then BitBlt.
    Returns None if both methods fail.
    """
    if sys.platform != "win32":
        logger.warning("windows_graphics_capture is Windows-only")
        return None

    hwnd = session.target.hwnd
    if not session.is_valid():
        logger.error("Target window %#010x no longer exists", hwnd)
        return None

    for warning in session.check_health():
        logger.warning(warning)

    # Get client area dimensions (what we actually want to capture)
    client_rect = session.get_client_rect()
    window_rect = session.get_rect()
    cl, ct, cr, cb = client_rect
    width = cr - cl
    height = cb - ct

    if width <= 0 or height <= 0:
        logger.error("Invalid client area dimensions: %dx%d", width, height)
        return None

    # Try PrintWindow first
    frame = _capture_printwindow(hwnd, width, height, window_rect, client_rect)
    if frame is not None and not frame.is_blank():
        return frame

    if frame is not None:
        logger.info("PrintWindow returned blank image, trying BitBlt")

    # Fallback to BitBlt
    frame = _capture_bitblt(hwnd, width, height, window_rect, client_rect)
    if frame is not None and not frame.is_blank():
        return frame

    logger.warning("Both PrintWindow and BitBlt produced blank images for %#010x", hwnd)
    return frame  # return the blank frame anyway so caller can decide


def _capture_printwindow(
    hwnd: int,
    width: int,
    height: int,
    window_rect: tuple[int, int, int, int],
    client_rect: tuple[int, int, int, int],
) -> Optional[CapturedFrame]:
    """Capture using PrintWindow with PW_RENDERFULLCONTENT.

    Always renders at full window size and crops to the client area.
    This avoids the clipping bug where a client-sized bitmap truncates
    the bottom of the content because PrintWindow includes the title bar.
    """
    hdc_window = None
    bitmap = None
    try:
        full_w = window_rect[2] - window_rect[0]
        full_h = window_rect[3] - window_rect[1]
        x_offset = client_rect[0] - window_rect[0]
        y_offset = client_rect[1] - window_rect[1]

        hdc_window = win32gui.GetWindowDC(hwnd)
        hdc_mem = win32ui.CreateDCFromHandle(hdc_window)
        hdc_compat = hdc_mem.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(hdc_mem, full_w, full_h)
        hdc_compat.SelectObject(bitmap)

        result = ctypes.windll.user32.PrintWindow(hwnd, hdc_compat.GetSafeHdc(), PW_RENDERFULLCONTENT)
        ts = time.time()
        cursor_position = capture_cursor_position()

        if not result:
            hdc_compat.DeleteDC()
            return None

        bmp_info = bitmap.GetInfo()
        bmp_bits = bitmap.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (bmp_info["bmWidth"], bmp_info["bmHeight"]), bmp_bits, "raw", "BGRX", 0, 1)

        # Crop to client area (removes title bar and window borders)
        if x_offset > 0 or y_offset > 0 or img.width != width or img.height != height:
            img = img.crop((x_offset, y_offset, x_offset + width, y_offset + height))

        hdc_compat.DeleteDC()
        return CapturedFrame(
            image=img,
            timestamp=ts,
            hwnd=hwnd,
            window_rect=window_rect,
            client_rect=client_rect,
            capture_method="printwindow",
            cursor_position=cursor_position,
        )
    except Exception as exc:
        logger.debug("PrintWindow failed: %s", exc)
        return None
    finally:
        if bitmap:
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
        if hdc_window:
            try:
                win32gui.ReleaseDC(hwnd, hdc_window)
            except Exception:
                pass


def _capture_bitblt(
    hwnd: int,
    width: int,
    height: int,
    window_rect: tuple[int, int, int, int],
    client_rect: tuple[int, int, int, int],
) -> Optional[CapturedFrame]:
    """Capture using BitBlt from the desktop DC, cropped to client area.

    Uses the desktop DC (``GetDC(0)``) instead of the window DC so that DWM-
    composited content is captured.  This matches what mss does internally and
    fixes the stale/black-frame problem for borderless-fullscreen and
    Fullscreen-Optimizations-enabled games.
    """
    hdc_src = None
    bitmap = None
    try:
        cl, ct = client_rect[0], client_rect[1]

        hdc_src = win32gui.GetDC(0)  # desktop DC — sees DWM composited output
        hdc_mem_handle = win32ui.CreateDCFromHandle(hdc_src)
        hdc_compat = hdc_mem_handle.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(hdc_mem_handle, width, height)
        hdc_compat.SelectObject(bitmap)
        # Copy the screen region at the client area's position
        hdc_compat.BitBlt((0, 0), (width, height), hdc_mem_handle, (cl, ct), win32con.SRCCOPY)
        ts = time.time()
        cursor_position = capture_cursor_position()

        bmp_info = bitmap.GetInfo()
        bmp_bits = bitmap.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (bmp_info["bmWidth"], bmp_info["bmHeight"]), bmp_bits, "raw", "BGRX", 0, 1)

        hdc_compat.DeleteDC()
        return CapturedFrame(
            image=img,
            timestamp=ts,
            hwnd=hwnd,
            window_rect=window_rect,
            client_rect=client_rect,
            capture_method="bitblt",
            cursor_position=cursor_position,
        )
    except Exception as exc:
        logger.debug("BitBlt failed: %s", exc)
        return None
    finally:
        if bitmap:
            try:
                win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
        if hdc_src:
            try:
                win32gui.ReleaseDC(0, hdc_src)
            except Exception:
                pass
