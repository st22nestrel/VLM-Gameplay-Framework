"""Cursor overlay helpers for Windows screenshots.

Windows capture APIs generally return the framebuffer without the hardware
cursor because the cursor is composited in a separate layer.  This module keeps
the capture backends raw and adds a cursor-marked screenshot variant at save
time, which makes it work consistently for PrintWindow, BitBlt, MSS, and DXCam.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

CursorMode = str
VLMImageMode = str
_VALID_CURSOR_MODES = {"raw", "cursor", "both"}
_VALID_VLM_IMAGE_MODES = {"raw", "cursor"}


@dataclasses.dataclass(frozen=True)
class CursorOverlayMetadata:
    """Metadata describing a cursor overlay attempt."""

    visible: bool
    inserted: bool
    screen_pos: Optional[tuple[int, int]] = None
    client_pos: Optional[tuple[int, int]] = None
    hotspot: Optional[tuple[int, int]] = None
    cursor_size: Optional[tuple[int, int]] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible": self.visible,
            "inserted": self.inserted,
            "screen_pos": list(self.screen_pos) if self.screen_pos else None,
            "client_pos": list(self.client_pos) if self.client_pos else None,
            "hotspot": list(self.hotspot) if self.hotspot else None,
            "cursor_size": list(self.cursor_size) if self.cursor_size else None,
            "reason": self.reason,
        }


@dataclasses.dataclass(frozen=True)
class CursorOverlayResult:
    image: Image.Image
    metadata: CursorOverlayMetadata


@dataclasses.dataclass(frozen=True)
class ScreenshotSaveResult:
    """Paths/images produced by saving raw and/or cursor screenshot variants."""

    selected_path: Optional[Path]
    selected_image: Image.Image
    selected_mode: VLMImageMode
    raw_path: Optional[Path] = None
    cursor_path: Optional[Path] = None
    cursor_metadata: Optional[CursorOverlayMetadata] = None

    def to_log_fields(self, key: str = "screenshot") -> dict[str, Any]:
        fields: dict[str, Any] = {
            key: str(self.selected_path) if self.selected_path else None,
            f"{key}_mode": self.selected_mode,
            f"{key}_raw": str(self.raw_path) if self.raw_path else None,
            f"{key}_cursor": str(self.cursor_path) if self.cursor_path else None,
        }
        if self.cursor_metadata is not None:
            fields["cursor_overlay"] = self.cursor_metadata.to_dict()
        return fields


def normalize_cursor_mode(value: Any, default: CursorMode = "both") -> CursorMode:
    """Normalize a config value into ``raw``, ``cursor``, or ``both``."""
    if value is None:
        value = default

    if isinstance(value, bool):
        return "cursor" if value else "raw"

    if isinstance(value, (list, tuple, set)):
        normalized = {normalize_cursor_mode(item, default="raw") for item in value}
        if "both" in normalized or {"raw", "cursor"}.issubset(normalized):
            return "both"
        if "cursor" in normalized:
            return "cursor"
        return "raw"

    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "false": "raw",
        "off": "raw",
        "none": "raw",
        "no": "raw",
        "true": "cursor",
        "on": "cursor",
        "yes": "cursor",
        "with-cursor": "cursor",
        "inserted": "cursor",
        "insert-cursor": "cursor",
        "all": "both",
        "raw-and-cursor": "both",
        "cursor-and-raw": "both",
    }
    mode = aliases.get(text, text)
    if mode not in _VALID_CURSOR_MODES:
        raise ValueError(
            f"Invalid cursor screenshot mode {value!r}; expected raw, cursor, or both"
        )
    return mode


def normalize_vlm_image_mode(
    value: Any,
    default: VLMImageMode = "cursor",
) -> VLMImageMode:
    """Normalize a config value into the single image variant sent to the VLM."""
    if value is None:
        value = default

    if isinstance(value, bool):
        return "cursor" if value else "raw"

    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "false": "raw",
        "off": "raw",
        "none": "raw",
        "no": "raw",
        "no-cursor": "raw",
        "without-cursor": "raw",
        "true": "cursor",
        "on": "cursor",
        "yes": "cursor",
        "with-cursor": "cursor",
        "inserted": "cursor",
        "insert-cursor": "cursor",
    }
    mode = aliases.get(text, text)
    if mode not in _VALID_VLM_IMAGE_MODES:
        raise ValueError(
            f"Invalid VLM image mode {value!r}; expected raw or cursor"
        )
    return mode


def cursor_mode_from_config(
    capture_cfg: Mapping[str, Any] | None,
    default: CursorMode = "both",
) -> CursorMode:
    """Resolve screenshot cursor mode from a capture config block.

    Configs should use ``cursor_mode: raw|cursor|both``.  Native MSS cursor
    capture is intentionally not supported because it does not work for this
    Windows-only pipeline.
    """
    cfg = capture_cfg or {}
    for key in ("cursor_mode", "screenshot_mode", "save_mode", "screenshot_variants"):
        if key in cfg:
            return normalize_cursor_mode(cfg.get(key), default=default)
    return normalize_cursor_mode(default, default="both")


def vlm_image_mode_from_config(
    capture_cfg: Mapping[str, Any] | None,
    default: VLMImageMode = "cursor",
) -> VLMImageMode:
    """Resolve which screenshot variant should be sent to the VLM."""
    cfg = capture_cfg or {}
    for key in (
        "vlm_image_mode",
        "vlm_screenshot_mode",
        "inference_image_mode",
        "model_image_mode",
        "input_image_mode",
    ):
        if key in cfg:
            return normalize_vlm_image_mode(cfg.get(key), default=default)
    return normalize_vlm_image_mode(default, default="cursor")


def cursor_path_for(raw_path: Path) -> Path:
    """Return sibling path for the cursor-marked variant."""
    return raw_path.with_name(f"{raw_path.stem}_cursor{raw_path.suffix}")


def save_frame_screenshots(
    frame: Any,
    raw_path: Path,
    cursor_mode: CursorMode = "both",
    selected_mode: Optional[VLMImageMode] = None,
) -> ScreenshotSaveResult:
    """Save a captured frame according to ``cursor_mode``.

    ``raw_path`` is always the filename for the unmodified image.  The cursor
    variant is saved next to it as ``<stem>_cursor<suffix>``.
    """
    mode = normalize_cursor_mode(cursor_mode)
    selection_default: VLMImageMode = (
        "cursor" if mode in ("cursor", "both") else "raw"
    )
    selected_mode = normalize_vlm_image_mode(selected_mode, default=selection_default)
    raw_image = frame.image

    saved_raw_path: Optional[Path] = None
    saved_cursor_path: Optional[Path] = None
    cursor_result: Optional[CursorOverlayResult] = None

    if mode in ("raw", "both"):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_image.save(str(raw_path))
        saved_raw_path = raw_path

    if mode in ("cursor", "both") or selected_mode == "cursor":
        cursor_result = overlay_cursor_on_frame(frame)

    if mode in ("cursor", "both"):
        saved_cursor_path = cursor_path_for(raw_path)
        saved_cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_result.image.save(str(saved_cursor_path))

    if selected_mode == "cursor":
        selected_image = cursor_result.image if cursor_result else raw_image
        selected_path = saved_cursor_path
    else:
        selected_image = raw_image
        selected_path = saved_raw_path

    try:
        if selected_path is not None:
            frame.saved_path = selected_path
    except Exception:
        pass

    return ScreenshotSaveResult(
        selected_path=selected_path,
        selected_image=selected_image,
        selected_mode=selected_mode,
        raw_path=saved_raw_path,
        cursor_path=saved_cursor_path,
        cursor_metadata=cursor_result.metadata if cursor_result else None,
    )


def overlay_cursor_on_frame(frame: Any) -> CursorOverlayResult:
    """Insert the cursor at the position recorded with ``frame``.

    The cursor artwork can be resolved at save time, but its coordinates must
    come from the framebuffer capture rather than the live cursor.
    """
    image: Image.Image = frame.image
    client_rect = getattr(frame, "client_rect", None)
    if client_rect is None:
        client_rect = (0, 0, image.width, image.height)
    return overlay_cursor_on_image(
        image,
        client_rect,
        screen_pos=getattr(frame, "cursor_position", None),
    )


def overlay_cursor_on_image(
    image: Image.Image,
    client_rect: tuple[int, int, int, int],
    *,
    screen_pos: Optional[tuple[int, int]] = None,
) -> CursorOverlayResult:
    """Overlay the cursor at an explicitly supplied screenshot-time position.

    ``screen_pos`` is optional so callers can represent a capture for which
    cursor coordinates were unavailable.  In that case no overlay is added;
    this function never substitutes the live cursor position.
    """
    if sys.platform != "win32":
        return CursorOverlayResult(
            image=image.copy(),
            metadata=CursorOverlayMetadata(
                visible=False,
                inserted=False,
                reason="cursor overlay is only available on Windows",
            ),
        )

    if screen_pos is None:
        return CursorOverlayResult(
            image=image.copy(),
            metadata=CursorOverlayMetadata(
                visible=False,
                inserted=False,
                reason="cursor position was unavailable at frame capture time",
            ),
        )

    cursor = _get_current_cursor_image()
    if cursor is None:
        return CursorOverlayResult(
            image=image.copy(),
            metadata=CursorOverlayMetadata(
                visible=False,
                inserted=False,
                reason="could not read current cursor",
            ),
        )

    cursor_image, hotspot, visible = cursor
    if not visible:
        return CursorOverlayResult(
            image=image.copy(),
            metadata=CursorOverlayMetadata(
                visible=False,
                inserted=False,
                screen_pos=screen_pos,
                hotspot=hotspot,
                cursor_size=cursor_image.size,
                reason="cursor is hidden",
            ),
        )

    client_left, client_top, _, _ = client_rect
    client_pos = (screen_pos[0] - client_left, screen_pos[1] - client_top)
    paste_xy = (client_pos[0] - hotspot[0], client_pos[1] - hotspot[1])

    if not _rects_intersect(
        paste_xy[0],
        paste_xy[1],
        paste_xy[0] + cursor_image.width,
        paste_xy[1] + cursor_image.height,
        0,
        0,
        image.width,
        image.height,
    ):
        return CursorOverlayResult(
            image=image.copy(),
            metadata=CursorOverlayMetadata(
                visible=True,
                inserted=False,
                screen_pos=screen_pos,
                client_pos=client_pos,
                hotspot=hotspot,
                cursor_size=cursor_image.size,
                reason="cursor is outside the captured client area",
            ),
        )

    output = image.convert("RGBA")
    _alpha_composite_clipped(output, cursor_image, paste_xy)
    if image.mode != "RGBA":
        output = output.convert("RGB")

    return CursorOverlayResult(
        image=output,
        metadata=CursorOverlayMetadata(
            visible=True,
            inserted=True,
            screen_pos=screen_pos,
            client_pos=client_pos,
            hotspot=hotspot,
            cursor_size=cursor_image.size,
        ),
    )


def _rects_intersect(
    left_a: int,
    top_a: int,
    right_a: int,
    bottom_a: int,
    left_b: int,
    top_b: int,
    right_b: int,
    bottom_b: int,
) -> bool:
    return left_a < right_b and right_a > left_b and top_a < bottom_b and bottom_a > top_b


def _alpha_composite_clipped(
    base: Image.Image,
    overlay: Image.Image,
    xy: tuple[int, int],
) -> None:
    x, y = xy
    src_left = max(0, -x)
    src_top = max(0, -y)
    dst_left = max(0, x)
    dst_top = max(0, y)
    width = min(overlay.width - src_left, base.width - dst_left)
    height = min(overlay.height - src_top, base.height - dst_top)
    if width <= 0 or height <= 0:
        return
    cropped = overlay.crop((src_left, src_top, src_left + width, src_top + height))
    base.alpha_composite(cropped, dest=(dst_left, dst_top))


if sys.platform == "win32":
    _user32 = ctypes.windll.user32
    _gdi32 = ctypes.windll.gdi32

    CURSOR_SHOWING = 0x00000001
    DI_NORMAL = 0x0003
    BI_RGB = 0
    DIB_RGB_COLORS = 0

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.wintypes.LONG), ("y", ctypes.wintypes.LONG)]

    class _CURSORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.wintypes.DWORD),
            ("flags", ctypes.wintypes.DWORD),
            ("hCursor", ctypes.c_void_p),
            ("ptScreenPos", _POINT),
        ]

    class _ICONINFO(ctypes.Structure):
        _fields_ = [
            ("fIcon", ctypes.wintypes.BOOL),
            ("xHotspot", ctypes.wintypes.DWORD),
            ("yHotspot", ctypes.wintypes.DWORD),
            ("hbmMask", ctypes.c_void_p),
            ("hbmColor", ctypes.c_void_p),
        ]

    class _BITMAP(ctypes.Structure):
        _fields_ = [
            ("bmType", ctypes.wintypes.LONG),
            ("bmWidth", ctypes.wintypes.LONG),
            ("bmHeight", ctypes.wintypes.LONG),
            ("bmWidthBytes", ctypes.wintypes.LONG),
            ("bmPlanes", ctypes.wintypes.WORD),
            ("bmBitsPixel", ctypes.wintypes.WORD),
            ("bmBits", ctypes.c_void_p),
        ]

    class _BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.wintypes.DWORD),
            ("biWidth", ctypes.wintypes.LONG),
            ("biHeight", ctypes.wintypes.LONG),
            ("biPlanes", ctypes.wintypes.WORD),
            ("biBitCount", ctypes.wintypes.WORD),
            ("biCompression", ctypes.wintypes.DWORD),
            ("biSizeImage", ctypes.wintypes.DWORD),
            ("biXPelsPerMeter", ctypes.wintypes.LONG),
            ("biYPelsPerMeter", ctypes.wintypes.LONG),
            ("biClrUsed", ctypes.wintypes.DWORD),
            ("biClrImportant", ctypes.wintypes.DWORD),
        ]

    class _RGBQUAD(ctypes.Structure):
        _fields_ = [
            ("rgbBlue", ctypes.c_ubyte),
            ("rgbGreen", ctypes.c_ubyte),
            ("rgbRed", ctypes.c_ubyte),
            ("rgbReserved", ctypes.c_ubyte),
        ]

    class _BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", _BITMAPINFOHEADER),
            ("bmiColors", _RGBQUAD * 1),
        ]

    _user32.GetCursorInfo.argtypes = [ctypes.POINTER(_CURSORINFO)]
    _user32.GetCursorInfo.restype = ctypes.wintypes.BOOL
    _user32.CopyIcon.argtypes = [ctypes.c_void_p]
    _user32.CopyIcon.restype = ctypes.c_void_p
    _user32.DestroyIcon.argtypes = [ctypes.c_void_p]
    _user32.DestroyIcon.restype = ctypes.wintypes.BOOL
    _user32.GetIconInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ICONINFO)]
    _user32.GetIconInfo.restype = ctypes.wintypes.BOOL
    _user32.DrawIconEx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _user32.DrawIconEx.restype = ctypes.wintypes.BOOL
    _user32.GetDC.argtypes = [ctypes.c_void_p]
    _user32.GetDC.restype = ctypes.c_void_p
    _user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _user32.ReleaseDC.restype = ctypes.c_int

    _gdi32.GetObjectW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    _gdi32.GetObjectW.restype = ctypes.c_int
    _gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    _gdi32.DeleteObject.restype = ctypes.wintypes.BOOL
    _gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    _gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    _gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    _gdi32.DeleteDC.restype = ctypes.wintypes.BOOL
    _gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _gdi32.SelectObject.restype = ctypes.c_void_p
    _gdi32.CreateDIBSection.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_BITMAPINFO),
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
    ]
    _gdi32.CreateDIBSection.restype = ctypes.c_void_p


def _get_current_cursor_image() -> Optional[
    tuple[Image.Image, tuple[int, int], bool]
]:
    """Return current cursor artwork, hotspot, and visibility.

    Cursor coordinates deliberately do not leave this helper; overlays use
    only the position stored with the captured framebuffer.
    """
    try:
        ci = _CURSORINFO()
        ci.cbSize = ctypes.sizeof(_CURSORINFO)
        if not _user32.GetCursorInfo(ctypes.byref(ci)):
            return None

        visible = bool(ci.flags & CURSOR_SHOWING)
        if not ci.hCursor:
            return (_fallback_cursor_image(), (0, 0), visible)

        hcursor = _user32.CopyIcon(ci.hCursor)
        if not hcursor:
            return (_fallback_cursor_image(), (0, 0), visible)

        icon_info = _ICONINFO()
        try:
            if not _user32.GetIconInfo(hcursor, ctypes.byref(icon_info)):
                return (_fallback_cursor_image(), (0, 0), visible)

            hotspot = (int(icon_info.xHotspot), int(icon_info.yHotspot))
            size = _cursor_bitmap_size(icon_info)
            if size is None:
                cursor_img = _fallback_cursor_image()
            else:
                rendered = _render_cursor_icon(hcursor, size[0], size[1])
                cursor_img = rendered if rendered is not None else _fallback_cursor_image()

            return (cursor_img, hotspot, visible)
        finally:
            if icon_info.hbmMask:
                _gdi32.DeleteObject(icon_info.hbmMask)
            if icon_info.hbmColor:
                _gdi32.DeleteObject(icon_info.hbmColor)
            _user32.DestroyIcon(hcursor)
    except Exception as exc:
        logger.debug("Cursor overlay failed: %s", exc)
        return None


def _cursor_bitmap_size(icon_info: Any) -> Optional[tuple[int, int]]:
    bitmap_handle = icon_info.hbmColor or icon_info.hbmMask
    if not bitmap_handle:
        return None

    bmp = _BITMAP()
    if not _gdi32.GetObjectW(bitmap_handle, ctypes.sizeof(bmp), ctypes.byref(bmp)):
        return None

    width = int(bmp.bmWidth)
    height = int(bmp.bmHeight)
    if not icon_info.hbmColor:
        height //= 2

    if width <= 0 or height <= 0:
        return None
    return width, height


def _render_cursor_icon(hcursor: Any, width: int, height: int) -> Optional[Image.Image]:
    transparent = _draw_icon_to_image(hcursor, width, height, (0, 0, 0, 0))
    if transparent is not None and transparent.getchannel("A").getextrema()[1] > 0:
        return transparent

    black = _draw_icon_to_image(hcursor, width, height, (0, 0, 0, 255))
    white = _draw_icon_to_image(hcursor, width, height, (255, 255, 255, 255))
    if black is None or white is None:
        return transparent
    return _extract_alpha_from_backgrounds(black, white)


def _draw_icon_to_image(
    hcursor: Any,
    width: int,
    height: int,
    bg_rgba: tuple[int, int, int, int],
) -> Optional[Image.Image]:
    hdc_screen = None
    hdc_mem = None
    bitmap = None
    old_obj = None

    try:
        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        bits = ctypes.c_void_p()
        hdc_screen = _user32.GetDC(None)
        if not hdc_screen:
            return None
        hdc_mem = _gdi32.CreateCompatibleDC(hdc_screen)
        if not hdc_mem:
            return None

        bitmap = _gdi32.CreateDIBSection(
            hdc_mem,
            ctypes.byref(bmi),
            DIB_RGB_COLORS,
            ctypes.byref(bits),
            None,
            0,
        )
        if not bitmap or not bits.value:
            return None

        old_obj = _gdi32.SelectObject(hdc_mem, bitmap)

        r, g, b, a = bg_rgba
        pixel = bytes((b, g, r, a))
        ctypes.memmove(bits.value, pixel * (width * height), width * height * 4)

        if not _user32.DrawIconEx(
            hdc_mem, 0, 0, hcursor, width, height, 0, None, DI_NORMAL
        ):
            return None

        data = ctypes.string_at(bits.value, width * height * 4)
        return Image.frombuffer("RGBA", (width, height), data, "raw", "BGRA", 0, 1).copy()
    finally:
        if old_obj and hdc_mem:
            _gdi32.SelectObject(hdc_mem, old_obj)
        if bitmap:
            _gdi32.DeleteObject(bitmap)
        if hdc_mem:
            _gdi32.DeleteDC(hdc_mem)
        if hdc_screen:
            _user32.ReleaseDC(None, hdc_screen)


def _extract_alpha_from_backgrounds(black: Image.Image, white: Image.Image) -> Image.Image:
    black_rgba = black.convert("RGBA")
    white_rgba = white.convert("RGBA")
    out = Image.new("RGBA", black_rgba.size, (0, 0, 0, 0))

    black_px = black_rgba.load()
    white_px = white_rgba.load()
    out_px = out.load()
    for y in range(out.height):
        for x in range(out.width):
            br, bg, bb, _ = black_px[x, y]
            wr, wg, wb, _ = white_px[x, y]
            diff = max(abs(wr - br), abs(wg - bg), abs(wb - bb))
            alpha = max(0, min(255, 255 - diff))
            if alpha == 0:
                out_px[x, y] = (0, 0, 0, 0)
                continue
            red = max(0, min(255, round(br * 255 / alpha)))
            green = max(0, min(255, round(bg * 255 / alpha)))
            blue = max(0, min(255, round(bb * 255 / alpha)))
            out_px[x, y] = (red, green, blue, alpha)
    return out


def _fallback_cursor_image() -> Image.Image:
    """Small arrow cursor used when the native cursor cannot be rendered."""
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    points = [(1, 1), (1, 23), (7, 17), (12, 29), (17, 27), (12, 16), (21, 16)]
    draw.polygon(points, fill=(0, 0, 0, 255))
    inner = [(3, 5), (3, 18), (8, 13), (13, 25), (15, 24), (10, 12), (16, 12)]
    draw.polygon(inner, fill=(255, 255, 255, 255))
    return img
