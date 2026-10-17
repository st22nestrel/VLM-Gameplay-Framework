"""Action executor: translates TickPlans into platform input via SendInput.

Two input modes:
- **foreground** (default): uses SendInput — requires the target window to be focused.
- **background** (experimental): uses PostMessage for keyboard and mouse buttons
  (these don't require focus), but falls back to SendInput for mouse movement
  (because FPS games read Raw Input deltas, not WM_MOUSEMOVE).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import sys
import time
from typing import Optional

from src.actions.state import HeldInput, TickPlan
from src.window_session import WindowSession

logger = logging.getLogger(__name__)

# ── Windows constants and structures ─────────────────────────────────────────

if sys.platform == "win32":
    import win32api
    import win32gui
    import win32con

    INPUT_KEYBOARD = 1
    INPUT_MOUSE = 0

    KEYEVENTF_SCANCODE = 0x0008
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_EXTENDEDKEY = 0x0001

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040

    # Scan codes for common game keys (Set 1 / hardware scan codes)
    _SCAN_CODES: dict[str, int] = {
        # Letters
        "a": 0x1E, "b": 0x30, "c": 0x2E, "d": 0x20, "e": 0x12, "f": 0x21,
        "g": 0x22, "h": 0x23, "i": 0x17, "j": 0x24, "k": 0x25, "l": 0x26,
        "m": 0x32, "n": 0x31, "o": 0x18, "p": 0x19, "q": 0x10, "r": 0x13,
        "s": 0x1F, "t": 0x14, "u": 0x16, "v": 0x2F, "w": 0x11, "x": 0x2D,
        "y": 0x15, "z": 0x2C,
        # Digits
        "0": 0x0B, "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05,
        "5": 0x06, "6": 0x07, "7": 0x08, "8": 0x09, "9": 0x0A,
        # F-keys
        "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E, "f5": 0x3F, "f6": 0x40,
        "f7": 0x41, "f8": 0x42, "f9": 0x43, "f10": 0x44, "f11": 0x57, "f12": 0x58,
        # Modifiers & common
        "space": 0x39, "shift": 0x2A, "ctrl": 0x1D, "alt": 0x38,
        "tab": 0x0F, "esc": 0x01, "enter": 0x1C, "backspace": 0x0E,
        "caps_lock": 0x3A,
        # Navigation
        "insert": 0xD2, "delete": 0xD3, "home": 0xC7, "end": 0xCF,
        "page_up": 0xC9, "page_down": 0xD1,
        "up": 0xC8, "down": 0xD0, "left_arrow": 0xCB, "right_arrow": 0xCD,
        # Punctuation
        "backtick": 0x29, "-": 0x0C, "=": 0x0D,
        "[": 0x1A, "]": 0x1B, "\\": 0x2B,
        ";": 0x27, "'": 0x28, ",": 0x33, ".": 0x34, "/": 0x35,
    }

    # Virtual key codes — needed for PostMessage background input
    _VK_CODES: dict[str, int] = {
        # Letters (VK_A=0x41 .. VK_Z=0x5A)
        "a": 0x41, "b": 0x42, "c": 0x43, "d": 0x44, "e": 0x45, "f": 0x46,
        "g": 0x47, "h": 0x48, "i": 0x49, "j": 0x4A, "k": 0x4B, "l": 0x4C,
        "m": 0x4D, "n": 0x4E, "o": 0x4F, "p": 0x50, "q": 0x51, "r": 0x52,
        "s": 0x53, "t": 0x54, "u": 0x55, "v": 0x56, "w": 0x57, "x": 0x58,
        "y": 0x59, "z": 0x5A,
        # Digits (VK_0=0x30 .. VK_9=0x39)
        "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
        "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
        # F-keys
        "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
        "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
        # Modifiers & common
        "space": 0x20, "shift": 0xA0, "ctrl": 0xA2, "alt": 0xA4,
        "tab": 0x09, "esc": 0x1B, "enter": 0x0D, "backspace": 0x08,
        "caps_lock": 0x14,
        # Navigation
        "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23,
        "page_up": 0x21, "page_down": 0x22,
        "up": 0x26, "down": 0x28, "left_arrow": 0x25, "right_arrow": 0x27,
        # Punctuation
        "backtick": 0xC0, "-": 0xBD, "=": 0xBB,
        "[": 0xDB, "]": 0xDD, "\\": 0xDC,
        ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
    }

    # ── ctypes structures for SendInput ──────────────────────────────────

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.wintypes.WORD),
            ("wScan", ctypes.wintypes.WORD),
            ("dwFlags", ctypes.wintypes.DWORD),
            ("time", ctypes.wintypes.DWORD),
            # ULONG_PTR is pointer-sized, but is a value rather than a pointer.
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.wintypes.DWORD),
            ("dwFlags", ctypes.wintypes.DWORD),
            ("time", ctypes.wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", ctypes.wintypes.DWORD),
            ("wParamL", ctypes.wintypes.WORD),
            ("wParamH", ctypes.wintypes.WORD),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("ki", KEYBDINPUT),
            ("mi", MOUSEINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.wintypes.DWORD),
            ("union", _INPUT_UNION),
        ]

    def _send_input(*inputs: INPUT) -> int:
        """Call Windows SendInput with an array of INPUT structs."""
        n = len(inputs)
        arr = (INPUT * n)(*inputs)
        return ctypes.windll.user32.SendInput(n, arr, ctypes.sizeof(INPUT))

    def _make_key_input(
        scan_code: int,
        key_up: bool = False,
        extended: bool = False,
    ) -> INPUT:
        """Create a keyboard INPUT struct using scan codes."""
        flags = KEYEVENTF_SCANCODE
        if key_up:
            flags |= KEYEVENTF_KEYUP
        if extended:
            flags |= KEYEVENTF_EXTENDEDKEY
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.union.ki.wVk = 0
        inp.union.ki.wScan = scan_code
        inp.union.ki.dwFlags = flags
        inp.union.ki.time = 0
        inp.union.ki.dwExtraInfo = 0
        return inp

    def _make_mouse_input(
        dx: int = 0,
        dy: int = 0,
        flags: int = 0,
    ) -> INPUT:
        """Create a mouse INPUT struct."""
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.union.mi.dx = dx
        inp.union.mi.dy = dy
        inp.union.mi.mouseData = 0
        inp.union.mi.dwFlags = flags
        inp.union.mi.time = 0
        inp.union.mi.dwExtraInfo = 0
        return inp

    _MOUSE_BUTTON_DOWN = {
        "left": MOUSEEVENTF_LEFTDOWN,
        "right": MOUSEEVENTF_RIGHTDOWN,
        "middle": MOUSEEVENTF_MIDDLEDOWN,
    }

    _MOUSE_BUTTON_UP = {
        "left": MOUSEEVENTF_LEFTUP,
        "right": MOUSEEVENTF_RIGHTUP,
        "middle": MOUSEEVENTF_MIDDLEUP,
    }

    # ── PostMessage helpers for background mode ──────────────────────────

    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_MOUSEMOVE = 0x0200
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_RBUTTONDOWN = 0x0204
    WM_RBUTTONUP = 0x0205
    WM_MBUTTONDOWN = 0x0207
    WM_MBUTTONUP = 0x0208
    MK_LBUTTON = 0x0001
    MK_RBUTTON = 0x0002
    MK_MBUTTON = 0x0010

    _WM_BUTTON_DOWN = {
        "left": (WM_LBUTTONDOWN, MK_LBUTTON),
        "right": (WM_RBUTTONDOWN, MK_RBUTTON),
        "middle": (WM_MBUTTONDOWN, MK_MBUTTON),
    }

    _WM_BUTTON_UP = {
        "left": (WM_LBUTTONUP, 0),
        "right": (WM_RBUTTONUP, 0),
        "middle": (WM_MBUTTONUP, 0),
    }

    def _make_key_lparam(scan_code: int, key_up: bool = False) -> int:
        """Build the lParam for WM_KEYDOWN/WM_KEYUP PostMessage."""
        lparam = 1 | (scan_code << 16)
        if key_up:
            lparam |= (1 << 30) | (1 << 31)
        return lparam

    def _makelparam(x: int, y: int) -> int:
        """Pack x, y into a single LPARAM (low-word=x, high-word=y)."""
        return (y << 16) | (x & 0xFFFF)


# ── Focus guard ──────────────────────────────────────────────────────────────


def check_focus(session: WindowSession) -> bool:
    """Return True if the target window is the foreground window.

    Logs a warning when focus is lost.
    """
    if sys.platform != "win32":
        logger.warning("Focus check not supported on non-Windows platforms")
        return True

    if session.is_foreground():
        return True

    fg_hwnd = win32gui.GetForegroundWindow()
    try:
        fg_title = win32gui.GetWindowText(fg_hwnd)
    except Exception:
        fg_title = "<unknown>"
    logger.warning(
        "Target window %#010x is NOT focused (foreground: %#010x %r) — skipping input",
        session.target.hwnd, fg_hwnd, fg_title,
    )
    return False


# ── Executor ─────────────────────────────────────────────────────────────────


class ActionExecutor:
    """Executes TickPlans by sending platform input events.

    Two input modes:
    - **foreground** (default): SendInput for everything. Requires focus.
    - **background** (experimental): PostMessage for keyboard/mouse-buttons
      (no focus needed), SendInput for mouse movement (hardware-level, moves
      the real cursor — FPS games read Raw Input deltas which PostMessage
      can't fabricate). Mouse aiming may still not work if the game ignores
      input when unfocused.
    """

    def __init__(
        self,
        session: WindowSession,
        dry_run: bool = False,
        background: bool = False,
        disable_center_delta_conversion: bool = False,
    ) -> None:
        self._session = session
        self._dry_run = dry_run
        self._background = background
        self._disable_center_delta_conversion = disable_center_delta_conversion
        if sys.platform != "win32":
            logger.warning(
                "ActionExecutor: non-Windows platform detected — "
                "input execution will be logged but not sent"
            )
        if background:
            logger.warning(
                "EXPERIMENTAL: background input mode enabled — "
                "keyboard/mouse-buttons via PostMessage (no focus required), "
                "mouse movement via SendInput (moves real cursor, game must "
                "process input when unfocused for aiming to work)."
            )

    def execute_tick(self, plan: TickPlan) -> dict:
        """Execute a single tick plan. Returns a summary dict for logging.

        In foreground mode: skips input when the target window is not focused.
        In background mode: always sends — focus not required.
        """
        summary = {
            "focus_ok": False,
            "released": [],
            "pressed": [],
            "tapped": [],
            "mouse_moves": [],
            "click_ats": [],
            "skipped": False,
            "dry_run": self._dry_run,
            "background": self._background,
        }

        # Focus guard — only enforced in foreground mode
        if not self._background and not check_focus(self._session):
            summary["skipped"] = True
            logger.warning("Tick skipped — target window not focused")
            return summary

        summary["focus_ok"] = True

        # 1. Release inputs no longer held
        for held in plan.to_release:
            self._release_input(held)
            summary["released"].append(held.canonical_name)

        # 2. Press new inputs
        for held in plan.to_press:
            self._press_input(held)
            summary["pressed"].append(held.canonical_name)

        # 3. Execute one-shot taps
        for held in plan.to_tap:
            self._press_input(held)
            time.sleep(0.03)
            self._release_input(held)
            summary["tapped"].append(held.canonical_name)

        # 4. Execute mouse moves
        for move in plan.mouse_moves:
            self._mouse_move(move["x"], move["y"])
            summary["mouse_moves"].append(move)

        # 5. Execute click_at actions
        for click in plan.click_ats:
            self.click_at(click["x"], click["y"])
            summary["click_ats"].append(click)

        return summary

    def release_all(self, held_inputs: list[HeldInput]) -> None:
        """Release a list of held inputs (used for cleanup)."""
        for held in held_inputs:
            self._release_input(held)

    # ── private input methods ────────────────────────────────────────────

    def _press_input(self, held: HeldInput) -> None:
        if held.input_type == "key" and held.key:
            self._key_down(held.key)
        elif held.input_type == "mouse_button" and held.button:
            self._mouse_button_down(held.button)

    def _release_input(self, held: HeldInput) -> None:
        if held.input_type == "key" and held.key:
            self._key_up(held.key)
        elif held.input_type == "mouse_button" and held.button:
            self._mouse_button_up(held.button)

    # ── keyboard ─────────────────────────────────────────────────────────

    def _key_down(self, key: str) -> None:
        logger.debug("Key down: %s (bg=%s)", key, self._background)
        if self._dry_run or sys.platform != "win32":
            return
        if self._background:
            self._post_key(key, up=False)
        else:
            scan = _SCAN_CODES.get(key.lower())
            if scan is None:
                logger.warning("No scan code for key %r — skipping", key)
                return
            _send_input(_make_key_input(scan, key_up=False))

    def _key_up(self, key: str) -> None:
        logger.debug("Key up: %s (bg=%s)", key, self._background)
        if self._dry_run or sys.platform != "win32":
            return
        if self._background:
            self._post_key(key, up=True)
        else:
            scan = _SCAN_CODES.get(key.lower())
            if scan is None:
                logger.warning("No scan code for key %r — skipping", key)
                return
            _send_input(_make_key_input(scan, key_up=True))

    def _post_key(self, key: str, up: bool) -> None:
        """Send a key event via PostMessage to the target HWND."""
        vk = _VK_CODES.get(key.lower())
        scan = _SCAN_CODES.get(key.lower())
        if vk is None or scan is None:
            logger.warning("No VK/scan code for key %r — skipping background send", key)
            return
        msg = WM_KEYUP if up else WM_KEYDOWN
        lparam = _make_key_lparam(scan, key_up=up)
        hwnd = self._session.target.hwnd
        win32api.PostMessage(hwnd, msg, vk, lparam)

    # ── mouse buttons ────────────────────────────────────────────────────

    def _mouse_button_down(self, button: str) -> None:
        logger.debug("Mouse button down: %s (bg=%s)", button, self._background)
        if self._dry_run or sys.platform != "win32":
            return
        if self._background:
            self._post_mouse_button(button, up=False)
        else:
            flag = _MOUSE_BUTTON_DOWN.get(button)
            if flag is None:
                logger.warning("Unknown mouse button %r — skipping", button)
                return
            _send_input(_make_mouse_input(flags=flag))

    def _mouse_button_up(self, button: str) -> None:
        logger.debug("Mouse button up: %s (bg=%s)", button, self._background)
        if self._dry_run or sys.platform != "win32":
            return
        if self._background:
            self._post_mouse_button(button, up=True)
        else:
            flag = _MOUSE_BUTTON_UP.get(button)
            if flag is None:
                logger.warning("Unknown mouse button %r — skipping", button)
                return
            _send_input(_make_mouse_input(flags=flag))

    def _post_mouse_button(self, button: str, up: bool) -> None:
        """Send a mouse button event via PostMessage."""
        hwnd = self._session.target.hwnd
        # Send click at center of client area
        client_rect = self._session.get_client_rect()
        client_w = client_rect[2] - client_rect[0]
        client_h = client_rect[3] - client_rect[1]
        cx, cy = client_w // 2, client_h // 2
        lparam = _makelparam(cx, cy)

        if up:
            entry = _WM_BUTTON_UP.get(button)
        else:
            entry = _WM_BUTTON_DOWN.get(button)
        if entry is None:
            logger.warning("Unknown mouse button %r — skipping background send", button)
            return
        msg, wparam = entry
        win32api.PostMessage(hwnd, msg, wparam, lparam)

    # ── click at coordinates ─────────────────────────────────────────────

    def click_at(self, x: int, y: int) -> dict:
        """Move cursor to (x, y) in client coordinates and left-click.

        Foreground mode: SetCursorPos (converted to screen coords) + SendInput click.
        Respects dry_run and focus guard.

        Returns a dict with execution details for debugging.
        """
        info: dict = {"client_target": (x, y)}
        logger.debug("Click at: (%d, %d)", x, y)
        if self._dry_run or sys.platform != "win32":
            info["status"] = "dry_run" if self._dry_run else "non_win32"
            return info
        if not self._background and not check_focus(self._session):
            logger.warning("click_at skipped — target window not focused")
            info["status"] = "skipped_no_focus"
            return info

        # Read cursor position before
        cursor_before = win32api.GetCursorPos()
        info["cursor_before"] = cursor_before

        # Convert client coords to screen coords
        client_rect = self._session.get_client_rect()
        screen_x = client_rect[0] + x
        screen_y = client_rect[1] + y
        info["client_rect_origin"] = (client_rect[0], client_rect[1])
        info["screen_target"] = (screen_x, screen_y)

        win32api.SetCursorPos((screen_x, screen_y))
        time.sleep(0.05)  # small delay for cursor to settle

        # Read cursor position after SetCursorPos
        cursor_after = win32api.GetCursorPos()
        info["cursor_after"] = cursor_after

        # Click down + up
        _send_input(_make_mouse_input(flags=MOUSEEVENTF_LEFTDOWN))
        time.sleep(0.02)
        _send_input(_make_mouse_input(flags=MOUSEEVENTF_LEFTUP))
        info["status"] = "ok"
        return info

    # ── mouse movement ───────────────────────────────────────────────────

    def _mouse_move(self, x: int, y: int) -> None:
        """Send a mouse movement.

        Foreground mode: relative delta via SendInput (for FPS camera control),
                         unless disable_center_delta_conversion is True.
        Background mode: WM_MOUSEMOVE to client coordinates via PostMessage.
          Note: most FPS games ignore WM_MOUSEMOVE for camera — this is
          experimental and mainly useful for menu/UI interaction.
        """
        client_rect = self._session.get_client_rect()
        client_w = client_rect[2] - client_rect[0]
        client_h = client_rect[3] - client_rect[1]

        if client_w <= 0 or client_h <= 0:
            logger.warning("Invalid client rect — skipping mouse move")
            return

        if self._background:
            # Background: send absolute client-area coords via PostMessage
            logger.debug(
                "Mouse move (bg): WM_MOUSEMOVE to (%d, %d) [client %dx%d]",
                x, y, client_w, client_h,
            )
            if self._dry_run:
                return
            hwnd = self._session.target.hwnd
            lparam = _makelparam(x, y)
            win32api.PostMessage(hwnd, WM_MOUSEMOVE, 0, lparam)
        else:
            # Foreground: convert to relative delta from center (unless disabled)
            if self._disable_center_delta_conversion:
                # Use absolute coordinates without conversion
                logger.debug(
                    "Mouse move: aim(%d, %d) [absolute, client %dx%d]",
                    x, y, client_w, client_h,
                )
                if self._dry_run:
                    return
                _send_input(_make_mouse_input(dx=x, dy=y, flags=MOUSEEVENTF_MOVE))
            else:
                # Convert to relative delta from center (default FPS behavior)
                center_x = client_w // 2
                center_y = client_h // 2
                dx = x - center_x
                dy = y - center_y
                logger.debug(
                    "Mouse move: aim(%d, %d) -> delta(%d, %d) [client %dx%d]",
                    x, y, dx, dy, client_w, client_h,
                )
                if self._dry_run:
                    return
                _send_input(_make_mouse_input(dx=dx, dy=dy, flags=MOUSEEVENTF_MOVE))
