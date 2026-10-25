"""Stateful, window-directed input without activating a window or moving the cursor.

This sends legacy Win32 input messages. It does not emulate a hardware device,
change GetAsyncKeyState, or provide Raw Input / DirectInput events.
"""

from __future__ import annotations

import sys
import time
import ctypes
from ctypes import wintypes

if sys.platform == "win32":
    import win32api
    import win32gui
    import win32process

    class _GUIThreadInfo(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                    ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                    ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                    ("rcCaret", wintypes.RECT)]

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(_GUIThreadInfo)]
    _user32.GetGUIThreadInfo.restype = wintypes.BOOL


def _thread_focus(thread: int) -> int | None:
    info = _GUIThreadInfo()
    info.cbSize = ctypes.sizeof(info)
    if not _user32.GetGUIThreadInfo(thread, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return info.hwndFocus


def pack_point(x: int, y: int) -> int:
    """Encode signed client coordinates in the two 16-bit LPARAM words."""
    if not (-32768 <= x <= 32767 and -32768 <= y <= 32767):
        raise ValueError("Window-message coordinates must fit signed 16-bit integers")
    return ((y & 0xFFFF) << 16) | (x & 0xFFFF)


class WindowMessageInput:
    """One virtual mouse and held-key state per attached target.

    Mouse events go to the deepest enabled, visible child at the virtual point.
    A drag stays on its original recipient, with coordinates remapped on every
    event. Keyboard events use the target thread's focused child when available,
    otherwise the last mouse recipient, otherwise the target itself. Releases
    always go to the HWND which received the corresponding press.
    """

    _BUTTONS = {
        "left": (0x0201, 0x0202, 0x0001),
        "right": (0x0204, 0x0205, 0x0002),
        "middle": (0x0207, 0x0208, 0x0010),
    }

    def __init__(self, session, vk_codes: dict[str, int], scan_codes: dict[str, int]):
        if sys.platform != "win32":
            raise RuntimeError("windowmessage input requires Windows")
        self._session = session
        self._hwnd = session.target.hwnd
        self._vk_codes = {**vk_codes, "shift": 0x10, "ctrl": 0x11, "alt": 0x12}
        self._scan_codes = scan_codes
        self._keys: dict[str, int] = {}
        self._buttons: set[str] = set()
        self._point: tuple[int, int] | None = None
        self._mouse_target: int | None = None
        self._drag_target: int | None = None

    def _root(self) -> int:
        if self._session.target.hwnd != self._hwnd:
            raise RuntimeError("Input target changed; create a new executor after reattaching")
        if not self._hwnd or not win32gui.IsWindow(self._hwnd):
            raise RuntimeError("Input target window no longer exists")
        if win32gui.IsIconic(self._hwnd):
            raise RuntimeError("Restore the minimized input target first")
        return self._hwnd

    def _belongs(self, hwnd: int | None) -> bool:
        return bool(hwnd and win32gui.IsWindow(hwnd)
                    and (hwnd == self._hwnd or win32gui.IsChild(self._hwnd, hwnd)))

    def _post(self, hwnd: int, message: int, wparam: int, lparam: int) -> None:
        self._root()
        if not self._belongs(hwnd):
            raise RuntimeError("Input recipient disappeared or left the target window")
        # Let pywin32 propagate access-denied (UIPI), queue-full and invalid-HWND
        # errors. Never fall back to the desktop input stream.
        win32api.PostMessage(hwnd, message, wparam, lparam)

    def _keyboard_target(self) -> int:
        root = self._root()
        thread, _ = win32process.GetWindowThreadProcessId(root)
        focus = _thread_focus(thread)
        if self._belongs(focus):
            return focus
        return self._mouse_target if self._belongs(self._mouse_target) else root

    def key(self, key: str, *, up: bool = False) -> None:
        key = key.lower()
        if key not in self._vk_codes or key not in self._scan_codes:
            raise ValueError(f"Unsupported windowmessage key: {key!r}")
        if up and key not in self._keys:
            return
        target = self._keys.get(key) or self._keyboard_target()
        scan = self._scan_codes[key]
        alt = "alt" in self._keys
        system = key in ("alt", "f10") or alt
        message = (0x0105 if up else 0x0104) if system else (0x0101 if up else 0x0100)
        # The old table uses bit 7 to mark E0-prefixed navigation scan codes.
        lparam = 1 | ((scan & 0x7F) << 16) | (int(bool(scan & 0x80)) << 24)
        lparam |= int(alt) << 29
        lparam |= int(up or key in self._keys) << 30
        lparam |= int(up) << 31
        self._post(target, message, self._vk_codes[key], lparam)
        if up:
            del self._keys[key]
        else:
            self._keys[key] = target

    def _flags(self, buttons: set[str] | None = None) -> int:
        flags = sum(self._BUTTONS[b][2] for b in (self._buttons if buttons is None else buttons))
        if "shift" in self._keys:
            flags |= 0x0004
        if "ctrl" in self._keys:
            flags |= 0x0008
        return flags

    def _position(self) -> tuple[int, int]:
        if self._point is None:
            left, top, right, bottom = win32gui.GetClientRect(self._root())
            if right <= left or bottom <= top:
                raise RuntimeError("Input target has an empty client area")
            self._point = ((right - left) // 2, (bottom - top) // 2)
        return self._point

    def _recipient(self) -> tuple[int, int]:
        root = self._root()
        point = self._position()
        screen = win32gui.ClientToScreen(root, point)
        target = self._drag_target
        if target is None:
            target = root
            while True:
                local = win32gui.ScreenToClient(target, screen)
                # Ignore invisible, disabled and transparent child windows.
                child = win32gui.ChildWindowFromPointEx(target, local, 0x0007)
                if not child or child == target:
                    break
                if not self._belongs(child):
                    raise RuntimeError("Mouse recipient is outside the target window")
                target = child
        if not self._belongs(target):
            raise RuntimeError("Mouse recipient no longer exists")
        local = win32gui.ScreenToClient(target, screen)
        return target, pack_point(*local)

    def move(self, x: int, y: int) -> None:
        pack_point(x, y)  # Validate before changing virtual state.
        previous = self._point
        self._point = (x, y)
        try:
            target, lparam = self._recipient()
            self._post(target, 0x0200, self._flags(), lparam)
        except BaseException:
            self._point = previous
            raise
        self._mouse_target = target

    def button(self, button: str, *, up: bool = False) -> None:
        if button not in self._BUTTONS:
            raise ValueError(f"Unsupported windowmessage mouse button: {button!r}")
        if up and button not in self._buttons:
            return
        if not up and button in self._buttons:
            return
        target, lparam = self._recipient()
        buttons = self._buttons - {button} if up else self._buttons | {button}
        message = self._BUTTONS[button][int(up)]
        self._post(target, message, self._flags(buttons), lparam)
        self._buttons = buttons
        self._mouse_target = target
        self._drag_target = target if buttons else None

    def click_at(self, x: int, y: int) -> dict:
        self.move(x, y)
        # A one-shot click must not release a separately held left button.
        if "left" in self._buttons:
            return {"status": "already_held", "client_target": (x, y)}
        self.button("left")
        try:
            time.sleep(0.02)
        finally:
            self.button("left", up=True)
        return {"status": "posted", "client_target": (x, y), "hwnd": self._mouse_target}

    def release_all(self) -> None:
        """Attempt every release even if one recipient has disappeared."""
        errors = []
        for button in list(self._buttons):
            try:
                self.button(button, up=True)
            except Exception as exc:
                errors.append(exc)
        # Release non-modifiers first, so Alt-up remains the last system event.
        keys = sorted(self._keys, key=lambda key: key in ("shift", "ctrl", "alt"))
        for key in keys:
            try:
                self.key(key, up=True)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
