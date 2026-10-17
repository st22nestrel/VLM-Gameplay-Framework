"""Global hotkey listener for pause/resume using Windows RegisterHotKey.

RegisterHotKey registers a system-wide hotkey that is consumed by Windows
before it reaches any application — so the game never sees the key press.
A dedicated thread runs a message pump to receive WM_HOTKEY messages.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Virtual key codes for the recommended pause/resume keys.
# These are relatively safe to intercept globally because games rarely need
# them for normal gameplay.
SAFE_VK_MAP: dict[str, int] = {
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
    "pause": 0x13,
    "scroll_lock": 0x91,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "page_up": 0x21,
    "page_down": 0x22,
}

_EXTRA_VK_MAP: dict[str, int] = {
    # Letters
    **{chr(code): 0x41 + code - ord("a") for code in range(ord("a"), ord("z") + 1)},
    # Digits
    **{str(i): 0x30 + i for i in range(10)},
    # F13-F24
    **{f"f{i}": 0x70 + (i - 1) for i in range(13, 25)},
    # Modifiers
    "shift": 0x10,
    "lshift": 0xA0,
    "rshift": 0xA1,
    "ctrl": 0x11,
    "lctrl": 0xA2,
    "rctrl": 0xA3,
    "alt": 0x12,
    "lalt": 0xA4,
    "ralt": 0xA5,
    "win": 0x5B,
    "lwin": 0x5B,
    "rwin": 0x5C,
    "apps": 0x5D,
    # Common keys
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "caps_lock": 0x14,
    "esc": 0x1B,
    "space": 0x20,
    "print_screen": 0x2C,
    "delete": 0x2E,
    "num_lock": 0x90,
    # Arrows
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "left_arrow": 0x25,
    "up_arrow": 0x26,
    "right_arrow": 0x27,
    "down_arrow": 0x28,
    # Numpad
    **{f"numpad{i}": 0x60 + i for i in range(10)},
    "numpad_multiply": 0x6A,
    "numpad_add": 0x6B,
    "numpad_separator": 0x6C,
    "numpad_subtract": 0x6D,
    "numpad_decimal": 0x6E,
    "numpad_divide": 0x6F,
    # Punctuation on a US keyboard layout.
    "semicolon": 0xBA,
    ";": 0xBA,
    "=": 0xBB,
    ",": 0xBC,
    "-": 0xBD,
    ".": 0xBE,
    "/": 0xBF,
    "backtick": 0xC0,
    "`": 0xC0,
    "[": 0xDB,
    "\\": 0xDC,
    "]": 0xDD,
    "quote": 0xDE,
    "'": 0xDE,
}

VK_MAP: dict[str, int] = {
    **_EXTRA_VK_MAP,
    **SAFE_VK_MAP,
}

_ALIASES: dict[str, str] = {
    "escape": "esc",
    "return": "enter",
    "control": "ctrl",
    "lcontrol": "lctrl",
    "rcontrol": "rctrl",
    "pageup": "page_up",
    "pgup": "page_up",
    "pagedown": "page_down",
    "pgdn": "page_down",
    "scrolllock": "scroll_lock",
    "capslock": "caps_lock",
    "numlock": "num_lock",
    "printscreen": "print_screen",
    "prtsc": "print_screen",
}

SAFE_KEYS = sorted(SAFE_VK_MAP.keys())
SUPPORTED_KEYS = sorted(VK_MAP.keys())

# Windows constants
_WM_HOTKEY = 0x0312
_MOD_NOREPEAT = 0x4000
_HOTKEY_ID = 1
_PM_REMOVE = 0x0001


def _normalize_key_name(key_name: str) -> str:
    key = key_name.lower().strip()
    key = key.replace(" ", "_").replace("-", "_")
    return _ALIASES.get(key, key)


def resolve_vk(key_name: str) -> Optional[int]:
    """Resolve a key name to a virtual key code, or None if unsupported."""
    return VK_MAP.get(_normalize_key_name(key_name))


def is_safe_key(key_name: str) -> bool:
    """Return True if a key is in the recommended global-hotkey set."""
    return _normalize_key_name(key_name) in SAFE_VK_MAP


class HotkeyListener:
    """Listens for a global hotkey and calls a callback on each press.

    The hotkey is registered system-wide via RegisterHotKey, which means
    the key press is consumed and never forwarded to the focused application.

    Parameters
    ----------
    vk_code : int
        Virtual key code to listen for.
    on_press : callable
        Called (from the listener thread) each time the hotkey is pressed.
    """

    def __init__(self, vk_code: int, on_press: Callable[[], None]) -> None:
        self._vk_code = vk_code
        self._on_press = on_press
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._registered = False

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the listener thread. Returns True if started successfully."""
        if sys.platform != "win32":
            logger.warning("HotkeyListener only works on Windows")
            return False
        if self.is_running:
            return True

        self._stop_event.clear()
        self._registered = False
        ready = threading.Event()

        self._thread = threading.Thread(
            target=self._run, args=(ready,), daemon=True, name="hotkey-listener"
        )
        self._thread.start()

        # Wait for registration to complete (up to 2s)
        ready.wait(timeout=2.0)
        return self._registered

    def stop(self) -> None:
        """Stop the listener thread and unregister the hotkey."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self, ready: threading.Event) -> None:
        """Thread target: register hotkey, pump messages, unregister."""
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32

        # Register the hotkey: no modifier keys, just the key itself
        result = user32.RegisterHotKey(None, _HOTKEY_ID, _MOD_NOREPEAT, self._vk_code)
        self._registered = bool(result)

        if not self._registered:
            logger.error(
                "Failed to register hotkey (VK=0x%02X). "
                "Another application may have claimed it.",
                self._vk_code,
            )
            ready.set()
            return

        logger.info("Global hotkey registered (VK=0x%02X)", self._vk_code)
        ready.set()

        msg = ctypes.wintypes.MSG()
        try:
            while not self._stop_event.is_set():
                # Non-blocking peek so we can check the stop event
                if user32.PeekMessageW(
                    ctypes.byref(msg), None, _WM_HOTKEY, _WM_HOTKEY, _PM_REMOVE
                ):
                    if msg.message == _WM_HOTKEY:
                        try:
                            self._on_press()
                        except Exception:
                            logger.exception("Error in hotkey callback")
                else:
                    self._stop_event.wait(0.05)
        finally:
            user32.UnregisterHotKey(None, _HOTKEY_ID)
            logger.info("Global hotkey unregistered")


class PauseController:
    """Manages pause/resume state via a global hotkey.

    Parameters
    ----------
    key_name : str
        Name of the key (e.g. "f1" or "p"). Must be in VK_MAP.
    """

    def __init__(self, key_name: str) -> None:
        self._key_name = _normalize_key_name(key_name)
        vk = resolve_vk(self._key_name)
        if vk is None:
            raise ValueError(
                f"Unsupported pause key: {key_name!r}. "
                f"Supported keys: {', '.join(SUPPORTED_KEYS)}"
            )
        if self._key_name not in SAFE_VK_MAP:
            logger.warning(
                "Pause key %r is outside the recommended safe set (%s). "
                "It will be intercepted globally and may disrupt normal input.",
                self._key_name,
                ", ".join(SAFE_KEYS),
            )
        self._vk_code = vk
        self._paused = threading.Event()  # set = paused
        self._resume_event = threading.Event()  # set = running (cleared while paused)
        self._resume_event.set()  # initially running
        self._listener = HotkeyListener(vk, self._toggle)

    @property
    def key_name(self) -> str:
        return self._key_name

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def wait_if_paused(self, timeout: float = 0.1) -> bool:
        """If paused, block up to *timeout* seconds. Returns True if paused."""
        if self._paused.is_set():
            # _resume_event is cleared while paused; wait for it (or timeout)
            self._resume_event.wait(timeout)
            return True
        return False

    def start(self) -> bool:
        """Start listening for the hotkey. Returns True on success."""
        return self._listener.start()

    def stop(self) -> None:
        """Stop listening and clean up."""
        self._listener.stop()

    def _toggle(self) -> None:
        """Toggle pause state (called from the listener thread)."""
        if self._paused.is_set():
            self._paused.clear()
            self._resume_event.set()
            logger.info("Gameplay RESUMED via hotkey")
        else:
            self._paused.set()
            self._resume_event.clear()
            logger.info("Gameplay PAUSED via hotkey")


class PhysicalPauseController:
    """Pause controller that reacts only to physical keyboard input.

    Unlike :class:`PauseController`, this uses ``WH_KEYBOARD_LL`` so injected
    ``SendInput`` events can be identified through ``LLKHF_INJECTED``.  The
    physical pause key is consumed before it reaches the foreground game;
    replayed instances of the same key pass through untouched.
    """

    _WH_KEYBOARD_LL = 13
    _WM_QUIT = 0x0012
    _WM_KEYDOWN = 0x0100
    _WM_KEYUP = 0x0101
    _WM_SYSKEYDOWN = 0x0104
    _WM_SYSKEYUP = 0x0105
    _LLKHF_INJECTED_MASK = 0x12

    def __init__(self, key_name: str) -> None:
        self._key_name = _normalize_key_name(key_name)
        vk = resolve_vk(self._key_name)
        if vk is None:
            raise ValueError(
                f"Unsupported pause key: {key_name!r}. "
                f"Supported keys: {', '.join(SUPPORTED_KEYS)}"
            )
        self._vk_code = vk
        self._paused = threading.Event()
        self._state_changed = threading.Event()
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._thread_id = 0
        self._hook = None
        self._callback = None
        self._physical_key_down = False
        self.error: Optional[str] = None

    @property
    def key_name(self) -> str:
        return self._key_name

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def start(self) -> bool:
        """Install the physical-only keyboard hook."""
        if sys.platform != "win32":
            self.error = "physical pause hotkeys are only available on Windows"
            return False
        if self._thread is not None and self._thread.is_alive():
            return True

        self._ready.clear()
        self._stop_event.clear()
        self.error = None
        self._thread = threading.Thread(
            target=self._message_loop,
            daemon=True,
            name="physical-pause-hotkey",
        )
        self._thread.start()
        if not self._ready.wait(timeout=2.0):
            self.error = "timed out while installing physical pause hotkey"
            return False
        return self._hook is not None

    def stop(self) -> None:
        """Remove the keyboard hook and stop its message loop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive() and self._thread_id:
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(
                self._thread_id,
                self._WM_QUIT,
                0,
                0,
            )
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "Physical pause hotkey did not stop within two seconds"
                )
        self._thread = None
        self._state_changed.set()

    def wait_for_change(self, timeout: float) -> bool:
        """Wait until pause state changes, returning whether it changed."""
        changed = self._state_changed.wait(timeout)
        if changed:
            self._state_changed.clear()
        return changed

    def force_pause(self) -> None:
        """Return to paused state, used when resume is unsafe."""
        self._paused.set()
        self._state_changed.set()

    def _toggle(self) -> None:
        if self._paused.is_set():
            self._paused.clear()
            logger.info("Playback RESUMED via physical hotkey")
        else:
            self._paused.set()
            logger.info("Playback PAUSED via physical hotkey")
        self._state_changed.set()

    def _message_loop(self) -> None:
        import ctypes
        import ctypes.wintypes

        class KeyboardData(ctypes.Structure):
            _fields_ = [
                ("vkCode", ctypes.wintypes.DWORD),
                ("scanCode", ctypes.wintypes.DWORD),
                ("flags", ctypes.wintypes.DWORD),
                ("time", ctypes.wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t),
            ]

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        # Ensure PostThreadMessageW can wake this thread even if stop() races
        # with hook installation.
        bootstrap_message = ctypes.wintypes.MSG()
        user32.PeekMessageW(
            ctypes.byref(bootstrap_message),
            None,
            0,
            0,
            0,
        )
        hook_proc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t,
            ctypes.c_int,
            ctypes.wintypes.WPARAM,
            ctypes.wintypes.LPARAM,
        )
        user32.SetWindowsHookExW.argtypes = [
            ctypes.c_int,
            hook_proc_type,
            ctypes.wintypes.HMODULE,
            ctypes.wintypes.DWORD,
        ]
        user32.SetWindowsHookExW.restype = ctypes.wintypes.HANDLE
        user32.CallNextHookEx.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.c_int,
            ctypes.wintypes.WPARAM,
            ctypes.wintypes.LPARAM,
        ]
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        kernel32.GetModuleHandleW.restype = ctypes.wintypes.HMODULE

        def callback(code: int, message: int, data_pointer: int) -> int:
            if code >= 0:
                data = ctypes.cast(
                    data_pointer,
                    ctypes.POINTER(KeyboardData),
                ).contents
                is_pause_key = int(data.vkCode) == self._vk_code
                is_injected = bool(
                    int(data.flags) & self._LLKHF_INJECTED_MASK
                )
                if is_pause_key and not is_injected:
                    if message in (self._WM_KEYDOWN, self._WM_SYSKEYDOWN):
                        if not self._physical_key_down:
                            self._physical_key_down = True
                            self._toggle()
                    elif message in (self._WM_KEYUP, self._WM_SYSKEYUP):
                        self._physical_key_down = False
                    # Consume the real pause key so the game does not see it.
                    return 1
            return user32.CallNextHookEx(
                self._hook,
                code,
                message,
                data_pointer,
            )

        self._callback = hook_proc_type(callback)
        self._thread_id = int(kernel32.GetCurrentThreadId())
        module = kernel32.GetModuleHandleW(None)
        self._hook = user32.SetWindowsHookExW(
            self._WH_KEYBOARD_LL,
            self._callback,
            module,
            0,
        )
        if not self._hook:
            self.error = (
                "SetWindowsHookExW failed with error "
                f"{kernel32.GetLastError()}"
            )
            self._ready.set()
            return

        self._ready.set()
        message = ctypes.wintypes.MSG()
        try:
            while (
                not self._stop_event.is_set()
                and user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0
            ):
                user32.TranslateMessage(ctypes.byref(message))
                user32.DispatchMessageW(ctypes.byref(message))
        finally:
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None
