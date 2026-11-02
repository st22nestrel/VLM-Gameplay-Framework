"""Temporary, process-local cursor compatibility for message-driven games.

Chess Titans reads GetCursorPos instead of the mouse message's LPARAM. Frida
lets that read see the virtual position without moving the desktop cursor.
Only the recipient thread, during synchronous mouse dispatch, is affected.
"""

from pathlib import Path
import threading


class ProcessCursor:
    def __init__(self, pid: int):
        try:
            import frida
        except ImportError as exc:
            raise RuntimeError(
                "Chess Titans windowmessage input needs the process-local cursor shim. "
                "Install it with: python -m pip install -r requirements-windowmessage.txt"
            ) from exc
        self._lock = threading.Lock()
        self._session = None
        self._script = None
        try:
            self._session = frida.attach(pid)
            self._script = self._session.create_script(
                Path(__file__).with_name("process_cursor.js").read_text(encoding="utf-8"))
            self._script.load()
            self._script.exports_sync.ready()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Cannot install the target-process cursor shim: {exc}") from exc

    def dispatch(self, hwnd: int, message: int, wparam: int, lparam: int,
                 screen: tuple[int, int]) -> None:
        with self._lock:
            try:
                self._script.exports_sync.dispatch(hex(hwnd), message, wparam, lparam, *screen)
            except Exception as exc:
                raise RuntimeError(f"Target-process mouse dispatch failed: {exc}") from exc

    def close(self) -> None:
        # Detaching unloads the script and restores the hooked API. It is also
        # safe after a target exits. No persistent game files are modified.
        session, self._session = self._session, None
        self._script = None
        if session is not None:
            try:
                session.detach()
            except Exception:
                pass  # Already detached / target exited.
