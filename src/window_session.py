"""Window session management: enumerate, select, and track a target game window."""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Optional

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    import ctypes.wintypes
    import win32gui
    import win32con
    import win32process

# ── data classes ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class WindowInfo:
    """Immutable snapshot of a discovered window."""

    hwnd: int
    title: str
    class_name: str
    pid: int
    rect: tuple[int, int, int, int]  # (left, top, right, bottom)
    is_visible: bool
    is_minimized: bool

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    def short_str(self) -> str:
        return f"[{self.hwnd:#010x}] {self.title!r}  ({self.width}x{self.height})  pid={self.pid}"


@dataclasses.dataclass
class WindowSession:
    """Tracks the currently-attached target window."""

    target: WindowInfo
    _title_hint: Optional[str] = dataclasses.field(default=None, repr=False)
    _warned_fullscreen: bool = dataclasses.field(default=False, repr=False)

    # ── queries ───────────────────────────────────────────────────────────

    def is_foreground(self) -> bool:
        if sys.platform != "win32":
            return True  # cannot check on non-Windows
        return win32gui.GetForegroundWindow() == self.target.hwnd

    def focus(self) -> bool:
        """Bring the target window to the foreground."""
        if sys.platform != "win32":
            return True
        if not self.is_valid():
            return False
        return focus_window(self.target.hwnd)

    def is_minimized(self) -> bool:
        if sys.platform != "win32":
            return False
        return bool(win32gui.IsIconic(self.target.hwnd))

    def is_valid(self) -> bool:
        if sys.platform != "win32":
            return True
        if win32gui.IsWindow(self.target.hwnd):
            return True
        # Window gone — try to reattach by title hint
        if self._title_hint and self._try_reattach():
            return True
        return False

    def _try_reattach(self) -> bool:
        """Try to find the window again by title hint. Returns True on success."""
        matches = find_window_by_title(self._title_hint)
        if len(matches) == 0:
            logger.warning(
                "Window %#010x gone, no window matching %r found — "
                "cannot reattach",
                self.target.hwnd, self._title_hint,
            )
            return False
        if len(matches) > 1:
            logger.error(
                "Window %#010x gone, multiple windows match %r — "
                "cannot reattach automatically",
                self.target.hwnd, self._title_hint,
            )
            for m in matches:
                logger.error("  %s", m.short_str())
            return False
        old_hwnd = self.target.hwnd
        self.target = matches[0]
        self._warned_fullscreen = False
        logger.info(
            "Window reattached: %#010x -> %#010x (%s)",
            old_hwnd, self.target.hwnd, self.target.title,
        )
        print(
            f"  Window reattached: {old_hwnd:#010x} -> "
            f"{self.target.hwnd:#010x} ({self.target.title!r})"
        )
        return True

    def get_rect(self) -> tuple[int, int, int, int]:
        if sys.platform != "win32":
            return self.target.rect
        return win32gui.GetWindowRect(self.target.hwnd)

    def get_client_rect(self) -> tuple[int, int, int, int]:
        if sys.platform != "win32":
            return self.target.rect
        cr = win32gui.GetClientRect(self.target.hwnd)
        lt = win32gui.ClientToScreen(self.target.hwnd, (cr[0], cr[1]))
        rb = win32gui.ClientToScreen(self.target.hwnd, (cr[2], cr[3]))
        return (lt[0], lt[1], rb[0], rb[1])

    def close(self, *, timeout_s: float = 5.0) -> bool:
        """Ask the target window to close and wait until its HWND disappears."""
        if sys.platform != "win32":
            return True
        return close_window(self.target.hwnd, timeout_s=timeout_s)

    def refresh_info(self) -> WindowInfo:
        """Return an updated WindowInfo snapshot."""
        if sys.platform != "win32":
            return self.target
        info = _build_window_info(self.target.hwnd)
        if info is None:
            logger.warning("Target window %#010x is no longer valid", self.target.hwnd)
            return self.target
        self.target = info
        return info

    # ── health checks with warnings ──────────────────────────────────────

    def check_health(self) -> list[str]:
        """Return a list of warning strings (empty = healthy)."""
        warnings: list[str] = []
        if not self.is_valid():
            warnings.append("Target window no longer exists")
            return warnings
        if self.is_minimized():
            warnings.append("Target window is minimized — capture may fail")
        if _is_likely_exclusive_fullscreen(self.target.hwnd) and not self._warned_fullscreen:
            warnings.append(
                "Target window appears to be exclusive fullscreen — "
                "capture may be less reliable; windowed/borderless-windowed is recommended"
            )
            self._warned_fullscreen = True
        return warnings


@dataclasses.dataclass
class LaunchCloseResult:
    """Outcome of closing a launched window/process."""

    window_closed: bool
    process_exited: bool
    terminated: bool


@dataclasses.dataclass
class LaunchedWindowSession:
    """A process launched by this framework and its attached game window."""

    process: subprocess.Popen
    session: WindowSession
    executable: Path

    def focus(self) -> bool:
        return self.session.focus()

    def close(
        self,
        *,
        timeout_s: float = 5.0,
        terminate_after: bool = True,
    ) -> LaunchCloseResult:
        return close_launched_window(
            self,
            timeout_s=timeout_s,
            terminate_after=terminate_after,
        )


# ── enumeration ───────────────────────────────────────────────────────────────


def enumerate_windows(*, visible_only: bool = True) -> list[WindowInfo]:
    """Return all top-level windows, optionally filtered to visible ones."""
    if sys.platform != "win32":
        logger.warning("Window enumeration is only supported on Windows")
        return []

    results: list[WindowInfo] = []

    def _callback(hwnd: int, _extra: object) -> bool:
        info = _build_window_info(hwnd)
        if info is None:
            return True
        if visible_only and not info.is_visible:
            return True
        if not info.title.strip():
            return True
        results.append(info)
        return True

    win32gui.EnumWindows(_callback, None)
    results.sort(key=lambda w: w.title.lower())
    return results


def find_window_by_title(substring: str) -> list[WindowInfo]:
    """Return windows whose title contains *substring* (case-insensitive)."""
    sub = substring.lower()
    return [w for w in enumerate_windows() if sub in w.title.lower()]


def find_windows_by_pid(
    pid: int,
    *,
    title_substring: Optional[str] = None,
) -> list[WindowInfo]:
    """Return visible top-level windows owned by *pid*.

    When *title_substring* is provided, the result is further filtered by
    case-insensitive title containment.
    """
    matches = [w for w in enumerate_windows() if w.pid == pid]
    if title_substring:
        sub = title_substring.lower()
        matches = [w for w in matches if sub in w.title.lower()]
    return matches


def wait_for_window(
    *,
    pid: Optional[int] = None,
    title_substring: Optional[str] = None,
    process: Optional[subprocess.Popen] = None,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.25,
) -> WindowInfo:
    """Wait until a matching top-level window appears.

    Prefer PID-based matching for launched processes. Title-only matching is
    supported for workflows that cannot reliably associate the final game
    window with the original launcher process.
    """
    if sys.platform != "win32":
        raise RuntimeError("Window launch/attach is only supported on Windows")
    if pid is None and not title_substring:
        raise ValueError("Specify pid and/or title_substring")

    deadline = time.monotonic() + max(0.0, timeout_s)
    poll_interval_s = max(0.01, poll_interval_s)

    while True:
        matches: list[WindowInfo] = []
        if pid is not None:
            matches = find_windows_by_pid(pid, title_substring=title_substring)
        elif title_substring:
            matches = find_window_by_title(title_substring)

        if matches:
            return _choose_best_window(matches)

        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Process pid={process.pid} exited before creating a matching "
                f"window (exit_code={process.returncode})"
            )

        if time.monotonic() >= deadline:
            target = []
            if pid is not None:
                target.append(f"pid={pid}")
            if title_substring:
                target.append(f"title contains {title_substring!r}")
            raise TimeoutError(
                "Timed out waiting for window matching " + ", ".join(target)
            )

        time.sleep(poll_interval_s)


def launch_and_attach(
    executable_path: str | Path,
    *,
    args: Optional[list[str]] = None,
    title_hint: Optional[str] = None,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.25,
    cwd: Optional[str | Path] = None,
    focus: bool = True,
) -> LaunchedWindowSession:
    """Launch an executable, attach to its first matching window, and focus it."""
    if sys.platform != "win32":
        raise RuntimeError("Launching game windows is only supported on Windows")

    executable = Path(executable_path)
    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")
    if not executable.is_file():
        raise ValueError(f"Executable path is not a file: {executable}")

    command = [str(executable)]
    if args:
        command.extend(args)

    working_dir = Path(cwd) if cwd is not None else executable.parent
    process = subprocess.Popen(command, cwd=str(working_dir))

    try:
        info = wait_for_window(
            pid=process.pid,
            title_substring=title_hint,
            process=process,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        session = attach(info.hwnd)
        if session is None:
            raise RuntimeError(f"Could not attach to launched window {info.hwnd:#010x}")
        session._title_hint = title_hint or info.title
        launched = LaunchedWindowSession(
            process=process,
            session=session,
            executable=executable,
        )
        if focus and not launched.focus():
            logger.warning("Launched window did not become foreground: %s", info.short_str())
        return launched
    except Exception:
        if process.poll() is None:
            process.terminate()
        raise


def focus_window(hwnd: int, *, timeout_s: float = 1.0) -> bool:
    """Best-effort foreground focus for a Win32 top-level window."""
    if sys.platform != "win32":
        return True
    if not win32gui.IsWindow(hwnd):
        return False

    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        else:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.BringWindowToTop(hwnd)
        _set_foreground_window(hwnd)
    except Exception as exc:
        logger.warning("Failed to focus window %#010x: %s", hwnd, exc)

    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if win32gui.GetForegroundWindow() == hwnd:
            return True
        time.sleep(0.05)
    return win32gui.GetForegroundWindow() == hwnd


def close_window(hwnd: int, *, timeout_s: float = 5.0) -> bool:
    """Post WM_CLOSE to *hwnd* and wait until the window is gone."""
    if sys.platform != "win32":
        return True
    if not win32gui.IsWindow(hwnd):
        return True

    try:
        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    except Exception as exc:
        logger.warning("Failed to post WM_CLOSE to %#010x: %s", hwnd, exc)
        return False

    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if not win32gui.IsWindow(hwnd):
            return True
        time.sleep(0.05)
    return not win32gui.IsWindow(hwnd)


def close_launched_window(
    launched: LaunchedWindowSession,
    *,
    timeout_s: float = 5.0,
    terminate_after: bool = True,
) -> LaunchCloseResult:
    """Close a launched game window, optionally terminating the process fallback."""
    window_closed = launched.session.close(timeout_s=timeout_s)

    process_exited = launched.process.poll() is not None
    terminated = False

    if not process_exited and window_closed:
        try:
            launched.process.wait(timeout=1.0)
        except TimeoutExpired:
            pass
        process_exited = launched.process.poll() is not None

    if not process_exited and terminate_after:
        terminated = True
        launched.process.terminate()
        try:
            launched.process.wait(timeout=max(0.1, timeout_s))
        except TimeoutExpired:
            launched.process.kill()
            launched.process.wait(timeout=max(0.1, timeout_s))
        process_exited = launched.process.poll() is not None

    return LaunchCloseResult(
        window_closed=window_closed,
        process_exited=process_exited,
        terminated=terminated,
    )


def attach(hwnd: int) -> Optional[WindowSession]:
    """Create a WindowSession for the given HWND, with health warnings."""
    if sys.platform != "win32":
        logger.warning("Non-Windows platform — window attachment has limited functionality")
        return None

    info = _build_window_info(hwnd)
    if info is None:
        logger.error("HWND %#010x is not a valid window", hwnd)
        return None

    session = WindowSession(target=info)
    for w in session.check_health():
        logger.warning(w)

    logger.info("Attached to window: %s", info.short_str())
    return session


# ── internal helpers ──────────────────────────────────────────────────────────


def _choose_best_window(windows: list[WindowInfo]) -> WindowInfo:
    """Choose the most likely main window from matching top-level windows."""
    if not windows:
        raise ValueError("Cannot choose from an empty window list")
    return sorted(
        windows,
        key=lambda w: (
            w.is_minimized,
            -(max(0, w.width) * max(0, w.height)),
            w.title.lower(),
        ),
    )[0]


def _set_foreground_window(hwnd: int) -> None:
    """Call SetForegroundWindow with temporary thread input attachment."""
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    current_tid = kernel32.GetCurrentThreadId()
    attached: list[int] = []

    candidate_hwnds = [hwnd]
    foreground = win32gui.GetForegroundWindow()
    if foreground and foreground != hwnd:
        candidate_hwnds.append(foreground)

    try:
        for candidate in candidate_hwnds:
            if not win32gui.IsWindow(candidate):
                continue
            thread_id, _ = win32process.GetWindowThreadProcessId(candidate)
            if thread_id and thread_id != current_tid and thread_id not in attached:
                if user32.AttachThreadInput(current_tid, thread_id, True):
                    attached.append(thread_id)

        win32gui.SetForegroundWindow(hwnd)
        try:
            win32gui.SetFocus(hwnd)
        except Exception:
            pass
    finally:
        for thread_id in reversed(attached):
            user32.AttachThreadInput(current_tid, thread_id, False)


def _build_window_info(hwnd: int) -> Optional[WindowInfo]:
    """Build a WindowInfo from an HWND, or None if the handle is invalid."""
    if not win32gui.IsWindow(hwnd):
        return None
    try:
        title = win32gui.GetWindowText(hwnd)
        class_name = win32gui.GetClassName(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        is_visible = bool(win32gui.IsWindowVisible(hwnd))
        is_minimized = bool(win32gui.IsIconic(hwnd))
    except Exception:
        return None
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name=class_name,
        pid=pid,
        rect=tuple(rect),
        is_visible=is_visible,
        is_minimized=is_minimized,
    )


def _is_likely_exclusive_fullscreen(hwnd: int) -> bool:
    """Heuristic: window covers the entire primary monitor with no border style."""
    try:
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        has_caption = bool(style & win32con.WS_CAPTION)
        rect = win32gui.GetWindowRect(hwnd)
        user32 = ctypes.windll.user32
        scr_w = user32.GetSystemMetrics(0)
        scr_h = user32.GetSystemMetrics(1)
        covers_screen = (
            rect[0] <= 0
            and rect[1] <= 0
            and rect[2] >= scr_w
            and rect[3] >= scr_h
        )
        return covers_screen and not has_caption
    except Exception:
        return False
