"""Run the real chess replay under an owned foreground cover (opt-in desktop test).

    python tests/background_capture/chess_moves_covered.py --new-game

All chess_moves options are accepted. Defaults select windowmessage and allow
unfocused input. Keep the desktop unlocked and avoid input during verification.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
import time

import chess_moves
from chess_titans import default_executable, open_game
import win32api
import win32con
import win32gui


@contextmanager
def foreground_cover(target):
    ready = threading.Event()
    done = threading.Event()
    state = {}

    def run():
        hwnd = None
        atom = None
        instance = win32api.GetModuleHandle(None)
        try:
            wc = win32gui.WNDCLASS()
            wc.hInstance = instance
            wc.lpszClassName = f"ChessInputVerification_{time.time_ns()}"
            wc.lpfnWndProc = win32gui.DefWindowProc
            wc.hbrBackground = win32con.COLOR_WINDOW + 1
            atom = win32gui.RegisterClass(wc)
            left, top, right, bottom = win32gui.GetWindowRect(target)
            hwnd = win32gui.CreateWindowEx(
                win32con.WS_EX_TOPMOST, atom, "Chess background input verification",
                win32con.WS_POPUP, left, top, right-left, bottom-top,
                0, 0, instance, None)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetForegroundWindow(hwnd)
            state["hwnd"] = hwnd
            ready.set()
            while not done.wait(.005):
                win32gui.PumpWaitingMessages()
        except Exception as exc:
            state["error"] = exc
        finally:
            ready.set()
            if hwnd and win32gui.IsWindow(hwnd):
                win32gui.DestroyWindow(hwnd)
            if atom:
                win32gui.UnregisterClass(atom, instance)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        if not ready.wait(5):
            raise RuntimeError("Timed out creating chess cover")
        if "error" in state:
            raise RuntimeError(f"Cannot create chess cover: {state['error']}")
        if win32gui.GetForegroundWindow() != state["hwnd"]:
            raise RuntimeError("Chess cover did not become foreground")
        yield state["hwnd"]
    finally:
        done.set()
        thread.join(5)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        return chess_moves.main(args)
    # Use the same executable override as the replay when opening before cover.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exe", type=Path, default=default_executable())
    known, _ = parser.parse_known_args(args)
    session = open_game(known.exe)
    original_class = chess_moves.GameSession
    observations = []
    out = None
    with foreground_cover(session.target.hwnd) as cover:
        cursor = win32gui.GetCursorPos()

        class VerifiedSession(original_class):
            def observe(self):
                nonlocal out
                result = super().observe()
                out = self.out
                check = {"hwnd": result["hwnd"], "foreground": result["foreground"],
                         "desktop_foreground": win32gui.GetForegroundWindow(),
                         "cursor": win32gui.GetCursorPos()}
                observations.append(check)
                if (check["foreground"] or check["desktop_foreground"] != cover
                        or check["cursor"] != cursor):
                    raise RuntimeError(f"Background input changed desktop state: {check}")
                return result

        chess_moves.GameSession = VerifiedSession
        try:
            return chess_moves.main(["--input-method", "windowmessage", "--allow-unfocused", *args])
        finally:
            chess_moves.GameSession = original_class
            if out is not None:
                (out / "background_verification.json").write_text(
                    json.dumps({"cover": cover, "cursor": cursor, "observations": observations},
                               indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
