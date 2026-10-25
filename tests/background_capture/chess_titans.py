"""Interactive Chess Titans background-capture smoke test (not a pytest test).

Run from the repository root with the transformers5 environment activated:
    python tests/background_capture/chess_titans.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control.session import ControlError, GameSession
from src.window_session import attach, find_window_by_title, launch_and_attach


def default_executable() -> Path:
    for folder in (os.environ.get("ProgramFiles", "C:/Program Files"),
                   os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")):
        candidate = Path(folder) / "Microsoft Games/Chess/chess.exe"
        if candidate.is_file():
            return candidate
    return Path("C:/Program Files/Microsoft Games/Chess/chess.exe")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=default_executable(),
                        help="Chess Titans executable (auto-detected in Microsoft Games)")
    parser.add_argument("--count", type=int, default=10, help="Capture attempts (default: 10)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between attempts (default: 1)")
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: outputs/background_capture/chess_<timestamp>)")
    return parser


def open_game(executable: Path):
    existing = [w for w in find_window_by_title("Chess Titans") if w.title == "Chess Titans"]
    if len(existing) > 1:
        raise RuntimeError("Multiple Chess Titans windows found; close the extra instances first")
    if existing:
        session = attach(existing[0].hwnd)
        if session is None:
            raise RuntimeError("Could not attach to the existing Chess Titans window")
        print("Using the already-open Chess Titans window.", flush=True)
        return session
    launched = launch_and_attach(executable, title_hint="Chess Titans", timeout_s=30)
    print(f"Opened Chess Titans: {executable}", flush=True)
    return launched.session


def background_problem(session) -> str | None:
    if not session.is_valid():
        return "Chess Titans was closed."
    if session.is_minimized():
        return "Chess Titans is minimized. Restore it, then cover it with another window."
    if session.is_foreground():
        return "Chess Titans is still in the foreground. Cover it with another window."
    return None


def capture_sequence(game, *, count: int, interval: float) -> list[dict]:
    """Only save observations while the target remains in the background."""
    results = []
    for index in range(count):
        record = {"attempt": index + 1, "time": time.time()}
        problem = background_problem(game.session)
        if problem:
            record.update(status="skipped", reason=problem)
            print(f"[{index + 1}/{count}] SKIPPED: {problem}", flush=True)
        else:
            try:
                observation = game.observe()
                # Background mode allows either focus state; this smoke test
                # specifically requires the game to stay unfocused.
                problem = background_problem(game.session)
                if observation["foreground"] or problem:
                    # These files belong only to this observation; discard any
                    # image that cannot demonstrate background capture.
                    for key in ("screenshot_raw", "screenshot_cursor"):
                        if observation.get(key):
                            Path(observation[key]).unlink(missing_ok=True)
                    record.update(status="skipped", reason=problem or "Game gained focus during capture")
                    print(f"[{index + 1}/{count}] SKIPPED: {record['reason']}", flush=True)
                else:
                    record.update(status="captured", observation=observation)
                    print(f"[{index + 1}/{count}] {observation['screenshot_raw']}", flush=True)
            except ControlError as exc:
                record.update(status="failed", error={"code": exc.code, "message": str(exc)})
                print(f"[{index + 1}/{count}] FAILED: {exc.code}: {exc}", flush=True)
        results.append(record)
        if index + 1 < count:
            time.sleep(interval)
    return results


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("This test requires Windows")
    if args.count < 1 or not 0 <= args.interval <= 3600:
        parser.error("--count must be positive; --interval must be between 0 and 3600 seconds")
    out = (args.out or ROOT / "outputs/background_capture" /
           f"chess_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    try:
        session = open_game(args.exe)
        print("\nPrepare the chess board and dismiss any game dialogs.")
        print("Cover the game partly or fully with another window. Do not minimize it.")
        print("Return to this terminal (it can cover the game) and press Enter to start.")
        print(f"Will take {args.count} screenshots, {args.interval:g}s apart.")
        print(f"Output: {out}", flush=True)
        while True:
            input("\nPress Enter when Chess Titans is in the background, or Ctrl+C to cancel: ")
            problem = background_problem(session)
            if problem is None:
                break
            print(problem, flush=True)
            if not session.is_valid():
                return 1

        game = GameSession(method="background", cursor_mode="raw", focus=False, dry_run=True, out=out)
        game.attach(hwnd=session.target.hwnd)
        try:
            results = capture_sequence(game, count=args.count, interval=args.interval)
        finally:
            game.stop()
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        captured = sum(r["status"] == "captured" for r in results)
        print(f"\nSaved {captured}/{args.count} screenshots. Results: {out / 'results.json'}")
        print("Open the PNGs and verify that they show the chess board, without the covering window.")
        print("Chess Titans is left open.")
        return 0 if captured == args.count else 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled. Chess Titans is left open.")
        return 130
    except (OSError, RuntimeError, ControlError) as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
