"""Replay three CLI-verified White moves against Chess Titans (opt-in desktop test)."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control.session import ControlError, GameSession
from src.actions.input_settings import INPUT_METHODS
from src.window_session import enumerate_windows

FIXTURES = Path(__file__).with_name("fixtures") / "chess_moves"
MOVES = (
    ("a2a3", (212, 346), (218, 308), "after_a3.png"),
    ("b2b3", (266, 346), (270, 308), "after_b3.png"),
    ("c2c3", (320, 346), (322, 308), "after_c3.png"),
)
PAWN_STRIP = (190, 270, 345, 360)
# Interiors of a2/b2/c2 and a3/b3/c3: avoid hover/last-move square borders.
PAWN_BOXES = ([(x - 10, 336, x + 10, 354) for x in (212, 266, 320)]
              + [(x - 10, 280, x + 10, 302) for x in (218, 270, 322)])
ENEMY_BOX = (220, 50, 580, 250)
THINKING_BOX = (700, 0, 765, 105)


def difference(left, right):
    """Mean absolute RGB error, in 0..255 channel values."""
    return sum(ImageStat.Stat(ImageChops.difference(left.convert("RGB"),
                                                   right.convert("RGB"))).mean) / 3


def frame(observation):
    with Image.open(observation["screenshot_raw"]) as image:
        if image.size != (800, 496):
            raise RuntimeError(f"Expected 800x496 client area, got {image.size}; restore the recorded layout")
        return image.convert("RGB")


def pawn_errors(image, reference):
    return [difference(image.crop(box), reference.crop((box[0] - 190, box[1] - 270,
                                                      box[2] - 190, box[3] - 270)))
            for box in PAWN_BOXES]


def starting_error(image):
    with Image.open(FIXTURES / "initial.png") as reference:
        return max(difference(image.crop((170, 50, 625, 430)), reference.crop((170, 50, 625, 430))),
                   *pawn_errors(image, reference.crop(PAWN_STRIP)))


def click(game, position, settle_ms):
    return game.step(actions=[{"name": "click_at", "params": dict(zip(("x", "y"), position))}],
                     duration_ms=0, settle_ms=settle_ms)


def reset_game(game):
    """F2, then the recorded 'end and start new game' confirmation, if present."""
    pid = game.session.target.pid
    game.step(actions=["new_game"], duration_ms=0, settle_ms=500)
    dialogs = [w for w in enumerate_windows() if w.pid == pid and w.class_name == "#32770"]
    if not dialogs:
        return  # F2 on a pristine board starts immediately.
    if len(dialogs) != 1:
        raise RuntimeError("Unexpected Chess Titans dialogs; dismiss them before replay")
    dialog = GameSession(method="background", cursor_mode="raw", out=game.out,
                         input_method=game.input_method, require_focus=game.require_focus)
    try:
        dialog.attach(hwnd=dialogs[0].hwnd)
        observation = dialog.observe()
        if (observation["width"], observation["height"]) != (350, 169):
            raise RuntimeError("Unexpected new-game dialog layout; dismiss it before replay")
        try:
            click(dialog, (175, 85), 500)
        except ControlError as exc:
            # The confirmation closes itself before step's post-action capture.
            if not (exc.code == "target_unavailable" and exc.details.get("actions_may_have_executed")
                    and not dialog.session.is_valid()):
                raise
    finally:
        dialog.stop()


def replay(game, *, opponent_wait=3.0, timeout=60.0, report=None):
    """Same click sequence and visual assertions can serve a future input backend."""
    if not 3 <= opponent_wait <= 5 or timeout <= opponent_wait:
        raise ValueError("opponent_wait must be 3..5 seconds and timeout must exceed it")
    report = report if report is not None else {}
    report.update(status="running", moves=[])
    observation = game.observe()
    report["initial"] = observation
    initial = frame(observation)
    error = starting_error(initial)
    if error > 4:
        raise RuntimeError(f"Starting board/layout differs from the recorded fresh White board ({error:.2f}); "
                           "start a new game, use the recorded theme/camera and 800x496 client size")
    idle = initial.crop(THINKING_BOX)
    previous = initial
    for name, source, destination, filename in MOVES:
        record = {"move": name, "source": source, "destination": destination}
        report["moves"].append(record)
        print(f"Playing {name}", flush=True)
        record["select"] = click(game, source, 400)
        started = time.monotonic()
        record["place"] = click(game, destination, round(opponent_wait * 1000))
        observation = record["place"]["observation"]
        record["polls"] = []
        last = None
        stable = 0
        with Image.open(FIXTURES / filename) as reference:
            while True:
                image = frame(observation)
                errors = pawn_errors(image, reference)
                thinking = difference(image.crop(THINKING_BOX), idle)
                enemy_change = difference(image.crop(ENEMY_BOX), previous.crop(ENEMY_BOX))
                change = difference(image.crop(ENEMY_BOX), last.crop(ENEMY_BOX)) if last else 255
                stable = stable + 1 if change < 0.5 else 0
                record["checks"] = dict(pawn_errors=errors, thinking_error=thinking,
                                        enemy_change=enemy_change, stable_samples=stable)
                if max(errors) < 4 and thinking < 4 and enemy_change > 1 and stable >= 2:
                    break
                if time.monotonic() - started >= timeout:
                    raise RuntimeError(f"{name}: move or settled opponent reply not verified: {record['checks']}")
                last = image
                time.sleep(0.5)
                observation = game.observe()
                record["polls"].append(observation)
        record.update(status="passed", elapsed_seconds=time.monotonic() - started,
                      final_observation=observation)
        previous = image
    report["status"] = "passed"
    return report


def main(argv=None):
    # Local import keeps the standalone launch helper shared with the capture test.
    from chess_titans import default_executable, open_game

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=default_executable())
    parser.add_argument("--new-game", action="store_true",
                        help="Reset using F2; abandoning an existing game can count as a loss")
    parser.add_argument("--opponent-wait", type=float, default=3.0, help="Minimum wait, 3..5 seconds")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-turn timeout in seconds")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--input-method", choices=INPUT_METHODS, default="sendinput")
    parser.add_argument("--allow-unfocused", action="store_true",
                        help="Skip automatic activation and all focus guards with either input sender")
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("Requires an unlocked Windows desktop")
    if not 3 <= args.opponent_wait <= 5 or not args.opponent_wait < args.timeout <= 300:
        parser.error("--opponent-wait must be 3..5 and --timeout must exceed it (maximum 300)")
    out = (args.out or ROOT / "outputs/background_capture" /
           f"chess_moves_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"input_method": args.input_method, "require_focus": not args.allow_unfocused}
    game = None
    try:
        session = open_game(args.exe)
        game = GameSession(action_config=FIXTURES / "actions.yaml", method="background",
                           cursor_mode="raw", out=out, input_method=args.input_method,
                           require_focus=not args.allow_unfocused)
        game.attach(hwnd=session.target.hwnd)
        if args.new_game:
            reset_game(game)
            # New-game camera rotation takes several seconds; never click through it.
            deadline = time.monotonic() + 15
            while starting_error(frame(game.observe())) > 4:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Fresh White board did not appear within 15 seconds")
                time.sleep(0.5)
        replay(game, opponent_wait=args.opponent_wait, timeout=args.timeout, report=report)
        print("Verified all three White moves and opponent replies. Chess Titans is left open.")
        return 0
    except KeyboardInterrupt:
        report.update(status="interrupted")
        return 130
    except (RuntimeError, OSError, ControlError) as exc:
        report.update(status="failed", error=str(exc))
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if game:
            game.stop()
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Results and screenshots: {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
