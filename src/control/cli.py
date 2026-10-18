"""JSON CLI and optional persistent JSON-lines command loop."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
from pathlib import Path

from .session import CAPTURE_METHODS, DEFAULT_ACTION_CONFIG, ControlError, GameSession, list_windows


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ControlError("invalid_request", message)


def build_parser():
    parser = Parser(description="Observe and control games without running a model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list-windows", help="List visible windows as JSON")
    actions = sub.add_parser("actions", help="Describe canonical actions and parameters")
    actions.add_argument("--action-config", default=str(DEFAULT_ACTION_CONFIG))
    for command in ("observe", "step", "serve"):
        p = sub.add_parser(command, help={
            "observe": "Save a screenshot and return its absolute path",
            "step": "Execute timed actions, release inputs, and take a screenshot",
            "serve": "Read one JSON request per stdin line; return one JSON response",
        }[command])
        target = p.add_mutually_exclusive_group(required=command != "serve")
        target.add_argument("--title", help="Unique window title substring")
        target.add_argument("--hwnd", help="Exact window handle (hex or decimal)")
        p.add_argument("--action-config", default=str(DEFAULT_ACTION_CONFIG))
        p.add_argument("--out", help="Artifact directory (default: outputs/control/<session-id>)")
        p.add_argument("--method", choices=CAPTURE_METHODS, default="auto")
        p.add_argument("--cursor-mode", choices=("raw", "cursor", "both"), default="both")
        p.add_argument("--no-focus", action="store_true", help="Require the game to already have focus")
        p.add_argument("--dry-run", action="store_true", help="Plan without changing focus or sending input")
        if command == "step":
            a = p.add_mutually_exclusive_group(required=True)
            a.add_argument("--action", action="append", help="Canonical action; repeat to combine actions")
            a.add_argument("--actions", help='JSON array, e.g. ["move_forward"]')
            a.add_argument("--actions-file", type=Path, help="UTF-8 file containing the JSON action array")
            p.add_argument("--duration-ms", type=int, default=250)
            p.add_argument("--settle-ms", type=int, default=100)
    return parser


def dispatch(game, request):
    if not isinstance(request, dict) or not isinstance(request.get("op"), str):
        raise ControlError("invalid_request", "Request must be an object with an op string")
    allowed = {
        "list-windows": set(), "actions": set(), "attach": {"title", "hwnd"},
        "observe": set(), "step": {"actions", "duration_ms", "settle_ms"}, "stop": set(),
    }
    op = request["op"]
    if op not in allowed or set(request) - allowed[op] - {"id", "op"}:
        raise ControlError("invalid_request", "Unknown operation or unexpected request fields")
    kwargs = {k: v for k, v in request.items() if k not in ("id", "op")}
    if op == "step" and "actions" not in kwargs:
        raise ControlError("invalid_request", "step requires actions (use [] to wait)")
    if op == "list-windows":
        return list_windows()
    return getattr(game, op)(**kwargs)


def error_response(exc, request=None):
    if isinstance(exc, ControlError):
        code, details = exc.code, exc.details
    elif isinstance(exc, (ValueError, TypeError)):
        code, details = "invalid_request", {}
    elif isinstance(exc, OSError):
        code, details = "io_error", {}
    else:
        code, details = "internal_error", {}
        logging.exception("Game control failed")
    response = {"ok": False, "error": {"code": code, "message": str(exc), **details}}
    if isinstance(request, dict) and "id" in request:
        response["id"] = request["id"]
    return response


def emit(response, output):
    print(json.dumps(response, ensure_ascii=True, allow_nan=False), file=output, flush=True)


def read_json(text):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("JSON numbers must be finite")
        return result
    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float)


def serve(game, source, output):
    try:
        for line in source:
            if not line.strip():
                continue
            request = None
            try:
                request = read_json(line)
                with contextlib.redirect_stdout(sys.stderr):
                    result = dispatch(game, request)
                response = {"ok": True, "result": result}
                if "id" in request:
                    response["id"] = request["id"]
            except KeyboardInterrupt:
                emit({"ok": False, "error": {"code": "interrupted", "message": "Interrupted"}}, output)
                return 130
            except Exception as exc:
                response = error_response(exc, request)
            emit(response, output)
            if response["ok"] and request["op"] == "stop":
                break
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            game.stop()
    return 0


def main(argv=None):
    output = sys.stdout
    game = None
    try:
        args = build_parser().parse_args(argv)
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "list-windows":
                result = list_windows()
            else:
                game = GameSession(
                    action_config=args.action_config, out=getattr(args, "out", None),
                    method=getattr(args, "method", "auto"),
                    cursor_mode=getattr(args, "cursor_mode", "both"),
                    dry_run=getattr(args, "dry_run", False),
                    focus=not getattr(args, "no_focus", False),
                )
                if args.command == "actions":
                    result = game.actions()
                else:
                    # Validate JSON before attaching or focusing any window.
                    if args.command == "step":
                        if args.action is not None:
                            actions = args.action
                        else:
                            text = (args.actions_file.read_text(encoding="utf-8-sig")
                                    if args.actions_file else args.actions)
                            actions = read_json(text)
                        game._parse_actions(actions)
                    if args.title is not None or args.hwnd is not None:
                        game.attach(title=args.title, hwnd=args.hwnd)
                    if args.command == "observe":
                        result = game.observe()
                    elif args.command == "step":
                        result = game.step(actions=actions, duration_ms=args.duration_ms,
                                           settle_ms=args.settle_ms)
        if args.command == "serve":
            return serve(game, sys.stdin, output)
        emit({"ok": True, "result": result}, output)
        return 0
    except KeyboardInterrupt:
        emit({"ok": False, "error": {"code": "interrupted", "message": "Interrupted"}}, output)
        return 130
    except Exception as exc:
        emit(error_response(exc), output)
        return 2 if isinstance(exc, (ValueError, TypeError)) or (
            isinstance(exc, ControlError) and exc.code == "invalid_request") else 1
    finally:
        if game is not None:
            with contextlib.redirect_stdout(sys.stderr):
                game.stop()
