"""Reusable synchronous game controls, without importing inference backends."""

from __future__ import annotations

import dataclasses
import math
import time
import uuid
from pathlib import Path

from src import window_session
from src.actions.executor import ActionExecutor
from src.actions.schema import ParsedAction, load_action_config
from src.actions.state import ActionState, HeldInput
from src.capture import dxcam_capture, fallback_capture, windows_graphics_capture
from src.capture.cursor_overlay import save_frame_screenshots

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACTION_CONFIG = ROOT / "configs/actions/control-actions.yaml"
CAPTURE_METHODS = ("auto", "dxcam", "win32", "mss")


class ControlError(Exception):
    def __init__(self, code: str, message: str, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def milliseconds(value, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ControlError("invalid_request", f"{name} must be an integer from 0 to {maximum}")
    return value


def window_info(window) -> dict:
    return {**dataclasses.asdict(window), "hwnd": hex(window.hwnd)}


def list_windows() -> dict:
    return {"windows": [window_info(w) for w in window_session.enumerate_windows()]}


def capture_frame(session, method):
    backends = {
        "dxcam": dxcam_capture.capture_frame,
        "win32": windows_graphics_capture.capture_frame,
        "mss": fallback_capture.capture_frame,
    }
    if method != "auto":
        return backends[method](session)
    # A blank image from an earlier backend must not prevent fallback.
    last = None
    for name, capture in backends.items():
        if name == "dxcam" and not dxcam_capture.is_available():
            continue
        frame = capture(session)
        if frame is not None:
            last = frame
            if not frame.is_blank():
                return frame
    return last


class GameSession:
    def __init__(self, *, action_config=DEFAULT_ACTION_CONFIG, out=None,
                 method="auto", cursor_mode="both", dry_run=False, focus=True):
        if method not in CAPTURE_METHODS:
            raise ControlError("invalid_request", f"Unknown capture method: {method}")
        if cursor_mode not in ("raw", "cursor", "both"):
            raise ControlError("invalid_request", "cursor_mode must be raw, cursor, or both")
        self.config = load_action_config(action_config)
        for action in self.config.actions.values():
            if action.execution.type not in {
                "hold_key", "hold_keys", "press_key", "press_keys", "mouse_button", "mouse_move", "click_at"
            }:
                raise ControlError("invalid_request", f"Unsupported execution type for {action.canonical_name}")
            if any(p.type not in {"int", "float"} for p in action.parameters):
                raise ControlError("invalid_request", f"Unsupported parameter type for {action.canonical_name}")
        self.out = Path(out or ROOT / "outputs/control" / uuid.uuid4().hex).resolve()
        self.method = method
        self.cursor_mode = cursor_mode
        self.dry_run = dry_run
        self.focus = focus
        self.session = None
        self.executor = None
        self.state = ActionState(self.config)
        self._identity = None

    def actions(self) -> dict:
        return {
            "actions": [dataclasses.asdict(a) for a in self.config.actions.values()],
            "coordinates": "Client-area pixels, origin at top left; aim is relative to client center.",
            "max_duration_ms": 10000,
        }

    def attach(self, *, title=None, hwnd=None) -> dict:
        if (title is None) == (hwnd is None):
            raise ControlError("invalid_request", "Provide exactly one of title or hwnd")
        if title is not None:
            if not isinstance(title, str) or not title.strip():
                raise ControlError("invalid_request", "title must be a nonempty string")
            matches = window_session.find_window_by_title(title)
            if len(matches) != 1:
                raise ControlError("target_not_unique", "Expected exactly one matching window",
                                   matches=[window_info(w) for w in matches])
            hwnd = matches[0].hwnd
        else:
            try:
                if type(hwnd) not in (str, int):
                    raise ValueError()
                hwnd = int(hwnd, 0) if isinstance(hwnd, str) else hwnd
                if hwnd <= 0:
                    raise ValueError()
            except ValueError:
                raise ControlError("invalid_request", "hwnd must be a positive integer or hex string")
        session = window_session.attach(hwnd)
        if session is None:
            raise ControlError("target_unavailable", "Cannot attach to target window")
        self.stop()
        self.session = session
        self._identity = (session.target.hwnd, session.target.pid)
        self.executor = ActionExecutor(session, dry_run=self.dry_run)
        return {"target": window_info(session.target), "dry_run": self.dry_run}

    def _prepare(self):
        if self.session is None:
            raise ControlError("not_attached", "Attach to a game first")
        if not self.session.is_valid():
            raise ControlError("target_unavailable", "Target window no longer exists")
        target = self.session.refresh_info()
        if (target.hwnd, target.pid) != self._identity:
            raise ControlError("target_unavailable", "Target window identity changed; attach again")
        if self.session.is_minimized():
            raise ControlError("target_unavailable", "Restore the minimized target window first")
        if self.focus and not self.dry_run and not self.session.focus():
            raise ControlError("focus_failed", "Could not focus target window")

    def _observation(self) -> dict:
        frame = capture_frame(self.session, self.method)
        if frame is None:
            raise ControlError("capture_failed", "No frame returned by capture backend")
        path = self.out / f"frame_{time.time_ns()}_{uuid.uuid4().hex[:8]}.png"
        saved = save_frame_screenshots(frame, path, cursor_mode=self.cursor_mode)
        return {
            **frame.to_metadata(),
            "screenshot": str(saved.selected_path),
            "screenshot_raw": str(saved.raw_path) if saved.raw_path else None,
            "screenshot_cursor": str(saved.cursor_path) if saved.cursor_path else None,
            "blank": frame.is_blank(),
            "target": window_info(self.session.target),
            "foreground": self.session.is_foreground(),
            "coordinate_space": "client_pixels",
        }

    def observe(self) -> dict:
        self._prepare()
        return self._observation()

    def _parse_actions(self, actions) -> list[ParsedAction]:
        if not isinstance(actions, list) or len(actions) > 32:
            raise ControlError("invalid_request", "actions must be an array of at most 32 actions")
        parsed = []
        for item in actions:
            if isinstance(item, str):
                item = {"name": item}
            if not isinstance(item, dict) or set(item) - {"name", "params"}:
                raise ControlError("invalid_request", "Each action needs name and optional params")
            name = item.get("name")
            definition = self.config.get_action(name) if isinstance(name, str) else None
            if definition is None:
                raise ControlError("invalid_request", f"Unknown canonical action: {name!r}")
            params = item.get("params", {})
            expected = {p.name: p for p in definition.parameters}
            if not isinstance(params, dict) or set(params) - set(expected):
                raise ControlError("invalid_request", f"Unexpected parameters for {name}")
            for key, spec in expected.items():
                if spec.required and key not in params:
                    raise ControlError("invalid_request", f"Missing {name}.{key}")
                if key in params:
                    value = params[key]
                    valid = type(value) is int if spec.type == "int" else (
                        type(value) is int or (type(value) is float and math.isfinite(value)))
                    if not valid:
                        raise ControlError("invalid_request", f"Invalid {name}.{key}: expected {spec.type}")
            if definition.execution.type in ("mouse_move", "click_at") and not {"x", "y"} <= set(params):
                raise ControlError("invalid_request", f"{name} requires x and y")
            if definition.execution.type in ("mouse_move", "click_at") and any(
                type(params[axis]) is not int for axis in ("x", "y")
            ):
                raise ControlError("invalid_request", f"{name} coordinates must be integer client pixels")
            parsed.append(ParsedAction(name, dict(params)))
        return parsed

    def step(self, *, actions, duration_ms=250, settle_ms=100) -> dict:
        parsed = self._parse_actions(actions)
        milliseconds(duration_ms, "duration_ms", 10000)
        milliseconds(settle_ms, "settle_ms", 5000)
        self._prepare()
        left, top, right, bottom = self.session.get_client_rect()
        for action in parsed:
            if self.config.get_action(action.canonical_name).execution.type in ("mouse_move", "click_at"):
                x, y = action.params["x"], action.params["y"]
                if not (0 <= x < right - left and 0 <= y < bottom - top):
                    raise ControlError("invalid_request", "Coordinates must lie inside the client area")
        plan = self.state.compute_tick(parsed)
        # Track taps too: interruption between tap-down and tap-up still needs cleanup.
        cleanup = plan.to_press + plan.to_tap
        if plan.click_ats:
            cleanup.append(HeldInput("click_at", "mouse_button", button="left"))
        started = time.perf_counter()
        execution = {"dry_run": True, "skipped": True}
        try:
            if not self.dry_run:
                execution = self.executor.execute_tick(plan)
                if execution["skipped"]:
                    raise ControlError("focus_failed", "Input skipped because the target lost focus")
                deadline = time.perf_counter() + duration_ms / 1000
                while time.perf_counter() < deadline:
                    if not self.session.is_valid() or not self.session.is_foreground():
                        raise ControlError("focus_lost", "Target lost focus during the step",
                                           actions_may_have_executed=True)
                    time.sleep(min(0.02, max(0, deadline - time.perf_counter())))
        finally:
            self.state.release_all()
            if not self.dry_run:
                self.executor.release_all(cleanup)
        if not self.dry_run:
            time.sleep(settle_ms / 1000)
        result = {
            "actions": [dataclasses.asdict(a) for a in parsed],
            "plan": dataclasses.asdict(plan),
            "duration_ms": duration_ms,
            "settle_ms": settle_ms,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "execution": execution,
            "inputs_released": True,
        }
        try:
            result["observation"] = self._observation()
        except Exception as exc:
            raise ControlError("capture_failed", str(exc), step=result,
                               actions_may_have_executed=not self.dry_run) from exc
        return result

    def stop(self) -> dict:
        if self.executor is not None:
            self.executor.release_all(self.state.release_all())
        self.session = None
        self.executor = None
        self._identity = None
        return {"stopped": True, "inputs_released": True}
