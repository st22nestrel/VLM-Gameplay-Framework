"""Opt-in cross-process delivery test on a Windows desktop (no game required)."""

import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

import pytest

from src.actions.executor import ActionExecutor
from src.actions.state import HeldInput, TickPlan
from src.actions.window_message import pack_point
from src.window_session import attach


@pytest.fixture
def target(request):
    if sys.platform != "win32" or not request.config.getoption("--live-background-input"):
        pytest.skip("requires --live-background-input and Windows desktop")
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("window_fixture.py"))],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    lines = queue.Queue()

    def read_lines():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read_lines, daemon=True).start()

    def receive():
        try:
            line = lines.get(timeout=10)
        except queue.Empty:
            raise AssertionError("Timed out waiting for the background input fixture")
        if line is None:
            raise AssertionError(f"Input fixture exited: {process.stderr.read()}")
        return json.loads(line)

    try:
        info = receive()

        def events():
            process.stdin.write("events\n")
            process.stdin.flush()
            return receive()["events"]

        yield info, events
    finally:
        if process.poll() is None:
            try:
                process.stdin.write("quit\n")
                process.stdin.flush()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def test_covered_window_receives_click_drag_keys_without_focus_or_cursor_change(target):
    import win32api
    import win32gui

    info, events = target
    session = attach(info["hwnd"])
    assert session is not None
    foreground = win32gui.GetForegroundWindow()
    cursor = win32api.GetCursorPos()
    assert foreground != info["hwnd"]
    left, top, right, bottom = session.get_client_rect()
    cover_rect = win32gui.GetWindowRect(info["cover"])
    assert cover_rect[0] <= left and cover_rect[1] <= top
    assert cover_rect[2] >= right and cover_rect[3] >= bottom

    executor = ActionExecutor(session, input_method="windowmessage", require_focus=False)
    executor.click_at(70, 90)
    held = [HeldInput("shift", "key", key="shift"), HeldInput("forward", "key", key="w"),
            HeldInput("drag", "mouse_button", button="left")]
    result = executor.execute_tick(TickPlan([], held, [{"x": 100, "y": 110}], []))
    executor.release_all(held)
    received = events()
    assert not result["skipped"] and not result["focus_ok"]
    assert [(event[0], event[1], event[2]) for event in received] == [
        (info["child"], 0x200, 0), (info["child"], 0x201, 1), (info["child"], 0x202, 0),
        (info["child"], 0x100, 0x10), (info["child"], 0x100, 0x57),
        (info["child"], 0x201, 5), (info["child"], 0x200, 5), (info["child"], 0x202, 4),
        (info["child"], 0x101, 0x57), (info["child"], 0x101, 0x10),
    ]
    assert received[0][3] == pack_point(50, 60)
    assert received[6][3] == pack_point(80, 80)
    assert win32gui.GetForegroundWindow() == foreground
    assert win32api.GetCursorPos() == cursor


def test_legacy_postmessage_buttons_revert_to_center_and_omit_drag_state(target):
    """Compare the old experiment on the same receiver, without its desktop click."""
    import win32api
    import win32gui

    info, events = target
    session = attach(info["hwnd"])
    foreground, cursor = win32gui.GetForegroundWindow(), win32api.GetCursorPos()
    executor = ActionExecutor(session, background=True, require_focus=False)
    held = [HeldInput("drag", "mouse_button", button="left")]
    executor.execute_tick(TickPlan([], [], [{"x": 70, "y": 90}], []))
    executor.execute_tick(TickPlan([], held, [{"x": 100, "y": 110}], []))
    executor.release_all(held)
    received = events()
    left, top, right, bottom = session.get_client_rect()
    center = pack_point((right - left) // 2, (bottom - top) // 2)
    assert received == [
        [info["hwnd"], 0x200, 0, pack_point(70, 90)],
        [info["hwnd"], 0x201, 1, center],
        [info["hwnd"], 0x200, 0, pack_point(100, 110)],
        [info["hwnd"], 0x202, 0, center],
    ]
    assert win32gui.GetForegroundWindow() == foreground
    assert win32api.GetCursorPos() == cursor


def test_control_cli_posts_click_and_captures_covered_target(target, tmp_path):
    import win32api
    import win32gui

    info, events = target
    foreground, cursor = win32gui.GetForegroundWindow(), win32api.GetCursorPos()
    command = [sys.executable, "-m", "src.control", "step", "--hwnd", str(info["hwnd"]),
               "--input-method", "windowmessage", "--allow-unfocused", "--method", "background",
               "--cursor-mode", "raw", "--out", str(tmp_path), "--duration-ms", "0", "--settle-ms", "0",
               "--actions", json.dumps([{"name": "click_at", "params": {"x": 70, "y": 90}}])]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    response = json.loads(completed.stdout)
    assert response["ok"]
    result = response["result"]
    assert result["input_method"] == "windowmessage"
    assert result["inputs_released"] and not result["require_focus"]
    assert int(result["observation"]["hwnd"], 0) == info["hwnd"]
    assert Path(result["observation"]["screenshot"]).is_file()
    assert [(e[0], e[1], e[2], e[3]) for e in events()] == [
        (info["child"], 0x200, 0, pack_point(50, 60)),
        (info["child"], 0x201, 1, pack_point(50, 60)),
        (info["child"], 0x202, 0, pack_point(50, 60)),
    ]
    assert win32gui.GetForegroundWindow() == foreground
    assert win32api.GetCursorPos() == cursor
