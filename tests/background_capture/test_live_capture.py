"""Pixel-based regression tests with real covered GDI and OpenGL windows."""

import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import pytest
from PIL import Image, ImageChops

from src.capture import background_capture, fallback_capture
from src.window_session import attach


@pytest.fixture(params=["gdi", "opengl"])
def target(request):
    if request.param == "opengl" and importlib.util.find_spec("pygame") is None:
        pytest.skip("Install tests/background_capture/requirements.txt for OpenGL coverage")
    command = [sys.executable, str(Path(__file__).with_name("window_fixture.py"))]
    if request.param == "opengl":
        command.append("--opengl")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    replies = queue.Queue()

    def read_output():
        for line in process.stdout:
            replies.put(line)

    threading.Thread(target=read_output, daemon=True).start()

    def reply():
        try:
            return json.loads(replies.get(timeout=10))
        except queue.Empty:
            if process.poll() is not None:
                pytest.fail(process.stderr.read())
            pytest.fail("Test window did not respond in 10s")

    def send(command):
        process.stdin.write(command + "\n")
        process.stdin.flush()
        assert reply() == {"done": command}
        time.sleep(0.15)  # Let the desktop compositor present the cover.

    try:
        info = reply()
        session = attach(info["hwnd"])
        assert session is not None
        yield session, send, info["cover_hwnd"]
    finally:
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


@pytest.mark.parametrize("coverage", ["partial", "full"])
def test_covered_window_pixels_and_freshness(target, coverage, tmp_path):
    import win32gui

    session, send, cover = target
    send(coverage)
    assert win32gui.GetForegroundWindow() == cover
    rect = session.get_rect()

    # Positive control: prove that a desktop screenshot really sees our cover.
    desktop = fallback_capture.capture_frame(session)
    assert desktop.image.getpixel((50, 50)) == (224, 32, 224)

    for color in [(32, 192, 64), (224, 176, 32)]:
        if color == (224, 176, 32):
            send("change")
        frame = background_capture.capture_frame(session)
        assert frame is not None
        frame.save(tmp_path / f"{coverage}_{color[0]}.png")
        assert frame.image.size == (320, 240)
        assert frame.image.getpixel((50, 50)) == color
        assert frame.image.getpixel((250, 150)) == (32, 64, 192)
        expected = Image.new("RGB", (320, 240), (32, 64, 192))
        expected.paste(color, (0, 0, 160, 240))
        assert ImageChops.difference(frame.image, expected).getbbox() is None
        assert frame.hwnd == session.target.hwnd
        assert frame.client_rect == session.get_client_rect()
        assert win32gui.GetForegroundWindow() == cover
        assert session.get_rect() == rect


def test_minimized_window_is_not_restored(target):
    session, send, _ = target
    send("minimize")
    assert session.is_minimized()
    assert background_capture.capture_frame(session) is None
    assert session.is_minimized()


def test_control_cli_observes_covered_window_without_focusing(target, tmp_path):
    import win32gui

    session, send, cover = target
    send("full")
    result = subprocess.run(
        [sys.executable, "-m", "src.control", "observe", "--hwnd", str(session.target.hwnd),
         "--method", "background", "--cursor-mode", "raw", "--out", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    response = json.loads(result.stdout)
    assert response["ok"]
    observation = response["result"]
    assert observation["foreground"] is False
    assert observation["capture_method"] == "printwindow"
    with Image.open(observation["screenshot_raw"]) as image:
        assert image.size == (320, 240)
        assert image.getpixel((50, 50)) == (32, 192, 64)
        assert image.getpixel((250, 150)) == (32, 64, 192)
    assert win32gui.GetForegroundWindow() == cover


@pytest.mark.parametrize("input_method", ["sendinput", "postmessage"])
def test_control_cli_allows_unfocused_wait_with_either_sender(target, tmp_path, input_method):
    import win32api
    import win32gui

    session, send, cover = target
    send("full")
    cursor = win32api.GetCursorPos()
    command = [sys.executable, "-m", "src.control", "step", "--hwnd", str(session.target.hwnd),
               "--method", "background", "--cursor-mode", "raw", "--out", str(tmp_path),
               "--input-method", input_method, "--actions", "[]", "--duration-ms", "25",
               "--settle-ms", "0", "--no-focus"]
    # Requiring focus still rejects a covered target for either input sender.
    blocked = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert blocked.returncode == 1, blocked.stderr + blocked.stdout
    assert json.loads(blocked.stdout)["error"]["code"] == "focus_lost"
    # Explicitly disable the requirement; this empty step sends no desktop input.
    allowed = subprocess.run(command + ["--allow-unfocused"], capture_output=True, text=True, timeout=15)
    assert allowed.returncode == 0, allowed.stderr + allowed.stdout
    result = json.loads(allowed.stdout)["result"]
    assert result["input_method"] == input_method
    assert result["require_focus"] is False
    assert result["execution"]["focus_ok"] is False
    assert result["execution"]["skipped"] is False
    assert result["observation"]["foreground"] is False
    assert result["inputs_released"]
    assert win32gui.GetForegroundWindow() == cover
    assert win32api.GetCursorPos() == cursor
