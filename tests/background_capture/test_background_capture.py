"""Offline contracts for occlusion-safe capture and its control interface."""

import dataclasses
import json
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.capture import background_capture, windows_graphics_capture as win32_capture
from src.capture.frame import CapturedFrame
from src.control import cli, session as control
from src.window_session import WindowInfo


@pytest.fixture
def window():
    session = MagicMock()
    session.target = WindowInfo(123, "Game", "test", 456, (0, 0, 100, 100), True, False)
    session.is_valid.return_value = True
    session.is_minimized.return_value = False
    session.is_foreground.return_value = False
    session.refresh_info.side_effect = lambda: session.target
    session.get_rect.return_value = (0, 0, 100, 100)
    session.get_client_rect.return_value = (0, 0, 100, 100)
    session.check_health.return_value = []
    return session


@pytest.fixture
def frame():
    return CapturedFrame(Image.new("RGB", (100, 100), "white"), 1, 123,
                         (0, 0, 100, 100), (0, 0, 100, 100), "printwindow")


@pytest.mark.parametrize("result", ["success", "blank", "failure"])
def test_printwindow_only_never_reads_desktop(window, frame, monkeypatch, result):
    monkeypatch.setattr(win32_capture.sys, "platform", "win32")
    expected = frame if result == "success" else None
    returned = None if result == "failure" else frame
    if result == "blank":
        frame.image = Image.new("RGB", (100, 100), "black")
    printwindow = MagicMock(return_value=returned)
    desktop = MagicMock(side_effect=AssertionError("Desktop pixels are not safe"))
    monkeypatch.setattr(win32_capture, "_capture_printwindow", printwindow)
    monkeypatch.setattr(win32_capture, "_capture_bitblt", desktop)
    assert background_capture.capture_frame(window) is expected
    printwindow.assert_called_once()
    desktop.assert_not_called()
    window.focus.assert_not_called()


@pytest.mark.parametrize("condition", ["invalid", "minimized", "closed_during_capture",
                                      "minimized_during_capture", "reattached", "geometry_error"])
def test_unavailable_target_is_rejected(window, frame, monkeypatch, condition):
    monkeypatch.setattr(background_capture.sys, "platform", "win32")
    backend = MagicMock(return_value=frame)
    monkeypatch.setattr(win32_capture, "capture_frame", backend)
    if condition == "invalid":
        window.is_valid.return_value = False
    elif condition == "minimized":
        window.is_minimized.return_value = True
    elif condition == "closed_during_capture":
        window.is_valid.side_effect = [True, False]
    elif condition == "minimized_during_capture":
        window.is_minimized.side_effect = [False, True]
    elif condition == "reattached":
        frame.hwnd = 999
    else:
        backend.side_effect = RuntimeError("Window was destroyed while reading geometry")
    assert background_capture.capture_frame(window) is None
    if condition in ("invalid", "minimized"):
        backend.assert_not_called()
    window.focus.assert_not_called()


def test_background_capture_is_windows_only(window, monkeypatch):
    monkeypatch.setattr(background_capture.sys, "platform", "linux")
    assert background_capture.capture_frame(window) is None
    window.is_valid.assert_not_called()


@pytest.fixture
def game(window, frame, tmp_path, monkeypatch):
    monkeypatch.setattr(control.window_session, "attach", lambda _: window)
    monkeypatch.setattr(control, "ActionExecutor", MagicMock())
    monkeypatch.setattr(background_capture, "capture_frame", MagicMock(return_value=frame))
    game = control.GameSession(method="background", out=tmp_path, cursor_mode="raw")
    game.attach(hwnd=123)
    game.executor.reset_mock()
    return game


def test_observe_captures_unfocused_without_focusing(game):
    result = game.observe()
    assert result["foreground"] is False
    assert result["capture_method"] == "printwindow"
    assert Image.open(result["screenshot"]).getpixel((50, 50)) == (255, 255, 255)
    game.session.focus.assert_not_called()
    game.executor.execute_tick.assert_not_called()


def test_step_keeps_foreground_input_guard(game):
    game.focus = False
    with pytest.raises(control.ControlError) as exc:
        game.step(actions=[], duration_ms=0, settle_ms=0)
    assert exc.value.code == "focus_lost"
    game.executor.execute_tick.assert_not_called()


def test_capture_failure_is_an_error(game, monkeypatch):
    monkeypatch.setattr(background_capture, "capture_frame", lambda _: None)
    with pytest.raises(control.ControlError) as exc:
        game.observe()
    assert exc.value.code == "capture_failed"
    assert not list(game.out.glob("*.png"))


@pytest.mark.parametrize("condition", ["minimized", "closed", "identity_changed"])
def test_target_change_during_observe_discards_frame(game, frame, monkeypatch, condition):
    def capture(_):
        if condition == "minimized":
            game.session.is_minimized.return_value = True
        elif condition == "closed":
            game.session.is_valid.return_value = False
        else:
            game.session.target = dataclasses.replace(game.session.target, hwnd=999)
        return frame

    monkeypatch.setattr(background_capture, "capture_frame", capture)
    with pytest.raises(control.ControlError) as exc:
        game.observe()
    assert exc.value.code == "target_unavailable"
    assert not list(game.out.glob("*.png"))


def test_cli_exposes_background_capture(game, monkeypatch, capsys):
    constructor = MagicMock(return_value=game)
    monkeypatch.setattr(cli, "GameSession", constructor)
    assert cli.main(["observe", "--hwnd", "123", "--method", "background"]) == 0
    assert constructor.call_args.kwargs["method"] == "background"
    assert json.loads(capsys.readouterr().out)["result"]["foreground"] is False


def test_other_capture_entry_points_use_the_same_backend(window, frame, monkeypatch):
    from src.cli.main import _do_capture as cli_capture, build_parser
    from src.loop.gameplay_loop import _do_capture as loop_capture

    backend = MagicMock(return_value=frame)
    monkeypatch.setattr(background_capture, "capture_frame", backend)
    args = build_parser().parse_args(["capture-window", "--hwnd", "123", "--method", "background"])
    assert args.method == "background"
    assert cli_capture(window, "background") is frame
    assert loop_capture(window, "background") is frame
    assert control.capture_frame(window, "background") is frame
    assert backend.call_count == 3
