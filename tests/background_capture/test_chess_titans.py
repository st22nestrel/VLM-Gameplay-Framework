"""Offline checks for the interactive game-launch/capture workflow."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.control.session import ControlError

spec = importlib.util.spec_from_file_location("chess_capture_smoke", Path(__file__).with_name("chess_titans.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.fixture
def game(tmp_path):
    game = MagicMock()
    game.session.is_valid.return_value = True
    game.session.is_minimized.return_value = False
    game.session.is_foreground.return_value = False
    game.session.target.hwnd = 123
    path = tmp_path / "frame.png"
    path.write_bytes(b"test screenshot")
    game.observe.return_value = {
        "foreground": False, "screenshot_raw": str(path), "screenshot_cursor": None,
    }
    return game


def test_launches_chess_when_not_open(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke, "find_window_by_title", lambda _: [])
    launch = MagicMock()
    monkeypatch.setattr(smoke, "launch_and_attach", launch)
    executable = tmp_path / "chess.exe"
    assert smoke.open_game(executable) is launch.return_value.session
    launch.assert_called_once_with(executable, title_hint="Chess Titans", timeout_s=30)


def test_reuses_existing_chess_without_launching_or_focusing(monkeypatch, game):
    existing = MagicMock(title="Chess Titans", hwnd=123)
    monkeypatch.setattr(smoke, "find_window_by_title", lambda _: [existing])
    monkeypatch.setattr(smoke, "attach", lambda _: game.session)
    launch = MagicMock()
    monkeypatch.setattr(smoke, "launch_and_attach", launch)
    assert smoke.open_game(Path("unused.exe")) is game.session
    launch.assert_not_called()
    game.session.focus.assert_not_called()


@pytest.mark.parametrize("condition", ["foreground", "minimized", "closed"])
def test_sequence_skips_unsuitable_window_without_capturing(game, condition):
    if condition == "foreground":
        game.session.is_foreground.return_value = True
    elif condition == "minimized":
        game.session.is_minimized.return_value = True
    else:
        game.session.is_valid.return_value = False
    result = smoke.capture_sequence(game, count=1, interval=0)
    assert result[0]["status"] == "skipped"
    game.observe.assert_not_called()
    game.session.focus.assert_not_called()


def test_sequence_records_success_without_changing_focus(game):
    result = smoke.capture_sequence(game, count=2, interval=0)
    assert [r["status"] for r in result] == ["captured", "captured"]
    assert game.observe.call_count == 2
    game.session.focus.assert_not_called()


def test_sequence_discards_screenshot_if_game_gains_focus(game):
    game.session.is_foreground.side_effect = [False, True]
    path = Path(game.observe.return_value["screenshot_raw"])
    result = smoke.capture_sequence(game, count=1, interval=0)
    assert result[0]["status"] == "skipped"
    assert not path.exists()


def test_capture_failure_is_recorded(game):
    game.observe.side_effect = ControlError("capture_failed", "No frame")
    result = smoke.capture_sequence(game, count=1, interval=0)
    assert result[0]["status"] == "failed"
    assert result[0]["error"]["code"] == "capture_failed"


def test_interactive_main_waits_for_user_then_writes_results(game, monkeypatch, tmp_path):
    monkeypatch.setattr(smoke.sys, "platform", "win32")
    monkeypatch.setattr(smoke, "open_game", lambda _: game.session)
    constructor = MagicMock(return_value=game)
    monkeypatch.setattr(smoke, "GameSession", constructor)

    def ready(_):
        game.observe.assert_not_called()
        game.attach.assert_not_called()
        return ""

    monkeypatch.setattr("builtins.input", ready)
    assert smoke.main(["--count", "1", "--out", str(tmp_path)]) == 0
    assert constructor.call_args.kwargs["method"] == "background"
    assert constructor.call_args.kwargs["focus"] is False
    assert constructor.call_args.kwargs["dry_run"] is True
    game.attach.assert_called_once_with(hwnd=123)
    game.stop.assert_called_once()
    assert json.loads((tmp_path / "results.json").read_text())[0]["status"] == "captured"
    game.session.close.assert_not_called()
