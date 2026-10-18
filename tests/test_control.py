"""External control contract and input cleanup, without touching real windows."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.capture.frame import CapturedFrame
from src.control import cli, session as control
from src.window_session import WindowInfo

CAPTURE_FRAME = control.capture_frame


@pytest.fixture
def game(tmp_path, monkeypatch):
    window = MagicMock()
    window.target = WindowInfo(123, "Test Game", "test", 456, (0, 0, 640, 480), True, False)
    window.refresh_info.side_effect = lambda: window.target
    window.is_valid.return_value = True
    window.is_minimized.return_value = False
    window.is_foreground.return_value = True
    window.focus.return_value = True
    window.get_client_rect.return_value = (10, 20, 650, 500)
    monkeypatch.setattr(control.window_session, "attach", lambda hwnd: window)
    monkeypatch.setattr(control.window_session, "find_window_by_title", lambda title: [window.target])
    executor = MagicMock()
    executor.execute_tick.return_value = {"skipped": False, "dry_run": False}
    monkeypatch.setattr(control, "ActionExecutor", lambda *a, **kw: executor)
    monkeypatch.setattr(control, "capture_frame", lambda *a: CapturedFrame(
        Image.new("RGB", (640, 480), "white"), 123.0, 123,
        (0, 0, 640, 480), (10, 20, 650, 500), "test"))
    game = control.GameSession(out=tmp_path, cursor_mode="raw")
    game.attach(title="Test")
    executor.reset_mock()
    window.focus.reset_mock()
    return game


def test_step_returns_screenshot_and_releases_chord(game):
    result = game.step(actions=["move_forward", "sprint", "enter"], duration_ms=0, settle_ms=0)
    assert result["inputs_released"]
    assert Path(result["observation"]["screenshot"]).is_file()
    assert result["observation"]["coordinate_space"] == "client_pixels"
    released = game.executor.release_all.call_args.args[0]
    assert {i.key for i in released} == {"w", "shift", "enter"}
    assert not game.state.held_inputs


def test_each_observation_has_unique_path(game):
    assert game.observe()["screenshot"] != game.observe()["screenshot"]


@pytest.mark.parametrize("actions", [None, {}, ["unknown"], [{"name": "aim"}],
    [{"name": "aim", "params": {"x": True, "y": 2}}],
    [{"name": "aim", "params": {"x": 1.5, "y": 2}}],
    [{"name": "move_forward", "params": {"oops": 1}}],
    [{"name": "move_forward", "extra": 1}], ["move_forward"] * 33])
def test_invalid_actions_do_not_focus_or_execute(game, actions):
    with pytest.raises(control.ControlError, match=".+") as error:
        game.step(actions=actions)
    assert error.value.code == "invalid_request"
    game.session.focus.assert_not_called()
    game.executor.execute_tick.assert_not_called()


@pytest.mark.parametrize("field,value", [("duration_ms", -1), ("duration_ms", 10001),
    ("duration_ms", True), ("settle_ms", 5001), ("settle_ms", 0.5)])
def test_duration_validation_precedes_focus(game, field, value):
    with pytest.raises(control.ControlError):
        game.step(actions=[], **{field: value})
    game.session.focus.assert_not_called()


@pytest.mark.parametrize("x,y", [(-1, 0), (640, 0), (0, 480)])
def test_client_coordinates_are_checked_before_input(game, x, y):
    with pytest.raises(control.ControlError):
        game.step(actions=[{"name": "click_at", "params": {"x": x, "y": y}}])
    game.executor.execute_tick.assert_not_called()


def test_client_coordinates_are_not_screen_coordinates(game):
    result = game.step(actions=[{"name": "click_at", "params": {"x": 0, "y": 0}}],
                       duration_ms=0, settle_ms=0)
    assert result["plan"]["click_ats"] == [{"x": 0, "y": 0}]


@pytest.mark.parametrize("failure", [RuntimeError("input failed"), KeyboardInterrupt()])
def test_partial_execution_always_releases_holds_and_taps(game, failure):
    game.executor.execute_tick.side_effect = failure
    with pytest.raises(type(failure)):
        game.step(actions=["move_forward", "enter"])
    assert {i.key for i in game.executor.release_all.call_args.args[0]} == {"w", "enter"}
    assert not game.state.held_inputs


def test_focus_loss_aborts_and_releases(game):
    game.focus = False
    game.session.is_foreground.return_value = False
    with pytest.raises(control.ControlError) as error:
        game.step(actions=["move_forward"], duration_ms=1000)
    assert error.value.code == "focus_lost"
    game.executor.release_all.assert_called_once()


def test_interrupted_click_releases_mouse_button(game):
    game.executor.execute_tick.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        game.step(actions=[{"name": "click_at", "params": {"x": 10, "y": 10}}])
    released = game.executor.release_all.call_args.args[0]
    assert any(i.input_type == "mouse_button" and i.button == "left" for i in released)


def test_executor_focus_guard_is_an_error(game):
    game.executor.execute_tick.return_value = {"skipped": True}
    with pytest.raises(control.ControlError) as error:
        game.step(actions=["move_forward"])
    assert error.value.code == "focus_failed"
    game.executor.release_all.assert_called_once()


def test_capture_failure_reports_that_action_may_have_executed(game, monkeypatch):
    monkeypatch.setattr(control, "capture_frame", lambda *a: None)
    with pytest.raises(control.ControlError) as error:
        game.step(actions=["move_forward"], duration_ms=0, settle_ms=0)
    assert error.value.code == "capture_failed"
    assert error.value.details["actions_may_have_executed"]
    assert error.value.details["step"]["inputs_released"]
    game.executor.release_all.assert_called_once()


def test_dry_run_does_not_focus_send_or_wait(game, monkeypatch):
    game.dry_run = True
    monkeypatch.setattr(control.time, "sleep", MagicMock(side_effect=AssertionError("slept")))
    result = game.step(actions=["move_forward", "enter"], duration_ms=10000)
    assert result["execution"]["dry_run"]
    game.session.focus.assert_not_called()
    game.executor.execute_tick.assert_not_called()
    game.executor.release_all.assert_not_called()


@pytest.mark.parametrize("condition", ["gone", "minimized", "reused", "focus_failed"])
def test_invalid_target_prevents_input(game, condition):
    if condition == "gone":
        game.session.is_valid.return_value = False
    elif condition == "minimized":
        game.session.is_minimized.return_value = True
    elif condition == "reused":
        game.session.target = WindowInfo(123, "Other", "test", 999, (0, 0, 640, 480), True, False)
    else:
        game.session.focus.return_value = False
    with pytest.raises(control.ControlError):
        game.step(actions=["move_forward"])
    game.executor.execute_tick.assert_not_called()


def test_title_requires_unique_match(game, monkeypatch):
    monkeypatch.setattr(control.window_session, "find_window_by_title", lambda title: [])
    with pytest.raises(control.ControlError) as error:
        game.attach(title="Missing")
    assert error.value.code == "target_not_unique"
    assert game.session is not None  # failed attach preserves existing session


def test_stop_detaches_without_closing_game(game):
    window = game.session
    game.stop()
    assert game.session is None
    window.close.assert_not_called()
    with pytest.raises(control.ControlError) as error:
        game.observe()
    assert error.value.code == "not_attached"


def test_auto_capture_falls_back_on_blank(game, monkeypatch):
    monkeypatch.setattr(control.dxcam_capture, "is_available", lambda: True)
    blank = MagicMock()
    blank.is_blank.return_value = True
    good = MagicMock()
    good.is_blank.return_value = False
    monkeypatch.setattr(control.dxcam_capture, "capture_frame", lambda _: blank)
    monkeypatch.setattr(control.windows_graphics_capture, "capture_frame", lambda _: None)
    monkeypatch.setattr(control.fallback_capture, "capture_frame", lambda _: good)
    assert CAPTURE_FRAME(game.session, "auto") is good


def test_serve_recovers_from_bad_request_preserves_ids_and_stops(game):
    source = io.StringIO('\nnot-json\n{"id":2,"op":"step","actions":["bad"]}\n'
                         '{"id":3,"op":"observe"}\n{"id":4,"op":"stop"}\n'
                         '{"op":"observe"}\n')
    out = io.StringIO()
    assert cli.serve(game, source, out) == 0
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["ok"] for r in responses] == [False, False, True, True]
    assert [r.get("id") for r in responses] == [None, 2, 3, 4]
    assert game.session is None


def test_serve_eof_releases_session(game):
    cli.serve(game, io.StringIO(""), io.StringIO())
    assert game.session is None


@pytest.mark.parametrize("bad_number", ["NaN", "Infinity", "1e999"])
def test_serve_rejects_nonfinite_ids_and_continues(game, bad_number):
    out = io.StringIO()
    cli.serve(game, io.StringIO('{"op":"actions","id":' + bad_number + '}\n'
                               '{"op":"stop","id":2}\n'), out)
    responses = [json.loads(line) for line in out.getvalue().splitlines()]
    assert not responses[0]["ok"]
    assert responses[1]["ok"] and responses[1]["id"] == 2


def test_cli_repeat_action_flags(game, monkeypatch, capsys):
    monkeypatch.setattr(cli, "GameSession", lambda **kw: game)
    assert cli.main(["step", "--title", "Test", "--action", "move_forward", "--action", "sprint",
                     "--duration-ms", "0", "--settle-ms", "0"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert [a["canonical_name"] for a in result["actions"]] == ["move_forward", "sprint"]


def test_serve_stdout_stays_json(game, monkeypatch):
    def noisy_actions():
        print("legacy output")
        return {"actions": []}
    monkeypatch.setattr(game, "actions", noisy_actions)
    out = io.StringIO()
    cli.serve(game, io.StringIO('{"op":"actions"}\n'), out)
    assert json.loads(out.getvalue())["ok"]


@pytest.mark.parametrize("payload", [[], {}, {"op": "x"}, {"op": "observe", "extra": 1},
    {"op": "step"}, {"op": "attach", "title": "x", "hwnd": 123}])
def test_dispatch_rejects_malformed_requests(game, payload):
    with pytest.raises(control.ControlError):
        cli.dispatch(game, payload)


def test_cli_invalid_args_are_json(capsys):
    assert cli.main(["step"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_request"


def test_cli_actions_does_not_import_backends(capsys):
    before = set(sys.modules)
    assert cli.main(["actions"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert not any(name.startswith("src.backends") for name in set(sys.modules) - before)


def test_cli_step_and_file_json(game, tmp_path, monkeypatch, capsys):
    path = tmp_path / "actions.json"
    path.write_text('["move_forward"]', encoding="utf-8-sig")
    monkeypatch.setattr(cli, "GameSession", lambda **kw: game)
    assert cli.main(["step", "--hwnd", "0x7b", "--actions-file", str(path),
                     "--duration-ms", "0", "--settle-ms", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["inputs_released"]
    assert game.session is None


@pytest.mark.parametrize("text", ['["unknown"]', '[NaN]', '{'])
def test_bad_cli_json_does_not_attach(game, monkeypatch, capsys, text):
    monkeypatch.setattr(cli, "GameSession", lambda **kw: game)
    game.attach = MagicMock()
    assert cli.main(["step", "--title", "Game", "--actions", text]) == 2
    assert not json.loads(capsys.readouterr().out)["ok"]
    game.attach.assert_not_called()
