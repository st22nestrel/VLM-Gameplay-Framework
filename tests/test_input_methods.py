"""Transport and focus-policy contracts; all platform input is mocked."""

import sys
from unittest.mock import MagicMock

import pytest

from src.actions import executor as platform
from src.actions.input_settings import InputSettings
from src.actions.schema import ParsedAction, load_action_config
from src.actions.state import ActionState
from src.control.session import ControlError, GameSession
from src.loop.gameplay_config import GameplayRuntimeConfig


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("require_focus", [False, True])
@pytest.mark.parametrize("foreground", [False, True])
def test_transport_and_focus_policy_are_independent(monkeypatch, background, require_focus, foreground):
    if sys.platform != "win32":
        pytest.skip("Windows input adapters")
    window = MagicMock()
    window.is_foreground.return_value = foreground
    monkeypatch.setattr(platform, "check_focus", lambda _: foreground)
    send = MagicMock()
    post = MagicMock()
    monkeypatch.setattr(platform, "_send_input", send)
    monkeypatch.setattr(platform.win32api, "PostMessage", post)
    config = load_action_config("configs/actions/control-actions.yaml")
    plan = ActionState(config).compute_tick([ParsedAction("enter", {})])
    executor = platform.ActionExecutor(window, background=background, require_focus=require_focus)
    result = executor.execute_tick(plan)
    blocked = require_focus and not foreground
    assert result["skipped"] == blocked
    assert result["focus_ok"] == foreground
    assert result["require_focus"] == require_focus
    assert send.call_count == (2 if not blocked and not background else 0)
    assert post.call_count == (2 if not blocked and background else 0)


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("require_focus", [False, True])
def test_click_at_retains_direct_sender_and_obeys_focus_policy(monkeypatch, background, require_focus):
    if sys.platform != "win32":
        pytest.skip("Windows input adapters")
    window = MagicMock()
    window.get_client_rect.return_value = (100, 200, 740, 680)
    monkeypatch.setattr(platform, "check_focus", lambda _: False)
    monkeypatch.setattr(platform.win32api, "GetCursorPos", lambda: (0, 0))
    move, send, post = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(platform.win32api, "SetCursorPos", move)
    monkeypatch.setattr(platform, "_send_input", send)
    monkeypatch.setattr(platform.win32api, "PostMessage", post)
    executor = platform.ActionExecutor(window, background=background, require_focus=require_focus)
    result = executor.click_at(21, 42)
    if require_focus:
        assert result["status"] == "skipped_no_focus"
        move.assert_not_called()
        send.assert_not_called()
    else:
        assert result["status"] == "ok"
        move.assert_called_once_with((121, 242))
        assert send.call_count == 2
    post.assert_not_called()


@pytest.mark.parametrize("bad", [{"input_method": "unknown"}, {"require_focus": "false"}])
def test_bad_settings_fail_before_attaching_or_input(bad):
    with pytest.raises(ValueError):
        InputSettings(**bad)
    with pytest.raises(ControlError) as error:
        GameSession(**bad)
    assert error.value.code == "invalid_request"
    with pytest.raises(ValueError):
        GameplayRuntimeConfig.from_raw({"actions": bad})


@pytest.mark.parametrize("input_method", ["sendinput", "postmessage", "windowmessage"])
@pytest.mark.parametrize("require_focus", [False, True])
def test_runtime_and_cli_preserve_independent_settings(input_method, require_focus):
    from src.cli.main import _override_input_settings, build_parser

    raw = {"actions": {"input_method": input_method, "require_focus": require_focus}}
    args = build_parser().parse_args(["gameplay", "--title", "Game"])
    _override_input_settings(raw, args)
    runtime = GameplayRuntimeConfig.from_raw(raw)
    assert runtime.input_method == input_method
    assert runtime.require_focus == require_focus
    args = build_parser().parse_args(["gameplay", "--title", "Game", "--input-method", "postmessage",
                                      "--allow-unfocused"])
    _override_input_settings(raw, args)
    assert raw["actions"] == {"input_method": "postmessage", "require_focus": False}


def test_legacy_runtime_defaults():
    runtime = GameplayRuntimeConfig.from_raw({})
    assert runtime.input_method == "sendinput"
    assert runtime.require_focus is True


@pytest.mark.parametrize("input_method", ["sendinput", "postmessage", "windowmessage"])
def test_discovery_cli_passes_overrides_into_loop(tmp_path, monkeypatch, input_method):
    from src.cli import main as cli
    from src.loop import discovery_loop

    config = tmp_path / "discovery.yaml"
    config.write_text("discovery:\n  system_prompts: {specific: test}\n  user_prompts: {specific: test}\n",
                      encoding="utf-8")
    factory = MagicMock()
    monkeypatch.setattr(cli, "_resolve_target", lambda _: MagicMock())
    monkeypatch.setattr(discovery_loop, "DiscoveryLoop", factory)
    args = cli.build_parser().parse_args(["discover-actions", "--title", "Game", "--config", str(config),
                                         "--input-method", input_method, "--allow-unfocused"])
    cli.cmd_discover_actions(args)
    assert factory.call_args.kwargs["input_settings"] == InputSettings(input_method, False)
