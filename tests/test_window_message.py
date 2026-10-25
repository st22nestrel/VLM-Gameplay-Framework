"""Window-directed input contracts, including failure and cleanup behavior."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.actions import executor as platform
from src.actions import window_message as wm
from src.actions.state import HeldInput, TickPlan

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows input adapter")


@pytest.fixture
def sender(monkeypatch):
    session = MagicMock()
    session.target.hwnd = 100
    session.is_foreground.return_value = False
    gui = SimpleNamespace(
        GetClassName=lambda hwnd: "BackgroundInputTestReceiver",
        IsWindow=lambda hwnd: hwnd in (100, 101, 102),
        IsIconic=lambda hwnd: False,
        IsChild=lambda root, child: root == 100 and child in (101, 102),
        GetClientRect=lambda hwnd: (0, 0, 640, 480),
        ClientToScreen=lambda hwnd, pt: (pt[0] + 1000, pt[1] + 2000),
        ScreenToClient=lambda hwnd, pt: (pt[0] - 1000 - (20 if hwnd == 101 else 0),
                                        pt[1] - 2000 - (30 if hwnd == 101 else 0)),
        ChildWindowFromPointEx=lambda hwnd, pt, flags: 101 if hwnd == 100 else hwnd,
    )
    monkeypatch.setattr(wm, "win32gui", gui)
    monkeypatch.setattr(wm, "win32process", SimpleNamespace(GetWindowThreadProcessId=lambda hwnd: (77, 88)))
    monkeypatch.setattr(wm, "_thread_focus", lambda thread: None)
    post = MagicMock()
    monkeypatch.setattr(wm, "win32api", SimpleNamespace(PostMessage=post))
    monkeypatch.setattr(wm.time, "sleep", lambda _: None)
    executor = platform.ActionExecutor(session, input_method="windowmessage", require_focus=False)
    # Any fallback to global desktop input is a test failure.
    monkeypatch.setattr(platform, "_send_input", MagicMock(side_effect=AssertionError("desktop input")))
    monkeypatch.setattr(platform.win32api, "SetCursorPos", MagicMock(side_effect=AssertionError("cursor moved")))
    return executor, post, session, gui


def test_background_click_routes_to_child_without_desktop_input(sender):
    executor, post, session, _ = sender
    result = executor.execute_tick(TickPlan([], [], [], [{"x": 70, "y": 90}]))
    assert result["input_method"] == "windowmessage"
    assert result["background"] and not result["focus_ok"] and not result["skipped"]
    assert [c.args for c in post.call_args_list] == [
        (101, 0x200, 0, wm.pack_point(50, 60)),
        (101, 0x201, 1, wm.pack_point(50, 60)),
        (101, 0x202, 0, wm.pack_point(50, 60)),
    ]
    session.focus.assert_not_called()


def test_drag_retains_position_recipient_buttons_and_modifiers(sender):
    executor, post, _, gui = sender
    backend = executor._window_input
    backend.key("shift")
    backend.key("ctrl")
    backend.move(70, 90)
    backend.button("left")
    # Crossing another child must not redirect an active drag.
    gui.ChildWindowFromPointEx = lambda hwnd, pt, flags: 102 if hwnd == 100 else hwnd
    backend.move(200, 210)
    backend.button("right")
    backend.button("left", up=True)
    backend.button("right", up=True)
    mouse = [c.args for c in post.call_args_list if c.args[1] >= 0x200]
    assert mouse == [
        (101, 0x200, 12, wm.pack_point(50, 60)),
        (101, 0x201, 13, wm.pack_point(50, 60)),
        (101, 0x200, 13, wm.pack_point(180, 180)),
        (101, 0x204, 15, wm.pack_point(180, 180)),
        (101, 0x202, 14, wm.pack_point(180, 180)),
        (101, 0x205, 12, wm.pack_point(180, 180)),
    ]
    backend.move(70, 90)
    assert post.call_args.args[0] == 102


def test_keyboard_flags_and_release_recipient(sender, monkeypatch):
    executor, post, _, _ = sender
    backend = executor._window_input
    monkeypatch.setattr(wm, "_thread_focus", lambda _: 101)
    backend.key("up")
    backend.key("up")
    monkeypatch.setattr(wm, "_thread_focus", lambda _: 102)
    backend.key("up", up=True)
    assert [c.args for c in post.call_args_list] == [
        (101, 0x100, 0x26, 1 | (0x48 << 16) | (1 << 24)),
        (101, 0x100, 0x26, 1 | (0x48 << 16) | (1 << 24) | (1 << 30)),
        (101, 0x101, 0x26, 1 | (0x48 << 16) | (1 << 24) | (3 << 30)),
    ]
    post.reset_mock()
    backend.key("alt")
    backend.key("a")
    executor.release_all([])
    events = [c.args for c in post.call_args_list]
    assert [(e[1], e[2]) for e in events] == [(0x104, 0x12), (0x104, 0x41),
                                                            (0x105, 0x41), (0x105, 0x12)]
    assert not events[0][3] & (1 << 29)
    assert all(e[3] & (1 << 29) for e in events[1:])


def test_foreign_focus_is_ignored_and_last_mouse_child_is_used(sender, monkeypatch):
    executor, post, _, _ = sender
    monkeypatch.setattr(wm, "_thread_focus", lambda _: 999)
    executor._window_input.key("enter")
    assert post.call_args.args[0] == 100
    executor._window_input.move(70, 90)
    executor._window_input.key("space")
    assert post.call_args.args[0] == 101


def test_focus_policy_and_dry_run(sender, monkeypatch):
    executor, post, session, _ = sender
    monkeypatch.setattr(platform, "check_focus", lambda _: False)
    for dry_run, require_focus in ((False, True), (True, False)):
        executor = platform.ActionExecutor(session, input_method="windowmessage", dry_run=dry_run,
                                           require_focus=require_focus)
        result = executor.execute_tick(TickPlan([], [HeldInput("forward", "key", key="w")],
                                                [{"x": 10, "y": 20}], [{"x": 30, "y": 40}]))
        assert result["skipped"] == require_focus
        executor.release_all([])
    post.assert_not_called()


def test_post_failure_propagates_without_recording_a_press(sender):
    executor, post, _, _ = sender
    post.side_effect = OSError("Access denied")
    with pytest.raises(OSError, match="Access denied"):
        executor._window_input.key("w")
    post.side_effect = None
    executor.release_all([])
    assert post.call_count == 1


def test_cleanup_attempts_remaining_releases_after_failure(sender):
    executor, post, _, _ = sender
    backend = executor._window_input
    backend.key("w")
    backend.key("shift")
    backend.button("left")
    post.reset_mock()
    post.side_effect = [OSError("recipient gone"), None, None]
    with pytest.raises(OSError, match="recipient gone"):
        executor.release_all([])
    assert post.call_count == 3
    assert not backend._keys
    assert backend._buttons == {"left"}
    post.side_effect = None
    executor.release_all([])
    assert not backend._buttons


def test_click_does_not_release_held_button_and_failed_click_is_released(sender, monkeypatch):
    executor, post, _, _ = sender
    backend = executor._window_input
    backend.button("left")
    assert executor.click_at(60, 70)["status"] == "already_held"
    assert backend._buttons == {"left"}
    backend.button("left", up=True)
    post.reset_mock()

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(wm.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        executor.click_at(60, 70)
    assert [c.args[1] for c in post.call_args_list] == [0x200, 0x201, 0x202]
    assert not backend._buttons


@pytest.mark.parametrize("condition", ["closed", "minimized", "reattached", "child_destroyed"])
def test_invalid_target_never_falls_back_to_global_input(sender, condition):
    executor, post, session, gui = sender
    backend = executor._window_input
    backend.button("left")
    post.reset_mock()
    if condition == "closed":
        gui.IsWindow = lambda _: False
    elif condition == "minimized":
        gui.IsIconic = lambda _: True
    elif condition == "reattached":
        session.target.hwnd = 102
    else:
        gui.IsWindow = lambda hwnd: hwnd == 100
    with pytest.raises(RuntimeError):
        backend.move(60, 70)
    post.assert_not_called()


def test_signed_point_packing(sender):
    executor, post, _, _ = sender
    executor._window_input.move(-1, -2)
    assert post.call_args.args[3] == wm.pack_point(-21, -32)
    assert wm.pack_point(-1, -2) == 0xFFFEFFFF
    with pytest.raises(ValueError, match="16-bit"):
        executor._window_input.move(32768, 0)


def test_unknown_inputs_fail_visibly(sender):
    executor, post, _, _ = sender
    with pytest.raises(ValueError, match="key"):
        executor._window_input.key("unsupported")
    with pytest.raises(ValueError, match="button"):
        executor._window_input.button("unsupported")
    post.assert_not_called()
