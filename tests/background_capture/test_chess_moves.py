"""Visual replay regressions, plus an explicitly enabled real-game run."""

import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

from PIL import Image
import pytest

SCRIPT = Path(__file__).with_name("chess_moves.py")
spec = importlib.util.spec_from_file_location("chess_moves", SCRIPT)
moves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moves)


@pytest.mark.parametrize("state", [0, 1, 2, 3])
def test_visual_check_distinguishes_every_move_from_missing_or_extra_moves(state):
    references = [Image.open(moves.FIXTURES / "initial.png").crop(moves.PAWN_STRIP)]
    references += [Image.open(moves.FIXTURES / move[3]) for move in moves.MOVES]
    with Image.open(moves.FIXTURES / "initial.png") as initial:
        image = initial.copy()
    image.paste(references[state], moves.PAWN_STRIP[:2])
    for other, reference in enumerate(references):
        error = max(moves.pawn_errors(image, reference))
        assert (error < 4) == (state == other)


def test_no_op_input_cannot_pass_replay_or_advance_to_second_move(monkeypatch):
    game = MagicMock()
    observation = {"screenshot_raw": str(moves.FIXTURES / "initial.png")}
    game.observe.return_value = observation
    game.step.return_value = {"observation": observation}
    monkeypatch.setattr(moves.time, "sleep", lambda _: None)
    monkeypatch.setattr(moves.time, "monotonic", lambda: next(ticks))
    ticks = itertools.count()
    report = {}
    with pytest.raises(RuntimeError, match="a2a3: move or settled opponent reply not verified"):
        moves.replay(game, timeout=4, report=report)
    assert len(report["moves"]) == 1
    assert game.step.call_count == 2
    assert game.step.call_args.kwargs["settle_ms"] == 3000


@pytest.mark.parametrize("condition", ["thinking", "no_reply"])
def test_correct_white_move_does_not_advance_before_opponent_finishes(condition, tmp_path, monkeypatch):
    with Image.open(moves.FIXTURES / "initial.png") as initial:
        image = initial.copy()
    with Image.open(moves.FIXTURES / "after_a3.png") as reference:
        image.paste(reference, moves.PAWN_STRIP[:2])
    if condition == "thinking":
        image.paste((0, 0, 0), (360, 195, 390, 230))  # Opponent's board has changed.
        image.paste((255, 120, 0), moves.THINKING_BOX)  # But the thinking area is still busy.
    path = tmp_path / "opponent_unfinished.png"
    image.save(path)
    unfinished = {"screenshot_raw": str(path)}
    game = MagicMock()
    game.observe.side_effect = itertools.chain(
        [{"screenshot_raw": str(moves.FIXTURES / "initial.png")}], itertools.repeat(unfinished))
    game.step.return_value = {"observation": unfinished}
    ticks = itertools.count()
    monkeypatch.setattr(moves.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(moves.time, "sleep", lambda _: None)
    report = {}
    with pytest.raises(RuntimeError, match="a2a3: move or settled opponent reply not verified"):
        moves.replay(game, timeout=4, report=report)
    assert max(report["moves"][0]["checks"]["pawn_errors"]) == 0
    assert game.step.call_count == 2


def test_wrong_starting_board_sends_no_input(tmp_path):
    image = Image.open(moves.FIXTURES / "initial.png")
    image.paste(Image.open(moves.FIXTURES / "after_c3.png"), moves.PAWN_STRIP[:2])
    path = tmp_path / "wrong_start.png"
    image.save(path)
    game = MagicMock()
    game.observe.return_value = {"screenshot_raw": str(path)}
    with pytest.raises(RuntimeError, match="Starting board/layout differs"):
        moves.replay(game)
    game.step.assert_not_called()


def test_real_chess_three_moves(request, tmp_path):
    if sys.platform != "win32" or not request.config.getoption("--live-chess-moves"):
        pytest.skip("Use --live-chess-moves on an unlocked Windows desktop with Chess Titans")
    result = subprocess.run([sys.executable, str(SCRIPT), "--new-game", "--out", str(tmp_path)],
                            cwd=moves.ROOT, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert [move["move"] for move in report["moves"]] == ["a2a3", "b2b3", "c2c3"]
    assert all(move["status"] == "passed" for move in report["moves"])
