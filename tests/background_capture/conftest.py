import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption("--live-background-capture", action="store_true",
                     help="Create temporary windows to test real occlusion (Windows desktop required)")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "test_live_capture.py" in str(item.path) and (
            sys.platform != "win32" or not config.getoption("--live-background-capture")
        ):
            item.add_marker(pytest.mark.skip(reason="Use --live-background-capture on Windows"))
