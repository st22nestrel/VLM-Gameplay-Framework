"""Register desktop-test opt-ins before collecting any test subdirectory."""


def pytest_addoption(parser):
    parser.addoption("--live-chess-moves", action="store_true",
                     help="Reset Chess Titans and replay three real moves (Windows desktop required)")
    parser.addoption("--live-background-capture", action="store_true",
                     help="Create temporary windows to test real occlusion (Windows desktop required)")
