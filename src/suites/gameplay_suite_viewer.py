"""Local web viewer for gameplay-suite results.

The viewer deliberately serves screenshots from their run directories instead of
embedding them in generated HTML.  This keeps startup fast even for large suites
and gives the browser only the image currently being inspected.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import errno
import json
import math
import mimetypes
import os
import re
import subprocess
import threading
import webbrowser
from collections import Counter
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import yaml


ASSET_DIR = Path(__file__).with_name("gameplay_suite_viewer_assets")
CLICK_MARKERS = ("cross", "circle")


class ViewerHTTPServer(ThreadingHTTPServer):
    """Threaded local server that never shares a port with another viewer."""

    allow_reuse_address = False


def find_latest_suite(root: str | Path = "outputs/gameplay-suites") -> Path:
    """Return the newest suite directory that contains a manifest."""
    root_path = Path(root).expanduser().resolve()
    candidates = [
        path
        for path in root_path.iterdir()
        if path.is_dir() and (path / "suite_manifest.yaml").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No gameplay suites found in {root_path}")
    # Suite directories use sortable timestamp names. Prefer that stable signal;
    # mtimes can change later when summaries or notes are added to older suites.
    return max(candidates, key=lambda path: (path.name, path.stat().st_mtime))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _relative_media_name(value: Any, run_dir: Path) -> str | None:
    if not value:
        return None
    candidate = Path(str(value).replace("\\", "/"))
    direct = run_dir / candidate.name
    if direct.is_file():
        return direct.name
    return None


def _action_label(tick: dict[str, Any]) -> str:
    response = str(tick.get("vlm_response") or "").strip()
    if response:
        return response
    actions = tick.get("parsed_actions") or []
    if not actions:
        return "No action"
    action = actions[0] or {}
    name = action.get("canonical_name", "action")
    params = action.get("params") or {}
    suffix = " ".join(str(value) for value in params.values())
    return f"{name} {suffix}".strip()


def _click_target(tick: dict[str, Any]) -> dict[str, float] | None:
    """Return the first parsed click-at coordinate, if the tick has one."""
    for action in tick.get("parsed_actions") or []:
        if not isinstance(action, dict):
            continue
        name = str(action.get("canonical_name") or "").strip().lower()
        if name not in {"click at", "click_at"}:
            continue
        params = action.get("params") or {}
        x, y = _number(params.get("x")), _number(params.get("y"))
        if x is not None and y is not None:
            return {"x": x, "y": y}
    return None


def _validate_click_marker(value: str) -> str:
    if value not in CLICK_MARKERS:
        raise ValueError(f"Click marker must be one of: {', '.join(CLICK_MARKERS)}")
    return value


def _max_action_streak(actions: list[str]) -> int:
    best = current = 0
    previous: str | None = None
    for action in actions:
        if action == previous:
            current += 1
        else:
            previous = action
            current = 1
        best = max(best, current)
    return best


def _safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 2) if values else None


def _summarize_ticks(ticks: list[dict[str, Any]], total_progress: Any) -> dict[str, Any]:
    latencies = [value for tick in ticks if (value := _number(tick.get("inference_latency_ms"))) is not None]
    actions = [_action_label(tick) for tick in ticks]
    valid = sum(bool(tick.get("parsed_actions")) for tick in ticks)
    stuck = sum(bool((tick.get("state_context") or {}).get("stuck_detected")) for tick in ticks)
    progress_events = sum(bool((tick.get("progress_tracking") or {}).get("progress_event")) for tick in ticks)
    errors = sum(bool(tick.get("inference_error")) for tick in ticks)
    skipped = sum(bool(tick.get("skipped")) for tick in ticks)
    return {
        "ticks": len(ticks),
        "progress": int(total_progress) if _number(total_progress) is not None else progress_events,
        "progressEvents": progress_events,
        "validActions": valid,
        "validRate": round(valid / len(ticks), 4) if ticks else 0.0,
        "errors": errors,
        "skipped": skipped,
        "stuckTicks": stuck,
        "stuckRate": round(stuck / len(ticks), 4) if ticks else 0.0,
        "avgLatency": _safe_mean(latencies),
        "maxLatency": round(max(latencies), 2) if latencies else None,
        "uniqueActions": len(set(actions)),
        "maxActionStreak": _max_action_streak(actions),
    }


def _media_url(value: Any, run_id: str, run_dir: Path) -> str | None:
    name = _relative_media_name(value, run_dir)
    if not name:
        return None
    return f"/media/{quote(run_id, safe='')}/{quote(name)}"


def _vlm_dispatch_payload(
    item: dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    """Preserve the effective VLM conversation in display-safe JSON."""
    messages = []
    for index, message in enumerate(item.get("messages") or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    source = part.get("image")
                    current = source in (None, "current_vlm_image")
                    image = item.get("image") if current else source
                    part = {
                        **part,
                        "imageUrl": _media_url(image, run_id, run_dir),
                        "imageName": _relative_media_name(image, run_dir),
                        "isCurrent": current,
                    }
                parts.append(part)
            content = parts
        messages.append(
            {
                "index": message.get("index", index),
                "role": str(message.get("role") or "unknown"),
                "content": content,
            }
        )
    return {
        "api": item.get("api"),
        "image": _media_url(item.get("image"), run_id, run_dir),
        "imageName": _relative_media_name(item.get("image"), run_dir),
        "messages": messages,
        "response": item.get("response"),
        "latency": _number(item.get("latency_ms")),
        "generationParams": item.get("generation_params") or {},
    }


def _tick_payload(
    tick: dict[str, Any],
    run_id: str,
    run_dir: Path,
    dispatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def media(key: str) -> str | None:
        name = _relative_media_name(tick.get(key), run_dir)
        if not name:
            return None
        return f"/media/{quote(run_id, safe='')}/{quote(name)}"

    progress = tick.get("progress_tracking") or {}
    state = tick.get("state_context") or {}
    generation = tick.get("generation_params") or {}
    progress_name = f"tick_{int(tick.get('tick', 0)):04d}_progress_tracking.png"
    progress_url = (
        f"/media/{quote(run_id, safe='')}/{quote(progress_name)}"
        if (run_dir / progress_name).is_file()
        else None
    )
    return {
        "tick": tick.get("tick"),
        "timestamp": tick.get("timestamp"),
        "screens": {
            "raw": media("screenshot_raw") or media("screenshot"),
            "cursor": media("screenshot_cursor"),
            "vlm": media("vlm_screenshot"),
            "progress": media("progress_tracking_debug")
            or media("screenshot_progress_tracking")
            or progress_url,
        },
        "action": _action_label(tick),
        "response": tick.get("vlm_response"),
        "parsedActions": tick.get("parsed_actions") or [],
        "clickTarget": _click_target(tick),
        "latency": _number(tick.get("inference_latency_ms")),
        "error": tick.get("inference_error"),
        "skipped": bool(tick.get("skipped")),
        "framesDropped": tick.get("frames_dropped"),
        "progress": {
            "event": bool(progress.get("progress_event")),
            "count": progress.get("progress_count"),
            "changed": progress.get("region_changed"),
            "score": _number(progress.get("change_score")),
        },
        "state": {
            "stuck": bool(state.get("stuck_detected")),
            "actionStuck": bool(state.get("action_stuck")),
            "screenStuck": bool(state.get("screen_stuck")),
            "reason": state.get("stuck_reason"),
            "repeated": state.get("repeated_action_count"),
            "screenChange": _number(state.get("screen_change_score_raw")),
        },
        "generation": {
            "key": generation.get("key"),
            "source": generation.get("source"),
            "effective": generation.get("effective") or {},
        },
        "dispatch": dispatch,
    }


def _read_vlm_items(
    log: dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> dict[int, dict[str, Any]]:
    vlm_io_name = Path(str(log.get("vlm_io_log") or "vlm_io.json")).name
    vlm_io_path = run_dir / vlm_io_name
    if not vlm_io_path.is_file():
        return {}
    try:
        vlm_document = json.loads(vlm_io_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    items: dict[int, dict[str, Any]] = {}
    for item in vlm_document.get("items") or []:
        if not isinstance(item, dict) or _number(item.get("tick")) is None:
            continue
        items[int(item["tick"])] = _vlm_dispatch_payload(item, run_id, run_dir)
    return items


def _run_payload(
    run_id: str,
    run_dir: Path,
    record: dict[str, Any],
    log: dict[str, Any],
) -> dict[str, Any]:
    ticks = log.get("ticks") or []
    vlm_items = _read_vlm_items(log, run_id, run_dir)
    return {
        "id": run_id,
        "name": record.get("name") or run_dir.parent.name or run_dir.name,
        "repeatIndex": record.get("repeat_index"),
        "repeatCount": record.get("repeat_count"),
        "status": record.get("status") or "completed",
        "stage": record.get("stage") or "gameplay",
        "startedAt": record.get("started_at"),
        "completedAt": record.get("completed_at"),
        "outputDir": record.get("output_dir") or run_dir.name,
        "sourceDirectory": str(run_dir),
        "backend": log.get("backend"),
        "model": log.get("model"),
        "runId": log.get("run_id") or run_dir.name,
        "error": record.get("error"),
        "summary": _summarize_ticks(ticks, log.get("total_progress_count")),
        "ticks": [
            _tick_payload(
                tick,
                run_id,
                run_dir,
                vlm_items.get(int(tick.get("tick", tick_index))),
            )
            for tick_index, tick in enumerate(ticks)
        ],
    }


def _group_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for name in dict.fromkeys(run["name"] for run in runs):
        members = [run for run in runs if run["name"] == name]
        numeric_keys = ("progress", "validRate", "avgLatency", "stuckRate", "errors", "maxActionStreak")
        aggregate: dict[str, Any] = {"runs": len(members)}
        for key in numeric_keys:
            values = [float(value) for run in members if (value := run["summary"].get(key)) is not None]
            aggregate[key] = _safe_mean(values)
        groups.append(
            {
                "name": name,
                "runIds": [run["id"] for run in members],
                "model": next((run["model"] for run in members if run["model"]), None),
                "aggregate": aggregate,
            }
        )
    return groups


def build_suite_payload(
    suite_dir: str | Path,
    click_marker: str = "cross",
) -> dict[str, Any]:
    """Read a suite into a compact, browser-friendly JSON document."""
    suite_path = Path(suite_dir).expanduser().resolve()
    manifest_path = suite_path / "suite_manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"suite_manifest.yaml not found in {suite_path}")
    manifest = _read_yaml(manifest_path)
    runs: list[dict[str, Any]] = []

    for index, record in enumerate(manifest.get("runs") or []):
        output_name = str(record.get("output_dir") or "")
        run_dir = (suite_path / output_name).resolve()
        if suite_path not in run_dir.parents or not run_dir.is_dir():
            continue
        log: dict[str, Any] = {}
        log_rel = record.get("gameplay_log")
        if log_rel:
            log_candidate = (suite_path / str(log_rel).replace("\\", os.sep)).resolve()
            if suite_path in log_candidate.parents and log_candidate.is_file():
                log = _read_yaml(log_candidate)
        runs.append(_run_payload(f"run-{index}", run_dir, record, log))

    return {
        "suite": {
            "name": manifest.get("suite_name") or suite_path.name,
            "directory": str(suite_path),
            "folderName": suite_path.name,
            "status": manifest.get("status"),
            "startedAt": manifest.get("started_at"),
            "completedAt": manifest.get("completed_at"),
            "repeatCount": manifest.get("repeat_count"),
            "config": manifest.get("suite_config"),
            "sourceType": "suite",
            "compareEnabled": len(runs) > 1,
            "clickMarker": _validate_click_marker(click_marker),
        },
        "groups": _group_runs(runs),
        "runs": runs,
    }


def _tick_datetime(tick: dict[str, Any] | None) -> str | None:
    timestamp = _number((tick or {}).get("timestamp"))
    if timestamp is None:
        return None
    return _dt.datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def _standalone_run_name(path: Path) -> str:
    """Infer the configuration label from common run directory layouts."""
    if (path.parent / "suite_manifest.yaml").is_file():
        return re.sub(r"_\d+$", "", path.name) or path.name
    return path.parent.name or path.name


def build_runs_payload(
    run_dirs: list[str | Path],
    click_marker: str = "cross",
) -> dict[str, Any]:
    """Build the gallery payload from independent gameplay run directories."""
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    if not paths:
        raise ValueError("At least one standalone run directory is required")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate standalone run directories are not allowed")

    prepared: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        log_path = path / "gameplay_log.yaml"
        if not log_path.is_file():
            raise FileNotFoundError(f"gameplay_log.yaml not found in {path}")
        prepared.append((path, _read_yaml(log_path)))

    names = [_standalone_run_name(path) for path, _ in prepared]
    counts = Counter(names)
    seen: Counter[str] = Counter()
    runs: list[dict[str, Any]] = []
    for index, ((path, log), name) in enumerate(zip(prepared, names)):
        seen[name] += 1
        ticks = log.get("ticks") or []
        record = {
            "name": name,
            "repeat_index": seen[name],
            "repeat_count": counts[name],
            "status": "completed",
            "stage": "gameplay",
            "started_at": _tick_datetime(ticks[0] if ticks else None),
            "completed_at": _tick_datetime(ticks[-1] if ticks else None),
            "output_dir": path.name,
        }
        runs.append(_run_payload(f"run-{index}", path, record, log))

    starts = [run["startedAt"] for run in runs if run.get("startedAt")]
    completions = [run["completedAt"] for run in runs if run.get("completedAt")]
    common_path = Path(os.path.commonpath([str(path) for path in paths]))
    collection_name = (
        f"{runs[0]['name']} / {paths[0].name}"
        if len(runs) == 1
        else "Curated gameplay runs"
    )
    return {
        "suite": {
            "name": collection_name,
            "directory": str(common_path),
            "folderName": paths[0].name if len(paths) == 1 else f"{len(paths)} selected runs",
            "status": "completed",
            "startedAt": min(starts) if starts else None,
            "completedAt": max(completions) if completions else None,
            "repeatCount": len(runs),
            "config": "Standalone run collection",
            "sourceType": "runs",
            "compareEnabled": len(runs) > 1,
            "clickMarker": _validate_click_marker(click_marker),
        },
        "groups": _group_runs(runs),
        "runs": runs,
    }


@dataclass
class ViewerApplication:
    suite_dir: Path
    payload: dict[str, Any]
    _run_directories: dict[str, Path]

    @classmethod
    def load(
        cls,
        suite_dir: str | Path,
        click_marker: str = "cross",
    ) -> "ViewerApplication":
        suite_path = Path(suite_dir).expanduser().resolve()
        payload = build_suite_payload(suite_path, click_marker)
        run_directories = {
            run["id"]: Path(run["sourceDirectory"])
            for run in payload["runs"]
        }
        return cls(suite_path, payload, run_directories)

    @classmethod
    def load_runs(
        cls,
        run_dirs: list[str | Path],
        click_marker: str = "cross",
    ) -> "ViewerApplication":
        payload = build_runs_payload(run_dirs, click_marker)
        collection_path = Path(payload["suite"]["directory"])
        run_directories = {
            run["id"]: Path(run["sourceDirectory"])
            for run in payload["runs"]
        }
        return cls(collection_path, payload, run_directories)

    @property
    def run_directories(self) -> dict[str, Path]:
        return self._run_directories.copy()

    def open_folder(self, run_id: str | None = None) -> Path:
        target = self.suite_dir if not run_id else self.run_directories.get(run_id)
        if target is None or not target.is_dir():
            raise FileNotFoundError("Run folder not found")
        if os.name != "nt":
            raise OSError("Opening a folder is currently supported on Windows only")
        subprocess.Popen(["explorer.exe", str(target)])
        return target


def make_handler(application: ViewerApplication) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        server_version = "GameplayGallery/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

        def _send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "public, max-age=3600")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, value: Any, status: int = 200) -> None:
            self._send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/suite":
                self._json(application.payload)
                return
            if parsed.path.startswith("/media/"):
                self._serve_media(parsed.path)
                return
            asset_name = "index.html" if parsed.path in ("/", "/index.html") else parsed.path.lstrip("/")
            if asset_name not in {"index.html", "styles.css", "app.js"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = ASSET_DIR / asset_name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix in {".html", ".css", ".js"}:
                content_type += "; charset=utf-8"
            self._send_bytes(path.read_bytes(), content_type)

        def _serve_media(self, url_path: str) -> None:
            parts = url_path.split("/", 3)
            if len(parts) != 4:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            run_id, filename = unquote(parts[2]), unquote(parts[3])
            run_dir = application.run_directories.get(run_id)
            if run_dir is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            candidate = (run_dir / filename).resolve()
            if candidate.parent != run_dir or not candidate.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._send_bytes(candidate.read_bytes(), content_type)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/open-folder":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            run_id = (parse_qs(parsed.query).get("run") or [None])[0]
            try:
                target = application.open_folder(run_id)
            except (FileNotFoundError, OSError) as exc:
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "path": str(target)})

    return ViewerHandler


def serve_gameplay_suite(
    suite_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    click_marker: str = "cross",
) -> None:
    """Serve a gameplay suite until interrupted."""
    application = ViewerApplication.load(suite_dir, click_marker)
    _serve_application(application, host, port, open_browser)


def serve_gameplay_runs(
    run_dirs: list[str | Path],
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    click_marker: str = "cross",
) -> None:
    """Serve one or more independent gameplay runs until interrupted."""
    application = ViewerApplication.load_runs(run_dirs, click_marker)
    _serve_application(application, host, port, open_browser)


def _serve_application(
    application: ViewerApplication,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The viewer must bind to a loopback address")
    server = _bind_server(host, port, make_handler(application))
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"Gameplay gallery: {url}")
    source_label = "Suite" if application.payload["suite"]["sourceType"] == "suite" else "Runs"
    print(f"{source_label}: {application.suite_dir}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


def _bind_server(
    host: str,
    starting_port: int,
    handler: type[BaseHTTPRequestHandler],
) -> ViewerHTTPServer:
    """Bind on the requested port, trying higher ports when it is occupied."""
    if starting_port == 0:
        return ViewerHTTPServer((host, starting_port), handler)
    if not 1 <= starting_port <= 65535:
        raise ValueError("Viewer port must be between 1 and 65535")

    for port in range(starting_port, 65536):
        try:
            return ViewerHTTPServer((host, port), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE and getattr(exc, "winerror", None) != 10048:
                raise

    raise OSError(f"No available viewer port found at or above {starting_port}")


def build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse gameplay-suite results")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--suite-dir", help="Suite directory (default: newest suite)")
    sources.add_argument(
        "--run-dir",
        action="append",
        help="Standalone gameplay run directory; repeat for comparison",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--click-marker",
        choices=CLICK_MARKERS,
        default="cross",
        help="Click overlay shape (default: cross)",
    )
    return parser


def main() -> None:
    args = build_standalone_parser().parse_args()
    if args.run_dir:
        serve_gameplay_runs(
            args.run_dir,
            args.host,
            args.port,
            not args.no_browser,
            args.click_marker,
        )
    else:
        suite_dir = Path(args.suite_dir) if args.suite_dir else find_latest_suite()
        serve_gameplay_suite(
            suite_dir,
            args.host,
            args.port,
            not args.no_browser,
            args.click_marker,
        )


if __name__ == "__main__":
    main()
