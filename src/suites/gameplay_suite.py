"""Live gameplay suite runner for comparing runtime/model configurations."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import yaml

from src.backends.factory import (
    create_backend_from_gameplay_config,
    loaded_backend,
)
from src.backends.base import VLMBackend
from src.loop.gameplay_config import (
    GameplayRuntimeConfig,
    deep_merge_config,
    load_gameplay_config,
)
from src.loop.generation_params import derive_suite_generation_params_seed
from src.loop.gameplay_loop import GameplayLoop
from src.window_session import LaunchedWindowSession, launch_and_attach


@dataclasses.dataclass(frozen=True)
class SuiteGameConfig:
    executable: Path
    title: Optional[str] = None
    args: tuple[str, ...] = ()
    cwd: Optional[Path] = None
    post_launch_wait_ms: int = 5000
    launch_timeout_ms: int = 20000
    close_timeout_ms: int = 5000
    terminate_after_close_timeout: bool = True
    focus: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SuiteGameConfig":
        executable = _required_path(data, "executable")
        args_raw = data.get("args", [])
        if args_raw is None:
            args_raw = []
        if not isinstance(args_raw, list):
            raise ValueError("game.args must be a list when provided")
        cwd_raw = data.get("cwd")
        return cls(
            executable=executable,
            title=_optional_str(data.get("title")),
            args=tuple(str(arg) for arg in args_raw),
            cwd=_resolve_path(cwd_raw) if cwd_raw else None,
            post_launch_wait_ms=_non_negative_int(
                data.get("post_launch_wait_ms", 5000),
                "game.post_launch_wait_ms",
            ),
            launch_timeout_ms=_positive_int(
                data.get("launch_timeout_ms", 20000),
                "game.launch_timeout_ms",
            ),
            close_timeout_ms=_positive_int(
                data.get("close_timeout_ms", 5000),
                "game.close_timeout_ms",
            ),
            terminate_after_close_timeout=_bool_value(
                data.get("terminate_after_close_timeout", True),
                True,
            ),
            focus=_bool_value(data.get("focus", True), True),
        )


@dataclasses.dataclass(frozen=True)
class SuiteRunPlan:
    name: str
    dir_name: str
    repeat_index: int
    repeat_count: int
    gameplay_config_path: Path
    merged_raw_config: dict[str, Any]
    runtime_config: GameplayRuntimeConfig
    game: SuiteGameConfig
    max_ticks: int
    generation_params_seed: Optional[int]


class GameplaySuiteRunner:
    """Execute a gameplay suite sequentially, restarting the game per run."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        selected_runs: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> None:
        self.config_path = _resolve_path(config_path)
        self.raw_suite = load_suite_config(self.config_path)
        self.selected_runs = selected_runs
        self.dry_run = dry_run
        self.suite_name = str(self.raw_suite.get("name", self.config_path.stem))
        self.output_root = _resolve_path(
            self.raw_suite.get("output_dir", "outputs/gameplay-suites")
        )
        self.debug_cfg = self.raw_suite.get("debug", {}) or {}
        self.execution_cfg = self.raw_suite.get("execution", {}) or {}
        self.fail_fast = _bool_value(self.execution_cfg.get("fail_fast", False), False)
        self.repeat_count = _positive_int(
            self.execution_cfg.get("repeats", 1),
            "execution.repeats",
        )
        self.validation_warnings: list[str] = []

    def validate(self) -> list[str]:
        """Return validation errors for the selected runs."""
        errors: list[str] = []
        self.validation_warnings = []
        try:
            self._selected_run_entries()
        except Exception as exc:
            return [str(exc)]

        for entry in self._selected_run_entries():
            try:
                plan = self._build_run_plan(entry, repeat_index=1)
            except Exception as exc:
                run_name = entry.get("name", "<unnamed>")
                errors.append(f"{run_name}: {exc}")
                continue
            self.validation_warnings.extend(self._run_plan_warnings(plan))
        return errors

    def run(self) -> Path:
        """Run the selected suite entries. Returns suite output directory."""
        errors = self.validate()
        if errors:
            raise ValueError("Suite validation failed:\n" + "\n".join(errors))

        suite_dir = self.output_root / _timestamp()
        suite_dir.mkdir(parents=True, exist_ok=False)

        manifest: dict[str, Any] = {
            "suite_name": self.suite_name,
            "suite_config": str(self.config_path),
            "started_at": _iso_now(),
            "completed_at": None,
            "status": "running",
            "dry_run": self.dry_run,
            "repeat_count": self.repeat_count,
            "output_dir": str(suite_dir),
            "selected_runs": [entry["name"] for entry in self._selected_run_entries()],
            "runs": [],
        }
        _write_yaml(suite_dir / "suite_manifest.yaml", manifest)

        interrupted = False
        for entry, repeat_index in self._expanded_run_entries():
            run_record = self._initial_run_record(
                entry,
                repeat_index=repeat_index,
            )
            manifest["runs"].append(run_record)
            _write_yaml(suite_dir / "suite_manifest.yaml", manifest)

            try:
                self._execute_run(
                    entry,
                    suite_dir,
                    run_record,
                    repeat_index=repeat_index,
                )
            except KeyboardInterrupt:
                interrupted = True
                run_record["status"] = "interrupted"
                run_record["completed_at"] = _iso_now()
                run_record["error"] = "Interrupted by user"
                manifest["status"] = "interrupted"
                _write_yaml(suite_dir / "suite_manifest.yaml", manifest)
                break
            except Exception as exc:
                run_record["status"] = "failed"
                run_record["completed_at"] = _iso_now()
                run_record["error"] = str(exc)
                run_record["traceback"] = traceback.format_exc()
                _write_yaml(suite_dir / "suite_manifest.yaml", manifest)
                if self.fail_fast:
                    break
            else:
                run_record["completed_at"] = run_record.get("completed_at") or _iso_now()
                _write_yaml(suite_dir / "suite_manifest.yaml", manifest)

        if not interrupted:
            statuses = [run.get("status") for run in manifest["runs"]]
            manifest["status"] = "failed" if any(s == "failed" for s in statuses) else "completed"
        manifest["completed_at"] = _iso_now()
        _write_yaml(suite_dir / "suite_manifest.yaml", manifest)

        if interrupted:
            raise KeyboardInterrupt
        return suite_dir

    def _selected_run_entries(self) -> list[dict[str, Any]]:
        runs = self.raw_suite.get("runs", [])
        if not isinstance(runs, list) or not runs:
            raise ValueError("Suite config must contain a non-empty runs list")

        seen_names: set[str] = set()
        seen_dirs: set[str] = set()
        by_name: dict[str, dict[str, Any]] = {}
        for entry in runs:
            if not isinstance(entry, dict):
                raise ValueError("Each runs item must be a mapping")
            name = str(entry.get("name", "")).strip()
            if not name:
                raise ValueError("Each run must have a non-empty name")
            if name in seen_names:
                raise ValueError(f"Duplicate run name: {name}")
            dir_name = sanitize_run_name(name)
            if not dir_name:
                raise ValueError(f"Run name has no filesystem-safe characters: {name!r}")
            if dir_name in seen_dirs:
                raise ValueError(
                    f"Run names sanitize to the same directory: {dir_name}"
                )
            seen_names.add(name)
            seen_dirs.add(dir_name)
            by_name[name] = entry

        if not self.selected_runs:
            return runs

        selected: list[dict[str, Any]] = []
        for name in self.selected_runs:
            if name not in by_name:
                raise ValueError(f"Selected run not found: {name}")
            selected.append(by_name[name])
        return selected

    def _expanded_run_entries(self) -> list[tuple[dict[str, Any], int]]:
        selected = self._selected_run_entries()
        return [
            (entry, repeat_index)
            for repeat_index in range(1, self.repeat_count + 1)
            for entry in selected
        ]

    def _run_dir_name(self, name: str, repeat_index: int) -> str:
        base = sanitize_run_name(name)
        if self.repeat_count <= 1:
            return base
        width = max(2, len(str(self.repeat_count)))
        return f"{base}_{repeat_index:0{width}d}"

    def _build_run_plan(
        self,
        entry: dict[str, Any],
        *,
        repeat_index: int,
    ) -> SuiteRunPlan:
        name = str(entry["name"]).strip()
        gameplay_config_value = entry.get("gameplay_config")
        if not gameplay_config_value:
            raise ValueError("run.gameplay_config is required")
        gameplay_config_path = _resolve_path(gameplay_config_value)
        if not gameplay_config_path.exists():
            raise FileNotFoundError(f"gameplay_config not found: {gameplay_config_path}")

        raw_gameplay = load_gameplay_config(gameplay_config_path)
        defaults = self.raw_suite.get("defaults", {}) or {}
        if not isinstance(defaults, dict):
            raise ValueError("defaults must be a mapping")
        merged = deep_merge_config(raw_gameplay, defaults.get("overrides", {}))
        merged = deep_merge_config(merged, entry.get("overrides", {}))
        runtime_config = GameplayRuntimeConfig.from_raw(merged)
        _validate_lora_path(runtime_config)
        create_backend_from_gameplay_config(runtime_config)

        game_raw = deep_merge_config(
            self.raw_suite.get("game", {}) or {},
            entry.get("game", {}) or {},
        )
        game = SuiteGameConfig.from_dict(game_raw)
        _validate_game_config(game)

        max_ticks = entry.get("max_ticks", defaults.get("max_ticks", 20))
        max_ticks = _positive_int(max_ticks, f"runs.{name}.max_ticks")

        return SuiteRunPlan(
            name=name,
            dir_name=self._run_dir_name(name, repeat_index),
            repeat_index=repeat_index,
            repeat_count=self.repeat_count,
            gameplay_config_path=gameplay_config_path,
            merged_raw_config=merged,
            runtime_config=runtime_config,
            game=game,
            max_ticks=max_ticks,
            generation_params_seed=derive_suite_generation_params_seed(
                runtime_config.model_settings.generation_params,
                name,
                repeat_index,
            ),
        )

    def _run_plan_warnings(self, plan: SuiteRunPlan) -> list[str]:
        config = plan.runtime_config.model_settings.generation_params
        warnings = [f"{plan.name}: {warning}" for warning in config.warnings]
        backend = create_backend_from_gameplay_config(plan.runtime_config)
        unsupported = sorted(
            config.configured_param_names - set(backend.supported_generation_params)
        )
        if unsupported:
            warnings.append(
                f"{plan.name}: backend {backend.backend_name} does not support "
                f"generation parameter(s): {', '.join(unsupported)}; "
                "they will be ignored"
            )
        return warnings

    def _initial_run_record(
        self,
        entry: dict[str, Any],
        *,
        repeat_index: int,
    ) -> dict[str, Any]:
        name = str(entry["name"]).strip()
        dir_name = self._run_dir_name(name, repeat_index)
        return {
            "name": name,
            "repeat_index": repeat_index,
            "repeat_count": self.repeat_count,
            "status": "pending",
            "stage": "validate",
            "started_at": None,
            "completed_at": None,
            "output_dir": str(Path(dir_name)),
            "gameplay_log": None,
            "game": {},
        }

    def _execute_run(
        self,
        entry: dict[str, Any],
        suite_dir: Path,
        run_record: dict[str, Any],
        *,
        repeat_index: int,
    ) -> None:
        plan = self._build_run_plan(entry, repeat_index=repeat_index)
        run_dir = suite_dir / plan.dir_name
        run_dir.mkdir(parents=True, exist_ok=False)

        run_record["status"] = "running"
        run_record["started_at"] = _iso_now()
        run_record["output_dir"] = str(Path(plan.dir_name))
        run_record["gameplay_config"] = str(plan.gameplay_config_path)
        run_record["max_ticks"] = plan.max_ticks

        if _bool_value(self.debug_cfg.get("dump_resolved_gameplay_config", False), False):
            resolved_path = run_dir / "resolved_gameplay.yaml"
            _write_yaml(resolved_path, plan.merged_raw_config)
            run_record["resolved_gameplay_config"] = str(Path(plan.dir_name) / resolved_path.name)

        backend: Optional[VLMBackend] = None
        launched: Optional[LaunchedWindowSession] = None
        try:
            run_record["stage"] = "model_load"
            backend = create_backend_from_gameplay_config(plan.runtime_config)
            with loaded_backend(backend, announce=True):
                run_record["stage"] = "game_launch"
                launched = launch_and_attach(
                    plan.game.executable,
                    args=list(plan.game.args),
                    title_hint=plan.game.title,
                    timeout_s=plan.game.launch_timeout_ms / 1000.0,
                    cwd=plan.game.cwd,
                    focus=plan.game.focus,
                )
                run_record["game"] = {
                    "executable": str(plan.game.executable),
                    "pid": launched.process.pid,
                    "hwnd": f"{launched.session.target.hwnd:#010x}",
                    "title": launched.session.target.title,
                }

                run_record["stage"] = "post_launch_wait"
                if plan.game.post_launch_wait_ms > 0:
                    time.sleep(plan.game.post_launch_wait_ms / 1000.0)
                if plan.game.focus:
                    launched.focus()

                run_record["stage"] = "gameplay"
                loop = GameplayLoop(
                    session=launched.session,
                    config=plan.runtime_config,
                    backend=backend,
                    output_dir=run_dir,
                    dry_run=self.dry_run,
                    no_wait=True,
                    max_ticks=plan.max_ticks,
                    delay_ms=0,
                    generation_params_seed=plan.generation_params_seed,
                    propagate_keyboard_interrupt=True,
                )
                try:
                    gameplay_log = loop.run()
                except KeyboardInterrupt:
                    partial_log = run_dir / "gameplay_log.yaml"
                    if partial_log.exists():
                        run_record["gameplay_log"] = str(
                            Path(plan.dir_name) / partial_log.name
                        )
                    raise
                run_record["gameplay_log"] = str(Path(plan.dir_name) / gameplay_log.name)
                run_record["status"] = "completed"
        finally:
            if launched is not None:
                close_result = launched.close(
                    timeout_s=plan.game.close_timeout_ms / 1000.0,
                    terminate_after=plan.game.terminate_after_close_timeout,
                )
                run_record.setdefault("game", {})["close"] = dataclasses.asdict(
                    close_result
                )
            if backend is not None:
                run_record["cleanup"] = "model_unload"
                # loaded_backend unloads backends it loaded. This stage is kept
                # explicit in the manifest for failure triage.
            if run_record.get("status") == "running":
                run_record["status"] = "failed"
            run_record["completed_at"] = _iso_now()


def load_suite_config(path: str | Path) -> dict[str, Any]:
    with open(_resolve_path(path), "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def parse_run_selectors(values: Optional[list[str]]) -> Optional[list[str]]:
    if not values:
        return None
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            name = part.strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names or None


def sanitize_run_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.strip("._ ")


def _validate_game_config(game: SuiteGameConfig) -> None:
    if not game.executable.exists():
        raise FileNotFoundError(f"game.executable not found: {game.executable}")
    if not game.executable.is_file():
        raise ValueError(f"game.executable is not a file: {game.executable}")
    if game.cwd is not None and not game.cwd.exists():
        raise FileNotFoundError(f"game.cwd not found: {game.cwd}")


def _validate_lora_path(config: GameplayRuntimeConfig) -> None:
    adapter = config.model_settings.lora_adapter_path
    if not adapter or not _looks_like_local_path(adapter):
        return
    path = _resolve_path(adapter)
    if not path.exists():
        raise FileNotFoundError(f"lora_adapter_path not found: {path}")


def _looks_like_local_path(value: str) -> bool:
    expanded = os.path.expanduser(os.path.expandvars(value))
    path = Path(expanded)
    if path.is_absolute():
        return True
    if expanded.startswith((".", "~")):
        return True
    if "\\" in expanded:
        return True
    if "/" in expanded:
        first = expanded.split("/", 1)[0]
        return Path(first).exists()
    return False


def _required_path(data: dict[str, Any], key: str) -> Path:
    value = data.get(key)
    if not value:
        raise ValueError(f"game.{key} is required")
    return _resolve_path(value)


def _resolve_path(value: str | Path) -> Path:
    text = os.path.expanduser(os.path.expandvars(str(value)))
    return Path(text)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
