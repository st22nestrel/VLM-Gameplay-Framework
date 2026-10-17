"""Analysis helpers for completed gameplay suite runs."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import yaml


def analyze_gameplay_suite(suite_dir: str | Path) -> Path:
    """Read a suite run directory and write suite_summary.yaml."""
    suite_path = Path(suite_dir)
    manifest_path = suite_path / "suite_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"suite_manifest.yaml not found in {suite_path}")

    manifest = _read_yaml(manifest_path)
    summary = {
        "suite_name": manifest.get("suite_name"),
        "suite_dir": str(suite_path),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest_path),
        "runs": [],
    }

    for run in manifest.get("runs", []):
        summary["runs"].append(_summarize_run(suite_path, run))

    out_path = suite_path / "suite_summary.yaml"
    with open(out_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    return out_path


def _summarize_run(suite_path: Path, manifest_run: dict[str, Any]) -> dict[str, Any]:
    name = manifest_run.get("name")
    status = manifest_run.get("status")
    gameplay_log_rel = manifest_run.get("gameplay_log")
    result: dict[str, Any] = {
        "name": name,
        "repeat_index": manifest_run.get("repeat_index"),
        "repeat_count": manifest_run.get("repeat_count"),
        "status": status,
        "error": manifest_run.get("error"),
        "backend": None,
        "model": None,
        "total_ticks": 0,
        "parsed_action_count": 0,
        "invalid_response_count": 0,
        "parsed_action_rate": 0.0,
        "inference_error_count": 0,
        "skipped_input_count": 0,
        "total_progress_count": None,
        "avg_inference_latency_ms": None,
        "max_inference_latency_ms": None,
        "blank_frame_count": None,
        "generation_params_strategy": None,
        "generation_params_total_prompts": 0,
        "generation_params_unique_count": 0,
        "generation_params_top": [],
        "output_dir": manifest_run.get("output_dir"),
        "gameplay_log": gameplay_log_rel,
    }

    if not gameplay_log_rel:
        return result

    log_path = suite_path / gameplay_log_rel
    if not log_path.exists():
        result["error"] = result.get("error") or f"gameplay_log not found: {log_path}"
        return result

    log = _read_yaml(log_path)
    ticks = log.get("ticks", []) or []
    total_ticks = int(log.get("total_ticks", len(ticks)))
    parsed_count = sum(1 for tick in ticks if tick.get("parsed_actions"))
    invalid_count = total_ticks - parsed_count
    latencies = [
        float(tick["inference_latency_ms"])
        for tick in ticks
        if _is_number(tick.get("inference_latency_ms"))
    ]
    generation_usage = log.get("generation_params_usage", {}) or {}

    result.update(
        {
            "backend": log.get("backend"),
            "model": log.get("model"),
            "total_ticks": total_ticks,
            "parsed_action_count": parsed_count,
            "invalid_response_count": invalid_count,
            "parsed_action_rate": (
                round(parsed_count / total_ticks, 4)
                if total_ticks > 0
                else 0.0
            ),
            "inference_error_count": sum(
                1 for tick in ticks if tick.get("inference_error")
            ),
            "skipped_input_count": sum(1 for tick in ticks if tick.get("skipped")),
            "total_progress_count": (
                int(log["total_progress_count"])
                if _is_number(log.get("total_progress_count"))
                else None
            ),
            "avg_inference_latency_ms": (
                round(mean(latencies), 1) if latencies else None
            ),
            "max_inference_latency_ms": max(latencies) if latencies else None,
            "blank_frame_count": _blank_frame_count(ticks),
            "generation_params_strategy": generation_usage.get("strategy"),
            "generation_params_total_prompts": int(
                generation_usage.get("total_prompts", 0)
            ),
            "generation_params_unique_count": int(
                generation_usage.get("unique_count", 0)
            ),
            "generation_params_top": generation_usage.get("top", []) or [],
        }
    )
    return result


def _blank_frame_count(ticks: list[dict[str, Any]]) -> Optional[int]:
    keys = ("blank", "is_blank", "pre_blank", "post_blank")
    found = False
    count = 0
    for tick in ticks:
        for key in keys:
            if key in tick:
                found = True
                if tick.get(key):
                    count += 1
                break
    return count if found else None


def _is_number(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}
