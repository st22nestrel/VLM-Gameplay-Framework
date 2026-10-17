"""CLI entry point for the VLM gameplay framework."""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load environment variables from a .env file at the project root (e.g.
# API keys for the openai backend). Real environment variables already set
# take precedence over values from .env.
try:
    from dotenv import load_dotenv

    load_dotenv(_project_root / ".env")
except ImportError:
    pass

from src.window_session import enumerate_windows, find_window_by_title, attach, launch_and_attach
from src.capture import windows_graphics_capture, fallback_capture, dxcam_capture
from src.capture.cursor_overlay import normalize_cursor_mode, save_frame_screenshots
from src.capture.frame import CapturedFrame
from src.actions.schema import load_action_config
from src.actions.parser import parse_vlm_text
from src.suites.gameplay_suite import GameplaySuiteRunner, parse_run_selectors
from src.suites.gameplay_suite_analysis import analyze_gameplay_suite
from src.suites.gameplay_suite_viewer import (
    find_latest_suite,
    serve_gameplay_runs,
    serve_gameplay_suite,
)
from src.loop.gameplay_loop import GameplayLoop
from src.loop.gameplay_config import GameplayRuntimeConfig, load_gameplay_config
from src.backends.factory import create_backend_from_gameplay_config, loaded_backend


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── list-windows ──────────────────────────────────────────────────────────────


def cmd_list_windows(args: argparse.Namespace) -> None:
    windows = enumerate_windows()
    if not windows:
        print("No visible windows found.")
        return
    print(f"Found {len(windows)} visible window(s):\n")
    for w in windows:
        print(f"  {w.short_str()}")


# --- launch-window -----------------------------------------------------------


def cmd_launch_window(args: argparse.Namespace) -> None:
    """Launch a game executable, focus its window, hold for N ticks, then close."""
    timeout_s = max(0.1, args.timeout / 1000.0)
    tick_count = max(0, args.ticks)
    tick_interval_s = max(0, args.tick_interval) / 1000.0
    close_timeout_s = max(0.1, args.close_timeout / 1000.0)

    try:
        launched = launch_and_attach(
            args.exe,
            title_hint=args.title,
            timeout_s=timeout_s,
            focus=not args.no_focus,
        )
    except Exception as exc:
        print(f"ERROR: Could not launch and attach: {exc}")
        sys.exit(1)

    print(f"Launched: {launched.executable}")
    print(f"Process:  pid={launched.process.pid}")
    print(f"Target:   {launched.session.target.short_str()}")
    print(f"Focused:  {launched.session.is_foreground()}")

    try:
        for tick_idx in range(tick_count):
            if args.refocus_each_tick:
                launched.focus()
            valid = launched.session.is_valid()
            foreground = launched.session.is_foreground() if valid else False
            print(
                f"Tick {tick_idx + 1}/{tick_count}: "
                f"valid={valid} foreground={foreground}"
            )
            if tick_idx < tick_count - 1 and tick_interval_s > 0:
                time.sleep(tick_interval_s)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.no_close:
            print("Leaving launched process/window open (--no-close).")
        else:
            result = launched.close(
                timeout_s=close_timeout_s,
                terminate_after=not args.no_terminate,
            )
            print(
                "Close:    "
                f"window_closed={result.window_closed} "
                f"process_exited={result.process_exited} "
                f"terminated={result.terminated}"
            )


# ── capture-window ────────────────────────────────────────────────────────────


def _resolve_target(args: argparse.Namespace):
    """Resolve a target window from --hwnd or --title."""
    if args.hwnd:
        hwnd = int(args.hwnd, 0)  # accepts 0x… or plain int
        session = attach(hwnd)
        if session is None:
            print(f"ERROR: Could not attach to HWND {args.hwnd}")
            sys.exit(1)
        return session

    if args.title:
        matches = find_window_by_title(args.title)
        if not matches:
            print(f'ERROR: No window found matching "{args.title}"')
            sys.exit(1)
        if len(matches) > 1:
            print(f'Multiple windows match "{args.title}":')
            for w in matches:
                print(f"  {w.short_str()}")
            print("\nPlease use --hwnd to select one.")
            sys.exit(1)
        session = attach(matches[0].hwnd)
        if session is None:
            print("ERROR: Could not attach to the matched window")
            sys.exit(1)
        session._title_hint = args.title
        return session

    print("ERROR: Specify --hwnd or --title to identify the target window.")
    sys.exit(1)


def _do_capture(session, method: str) -> CapturedFrame | None:
    """Perform a single capture using the specified method."""
    if method == "dxcam":
        return dxcam_capture.capture_frame(session)
    if method == "mss":
        return fallback_capture.capture_frame(session)
    if method in ("printwindow", "bitblt"):
        return windows_graphics_capture.capture_frame(session)
    # auto: dxcam -> printwindow/bitblt -> mss
    if dxcam_capture.is_available():
        frame = dxcam_capture.capture_frame(session)
        if frame is not None:
            return frame
    frame = windows_graphics_capture.capture_frame(session)
    if frame is not None:
        return frame
    return fallback_capture.capture_frame(session)


def cmd_capture_window(args: argparse.Namespace) -> None:
    session = _resolve_target(args)
    method = getattr(args, "method", "auto") or "auto"
    cursor_mode = normalize_cursor_mode(getattr(args, "cursor_mode", "both"), default="both")

    # Build output directory
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_id if args.out else Path("outputs/captures") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    count = max(1, args.count)
    interval = max(0, args.interval) / 1000.0  # ms -> seconds

    print(f"Target: {session.target.short_str()}")
    print(
        f"Capturing {count} frame(s), interval={args.interval}ms, "
        f"method={method}, cursor_mode={cursor_mode}"
    )
    print(f"Output: {out_dir}\n")

    delay_ms = args.delay if args.delay is not None else 0
    if delay_ms > 0:
        delay_sec = delay_ms / 1000.0
        print(f"Starting in {delay_sec:.1f}s — switch to game window!")
        time.sleep(delay_sec)

    metadata_list = []
    for i in range(count):
        frame = _do_capture(session, method)
        if frame is None:
            print(f"  Frame {i+1}/{count}: FAILED (no image)")
            continue

        filename = f"frame_{i:04d}.png"
        screenshot_result = save_frame_screenshots(
            frame,
            out_dir / filename,
            cursor_mode=cursor_mode,
        )
        metadata = frame.to_metadata()
        metadata["saved_path_raw"] = (
            str(screenshot_result.raw_path)
            if screenshot_result.raw_path is not None
            else None
        )
        metadata["saved_path_cursor"] = (
            str(screenshot_result.cursor_path)
            if screenshot_result.cursor_path is not None
            else None
        )
        if screenshot_result.cursor_metadata is not None:
            metadata["cursor_overlay"] = screenshot_result.cursor_metadata.to_dict()
        metadata_list.append(metadata)

        blank_tag = " [BLANK]" if frame.is_blank() else ""
        variants = []
        if screenshot_result.raw_path is not None:
            variants.append(screenshot_result.raw_path.name)
        if screenshot_result.cursor_path is not None:
            variants.append(screenshot_result.cursor_path.name)
        variant_text = ", ".join(variants) or screenshot_result.selected_path.name
        print(
            f"  Frame {i+1}/{count}: {variant_text}  "
            f"{frame.width}x{frame.height}  "
            f"method={frame.capture_method}{blank_tag}"
        )

        if i < count - 1 and interval > 0:
            time.sleep(interval)

    # Write metadata YAML
    try:
        import yaml
        meta_path = out_dir / "capture_metadata.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(
                {
                    "run_id": run_id,
                    "target": session.target.short_str(),
                    "hwnd": f"{session.target.hwnd:#010x}",
                    "method": method,
                    "cursor_mode": cursor_mode,
                    "frame_count": len(metadata_list),
                    "frames": metadata_list,
                },
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"\nMetadata saved to {meta_path}")
    except ImportError:
        pass

    print(f"Done. {len(metadata_list)} frame(s) saved to {out_dir}")


# ── parse-text ───────────────────────────────────────────────────────────────


def cmd_parse_text(args: argparse.Namespace) -> None:
    """Parse VLM-style text into canonical actions (no execution)."""
    config_path = args.action_config or str(_project_root / "configs/actions/default-actions_chess.yaml")
    config = load_action_config(config_path)

    text = args.text
    print(f"Input text: {text!r}\n")

    actions = parse_vlm_text(text, config)
    if not actions:
        print("No actions parsed.")
        return

    print(f"Parsed {len(actions)} action(s):")
    for a in actions:
        print(f"  {a}")


def cmd_gameplay(args: argparse.Namespace) -> None:
    """Run the live VLM gameplay loop against a target window."""
    session = _resolve_target(args)
    out_dir = Path(args.out) if args.out else None
    max_ticks = args.max_ticks if args.max_ticks else None
    config_path = args.config or str(_project_root / "configs/runtime/gameplay.yaml")

    delay_ms = args.delay if args.delay is not None else None
    raw_config = load_gameplay_config(config_path)
    runtime_config = GameplayRuntimeConfig.from_raw(
        raw_config,
        lora_base_model_source_override=args.lora_base_model_source,
    )
    backend = create_backend_from_gameplay_config(runtime_config)

    with loaded_backend(backend, announce=True):
        loop = GameplayLoop(
            session=session,
            config=runtime_config,
            output_dir=out_dir,
            dry_run=args.dry_run,
            no_wait=args.no_wait,
            max_ticks=max_ticks,
            delay_ms=delay_ms,
            backend=backend,
        )
        loop.run()


# --- gameplay-suite ----------------------------------------------------------


def cmd_gameplay_suite(args: argparse.Namespace) -> None:
    """Run a live gameplay suite, restarting the game for each selected run."""
    selected_runs = parse_run_selectors(args.run)
    runner = GameplaySuiteRunner(
        config_path=args.config,
        selected_runs=selected_runs,
        dry_run=args.dry_run,
    )
    errors = runner.validate()
    if runner.validation_warnings:
        print("Suite validation warnings:")
        for warning in runner.validation_warnings:
            print(f"  - {warning}")
    if args.validate_only:
        if errors:
            print("Suite validation failed:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)
        selected = selected_runs or ["(all runs)"]
        print(f"Suite validation passed: {', '.join(selected)}")
        return

    if errors:
        print("Suite validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    try:
        suite_dir = runner.run()
    except KeyboardInterrupt:
        print("\nGameplay suite interrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: Gameplay suite failed: {exc}")
        sys.exit(1)

    print(f"\nGameplay suite complete. Output: {suite_dir}")


def cmd_analyze_gameplay_suite(args: argparse.Namespace) -> None:
    """Analyze a completed gameplay suite directory."""
    try:
        summary_path = analyze_gameplay_suite(args.suite_dir)
    except Exception as exc:
        print(f"ERROR: Could not analyze gameplay suite: {exc}")
        sys.exit(1)
    print(f"Suite summary written: {summary_path}")


def cmd_view_gameplay_suite(args: argparse.Namespace) -> None:
    """Launch the local gameplay-suite gallery."""
    try:
        if args.run_dir:
            serve_gameplay_runs(
                run_dirs=parse_run_selectors(args.run_dir),
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
                click_marker=args.click_marker,
            )
        else:
            suite_dir = Path(args.suite_dir) if args.suite_dir else find_latest_suite()
            serve_gameplay_suite(
                suite_dir=suite_dir,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
                click_marker=args.click_marker,
            )
    except Exception as exc:
        print(f"ERROR: Could not launch gameplay suite viewer: {exc}")
        sys.exit(1)


# ── discover-actions ─────────────────────────────────────────────────────────


def cmd_discover_actions(args: argparse.Namespace) -> None:
    """Discover action key bindings from a game's settings menu via VLM."""
    import yaml as _yaml
    from src.loop.discovery_loop import DiscoveryLoop

    session = _resolve_target(args)
    config_path = args.config or str(_project_root / "configs/runtime/discovery.yaml")

    # Validate config before creating the loop
    try:
        with open(config_path, "r") as f:
            config = _yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    except _yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in config: {e}")
        sys.exit(1)

    disc = config.get("discovery", {})

    # Validate prompt-type matches a key in system_prompts
    prompt_type = args.prompt_type
    system_prompts = disc.get("system_prompts", {})
    if prompt_type not in system_prompts:
        available = ", ".join(sorted(system_prompts.keys())) or "(none)"
        print(f"ERROR: --prompt-type '{prompt_type}' not found in config. "
              f"Available: {available}")
        sys.exit(1)

    user_prompts = disc.get("user_prompts", {})
    if prompt_type not in user_prompts:
        print(f"ERROR: --prompt-type '{prompt_type}' not found in user_prompts config.")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else None
    max_ticks = args.max_ticks if args.max_ticks else None
    delay_ms = args.delay if args.delay is not None else None

    loop = DiscoveryLoop(
        session=session,
        config_path=config_path,
        prompt_type=prompt_type,
        output_dir=out_dir,
        dry_run=args.dry_run,
        no_wait=args.no_wait,
        max_ticks=max_ticks,
        delay_ms=delay_ms,
        no_exit_keywords=args.no_exit_keywords,
        verbose=args.verbose,
        lora_base_model_source=args.lora_base_model_source,
    )
    loop.run()


# ── argument parser ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm-game",
        description="VLM gameplay framework CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    # list-windows
    sub.add_parser("list-windows", help="List all visible windows")

    # launch-window
    launch = sub.add_parser(
        "launch-window",
        help="Launch an executable, attach to its window, focus it, and close it",
    )
    launch.add_argument("--exe", type=str, required=True, help="Executable path to launch")
    launch.add_argument(
        "--title",
        type=str,
        help="Optional title substring to identify the launched top-level window",
    )
    launch.add_argument(
        "--ticks",
        type=int,
        default=5,
        help="Number of probe ticks to keep the window open (default: 5)",
    )
    launch.add_argument(
        "--tick-interval",
        type=int,
        default=1000,
        help="Interval between probe ticks in ms (default: 1000)",
    )
    launch.add_argument(
        "--timeout",
        type=int,
        default=15000,
        help="Window attach timeout in ms (default: 15000)",
    )
    launch.add_argument("--no-focus", action="store_true", help="Do not focus after attach")
    launch.add_argument(
        "--refocus-each-tick",
        action="store_true",
        help="Bring the window to foreground before every probe tick",
    )
    launch.add_argument("--no-close", action="store_true", help="Leave the process/window open")
    launch.add_argument(
        "--close-timeout",
        type=int,
        default=5000,
        help="WM_CLOSE wait timeout in ms before fallback handling (default: 5000)",
    )
    launch.add_argument(
        "--no-terminate",
        action="store_true",
        help="Do not terminate the process if WM_CLOSE does not exit it",
    )

    # capture-window
    cap = sub.add_parser("capture-window", help="Capture frames from a target window")
    cap.add_argument("--hwnd", type=str, help="Target window handle (hex or int)")
    cap.add_argument("--title", type=str, help="Substring to match in window title")
    cap.add_argument("--out", type=str, help="Output directory (default: outputs/captures)")
    cap.add_argument("--method", choices=["auto", "dxcam", "printwindow", "bitblt", "mss"], default="auto",
                     help="Capture method (default: auto)")
    cap.add_argument("--cursor-mode", choices=["raw", "cursor", "both"], default="both",
                     help="Screenshot variants to save: raw, cursor, or both (default: both)")
    cap.add_argument("--count", type=int, default=1, help="Number of frames to capture (default: 1)")
    cap.add_argument("--interval", type=int, default=500, help="Interval between frames in ms (default: 500)")
    cap.add_argument("--delay", type=int, default=None,
                     help="Delay before capture starts in ms (default: 0)")

    # parse-text
    pt = sub.add_parser("parse-text", help="Parse VLM text into canonical actions (no execution)")
    pt.add_argument("text", type=str, help="VLM text to parse")
    pt.add_argument("--action-config", type=str, help="Path to action config YAML")

    # gameplay
    gp = sub.add_parser("gameplay", help="Run the live VLM gameplay loop")
    gp.add_argument("--hwnd", type=str, help="Target window handle (hex or int)")
    gp.add_argument("--title", type=str, help="Substring to match in window title")
    gp.add_argument("--config", type=str,
                     help="Path to runtime config YAML (default: configs/runtime/gameplay.yaml)")
    gp.add_argument("--out", type=str, help="Output directory (default: outputs/gameplay-runs/<run-id>)")
    gp.add_argument("--dry-run", action="store_true", help="Parse and plan but don't send input")
    gp.add_argument("--no-wait", action="store_true",
                     help="Skip the wait-for-ready prompt")
    gp.add_argument("--max-ticks", type=int, default=None,
                     help="Stop after N ticks (default: unlimited)")
    gp.add_argument("--delay", type=int, default=None,
                     help="Delay before action execution starts in ms (default: config or 3000)")
    gp.add_argument(
        "--lora-base-model-source",
        choices=["adapter", "config"],
        default=None,
        help=(
            "For LoRA adapters, choose the base model from adapter metadata "
            "or from the config model field (default: config value or adapter)"
        ),
    )

    # gameplay-suite
    gs = sub.add_parser(
        "gameplay-suite",
        help="Run a YAML-defined live gameplay suite sequentially",
    )
    gs.add_argument("--config", type=str, required=True, help="Path to suite YAML config")
    gs.add_argument(
        "--run",
        action="append",
        help="Run name(s) to execute, repeatable and comma-separated",
    )
    gs.add_argument("--dry-run", action="store_true", help="Run inference but don't send input")
    gs.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate selected suite runs without creating outputs",
    )

    # analyze-gameplay-suite
    ags = sub.add_parser(
        "analyze-gameplay-suite",
        help="Create suite_summary.yaml for a completed gameplay suite run",
    )
    ags.add_argument("--suite-dir", type=str, required=True, help="Suite output directory")

    # view-gameplay-suite
    vgs = sub.add_parser(
        "view-gameplay-suite",
        help="Launch the local visual gallery for gameplay suite results",
    )
    viewer_sources = vgs.add_mutually_exclusive_group()
    viewer_sources.add_argument(
        "--suite-dir",
        type=str,
        help="Suite output directory (default: newest suite)",
    )
    viewer_sources.add_argument(
        "--run-dir",
        action="append",
        help="Standalone gameplay run directory; repeatable and comma-separated",
    )
    vgs.add_argument("--host", default="127.0.0.1", help="Viewer bind address")
    vgs.add_argument(
        "-p",
        "--port",
        type=int,
        default=8765,
        help="Starting viewer port (default: 8765; tries higher ports if occupied)",
    )
    vgs.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser",
    )
    vgs.add_argument(
        "--click-marker",
        choices=("cross", "circle"),
        default="cross",
        help="Click overlay shape (default: cross)",
    )

    # discover-actions
    da = sub.add_parser("discover-actions",
                         help="Discover action key bindings from a game's settings menu via VLM")
    da.add_argument("--hwnd", type=str, help="Target window handle (hex or int)")
    da.add_argument("--title", type=str, help="Substring to match in window title")
    da.add_argument("--config", type=str,
                     help="Path to discovery config YAML (default: configs/runtime/discovery.yaml)")
    da.add_argument("--prompt-type", choices=["specific", "universal"], default="specific",
                     help="Which prompt set to use (default: specific)")
    da.add_argument("--out", type=str,
                     help="Output directory (default: outputs/action-discovery/<run-id>)")
    da.add_argument("--dry-run", action="store_true", help="Parse VLM output but don't send input")
    da.add_argument("--no-wait", action="store_true",
                     help="Skip the wait-for-ready prompt")
    da.add_argument("--delay", type=int, default=None,
                     help="Delay before execution starts in ms (default: config or 3000)")
    da.add_argument("--max-ticks", type=int, default=None,
                     help="Stop after N ticks (default: unlimited, runs until exit condition)")
    da.add_argument("--no-exit-keywords", action="store_true",
                     help="Ignore exit keywords in VLM responses (stop only via F1+Ctrl+C or --max-ticks)")
    da.add_argument(
        "--lora-base-model-source",
        choices=["adapter", "config"],
        default=None,
        help=(
            "For LoRA adapters, choose the base model from adapter metadata "
            "or from the config model field (default: config value or adapter)"
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    if args.command == "list-windows":
        cmd_list_windows(args)
    elif args.command == "launch-window":
        cmd_launch_window(args)
    elif args.command == "capture-window":
        cmd_capture_window(args)
    elif args.command == "parse-text":
        cmd_parse_text(args)
    elif args.command == "gameplay":
        cmd_gameplay(args)
    elif args.command == "gameplay-suite":
        cmd_gameplay_suite(args)
    elif args.command == "analyze-gameplay-suite":
        cmd_analyze_gameplay_suite(args)
    elif args.command == "view-gameplay-suite":
        cmd_view_gameplay_suite(args)
    elif args.command == "discover-actions":
        cmd_discover_actions(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
