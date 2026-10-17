"""VLM action discovery loop: capture -> infer -> parse -> navigate/accumulate -> repeat.

Captures frames from a game's settings menu, sends them to the VLM with
discovery prompts, and routes responses to either navigation commands (clicks)
or action mapping accumulation. Stops when exit keywords are detected,
max ticks reached, or the user presses Ctrl+C.
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

from src.window_session import WindowSession
from src.capture.cursor_overlay import (
    cursor_mode_from_config,
    save_frame_screenshots,
    vlm_image_mode_from_config,
)
from src.capture.frame import CapturedFrame
from src.actions.executor import ActionExecutor
from src.backends.base import VLMBackend
from src.discovery.parser import (
    DiscoveredMapping,
    DiscoveryParseResult,
    parse_discovery_response,
)
from src.loop.coord_utils import normalize_xy_from_0_1000
from src.loop.frame_queue import FrameQueue
from src.loop.gameplay_loop import _do_capture
from src.loop.hotkey import PauseController

logger = logging.getLogger(__name__)


class DiscoveryLoop:
    """VLM action discovery loop.

    Parameters
    ----------
    session : WindowSession
        Attached target window.
    config_path : str | Path
        Path to the runtime YAML config (discovery.yaml).
    prompt_type : str
        Key selecting which system/user prompt pair to use.
    output_dir : Path | None
        Override output directory.
    dry_run : bool
        If True, parse VLM output but don't send input.
    no_wait : bool
        Skip the wait-for-ready prompt.
    max_ticks : int | None
        Stop after this many ticks (None = unlimited).
    delay_ms : int | None
        Delay before execution starts (ms). None = use config.
    backend : VLMBackend | None
        Inject a backend (useful for testing).
    """

    def __init__(
        self,
        session: WindowSession,
        config_path: str | Path = "configs/runtime/discovery.yaml",
        prompt_type: str = "specific",
        output_dir: Optional[Path] = None,
        dry_run: bool = False,
        no_wait: bool = False,
        max_ticks: Optional[int] = None,
        delay_ms: Optional[int] = None,
        no_exit_keywords: bool = False,
        verbose: bool = False,
        lora_base_model_source: Optional[str] = None,
        backend: Optional[VLMBackend] = None,
    ) -> None:
        self.verbose = verbose
        self.session = session
        with open(Path(config_path), "r") as f:
            raw = yaml.safe_load(f)
        self.raw_config = raw

        disc = raw.get("discovery", {})

        # Backend config
        self._backend_name = disc.get("backend", "transformers")
        self._model_id = disc.get("model", "Qwen/Qwen3-VL-2B-Instruct")
        self._max_new_tokens = disc.get("max_new_tokens", 1024)

        # Model-level settings (backend-agnostic).
        # Prefer generic model_settings, but keep legacy gemma4 block for
        # compatibility with existing configs.
        model_settings = disc.get("model_settings", {})
        gemma4_cfg = disc.get("gemma4", {})

        def _ms_get(key: str, default=None):
            return model_settings.get(key, gemma4_cfg.get(key, default))

        self._enable_thinking = bool(_ms_get("enable_thinking", False))
        self._temperature = _ms_get("temperature")
        self._top_p = _ms_get("top_p")
        self._top_k = _ms_get("top_k")
        self._normalize_coordinates_from_0_1000 = bool(
            _ms_get("normalize_coordinates_from_0_1000", False)
        )
        self._lora_adapter_path = _ms_get("lora_adapter_path")
        self._lora_base_model_source = (
            lora_base_model_source
            if lora_base_model_source is not None
            else _ms_get("lora_base_model_source", "adapter")
        )

        self._llama_cpp_cfg = disc.get("llama_cpp", {})
        self._openai_cfg = disc.get("openai", {})

        # Prompts
        self.prompt_type = prompt_type
        system_prompts = disc.get("system_prompts", {})
        user_prompts = disc.get("user_prompts", {})
        self.system_prompt = system_prompts.get(prompt_type, "").strip()
        self.user_prompt = user_prompts.get(prompt_type, "").strip()

        # Capture config
        capture_cfg = raw.get("capture", {})
        self.capture_method = capture_cfg.get("method", "auto")
        self.cursor_mode = cursor_mode_from_config(capture_cfg, default="both")
        self.vlm_image_mode = vlm_image_mode_from_config(capture_cfg, default="cursor")
        self.frame_cadence_ms = disc.get("frame_cadence_ms", 2000)

        # Exit keywords (empty list = disabled)
        if no_exit_keywords:
            self.exit_keywords: list[str] = []
        else:
            self.exit_keywords = disc.get("exit_keywords", [
                "no more mappings", "discovery complete",
                "all mappings found", "no additional mappings",
            ])

        # Repetition detection: stop after N consecutive identical responses
        self.repetition_limit: int = disc.get("repetition_limit", 3)

        # Pause/resume hotkey
        self.pause_key: Optional[str] = disc.get("pause_key", "f1")
        self._pause_controller: Optional[PauseController] = None

        # Runtime options
        self.dry_run = dry_run
        self.no_wait = no_wait
        self.max_ticks = max_ticks

        # Delay: CLI > config > default 3000
        if delay_ms is not None:
            self.delay_ms = delay_ms
        elif disc.get("delay_ms") is not None:
            self.delay_ms = int(disc["delay_ms"])
        else:
            self.delay_ms = 3000

        # Backend
        if backend is not None:
            self._backend = backend
        else:
            self._backend = self._create_backend()

        # Output directory
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            base = disc.get("output_dir", "outputs/action-discovery")
            self.output_dir = Path(base) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frame queue and capture thread
        self._frame_queue = FrameQueue()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # Discovery state
        self._discovered_mappings: list[DiscoveredMapping] = []
        self._tick_log: list[dict] = []
        self._navigation_count: int = 0
        self._should_exit: bool = False
        self._recent_responses: list[str] = []  # for repetition detection

    def _create_backend(self) -> VLMBackend:
        if self._backend_name == "transformers":
            from src.backends.transformers_backend import TransformersBackend

            return TransformersBackend(
                model_id=self._model_id,
                max_new_tokens=self._max_new_tokens,
                enable_thinking=self._enable_thinking,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                lora_adapter_path=self._lora_adapter_path,
                lora_base_model_source=self._lora_base_model_source,
            )
        if self._backend_name == "openai":
            from src.backends.openai_backend import OpenAIBackend

            return OpenAIBackend(
                model_id=self._model_id,
                max_new_tokens=self._max_new_tokens,
                api_key_env=self._openai_cfg.get("api_key_env", "OPENAI_API_KEY"),
                base_url=self._openai_cfg.get(
                    "base_url",
                    "https://api.openai.com/v1",
                ),
                timeout_sec=int(self._openai_cfg.get("timeout_sec", 60)),
                enable_thinking=self._enable_thinking,
                temperature=self._temperature,
                top_p=self._top_p,
            )
        if self._backend_name == "llama_cpp_server":
            from src.backends.llama_cpp_server_backend import LlamaCppServerBackend

            return LlamaCppServerBackend(
                model_id=self._model_id,
                max_new_tokens=self._max_new_tokens,
                enable_thinking=self._enable_thinking,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                base_url=self._llama_cpp_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
                auto_start=self._llama_cpp_cfg.get("auto_start", True),
                server_path=self._llama_cpp_cfg.get("server_path", "llama-server"),
                host=self._llama_cpp_cfg.get("host", "127.0.0.1"),
                port=int(self._llama_cpp_cfg.get("port", 8080)),
                startup_timeout_sec=int(self._llama_cpp_cfg.get("startup_timeout_sec", 300)),
                request_timeout_sec=int(self._llama_cpp_cfg.get("request_timeout_sec", 300)),
                hf_repo_id=self._llama_cpp_cfg.get("hf_repo_id"),
                hf_variant=self._llama_cpp_cfg.get("hf_variant"),
                hf_filename=self._llama_cpp_cfg.get("hf_filename"),
                hf_mmproj_filename=self._llama_cpp_cfg.get("hf_mmproj_filename"),
                local_model_path=self._llama_cpp_cfg.get("local_model_path"),
                local_mmproj_path=self._llama_cpp_cfg.get("local_mmproj_path"),
                n_gpu_layers=self._llama_cpp_cfg.get("n_gpu_layers", -1),
                ctx_size=int(self._llama_cpp_cfg.get("ctx_size", 8192)),
                parallel=int(self._llama_cpp_cfg.get("parallel", 1)),
                threads=self._llama_cpp_cfg.get("threads"),
                flash_attn=self._llama_cpp_cfg.get("flash_attn", True),
                jinja=self._llama_cpp_cfg.get("jinja", True),
                alias=self._llama_cpp_cfg.get("alias"),
                hf_token_env=self._llama_cpp_cfg.get("hf_token_env", "HF_TOKEN"),
            )
        raise ValueError(f"Unknown backend: {self._backend_name!r}")

    # -- capture thread -------------------------------------------------------

    def _capture_loop(self) -> None:
        interval = self.frame_cadence_ms / 1000.0
        while not self._stop_event.is_set():
            try:
                frame = _do_capture(self.session, self.capture_method)
                if frame is not None:
                    self._frame_queue.put(frame)
                else:
                    logger.debug("Capture returned None")
            except Exception as e:
                logger.error("Capture error: %s", e)
            self._stop_event.wait(interval)

    def _start_capture_thread(self) -> None:
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="discovery-capture"
        )
        self._capture_thread.start()
        logger.info("Capture thread started (cadence=%dms)", self.frame_cadence_ms)

    def _stop_capture_thread(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
            self._capture_thread = None
        logger.info("Capture thread stopped")

    # -- wait for ready -------------------------------------------------------

    def wait_for_ready(self) -> None:
        print("\n" + "=" * 60)
        print("DISCOVERY LOOP READY")
        print(f"  Backend:       {self._backend.backend_name}")
        print(f"  Model:         {self._backend.model_name}")
        print(f"  Prompt type:   {self.prompt_type}")
        print(f"  Frame cadence: {self.frame_cadence_ms}ms")
        print(f"  Screenshots:   save={self.cursor_mode}, vlm={self.vlm_image_mode}")
        print(f"  Target:        {self.session.target.short_str()}")
        print(f"  Output:        {self.output_dir}")
        if self.dry_run:
            print("  Dry run:       YES (no input will be sent)")
        if self.max_ticks:
            print(f"  Max ticks:     {self.max_ticks}")
        print(f"  Delay:         {self.delay_ms}ms")
        if self.pause_key:
            print(f"  Pause/resume:  {self.pause_key.upper()}")
        print("=" * 60)
        print("\nSwitch to the game and open the settings/controls menu.")
        user_input = input(
            "Press ENTER when ready to begin, or type delay in ms + ENTER: "
        ).strip()
        if user_input:
            try:
                self.delay_ms = int(user_input)
                print(f"  Delay overridden to {self.delay_ms}ms")
            except ValueError:
                print(f"  Ignoring invalid delay value: {user_input!r}")
        print()

    # -- interruptible inference ----------------------------------------------

    def _infer_interruptible(
        self, image, system_prompt: str, user_prompt: str
    ) -> str:
        result: list[Optional[str]] = [None]
        error: list[Optional[Exception]] = [None]

        def _worker() -> None:
            try:
                result[0] = self._backend.infer(image, system_prompt, user_prompt)
            except Exception as exc:
                error[0] = exc

        t = threading.Thread(target=_worker, daemon=True, name="infer-thread")
        t.start()

        while t.is_alive():
            t.join(timeout=0.1)

        if error[0] is not None:
            logger.error("Inference error: %s", error[0])
            return ""
        return result[0] or ""

    # -- single tick ----------------------------------------------------------

    def _run_tick(
        self,
        tick_idx: int,
        frame: CapturedFrame,
        executor: ActionExecutor,
    ) -> dict:
        """Execute a single discovery tick. Returns a tick record dict."""
        # Save screenshot
        screenshot_path = self.output_dir / f"tick_{tick_idx:04d}.png"
        screenshot_result = save_frame_screenshots(
            frame,
            screenshot_path,
            cursor_mode=self.cursor_mode,
            selected_mode=self.vlm_image_mode,
        )
        inference_image = screenshot_result.selected_image

        # Run inference
        t0 = time.time()
        response = self._infer_interruptible(
            inference_image, self.system_prompt, self.user_prompt
        )
        inference_ms = round((time.time() - t0) * 1000, 1)

        # VLM collision detection: check if pause key appears in response
        if self.pause_key and self._pause_controller:
            if self.pause_key.lower() in response.lower():
                logger.warning(
                    "VLM response contains pause key '%s' — "
                    "suppressing any pause toggle",
                    self.pause_key,
                )
                self._pause_controller._paused.clear()
                self._pause_controller._resume_event.set()

        # Parse response
        result = parse_discovery_response(response, self.exit_keywords)

        # Route: navigation actions
        nav_executed = []
        for nav in result.navigation_actions:
            if nav.type == "click":
                x, y = nav.x, nav.y
                if self._normalize_coordinates_from_0_1000:
                    x, y = normalize_xy_from_0_1000(x, y, frame.width, frame.height)

                click_info = executor.click_at(x, y)
                entry = {"type": "click", "x": x, "y": y}
                entry["exec"] = click_info
                nav_executed.append(entry)
                self._navigation_count += 1

                if self.verbose:
                    status = click_info.get("status", "?")
                    before = click_info.get("cursor_before", "?")
                    after = click_info.get("cursor_after", "?")
                    screen = click_info.get("screen_target", "?")
                    origin = click_info.get("client_rect_origin", "?")
                    print(
                        f"    click {x} {y} -> "
                        f"screen {screen} (origin {origin}) -> "
                        f"cursor: {before} -> {after} [{status}]"
                    )

        # Route: discovered mappings
        new_mappings = []
        for mapping in result.discovered_mappings:
            self._discovered_mappings.append(mapping)
            new_mappings.append({
                "action_name": mapping.action_name,
                "key_name": mapping.key_name,
            })
            logger.info(
                "Discovered: %s -> %s", mapping.action_name, mapping.key_name
            )

        # Route: exit condition
        if result.is_exit:
            self._should_exit = True
            logger.info("Exit condition detected in VLM response")

        # Repetition detection
        if self.repetition_limit > 0:
            normalized = response.strip().lower()
            self._recent_responses.append(normalized)
            # Keep only the last N responses
            if len(self._recent_responses) > self.repetition_limit:
                self._recent_responses = self._recent_responses[-self.repetition_limit:]
            # Check if all recent responses are identical
            if (
                len(self._recent_responses) >= self.repetition_limit
                and len(set(self._recent_responses)) == 1
            ):
                logger.warning(
                    "VLM repeated the same response %d times — stopping",
                    self.repetition_limit,
                )
                self._should_exit = True

        # Build tick record
        record: dict = {
            "tick": tick_idx,
            "timestamp": frame.timestamp,
            "screenshot": (
                str(screenshot_result.selected_path)
                if screenshot_result.selected_path is not None
                else None
            ),
            "vlm_image_mode": screenshot_result.selected_mode,
            "vlm_screenshot": (
                str(screenshot_result.selected_path)
                if screenshot_result.selected_path is not None
                else None
            ),
            "screenshot_raw": (
                str(screenshot_result.raw_path)
                if screenshot_result.raw_path is not None
                else None
            ),
            "screenshot_cursor": (
                str(screenshot_result.cursor_path)
                if screenshot_result.cursor_path is not None
                else None
            ),
            "vlm_response": response,
            "navigation_actions": nav_executed,
            "discovered_mappings": new_mappings,
            "is_exit": result.is_exit,
            "inference_latency_ms": inference_ms,
            "frames_dropped": self._frame_queue.total_dropped,
        }
        if screenshot_result.cursor_metadata is not None:
            record["cursor_overlay"] = screenshot_result.cursor_metadata.to_dict()

        return record

    # -- main run -------------------------------------------------------------

    def run(self) -> Path:
        """Execute the discovery loop. Returns path to the output directory."""
        # Load model
        if not self._backend.is_loaded:
            print(f"Loading model {self._backend.model_name}...")
            self._backend.load_model()
            print("Model loaded.")

        # Wait for user
        if not self.no_wait:
            self.wait_for_ready()
        else:
            print(
                f"Running discovery loop "
                f"(max_ticks={self.max_ticks}, dry_run={self.dry_run})..."
            )

        # Pre-execution delay (skip in dry-run)
        if not self.dry_run and self.delay_ms > 0:
            delay_sec = self.delay_ms / 1000.0
            print(f"Starting in {delay_sec:.1f}s — switch to game window!")
            time.sleep(delay_sec)

        # Executor
        executor = ActionExecutor(self.session, dry_run=self.dry_run)

        # Start pause/resume hotkey listener
        if self.pause_key:
            try:
                self._pause_controller = PauseController(self.pause_key)
                if self._pause_controller.start():
                    print(
                        f"Pause/resume hotkey registered: "
                        f"{self.pause_key.upper()}"
                    )
                else:
                    logger.warning(
                        "Could not register pause hotkey (%s) — "
                        "pause/resume disabled",
                        self.pause_key,
                    )
                    self._pause_controller = None
            except ValueError as e:
                logger.warning("Pause hotkey disabled: %s", e)
                self._pause_controller = None

        # Start capture thread
        self._start_capture_thread()

        tick_idx = 0
        was_paused = False

        try:
            while True:
                if self.max_ticks is not None and tick_idx >= self.max_ticks:
                    logger.info("Max ticks reached (%d), stopping", self.max_ticks)
                    break

                if self._should_exit:
                    logger.info("Exit condition met, stopping")
                    break

                # Handle pause/resume
                if self._pause_controller and self._pause_controller.is_paused:
                    if not was_paused:
                        self._stop_capture_thread()
                        self._frame_queue.clear()
                        print(
                            f"\n  ** PAUSED — press "
                            f"{self.pause_key.upper()} to resume **"
                        )
                        was_paused = True
                    time.sleep(0.1)
                    continue

                if was_paused:
                    print("  ** RESUMED **\n")
                    self._frame_queue.clear()
                    self._start_capture_thread()
                    was_paused = False

                # Get latest frame
                timeout = (self.frame_cadence_ms / 1000.0) * 2
                frame = self._frame_queue.get(timeout=timeout)
                if frame is None:
                    logger.warning(
                        "No frame available after %.1fs timeout", timeout
                    )
                    continue

                record = self._run_tick(tick_idx, frame, executor)
                self._tick_log.append(record)

                # Print progress
                nav_count = len(record["navigation_actions"])
                map_count = len(record["discovered_mappings"])
                parts = []
                if nav_count:
                    parts.append(f"{nav_count} nav")
                if map_count:
                    parts.append(f"{map_count} mappings")
                if record["is_exit"]:
                    parts.append("EXIT")
                summary = ", ".join(parts) or "no actions"

                print(
                    f"  Tick {tick_idx + 1}: [{summary}] "
                    f"[{record['inference_latency_ms']}ms]"
                )

                if self.verbose:
                    # Print VLM response text (indented, trimmed)
                    resp = record["vlm_response"].strip()
                    for line in resp.splitlines():
                        print(f"    VLM: {line.rstrip()}")
                    # Mapping details already printed inline above (nav)
                    for m in record["discovered_mappings"]:
                        print(
                            f"    mapping: {m['action_name']}: "
                            f"{m['key_name']}"
                        )

                tick_idx += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            if self._pause_controller:
                self._pause_controller.stop()
                self._pause_controller = None
            self._stop_capture_thread()

        log_path = self._write_log()
        print(
            f"\nDiscovery complete. {len(self._tick_log)} ticks, "
            f"{len(self._discovered_mappings)} mappings discovered. "
            f"Log: {log_path}"
        )
        return self.output_dir

    # -- log writing ----------------------------------------------------------

    def _write_log(self) -> Path:
        """Write the structured YAML log."""
        log_data = {
            "run_id": self.run_id,
            "backend": self._backend.backend_name,
            "model": self._backend.model_name,
            "prompt_type": self.prompt_type,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "frame_cadence_ms": self.frame_cadence_ms,
            "max_new_tokens": self._max_new_tokens,
            "capture_method": self.capture_method,
            "cursor_mode": self.cursor_mode,
            "vlm_image_mode": self.vlm_image_mode,
            "pause_key": self.pause_key,
            "dry_run": self.dry_run,
            "max_ticks": self.max_ticks,
            "target_window": self.session.target.short_str(),
            "target_hwnd": f"{self.session.target.hwnd:#010x}",
            "total_ticks": len(self._tick_log),
            "total_frames_captured": self._frame_queue.total_put,
            "total_frames_dropped": self._frame_queue.total_dropped,
            "discovered_mappings_count": len(self._discovered_mappings),
            "navigation_actions_count": self._navigation_count,
            "discovered_mappings": [
                {
                    "action_name": m.action_name,
                    "key_name": m.key_name,
                    "raw_line": m.raw_line,
                }
                for m in self._discovered_mappings
            ],
            "ticks": self._tick_log,
        }

        log_path = self.output_dir / "discovery_log.yaml"
        with open(log_path, "w") as f:
            yaml.dump(log_data, f, default_flow_style=False, sort_keys=False)

        return log_path
