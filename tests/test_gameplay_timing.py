"""Offline regression tests: no live windows, input, credentials, or API calls."""

import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import yaml

from src.actions.state import ActionState
from src.backends.base import VLMBackend
from src.backends.factory import create_backend_from_gameplay_config
from src.capture.frame import CapturedFrame
from src.loop.gameplay_config import GameplayRuntimeConfig
from src.loop.gameplay_loop import GameplayLoop
from src.suites.gameplay_suite import GameplaySuiteRunner


SUITE = "configs/eval/gameplay_suite_gemma4_google_chess_experiments.yaml"


def frame(color="black", timestamp=None):
    return CapturedFrame(
        image=Image.new("RGB", (64, 64), color),
        timestamp=time.time() if timestamp is None else timestamp,
        hwnd=1, window_rect=(0, 0, 64, 64), client_rect=(0, 0, 64, 64),
        capture_method="test",
    )


class RecordingBackend(VLMBackend):
    backend_name = "test"
    model_name = "offline"
    is_loaded = True

    def __init__(self, first_delay=0):
        self.first_delay = first_delay
        self.requests = []

    def load_model(self):
        pass

    def unload_model(self):
        pass

    def infer(self, image, system_prompt, user_prompt, generation_params=None):
        self.requests.append((time.monotonic(), image.getpixel((63, 63))[0]))
        if len(self.requests) == 1:
            time.sleep(self.first_delay)
        return "click at 500 500"


class GameplayTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make_loop(self, strategy="background", backend=None, max_ticks=3):
        raw = {
            "capture": {"strategy": strategy, "cursor_mode": "off", "vlm_image_mode": "raw",
                        "post_action_settle_ms": 10},
            "gameplay": {
                "pause_key": None, "log_vlm_io": False,
                "frame_cadence_ms": 1000, "inference_interval_ms": 2000,
                "progress_tracking": {
                    "enabled": True, "save_debug_crops": True,
                    "region": {"left": 0, "top": 0, "width": 8, "height": 8},
                },
            },
        }
        session = Mock()
        session.target.hwnd = 1
        session.target.short_str.return_value = "offline test"
        return GameplayLoop(
            session=session, config=GameplayRuntimeConfig.from_raw(raw),
            backend=backend or RecordingBackend(), output_dir=Path(self.temp.name),
            no_wait=True, max_ticks=max_ticks, delay_ms=0,
        )

    def test_progress_and_capture_continue_during_slow_inference(self):
        backend = RecordingBackend(first_delay=3.2)
        loop = self.make_loop(backend=backend)
        captures = []
        origin = time.monotonic()

        def capture(*args):
            elapsed = time.monotonic() - origin
            result = frame("white" if 0.8 <= elapsed < 1.8 else "black")
            result.image.putpixel((63, 63), (len(captures), 0, 0))
            captures.append(time.monotonic())
            return result

        executor = Mock()
        executor.execute_tick.return_value = {"skipped": False, "dry_run": False}
        with patch("src.loop.gameplay_loop._do_capture", side_effect=capture), \
                patch("src.loop.gameplay_loop.ActionExecutor", return_value=executor), \
                redirect_stdout(io.StringIO()):
            log_path = loop.run()
        log = yaml.safe_load(log_path.read_text())
        self.assertEqual(len(backend.requests), 3)
        self.assertTrue(all(b[0] - a[0] >= 2 for a, b in zip(backend.requests, backend.requests[1:])))
        # A transient spinner appears and disappears entirely within request 1.
        self.assertGreaterEqual(sum(t < backend.requests[0][0] + 3.2 for t in captures), 4)
        self.assertEqual(log["total_progress_count"], 1)
        self.assertTrue(log["ticks"][1]["progress_tracking"]["progress_event"])
        self.assertEqual(sum(s["progress_event"] for s in log["progress_samples"]), 1)
        self.assertEqual(len(log["progress_samples"]), len(captures))
        self.assertEqual(log["inference_interval_ms"], 2000)
        self.assertTrue(all("inference_started_at" in tick for tick in log["ticks"]))
        self.assertGreaterEqual(log["progress_samples"][-1]["timestamp"], loop._frame_not_before)
        # The second model image follows execution of the first action (3.2 s).
        self.assertGreaterEqual(backend.requests[1][1], 4)
        self.assertTrue((Path(self.temp.name) / "capture_0001_progress_tracking.png").exists())
        self.assertTrue((Path(self.temp.name) / "tick_0001_progress_tracking.png").exists())
        self.assertIsNone(loop._capture_thread)

    def test_queued_frames_from_before_the_action_are_rejected(self):
        loop = self.make_loop()
        loop._frame_not_before = 20
        older, fresh = frame(timestamp=19), frame(timestamp=21)
        with patch.object(loop, "_get_frame_with_action_release", side_effect=[older, fresh]):
            self.assertIs(loop._next_frame(1, Mock(), Mock()), fresh)

    def test_failed_requests_are_throttled_too(self):
        backend = RecordingBackend()
        starts = []

        def fail(*args, **kwargs):
            starts.append(time.monotonic())
            raise RuntimeError("offline simulated rate limit")

        backend.infer = fail
        loop = self.make_loop(backend=backend, max_ticks=2)
        loop.frame_cadence_ms = 25
        loop.inference_interval_ms = 200
        executor = Mock()
        executor.execute_tick.return_value = {"skipped": False, "dry_run": False}
        with patch("src.loop.gameplay_loop._do_capture", side_effect=lambda *args: frame()), \
                patch("src.loop.gameplay_loop.ActionExecutor", return_value=executor), \
                self.assertLogs("src.loop.gameplay_loop", level="ERROR"), \
                redirect_stdout(io.StringIO()):
            log = yaml.safe_load(loop.run().read_text())
        self.assertEqual(len(starts), 2)
        self.assertGreaterEqual(starts[1] - starts[0], 0.2)
        self.assertTrue(all(tick["inference_error"] for tick in log["ticks"]))

    def test_tick_snapshots_do_not_double_count_or_lose_transient_events(self):
        loop = self.make_loop()
        for color in ["black", "white", "white", "black", "white", "black"]:
            loop._sample_background_progress(frame(color))
        first = loop._progress_for_tick(0, frame().image)
        second = loop._progress_for_tick(1, frame().image)
        self.assertEqual(first["progress_events_since_tick"], 2)
        self.assertEqual(first["progress_count"], 2)
        self.assertFalse(second["progress_event"])
        self.assertEqual(second["progress_count"], 2)

    def test_after_action_mode_still_measures_each_tick(self):
        loop = self.make_loop(strategy="after_action")
        self.assertFalse(loop._progress_for_tick(0, frame().image)["progress_event"])
        self.assertTrue(loop._progress_for_tick(1, frame("white").image)["progress_event"])
        self.assertFalse(loop._progress_for_tick(2, frame("white").image)["progress_event"])
        self.assertEqual(loop._progress_samples, [])

    def test_pause_interrupts_rate_limit_wait(self):
        loop = self.make_loop()
        loop._last_inference_started_at = time.monotonic()
        loop._pause_controller = Mock(is_paused=True)
        self.assertFalse(loop._wait_for_inference_slot(ActionState(loop.action_config), Mock()))

    def test_legacy_config_has_no_inference_throttle(self):
        config = GameplayRuntimeConfig.from_raw({})
        self.assertEqual(config.inference_interval_ms, 0)


class GemmaSuiteTests(unittest.TestCase):
    def test_all_seven_conditions_are_matched_between_models(self):
        runner = GameplaySuiteRunner(SUITE)
        self.assertEqual(runner.validate(), [])
        self.assertEqual(runner.validation_warnings, [])
        self.assertEqual(len(runner._expanded_run_entries()), 28)
        entries = runner._selected_run_entries()
        self.assertEqual(len(entries), 14)
        for left, right in zip(entries[::2], entries[1::2]):
            plans = [runner._build_run_plan(entry, repeat_index=1) for entry in [left, right]]
            configs = [copy.deepcopy(plan.merged_raw_config) for plan in plans]
            self.assertEqual(configs[0]["gameplay"].pop("model"), "gemma-4-26b-a4b-it")
            self.assertEqual(configs[1]["gameplay"].pop("model"), "gemma-4-31b-it")
            self.assertEqual(configs[0], configs[1])
            for plan in plans:
                config = plan.runtime_config
                self.assertEqual(plan.max_ticks, 30)
                self.assertEqual(config.frame_cadence_ms, 1000)
                self.assertEqual(config.inference_interval_ms, 2000)
                self.assertEqual(config.capture_strategy, "background")
                self.assertIsNone(config.model_settings.lora_adapter_path)
                backend = create_backend_from_gameplay_config(config)
                with patch("src.backends.openai_backend.OpenAI") as client, \
                        patch.dict("os.environ", {"GEMINI_API_KEY": "offline-placeholder"}):
                    backend.load_model()
                self.assertEqual(client.call_args.kwargs["max_retries"], 0)
                self.assertEqual(client.call_args.kwargs["base_url"],
                                 "https://generativelanguage.googleapis.com/v1beta/openai")


if __name__ == "__main__":
    unittest.main()
