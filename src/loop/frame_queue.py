"""Thread-safe frame buffer with latest-frame-wins semantics.

A capture thread puts frames in; the main loop takes the latest one.
If inference is slow and multiple frames arrive, only the newest is retained.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from src.capture.frame import CapturedFrame

logger = logging.getLogger(__name__)


class FrameQueue:
    """Single-slot frame buffer: only the latest frame is kept.

    Designed for the gameplay loop where a background capture thread produces
    frames at a fixed cadence, and the inference loop consumes the latest one.
    """

    def __init__(self) -> None:
        self._frame: Optional[CapturedFrame] = None
        self._lock = threading.Lock()
        self._available = threading.Event()
        self._total_put = 0
        self._total_dropped = 0

    def put(self, frame: CapturedFrame) -> None:
        """Store a new frame, replacing any unconsumed one."""
        with self._lock:
            if self._frame is not None:
                self._total_dropped += 1
                logger.debug("Frame dropped (total dropped: %d)", self._total_dropped)
            self._frame = frame
            self._total_put += 1
            self._available.set()

    def get(self, timeout: Optional[float] = None) -> Optional[CapturedFrame]:
        """Get the latest frame, blocking until one is available.

        Returns None if timeout expires before a frame arrives.
        """
        if not self._available.wait(timeout=timeout):
            return None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._available.clear()
            return frame

    def get_nowait(self) -> Optional[CapturedFrame]:
        """Get the latest frame without blocking. Returns None if empty."""
        with self._lock:
            frame = self._frame
            self._frame = None
            self._available.clear()
            return frame

    def clear(self) -> None:
        """Discard any buffered frame."""
        with self._lock:
            self._frame = None
            self._available.clear()

    @property
    def total_put(self) -> int:
        """Total frames put into the queue."""
        return self._total_put

    @property
    def total_dropped(self) -> int:
        """Frames overwritten before being consumed."""
        return self._total_dropped

    @property
    def has_frame(self) -> bool:
        """Whether a frame is currently available."""
        return self._available.is_set()
