"""Optional game-freeze support via the xspeedhack library.

Freezes the target game process's perceived clock (via DLL injection) while
waiting for a VLM response, then restores normal speed before the chosen
action is executed. This is purely a UX/consistency feature: if the
``xspeedhack`` package is not installed, or attaching to the process fails,
freezing is silently disabled and the gameplay loop behaves as before.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FreezeController:
    """Wraps an xspeedhack client to freeze/unfreeze the target process.

    Parameters
    ----------
    pid : int
        Process id of the target game window.
    arch : str
        "x86" or "x64" — must match the target process bitness.
    freeze_speed : float
        Game speed multiplier applied while frozen (0.0 = fully frozen).
    normal_speed : float
        Game speed multiplier applied while unfrozen (1.0 = normal speed).
    """

    def __init__(
        self,
        pid: int,
        arch: str = "x64",
        freeze_speed: float = 0.0,
        normal_speed: float = 1.0,
    ) -> None:
        self._pid = pid
        self._arch = arch
        self._freeze_speed = max(0.0, freeze_speed)
        self._normal_speed = max(0.0, normal_speed)
        self._client = None
        self._frozen = False

    def attach(self) -> bool:
        """Inject the speedhack DLL into the target process.

        Returns True on success, False if xspeedhack is unavailable or
        attaching failed (in which case freezing is a no-op).
        """
        try:
            import xspeedhack as xsh
        except ImportError:
            logger.warning(
                "freeze feature enabled but 'xspeedhack' package is not "
                "installed; run 'pip install xspeedhack' to enable it. "
                "Freezing disabled for this run."
            )
            return False

        try:
            self._client = xsh.Client(process_id=self._pid, arch=self._arch)
        except Exception as exc:
            logger.warning(
                "Could not attach xspeedhack to pid=%s (arch=%s): %s. "
                "Freezing disabled for this run.",
                self._pid,
                self._arch,
                exc,
            )
            self._client = None
            return False

        logger.info(
            "xspeedhack attached to pid=%s (arch=%s)", self._pid, self._arch
        )
        return True

    @property
    def is_attached(self) -> bool:
        return self._client is not None

    def freeze(self) -> None:
        if self._client is None or self._frozen:
            return
        try:
            self._client.set_speed(self._freeze_speed)
            self._frozen = True
        except Exception as exc:
            logger.warning("Failed to freeze game process: %s", exc)

    def unfreeze(self) -> None:
        if self._client is None or not self._frozen:
            return
        try:
            self._client.set_speed(self._normal_speed)
        except Exception as exc:
            logger.warning("Failed to unfreeze game process: %s", exc)
        finally:
            self._frozen = False

    def close(self) -> None:
        """Restore normal speed and drop the client reference."""
        if self._client is None:
            return
        self.unfreeze()
        self._client = None
