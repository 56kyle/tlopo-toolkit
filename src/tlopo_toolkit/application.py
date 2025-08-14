"""Application class with process and window management."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import psutil
from pywinctl._main import BaseWindow

from tlopo_toolkit.exceptions import WindowNotFoundError
from tlopo_toolkit.process import Executable, spawn_process, terminate_process
from tlopo_toolkit.window import find_windows_for_process


@dataclass(frozen=True)
class Application:
    """Application instance with concrete process and window."""
    executable: ClassVar[Executable]
    process: psutil.Process
    window: BaseWindow

    @classmethod
    def spawn(cls, args: list[str] = None, timeout: float = 10.0) -> "Application":
        """Spawn application and wait for window."""
        process = spawn_process(cls.executable, args)
        window = cls._wait_for_window(process, timeout)
        return cls(process=process, window=window)

    @classmethod
    def _wait_for_window(cls, process: psutil.Process, timeout: float = 10.0) -> BaseWindow:
        """Wait for window to appear for process."""
        if not process.is_running():
            raise WindowNotFoundError("Process not running")

        start = time.time()
        while time.time() - start < timeout:
            windows = find_windows_for_process(process)
            if windows:
                return windows[0]
            time.sleep(0.1)

        raise WindowNotFoundError(f"Window not found in {timeout}s")

    @property
    def is_running(self) -> bool:
        """Check if process is running."""
        return self.process.is_running()

    def terminate(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the process."""
        terminate_process(self.process, force, timeout)
