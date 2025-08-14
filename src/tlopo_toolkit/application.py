"""Application class with process and window management."""
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Generator, Optional

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
    @contextmanager
    def managed(
        cls,
        args: Optional[list[str]] = None,
        spawn_timeout: float = 10.0,
        force: bool = False,
        terminate_timeout: float = 5.0
    ) -> Generator["Application", None, None]:
        """Context manager for application lifecycle."""
        app: "Application" = cls.spawn(args, spawn_timeout)
        try:
            yield app
        finally:
            app.terminate(force, terminate_timeout)

    @classmethod
    def spawn(cls, args: Optional[list[str]] = None, timeout: float = 10.0) -> "Application":
        """Spawn application and wait for window."""
        process: psutil.Process = spawn_process(cls.executable, args)
        window: BaseWindow = cls._wait_for_window(process, timeout)
        return cls(process=process, window=window)

    @classmethod
    def _wait_for_window(cls, process: psutil.Process, timeout: float = 10.0) -> BaseWindow:
        """Wait for window to appear for process."""
        if not process.is_running():
            raise WindowNotFoundError("Process not running")

        start: float = time.time()
        while time.time() - start < timeout:
            windows: list[BaseWindow] = find_windows_for_process(process)
            if windows:
                return windows[0]
            time.sleep(0.1)

        raise WindowNotFoundError(f"Window not found in {timeout}s")

    @property
    def is_running(self) -> bool:
        """Check if process is running."""
        return self.process.is_running()

    def __enter__(self) -> "Application":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.terminate()

    def terminate(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the process."""
        terminate_process(self.process, force, timeout)
