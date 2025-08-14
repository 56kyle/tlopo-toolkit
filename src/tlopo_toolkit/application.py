"""Module containing logic for spawning Processes that have a Window head associated to them."""
import time
from pathlib import Path
from typing import Any, Optional

from tlopo_toolkit.exceptions import WindowNotFoundError
from tlopo_toolkit.process import Process, ProcessInfo
from tlopo_toolkit.window import Window, WindowInfo


class Application(Process):
    """Process with window management capabilities."""

    def __init__(self, executable_path: Path, working_directory: Optional[Path] = None) -> None:
        super().__init__(executable_path, working_directory)
        self._window_info: Optional[WindowInfo] = None

    @property
    def window_info(self) -> Optional[WindowInfo]:
        """Get window information if discovered."""
        return self._window_info

    def find_window(self) -> Optional[WindowInfo]:
        """Find the main window for the running process."""
        if not self.is_running:
            self._window_info = None
            return None

        windows = Window.find_all_windows_for_process(self.process_info)
        self._window_info = windows[0] if windows else None
        return self._window_info

    def wait_for_window(self, timeout: float = 10.0) -> WindowInfo:
        """Wait for the application window to appear."""
        if not self.is_running:
            raise WindowNotFoundError("Process is not running")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.find_window():
                return self._window_info
            time.sleep(0.1)
        
        raise WindowNotFoundError(f"Window not found within {timeout}s timeout")

    def spawn_with_window(self, command_args: list[str] = None, window_timeout: float = 10.0) -> tuple[ProcessInfo, WindowInfo]:
        """Spawn process and wait for window to appear."""
        process_info = self.spawn(command_args)
        window_info = self.wait_for_window(window_timeout)
        return process_info, window_info

    def terminate(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the application process."""
        super().terminate(force, timeout)
        self._window_info = None

    def __enter__(self) -> 'Application':
        """Context manager entry - spawn process."""
        self.spawn()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - terminate process."""
        self.terminate()
