"""Process with window management."""
import time
from pathlib import Path
from typing import Any, Optional

from pywinctl._main import BaseWindow
from tlopo_toolkit.exceptions import WindowNotFoundError
from tlopo_toolkit.process import Process, ProcessInfo
from tlopo_toolkit.window import find_windows_for_process


class Application(Process):
    """Process with window discovery."""

    def __init__(self, executable_path: Path, working_directory: Optional[Path] = None):
        super().__init__(executable_path, working_directory)
        self._window: Optional[BaseWindow] = None

    @property
    def window(self) -> Optional[BaseWindow]:
        """Get discovered window."""
        return self._window

    def find_window(self) -> Optional[BaseWindow]:
        """Find main window for process."""
        if not self.is_running:
            self._window = None
            return None
        
        windows = find_windows_for_process(self.process_info)
        self._window = windows[0] if windows else None
        return self._window

    def wait_for_window(self, timeout: float = 10.0) -> BaseWindow:
        """Wait for window to appear."""
        if not self.is_running:
            raise WindowNotFoundError("Process not running")

        start = time.time()
        while time.time() - start < timeout:
            if self.find_window():
                return self._window
            time.sleep(0.1)
        
        raise WindowNotFoundError(f"Window not found in {timeout}s")

    def spawn_with_window(self, args: list[str] = None, timeout: float = 10.0) -> tuple[ProcessInfo, BaseWindow]:
        """Spawn and wait for window."""
        proc_info = self.spawn(args)
        window = self.wait_for_window(timeout)
        return proc_info, window

    def terminate(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate process and clear window."""
        super().terminate(force, timeout)
        self._window = None

    def __enter__(self) -> 'Application':
        self.spawn()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.terminate()
