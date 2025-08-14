"""Module containing logic for spawning Processes that have a Window head associated to them."""
import asyncio
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional

from tlopo_toolkit.exceptions import WindowNotFoundError
from tlopo_toolkit.process import Process
from tlopo_toolkit.process import ProcessInfo
from tlopo_toolkit.window import Window
from tlopo_toolkit.window import WindowInfo


class Application(Process):
    """Abstract base class for spawnable game applications with window management capabilities."""

    def __init__(self, executable_path: Path, working_directory: Optional[Path] = None) -> None:
        """Initialize application with executable path and optional working directory."""
        super().__init__(executable_path, working_directory)
        self._window_info: Optional[WindowInfo] = None

    @property
    def window_info(self) -> Optional[WindowInfo]:
        """Get immutable window information if window has been discovered."""
        return self._window_info

    def _build_command_arguments(self) -> List[str]:
        """Build command line arguments specific to this application type."""
        return []

    def _try_discover_window(self) -> bool:
        """Attempt to discover the main window for the running process."""
        if not self.is_running:
            self._window_info = None
            return False

        try:
            self._window_info = Window.find_all_windows_for_process(self.process_info)
            return True
        except WindowNotFoundError:
            self._window_info = None
            return False

    async def wait_for_window_async(self, timeout: float = 10.0, poll_interval: float = 0.1) -> WindowInfo:
        """Wait for the application window to appear asynchronously."""
        if not self.is_running:
            raise WindowNotFoundError("Process is not running")

        start_time = asyncio.get_event_loop().time()

        while True:
            if self._try_discover_window():
                return self._window_info

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise WindowNotFoundError(f"Window not found within {timeout}s timeout")

            await asyncio.sleep(poll_interval)

    def wait_for_window_sync(self, timeout: float = 10.0, poll_interval: float = 0.1) -> WindowInfo:
        """Wait for the application window to appear synchronously."""
        if not self.is_running:
            raise WindowNotFoundError("Process is not running")

        import time
        start_time = time.time()

        while True:
            if self._try_discover_window():
                return self._window_info

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise WindowNotFoundError(f"Window not found within {timeout}s timeout")

            time.sleep(poll_interval)

    async def spawn_process_async(self, additional_args: Optional[List[str]] = None) -> ProcessInfo:
        """Spawn the application process asynchronously and return process information."""
        command_args = self._build_command_arguments()
        if additional_args:
            command_args.extend(additional_args)

        return await self.spawn_async(command_args)

    def spawn_process_sync(self, additional_args: Optional[List[str]] = None) -> ProcessInfo:
        """Spawn the application process synchronously and return process information."""
        command_args = self._build_command_arguments()
        if additional_args:
            command_args.extend(additional_args)

        return self.spawn_sync(command_args)

    async def spawn_with_window_async(
        self,
        additional_args: Optional[List[str]] = None,
        window_timeout: float = 10.0
    ) -> tuple[ProcessInfo, WindowInfo]:
        """Spawn process and wait for window to appear asynchronously."""
        process_info = await self.spawn_process_async(additional_args)
        window_info = await self.wait_for_window_async(window_timeout)
        return process_info, window_info

    def spawn_with_window_sync(
        self,
        additional_args: Optional[List[str]] = None,
        window_timeout: float = 10.0
    ) -> tuple[ProcessInfo, WindowInfo]:
        """Spawn process and wait for window to appear synchronously."""
        process_info = self.spawn_process_sync(additional_args)
        window_info = self.wait_for_window_sync(window_timeout)
        return process_info, window_info

    async def terminate_process_async(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the application process asynchronously with optional force and timeout."""
        await self.terminate_async(force, timeout)
        self._window_info = None

    def terminate_process_sync(self, force: bool = False, timeout: float = 5.0) -> None:
        """Terminate the application process synchronously with optional force and timeout."""
        self.terminate_sync(force, timeout)
        self._window_info = None

    def rediscover_window(self) -> Optional[WindowInfo]:
        """Manually rediscover the window and return updated window information."""
        self._try_discover_window()
        return self._window_info

    async def __aenter__(self) -> 'Application':
        """Async context manager entry - spawn process."""
        await self.spawn_process_async()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - terminate process."""
        await self.terminate_process_async()

    def __enter__(self) -> 'Application':
        """Context manager entry - spawn process synchronously."""
        self.spawn_process_sync()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - terminate process synchronously."""
        self.terminate_process_sync()
