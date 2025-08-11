from dataclasses import dataclass

import pywinctl
import win32process

from tlopo_toolkit.exceptions import WindowNotFoundError
from tlopo_toolkit.process import ProcessInfo


@dataclass(frozen=True)
class WindowInfo:
    """Immutable container for window identification data."""
    hwnd: int
    pid: int


class Window:
    """Manages window discovery and identification for processes."""
    @staticmethod
    def find_all_windows_for_process(process_info: ProcessInfo) -> list[WindowInfo]:
        """Find all visible windows for a given process."""
        try:
            # Get all windows and filter by PID
            all_windows = pywinctl.getAllWindows()
            process_windows = [w for w in all_windows if hasattr(w, '_hWnd') and w.visible]

            # Filter by PID and convert to WindowInfo
            matching_windows = []
            for window in process_windows:
                try:
                    # Get the window's process ID
                    _, window_pid = win32process.GetWindowThreadProcessId(window._hWnd)
                    if window_pid == process_info.pid:
                        matching_windows.append(WindowInfo(hwnd=window._hWnd, pid=process_info.pid))
                finally:
                    pass

            return matching_windows

        except Exception as e:
            # Return empty list if no windows found rather than raising
            return []
