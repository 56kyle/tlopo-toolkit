import pywinctl
import win32process
from pywinctl._main import BaseWindow
import psutil


def find_windows_for_process(process: psutil.Process) -> list[BaseWindow]:
    """Find visible windows belonging to process."""
    windows = []
    for window in pywinctl.getAllWindows():
        if hasattr(window, '_hWnd') and window.visible:
            try:
                _, window_pid = win32process.GetWindowThreadProcessId(window._hWnd)
                if window_pid == process.pid:
                    windows.append(window)
            except (OSError, ValueError):
                # Skip windows we can't get PID for (access denied, invalid handle)
                continue
    return windows
