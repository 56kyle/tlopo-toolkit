from dataclasses import dataclass

import pywinctl
import win32process

from tlopo_toolkit.process import ProcessInfo


@dataclass(frozen=True)
class WindowInfo:
    """Window identification."""
    hwnd: int
    pid: int


class Window:
    """Window discovery."""
    
    @staticmethod
    def find_all_windows_for_process(process_info: ProcessInfo) -> list[WindowInfo]:
        """Find visible windows for process."""
        windows = []
        try:
            for window in pywinctl.getAllWindows():
                if hasattr(window, '_hWnd') and window.visible:
                    try:
                        _, pid = win32process.GetWindowThreadProcessId(window._hWnd)
                        if pid == process_info.pid:
                            windows.append(WindowInfo(window._hWnd, pid))
                    except:
                        continue
        except:
            pass
        return windows
