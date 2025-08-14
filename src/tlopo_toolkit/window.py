from dataclasses import dataclass

import pywinctl
import win32process

from tlopo_toolkit.process import ProcessInfo


@dataclass(frozen=True)
class WindowInfo:
    """Window identification data."""
    hwnd: int
    pid: int


class Window:
    """Window discovery for processes."""
    
    @staticmethod
    def find_all_windows_for_process(process_info: ProcessInfo) -> list[WindowInfo]:
        """Find all visible windows for a given process."""
        try:
            all_windows = pywinctl.getAllWindows()
            matching_windows = []
            
            for window in all_windows:
                if hasattr(window, '_hWnd') and window.visible:
                    try:
                        _, window_pid = win32process.GetWindowThreadProcessId(window._hWnd)
                        if window_pid == process_info.pid:
                            matching_windows.append(WindowInfo(hwnd=window._hWnd, pid=process_info.pid))
                    except:
                        continue
            
            return matching_windows
        except:
            return []
