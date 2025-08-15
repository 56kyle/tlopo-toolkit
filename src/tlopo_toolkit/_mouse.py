"""Module containing logic for interacting with the mouse."""

from typing import Union

import win32api
import win32con
import win32gui

from tlopo_toolkit.util import find_window_by_title


def mouse_move(hwnd: int, x: Union[int, float], y: Union[int, float]) -> None:
    """Moves the mouse to the given coordinates."""
    x: int = int(x)
    y: int = int(y)
    # Not sure why, but when the window is in foreground it doesn't seem to work well with post message
    if win32gui.GetForegroundWindow() == hwnd:
        win32api.SetCursorPos((x, y))
    else:
        l_param = win32api.MAKELONG(x, y)
        win32gui.SendMessage(hwnd, win32con.WM_MOUSEMOVE, 0, l_param)
        win32gui.SendMessage(hwnd, win32con.WM_MOUSEHOVER, 0, l_param)
        win32gui.SendMessage(hwnd, win32con.WM_MBUTTONDOWN, win32con.MK_MBUTTON, l_param)
        win32gui.SendMessage(hwnd, win32con.WM_MBUTTONUP, None, l_param)


def mouse_click(hwnd: int, x: Union[int, float], y: Union[int, float]) -> None:
    """Clicks the mouse at the given coordinates."""
    x: int = int(x)
    y: int = int(y)
    mouse_move(hwnd=hwnd, x=x, y=y)
    l_param = win32api.MAKELONG(x, y)
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, l_param)
    win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, None, l_param)


if __name__ == "__main__":
    hwnd: int = find_window_by_title("The Legend of Pirates Online [BETA]")
    # mouse.move(1500, 400)
    mouse_move(hwnd=hwnd, x=1700, y=400)
