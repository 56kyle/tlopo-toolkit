"""Module containing logic for interacting with the mouse."""

import win32api
import win32con
import win32gui
from window_input import Window


def mouse_move(hwnd: int, x: int, y: int) -> None:
    l_param = win32api.MAKELONG(x, y)
    # Not sure why, but when the window is in foreground it doesn't seem to work well with post message
    if win32gui.GetForegroundWindow() == hwnd:
        win32api.SetCursorPos((x, y))
    else:
        win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, None, l_param)
