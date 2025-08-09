import win32gui
import win32ui
import win32con
from PIL import Image
import numpy as np
from typing import Optional, Tuple
from contextlib import contextmanager


def find_window_by_title(title: str) -> Optional[int]:
    """Find first visible window containing title substring."""
    result = []

    def enum_callback(hwnd: int, _) -> bool:
        if win32gui.IsWindowVisible(hwnd) and title.lower() in win32gui.GetWindowText(hwnd).lower():
            result.append(hwnd)
            return False
        return True

    win32gui.EnumWindows(enum_callback, None)
    return result[0] if result else None


def get_client_dimensions(hwnd: int) -> Tuple[int, int]:
    """Calculate client area dimensions from window handle."""
    try:
        _, _, width, height = win32gui.GetClientRect(hwnd)
        return max(0, width), max(0, height)
    except:
        return 0, 0


def array_to_image(bitmap_data: bytes, width: int, height: int) -> Optional[Image.Image]:
    """Convert BGRA bitmap bytes to RGB PIL Image."""
    try:
        expected_size = width * height * 4
        if len(bitmap_data) < expected_size or width <= 0 or height <= 0:
            return None

        array = np.frombuffer(bitmap_data, dtype=np.uint8)
        array = array.reshape((height, width, 4))[:, :, :3][:, :, ::-1]
        return Image.fromarray(array)
    except:
        return None


@contextmanager
def memory_dc_context(width: int, height: int):
    """Create memory device context independent of screen coordinates."""
    if width <= 0 or height <= 0:
        yield None, None
        return

    screen_dc = None
    memory_dc = None
    bitmap = None

    try:
        screen_dc = win32gui.GetDC(0)  # Get screen DC
        memory_dc = win32ui.CreateDCFromHandle(screen_dc).CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(win32ui.CreateDCFromHandle(screen_dc), width, height)
        memory_dc.SelectObject(bitmap)

        yield memory_dc, bitmap
    finally:
        cleanup_actions = [
            lambda: win32gui.DeleteObject(bitmap.GetHandle()) if bitmap else None,
            lambda: memory_dc.DeleteDC() if memory_dc else None,
            lambda: win32gui.ReleaseDC(0, screen_dc) if screen_dc else None
        ]
        for cleanup in cleanup_actions:
            try:
                cleanup()
            except:
                pass


def try_print_window_capture(hwnd: int, dc_handle: int) -> bool:
    """Attempt PrintWindow API capture for complete window content."""
    try:
        # Try most common flags first
        return win32gui.PrintWindow(hwnd, dc_handle, 2) != 0  # PW_CLIENTONLY
    except:
        return False


def try_legacy_bitblt_capture(hwnd: int, memory_dc, width: int, height: int) -> bool:
    """Fallback to traditional BitBlt method for client area only."""
    try:
        # Get client area DC instead of window DC
        hwnd_dc = win32gui.GetDC(hwnd)  # Client area only
        if hwnd_dc:
            result = memory_dc.BitBlt(
                (0, 0), (width, height),
                win32ui.CreateDCFromHandle(hwnd_dc), (0, 0), win32con.SRCCOPY
            )
            win32gui.ReleaseDC(hwnd, hwnd_dc)
            return True
        return False
    except:
        return False


def extract_bitmap_data(bitmap) -> Optional[bytes]:
    """Extract raw bitmap data as bytes."""
    try:
        return bitmap.GetBitmapBits(True) if bitmap else None
    except:
        return None


def is_bitmap_valid(bitmap_data: bytes, width: int, height: int) -> bool:
    """Check if bitmap data exists and has correct size."""
    try:
        expected_size = width * height * 4
        return bitmap_data is not None and len(bitmap_data) >= expected_size
    except:
        return False


def screenshot_window(window_title: str) -> Optional[Image.Image]:
    """Capture complete window client area including offscreen content."""
    hwnd = find_window_by_title(window_title)
    if not hwnd or not win32gui.IsWindow(hwnd):
        return None

    width, height = get_client_dimensions(hwnd)
    if width <= 0 or height <= 0:
        return None

    with memory_dc_context(width, height) as (memory_dc, bitmap):
        if not memory_dc or not bitmap:
            return None

        dc_handle = memory_dc.GetSafeHdc()

        # Try PrintWindow first (best for offscreen content)
        success = try_print_window_capture(hwnd, dc_handle)

        # Fallback to BitBlt if PrintWindow fails
        if not success:
            success = try_legacy_bitblt_capture(hwnd, memory_dc, width, height)

        if success:
            bitmap_data = extract_bitmap_data(bitmap)
            if is_bitmap_valid(bitmap_data, width, height):
                return array_to_image(bitmap_data, width, height)

    return None
