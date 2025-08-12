import subprocess
from contextlib import contextmanager
from contextlib import suppress
from math import floor
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import win32con
import win32gui
import win32ui
from PIL import Image
from pywinctl import getAllWindows
from pywinctl._main import BaseWindow

from tlopo_toolkit.geometry import Rect
from tlopo_toolkit.geometry import Region


def get_window_from_pid(pid: int) -> BaseWindow:
    """Run executable at the provided path and return its window."""
    for window in getAllWindows():
        if window.getPID() == pid:
            return window
    raise ValueError("Could not find window with given PID for the Launcher.")


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


def get_brewing_board_region(hwnd: int) -> Region:
    """Returns the brewing hexboard region."""
    brewing_minigame_region: Region = get_brewing_minigame_region(hwnd=hwnd)

    gray_minigame_region: Region = _get_gray_minigame_region(minigame_region=brewing_minigame_region)
    blurred_minigame_region: Region = _get_blurred_minigame_region(minigame_region=gray_minigame_region)
    edged_minigame_region: Region = _get_edged_minigame_region(minigame_region=blurred_minigame_region)

    bottom_edge_region: Region = _get_bottom_edge_region(edged_minigame_region=edged_minigame_region)
    print(f"{bottom_edge_region.rect=}")
    bottom_edge_region.show()
    right_edge_region: Region = _get_right_edge_region(edged_minigame_region=edged_minigame_region)
    print(f"{right_edge_region.rect=}")
    right_edge_region.show()

    dxi_inner: int = __find_brewing_board_left_edge_offset(bottom_edge=bottom_edge_region)
    dxf_inner: int = __find_brewing_board_right_edge_offset(bottom_edge=bottom_edge_region)
    dyi_inner: int = __find_brewing_board_top_edge_offset(right_edge=right_edge_region)
    dyf_inner: int = __find_brewing_board_bottom_edge_offset(right_edge=right_edge_region)

    dx_inner: int = dxf_inner - dxi_inner
    dx_hex_outer: float = dx_inner / 5.75
    dxi: int = round(dxi_inner - dx_hex_outer)
    dxf: int = round(dxf_inner + dx_hex_outer)

    dy_inner: int = dyf_inner - dyi_inner
    dy_hex_inner: float = dy_inner / 19
    dyi: int = round(dyi_inner - dy_hex_inner)
    dyf: int = round(dyf_inner + dy_hex_inner)

    relative_rect: Rect = Rect(x=dxi, y=dyi, w=dxf, h=dyf)
    print(f"{relative_rect=}")

    board_region: Region = brewing_minigame_region.crop_relative(rect=relative_rect)
    print(f"{board_region.rect=}")
    board_region.show()
    return board_region


def _get_gray_minigame_region(minigame_region: Region) -> Region:
    """Converts the given region's image to grayscale and returns a new region with the grayscale image."""
    gray_image: np.ndarray = cv2.cvtColor(minigame_region.image, cv2.COLOR_BGR2GRAY)
    return Region(image=gray_image, rect=minigame_region.rect)


def _get_blurred_minigame_region(minigame_region: Region) -> Region:
    """Blurs the provided region's image and returns a new region with the blurred image."""
    blurred_image: np.ndarray = cv2.GaussianBlur(minigame_region.image, (19, 19), 0)
    return Region(image=blurred_image, rect=minigame_region.rect)


def _get_edged_minigame_region(minigame_region: Region) -> Region:
    """Get the edged version of the minigame region."""
    edged_image: np.ndarray = cv2.Canny(minigame_region.image, 50, 120)
    return Region(image=edged_image, rect=minigame_region.rect)


def _get_right_edge_region(edged_minigame_region: Region) -> Region:
    """Get the right edge of the hex board in the given region."""
    print("Pre get right")
    w_half: int = edged_minigame_region.rect.w // 2
    y_half: int = edged_minigame_region.rect.h // 2

    xi: int = floor(edged_minigame_region.rect.x * 0.75)
    dx: int = edged_minigame_region.rect.w - xi
    yi: int = 0
    dy: int = edged_minigame_region.rect.h
    right_edge_rect: Rect = Rect(x=xi, y=yi, w=dx, h=dy)
    print(f"{right_edge_rect=}")
    right_edge_region: Region = edged_minigame_region.crop_relative(rect=right_edge_rect)
    print(f"{right_edge_region.rect=}")
    right_edge_region.show()
    return right_edge_region


def _get_bottom_edge_region(edged_minigame_region: Region) -> Region:
    """Get the bottom edge of the hex board in the given region."""
    w_half: int = edged_minigame_region.rect.w // 2
    y_half: int = edged_minigame_region.rect.h // 2

    xi: int = edged_minigame_region.rect.x + w_half
    yi: int = edged_minigame_region.rect.y + y_half
    dx: int = edged_minigame_region.rect.right - w_half
    dy: int = edged_minigame_region.rect.bottom - y_half

    bottom_edge_rect: Rect = Rect(x=xi, y=yi, w=dx, h=dy)
    print(f"{bottom_edge_rect=}")
    bottom_edge_region: Region = edged_minigame_region.crop_relative(rect=bottom_edge_rect)
    bottom_edge_region.show()
    return bottom_edge_region


def __find_brewing_board_left_edge_offset(bottom_edge: Region) -> int:
    """Find the left edge of the brewing board."""
    x_counts: np.ndarray = bottom_edge.image.astype(bool, copy=True).sum(axis=0)
    half_width: int = bottom_edge.rect.w // 2
    return int(np.argmax(x_counts[:half_width]))


def __find_brewing_board_right_edge_offset(bottom_edge: Region) -> int:
    x_counts: np.ndarray = bottom_edge.image.astype(bool, copy=True).sum(axis=0)
    half_width: int = bottom_edge.rect.w // 2
    return int(np.argmax(x_counts[half_width:])) + half_width


def __find_brewing_board_top_edge_offset(right_edge: Region) -> int:
    y_counts: np.ndarray = right_edge.image.astype(bool, copy=True).sum(axis=1)
    half_height: int = right_edge.rect.h // 2
    return int(np.argmax(y_counts[:half_height]))


def __find_brewing_board_bottom_edge_offset(right_edge: Region) -> int:
    y_counts: np.ndarray = right_edge.image.astype(bool, copy=True).sum(axis=1)
    half_height: int = right_edge.rect.h // 2
    return int(np.argmax(y_counts[half_height:])) + half_height


def get_brewing_minigame_region(hwnd: int) -> Region:
    """Returns the brewing minigame region."""
    client_region: Region = get_client_region(hwnd=hwnd)
    half_height: int = client_region.rect.h // 2

    row_sums: np.ndarray = client_region.image[half_height].sum(axis=1)

    midpoint: int = len(row_sums) // 2
    left_bar_width: int = sum(np.array(row_sums[:midpoint] == 0))
    right_bar_width: int = sum(np.array(row_sums[midpoint:] == 0))

    minigame_region: Region = client_region.crop_relative(
        rect=Rect(
            x=left_bar_width, y=0, w=client_region.rect.w - (left_bar_width + right_bar_width), h=client_region.rect.h
        )
    )
    print(f"{minigame_region.rect=}")
    minigame_region.show()
    return minigame_region


def get_client_region(hwnd: int) -> Region:
    """Returns the client Region."""
    client_img: np.ndarray = get_client_image(hwnd=hwnd)
    print(f"{client_img.shape=}")
    client_rect: Rect = get_client_rect(hwnd=hwnd)
    print(f"{client_rect=}")
    client_region: Region = Region(image=client_img, rect=client_rect)
    print(f"{client_region.rect=}")
    client_region.show()
    return client_region


def get_client_image(hwnd: int) -> np.ndarray:
    """Returns the client image."""
    client_img: Optional[Image] = screenshot_window(hwnd=hwnd)
    if client_img is None:
        raise ValueError("Could not capture client image.")
    client_img_arr: np.ndarray = np.array(client_img)
    bgr: np.ndarray = cv2.cvtColor(client_img_arr, cv2.COLOR_RGB2BGR)
    return bgr[:, :, :3]


def get_client_rect(hwnd: int) -> Rect:
    """Returns the client Rect."""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    return Rect(left, top, right - left, bottom - top)


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
            lambda: win32gui.ReleaseDC(0, screen_dc) if screen_dc else None,
        ]
        for cleanup in cleanup_actions:
            with suppress(Exception):
                cleanup()


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
            memory_dc.BitBlt((0, 0), (width, height), win32ui.CreateDCFromHandle(hwnd_dc), (0, 0), win32con.SRCCOPY)
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


def screenshot_window_from_title(window_title: str) -> Optional[Image.Image]:
    """Capture complete window client area including offscreen content."""
    hwnd: int = find_window_by_title(window_title)
    if not hwnd or not win32gui.IsWindow(hwnd):
        return None
    return screenshot_window(hwnd)


def screenshot_window(hwnd: int) -> Optional[Image.Image]:
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


if __name__ == "__main__":
    hwnd: int = find_window_by_title("The Legend of Pirates Online [BETA]")
    board_region: Region = get_brewing_board_region(hwnd=hwnd)
    board_region.show()
