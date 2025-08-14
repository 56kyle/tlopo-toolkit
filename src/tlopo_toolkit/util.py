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
from PIL.Image import Image
from PIL.Image import fromarray
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

    bottom_edge_region: Region = _get_bottom_edge_region(minigame_region=brewing_minigame_region)
    right_edge_region: Region = _get_right_edge_region(minigame_region=brewing_minigame_region)

    bottom_edge_img: np.ndarray = bottom_edge_region.export()
    right_edge_img: np.ndarray = right_edge_region.export()

    edged_bottom_edge_img: np.ndarray = _get_hex_edges_from_img(img=bottom_edge_img)
    edged_right_edge_img: np.ndarray = _get_hex_edges_from_img(img=right_edge_img)

    dxi_inner: int = __find_brewing_board_left_edge_offset(bottom_edge=edged_bottom_edge_img)
    dxf_inner: int = __find_brewing_board_right_edge_offset(bottom_edge=edged_bottom_edge_img)
    dyi_inner: int = __find_brewing_board_top_edge_offset(right_edge=edged_right_edge_img)
    dyf_inner: int = __find_brewing_board_bottom_edge_offset(right_edge=edged_right_edge_img)

    dx_inner: int = dxf_inner - dxi_inner
    dx_hex_outer: float = 0 * dx_inner / 5.75

    dxi: int = round(dxi_inner)
    dxf: int = round(dxf_inner + (dx_hex_outer / 4))

    # Convert bottom_edge offsets to brewing_minigame_region coordinate system
    # bottom_edge starts at (w_half, y_half) within minigame_region
    # The offsets dxi, dxf are already relative to the bottom_edge start position
    bottom_edge_start_x = brewing_minigame_region.rect.w // 2
    xi: int = brewing_minigame_region.rect.x + bottom_edge_start_x + dxi
    xf: int = brewing_minigame_region.rect.x + bottom_edge_start_x + dxf

    dy_inner: int = dyf_inner - dyi_inner
    dy_hex_inner: float = dy_inner // 19
    dyi: int = round(dyi_inner - dy_hex_inner)
    dyf: int = round(dyf_inner)

    # Convert right_edge offsets to brewing_minigame_region coordinate system
    # right_edge starts at (w_quarter * 3, 0) within minigame_region
    right_edge_start_y = 0
    yi: int = brewing_minigame_region.rect.y + right_edge_start_y + dyi
    yf: int = brewing_minigame_region.rect.y + right_edge_start_y + dyf

    absolute_rect: Rect = Rect(x=xi, y=yi, w=xf - xi, h=yf - yi)
    print(f"{absolute_rect=}")

    # Convert absolute coordinates to relative coordinates within brewing_minigame_region
    relative_rect: Rect = Rect(
        x=xi - brewing_minigame_region.rect.x, y=yi - brewing_minigame_region.rect.y, w=xf - xi, h=yf - yi
    )
    board_region: Region = brewing_minigame_region.crop_relative(rect=relative_rect)
    print(f"{board_region.rect=}")
    board_region.show()

    return board_region


def _get_right_edge_region(minigame_region: Region) -> Region:
    """Get the right edge of the hex board in the given region."""
    print("Pre get right")
    w_quarter: int = minigame_region.rect.w // 4

    xi: int = w_quarter * 3
    yi: int = 0
    dx: int = minigame_region.rect.w - xi
    dy: int = minigame_region.rect.h
    right_edge_rect: Rect = Rect(x=xi, y=yi, w=dx, h=dy)
    print(f"{right_edge_rect=}")
    right_edge_region: Region = minigame_region.crop_relative(rect=right_edge_rect)
    print(f"{right_edge_region.rect=}")
    right_edge_region.show()
    return right_edge_region


def _get_bottom_edge_region(minigame_region: Region) -> Region:
    """Get the bottom edge of the hex board in the given region."""
    w_half: int = minigame_region.rect.w // 2
    y_half: int = minigame_region.rect.h // 2

    xi: int = w_half
    yi: int = y_half
    dx: int = minigame_region.rect.w - w_half
    dy: int = minigame_region.rect.h - y_half

    bottom_edge_rect: Rect = Rect(x=xi, y=yi, w=dx, h=dy)
    print(f"{bottom_edge_rect=}")
    bottom_edge_region: Region = minigame_region.crop_relative(rect=bottom_edge_rect)
    bottom_edge_region.show()
    return bottom_edge_region


def _get_hex_edges_from_img(img: np.ndarray) -> np.ndarray:
    """Returns the edges of the hex board as a mask of the given image."""
    gray_image: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image: np.ndarray = cv2.GaussianBlur(gray_image, (19, 19), 0)
    edged_image: np.ndarray = cv2.Canny(blurred_image, 50, 120)
    return edged_image


def __find_brewing_board_left_edge_offset(bottom_edge: np.ndarray) -> int:
    """Find the left edge of the brewing board."""
    x_counts: np.ndarray = bottom_edge.astype(bool, copy=True).sum(axis=0)
    half_width: int = bottom_edge.shape[0] // 2
    return int(np.argmax(x_counts[:half_width]))


def __find_brewing_board_right_edge_offset(bottom_edge: np.ndarray) -> int:
    x_counts: np.ndarray = bottom_edge.astype(bool, copy=True).sum(axis=0)
    half_width: int = bottom_edge.shape[0] // 2
    return int(np.argmax(x_counts[half_width:])) + half_width


def __find_brewing_board_top_edge_offset(right_edge: np.ndarray) -> int:
    y_counts: np.ndarray = right_edge.astype(bool, copy=True).sum(axis=1)
    half_height: int = right_edge.shape[1] // 2
    return int(np.argmax(y_counts[:half_height]))


def __find_brewing_board_bottom_edge_offset(right_edge: np.ndarray) -> int:
    y_counts: np.ndarray = right_edge.astype(bool, copy=True).sum(axis=1)
    half_height: int = right_edge.shape[1] // 2
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


def array_to_image(bitmap_data: bytes, width: int, height: int) -> Optional[Image]:
    """Convert BGRA bitmap bytes to RGB PIL Image."""
    try:
        expected_size = width * height * 4
        if len(bitmap_data) < expected_size or width <= 0 or height <= 0:
            return None

        array = np.frombuffer(bitmap_data, dtype=np.uint8)
        array = array.reshape((height, width, 4))[:, :, :3][:, :, ::-1]
        return fromarray(array)
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


def screenshot_window_from_title(window_title: str) -> Optional[Image]:
    """Capture complete window client area including offscreen content."""
    hwnd: int = find_window_by_title(window_title)
    if not hwnd or not win32gui.IsWindow(hwnd):
        return None
    return screenshot_window(hwnd)


def screenshot_window(hwnd: int) -> Optional[Image]:
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


def as_cv_img(img: Union[Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, Image):
        img_array: np.ndarray = np.array(img)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    hwnd: int = find_window_by_title("The Legend of Pirates Online [BETA]")
    board_region: Region = get_brewing_board_region(hwnd=hwnd)
    board_region.show()
