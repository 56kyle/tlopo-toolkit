"""Module containing logic for interacting with the potion brewing minigame's interface."""

from typing import Optional
from typing import Union

import cv2
import numpy as np
from PIL.Image import Image

from tlopo_toolkit.geometry import Rect
from tlopo_toolkit.util import as_cv_img
from tlopo_toolkit.util import find_window_by_title
from tlopo_toolkit.util import screenshot_client


PIECE_WINDOW_WIDTH_RATIO: float = 91 / 1936
PIECE_WINDOW_HEIGHT_RATIO: float = 81 / 1056

PLAY_AREA_OFFSET_RATIO_OF_HALF: float = 66 / 678


def get_minigame_img(hwnd: int) -> np.ndarray:
    """Returns an image of the game area."""
    img: Optional[Image] = screenshot_client(hwnd=hwnd)
    if img is None:
        raise ValueError("No image found.")
    img_arr: np.ndarray = np.array(img)
    bgr: np.ndarray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    return crop_to_minigame_img(bgr)


def crop_to_minigame_img(img: np.ndarray) -> np.ndarray:
    """Crops the given image to the minigame area."""

    y_half: int = img.shape[0] // 2
    row_sums: np.ndarray = img[y_half].sum(axis=1)

    midpoint: int = len(row_sums) // 2
    left_bar_width: int = sum(np.array(row_sums[:midpoint] == 0))
    right_bar_width: int = sum(np.array(row_sums[midpoint:] == 0))

    xi: int = left_bar_width
    xf: int = img.shape[1] - right_bar_width
    yi: int = 0
    yf: int = img.shape[0]

    return img[yi:yf, xi:xf]


def find_board_range_x(bottom_half_section):
    """Use the bottom half of the image to find horizontal bounds using most common x coordinate."""
    h: int
    w: int
    print(bottom_half_section.shape)
    h, w = bottom_half_section.shape[:2]
    gray: np.ndarray = cv2.cvtColor(bottom_half_section, cv2.COLOR_BGR2LAB)

    # Use larger kernel size for much better edge detection
    blurred: np.ndarray = cv2.GaussianBlur(gray, (9, 9), 3.2)
    edges: np.ndarray = cv2.Canny(blurred, 30, 120)

    # Sum edge pixels along each column to get x-coordinate counts
    x_counts: np.ndarray = edges.astype(bool, copy=True).sum(axis=0)
    print(x_counts)

    w_half = w // 2
    # dxi_inner: int = int(np.argmax(x_counts[:w_half]))
    # dxf_inner: int = int(np.argmax(x_counts[w_half:])) + w_half

    dxi_inner, dxf_inner = find_outer_bounds(x_counts)

    dx_inner: int = dxf_inner - dxi_inner
    hex_outer: float = 0 * dx_inner / 5.75

    dxi: int = round(dxi_inner - (hex_outer / 4))
    dxf: int = round(dxf_inner + (hex_outer / 4))

    return dxi, dxf


def find_board_range_y(board_right_section: np.ndarray) -> tuple[int, int]:
    """Use the right edge of the board area to find vertical bounds."""
    h: int
    w: int
    print(board_right_section.shape)
    h, w = board_right_section.shape[:2]
    gray: np.ndarray = cv2.cvtColor(board_right_section, cv2.COLOR_BGR2LAB)

    # Use larger kernel size for much better edge detection
    blurred: np.ndarray = cv2.GaussianBlur(gray, (19, 19), 3.2)
    edges: np.ndarray = cv2.Canny(blurred, 30, 120)

    # Sum edge pixels along each column to get x-coordinate counts
    y_counts: np.ndarray = edges.astype(bool, copy=True).sum(axis=1)
    print(y_counts)

    h_half: int = h // 2
    # dyi_inner: int = int(np.argmax(y_counts[:h_half]))
    # dyf_inner: int = int(np.argmax(y_counts[h_half:])) + h_half
    dyi_inner, dyf_inner = find_outer_bounds(y_counts)

    dy_inner: int = dyf_inner - dyi_inner
    hex_inner: float = 0 * dy_inner / 19

    dyi: int = round(dyi_inner - hex_inner)
    dyf: int = round(dyf_inner + hex_inner)

    return dyi, dyf


def find_outer_bounds(axis_sums: np.ndarray) -> tuple[int, int]:
    # Find all positions above a relative threshold
    max_val: int = np.max(axis_sums)
    threshold: float = 0.3 * max_val

    significant_indices: np.ndarray = np.where(axis_sums >= threshold)[0]

    if len(significant_indices) == 0:
        raise ValueError("No significant indices found.")

    # First and last significant indices are your outer bounds
    left_bound = significant_indices[0]
    right_bound = significant_indices[-1]

    return int(left_bound), int(right_bound)


def crop_hexagonal_board(image: Union[Image, np.ndarray]) -> np.ndarray:
    """Detect hexagonal board using separate X and Y range finding."""
    if isinstance(image, Image):
        img_array: np.ndarray = np.array(image)
        img: np.ndarray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img: np.ndarray = image

    # Convert PIL to OpenCV format
    board_rect: Rect = get_board_rect(img)

    cropped: np.ndarray = img[board_rect.y : board_rect.bottom, board_rect.x : board_rect.right]

    return cropped


def get_board_rect(image: Union[Image, np.ndarray]) -> Rect:
    img: np.ndarray = as_cv_img(image)
    h: int
    w: int
    h, w = img.shape[:2]
    # Bottom half of image height, but also halve the width (right half) - for X detection
    bottom_half_start: int = h // 2
    width_half_start: int = w // 2
    bottom_half_section: np.ndarray = img[bottom_half_start:, width_half_start:]
    # Board area starts at roughly 3/4 width (right quarter of image) - for Y detection
    board_area_start: int = int(w * 0.75)
    board_right_section: np.ndarray = img[:, board_area_start:]
    # Find the board boundaries
    board_left, board_right = find_board_range_x(bottom_half_section)
    board_top, board_bottom = find_board_range_y(board_right_section)
    # Adjust coordinates back to full image
    board_left += width_half_start
    board_right += width_half_start

    crop_x: int = max(0, board_left)
    crop_y: int = max(0, board_top)
    crop_w: int = min(w - crop_x, board_right - board_left)
    crop_h: int = min(h - crop_y, board_bottom - board_top)
    return Rect(crop_x, crop_y, crop_w, crop_h)


if __name__ == "__main__":
    # img: Image = screenshot_window_from_title("The Legend of Pirates Online [BETA]")
    hwnd: Optional[int] = find_window_by_title("The Legend of Pirates Online [BETA]")
    if hwnd is None:
        raise ValueError("No window found.")
    img: np.ndarray = get_minigame_img(hwnd=hwnd)

    crop_hexagonal_board(img).show()
