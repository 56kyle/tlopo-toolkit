"""Module containing logic for interacting with the potion brewing minigame's interface."""
from pathlib import Path
from typing import Iterable

import numpy as np
import pyscreeze
import win32con
import win32gui
import win32ui
from PIL import Image
from window_input import Window

import cv2

from tlopo_toolkit.geometry import Layout
from tlopo_toolkit.geometry import Point
from tlopo_toolkit.geometry import Rect
from tlopo_toolkit.potion_brewing.board import draw_bounding_box
from tlopo_toolkit.potion_brewing.board import show
from tlopo_toolkit.util import find_window_by_title
from tlopo_toolkit.util import screenshot_window


PIECE_WINDOW_WIDTH_RATIO: float = 91 / 1936
PIECE_WINDOW_HEIGHT_RATIO: float = 81 / 1056

PLAY_AREA_OFFSET_RATIO_OF_HALF: float = 66 / 678




def get_play_area_from_image(img: Image) -> None:
    """Returns the play area box coordinates."""



def exit_minigame() -> None:
    """Exits the potion brewing minigame."""
    raise NotImplementedError


def get_scaled_reference_image(window: Window, img: np.ndarray) -> np.ndarray:
    """Returns the scaled reference image."""
    window: Window = Window(hwnd="The Legend of Pirates Online [BETA]")
    left, top, right, bottom = win32gui.GetWindowRect(window.hwnd)
    width: int = right - left
    height: int = bottom - top
    ref_width: int = round(width * PIECE_WINDOW_WIDTH_RATIO)
    ref_height: int = round(height * PIECE_WINDOW_HEIGHT_RATIO)
    return cv2.resize(
        img,
        (ref_width, ref_height),
        interpolation=cv2.INTER_LINEAR,
    )


def get_piece_slots(window: Window) -> list[Point]:
    left, top, right, bottom = win32gui.GetWindowRect(window.hwnd)

def get_minigame_area() -> Image:
    """Returns an image of the game area."""
    img: Image = screenshot_window("The Legend of Pirates Online [BETA]")
    img_arr: np.ndarray = np.array(img)

    y_half: int = img.size[1] // 2
    row_sums: np.ndarray = img_arr[y_half].sum(axis=1)

    midpoint: int = len(row_sums) // 2
    left_bar_width: int = sum(np.array(row_sums[:midpoint] == 0))
    right_bar_width: int = sum(np.array(row_sums[midpoint:] == 0))

    return img.crop((left_bar_width, 0, img.size[0] - right_bar_width, img.size[1]))


def get_board_area() -> Image:
    """Returns the play area relative to the window client."""
    minigame_area: Image = get_minigame_area()


def blur_relative_to_size(img: np.ndarray, size: int) -> np.ndarray:
    """Blurs the image based on the resolution of the given image."""
    kernel_scale: int = 20 / (size / 100)
    if size < 200:
        print(200)
        kernel_scale: int = 5
    elif size < 400:
        print(400)
        kernel_scale: int = 5
    elif size < 800:
        print(800)
        kernel_scale: int = 13
    else:
        kernel_scale: int = 19
    return cv2.GaussianBlur(img, (kernel_scale, kernel_scale), 0)



def find_board_range_x(bottom_half_section):
    """Use the bottom half of the image to find horizontal bounds using most common x coordinate."""
    h: int
    w: int
    h, w = bottom_half_section.shape[:2]
    gray: np.ndarray = cv2.cvtColor(bottom_half_section, cv2.COLOR_BGR2GRAY)

    # Use larger kernel size for much better edge detection
    blurred: np.ndarray = blur_relative_to_size(gray, w)
    show(blurred)
    edges: np.ndarray = cv2.Canny(blurred, 50, 120)
    show(edges)

    # Sum edge pixels along each column to get x-coordinate counts
    x_counts: np.ndarray = edges.astype(bool, copy=True).sum(axis=0)

    w_half = w // 2
    dxi_inner: int = int(np.argmax(x_counts[:w_half]))
    dxf_inner: int = int(np.argmax(x_counts[w_half:])) + w_half

    dx_inner: int = dxf_inner - dxi_inner
    hex_outer: float = dx_inner / 5.75

    dxi: int = round(dxi_inner - (hex_outer / 4))
    dxf: int = round(dxf_inner + (hex_outer / 4))

    show(draw_bounding_box(bottom_half_section, Rect(dxi, 0, dxf - dxi, h)))
    return dxi, dxf


def find_board_range_y(board_right_section):
    """Use the right edge of the board area to find vertical bounds."""
    h: int
    w: int
    h, w = board_right_section.shape[:2]
    gray: np.ndarray = cv2.cvtColor(board_right_section, cv2.COLOR_BGR2GRAY)

    # Use larger kernel size for much better edge detection
    blurred: np.ndarray = blur_relative_to_size(gray, h)
    show(blurred)
    edges: np.ndarray = cv2.Canny(blurred, 50, 120)
    show(edges)

    # Sum edge pixels along each column to get x-coordinate counts
    y_counts: np.ndarray = edges.astype(bool, copy=True).sum(axis=1)

    h_half: int = h // 2
    dyi_inner: int = int(np.argmax(y_counts[:h_half]))
    dyf_inner: int = int(np.argmax(y_counts[h_half:])) + h_half

    dy_inner: int = dyf_inner - dyi_inner
    hex_inner: float = dy_inner / 19

    dyi: int = round(dyi_inner - hex_inner)
    dyf: int = round(dyf_inner + hex_inner)

    show(draw_bounding_box(board_right_section, Rect(0, dyi, w, dyf - dyi)))

    return dyi, dyf


def crop_hexagonal_board(pil_image):
    """Detect hexagonal board using separate X and Y range finding."""
    # Convert PIL to OpenCV format
    img_array: np.ndarray = np.array(pil_image)
    img: np.ndarray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if len(img_array.shape) == 3 else img_array

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

    # Minimal padding
    crop_x: int = max(0, board_left)
    crop_y: int = max(0, board_top)
    crop_w: int = min(w - crop_x, board_right - board_left)
    crop_h: int = min(h - crop_y, board_bottom - board_top)
    show(draw_bounding_box(img, Rect(crop_x, crop_y, crop_w, crop_h)))

    cropped: np.ndarray = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    # Convert back to PIL
    cropped_rgb: np.ndarray = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)


if __name__ == "__main__":
    #img: Image = screenshot_window("The Legend of Pirates Online [BETA]")
    img: Image = get_minigame_area()
    img_arr: np.ndarray = np.array(img)

    crop_hexagonal_board(img_arr).show()

    # xp = (img.size[0] // 2) // 9
    # yp = (img.size[1] // 2) // 9

    # xp = img.size[0] - round((img.size[0] // 2) * PLAY_AREA_OFFSET_RATIO_OF_HALF)
    # yp = img.size[1] - round((img.size[1] // 2) * PLAY_AREA_OFFSET_RATIO_OF_HALF)
    #
    # xi = img.size[0] // 2
    # for y in range(img.size[1] // 2, img.size[1], yp):
    #     for xi in range(img.size[0]):
    #         img.putpixel((xi, y), (255, 0, 0))
    # for x in range(img.size[0] // 2, img.size[0], xp):
    #     for yi in range(img.size[1]):
    #         img.putpixel((x, yi), (255, 0, 0))
    # img.show()


    # window: Window = Window(hwnd="The Legend of Pirates Online [BETA]")
    # path: Path = Path(r"C:\Users\56kyl\source\repos\tlopo-toolkit\data\example.PNG")
    # ref_path: Path = Path(r"C:\Users\56kyl\source\repos\tlopo-toolkit\data\reference\blue_0_clear.png")
    # img: np.ndarray = cv2.imread(str(path))
    # ref_img: np.ndarray = cv2.imread(str(ref_path))
    # scaled_ref_img: np.ndarray = get_scaled_reference_image(window, img=ref_img)
    # boxes: Iterable[Rect] = pyscreeze.locateAll(scaled_ref_img, img, grayscale=False, confidence=0.7)
    # for box in boxes:
    #     img = draw_bounding_box(img, Rect(*box))
    # show(scaled_ref_img)
    # show(img)

    # mask: np.ndarray = cv2.inRange(img, np.array([160, 160, 160]), np.array([255, 255, 255]))
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # output = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # show(output)

    # path: Path = Path(r"C:\Users\56kyl\source\repos\tlopo-toolkit\data\reference\blue_0.PNG")
    # path: Path = Path(r"C:\Users\56kyl\source\repos\tlopo-toolkit\data\example.PNG")
    # img: np.ndarray = cv2.imread(str(path))
    # mask: np.ndarray = cv2.inRange(img, np.array([160, 160, 160]), np.array([255, 255, 255]))
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # output = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # show(output)


