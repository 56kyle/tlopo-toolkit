"""Module containing logic for interacting with the potion brewing minigame's board."""

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Callable
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Union

import cv2
import mouse
import numpy as np

from PIL.Image import Image
from PIL.Image import fromarray
from cv2.aruco import Board
from shapely import Point

from tlopo_toolkit.geometry import Hex
from tlopo_toolkit.geometry import Layout
from tlopo_toolkit.geometry import ORIENTATION_FLAT
from tlopo_toolkit.geometry import OffsetCoord
from tlopo_toolkit.geometry import Rect
from tlopo_toolkit.geometry import Region
from tlopo_toolkit.geometry import hex_to_pixel

from tlopo_toolkit.geometry import polygon_lines
from tlopo_toolkit.geometry import qoffset_to_cube
from tlopo_toolkit.potion_brewing.interface import get_board_rect
from tlopo_toolkit.potion_brewing.interface import get_board_region_from_minigame_region
from tlopo_toolkit.potion_brewing.interface import get_client_region
from tlopo_toolkit.potion_brewing.interface import get_minigame_region_from_client_region
from tlopo_toolkit.util import as_cv_img
from tlopo_toolkit.util import find_window_by_title


def get_contour_bounding_box(contour: np.ndarray) -> Rect:
    """Returns the bounding box of the given contour."""
    x, y, w, h = cv2.boundingRect(contour)
    return Rect(x, y, w, h)


def get_contours(img: np.ndarray):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply binary thresholding (adjust threshold value as needed)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return sorted(contours, key=cv2.contourArea, reverse=True)


@dataclass(frozen=True)
class Board:
    """Class representing the potion brewing minigame's board geometry."""

    layout: Layout
    q_offset: Literal[-1, 1]
    grid: np.ndarray[tuple[10, 8], np.dtype[Hex]]


def get_board_from_window(hwnd: int) -> Board:
    """Returns the potion brewing minigame's board geometry from the given window handle."""
    minigame_region: Region = get_minigame_region_from_client_region(hwnd=hwnd)
    return get_board_from_image(img=minigame_region.export())


def get_board_from_image(img: Union[Image, np.ndarray]) -> Board:
    """Returns the potion brewing minigame's board geometry from the given img."""
    img: np.ndarray = as_cv_img(img=img)
    board_rect: Rect = get_board_rect(img)
    return get_board_from_bounds(rect=board_rect)


def get_board_from_bounds(rect: Rect) -> Board:
    """Returns the potion brewing minigame's board geometry from the given layout."""
    layout: Layout = get_layout_from_bounds(rect=rect)
    hex_grid: np.ndarray[tuple[10, 8], np.dtype[Hex]] = build_hex_grid()

    return Board(layout=layout, q_offset=-1, grid=hex_grid)


def get_layout_from_bounds(rect: Rect) -> Layout:
    """Returns the potion brewing minigame's board layout from the given bounds."""
    x_hex_size: float = rect.w / 12.5
    y_hex_size: float = (rect.h / 10.5) / math.sqrt(3)
    print(f"{x_hex_size=}, {y_hex_size=}")
    return Layout(
        orientation=ORIENTATION_FLAT,
        size=Point(x_hex_size, y_hex_size),
        origin=Point(rect.x + x_hex_size, rect.y + ((y_hex_size * math.sqrt(3)) / 2)),
    )


def build_hex_grid() -> np.ndarray[tuple[10, 8], np.dtype[Hex]]:
    """Builds a hex grid for the board ignoring scale."""
    hex_grid: np.ndarray[tuple[10, 8], np.dtype[Hex]] = np.empty((10, 8), dtype=Hex)

    for x in range(0, 8):
        for y in range(0, 10):
            offset: OffsetCoord = OffsetCoord(col=x, row=y)
            hex_grid[y, x] = qoffset_to_cube(-1, offset)
    return hex_grid


def get_and_display_board() -> None:
    """Displays the potion brewing minigame's board.

    This mainly just exists for debugging purposes.
    """
    hwnd: Optional[int] = find_window_by_title("The Legend of Pirates Online [BETA]")
    client_region: Region = get_client_region(hwnd=hwnd)
    minigame_region: Region = get_minigame_region_from_client_region(client_region=client_region)

    img: np.ndarray = minigame_region.export()

    board_rect: Rect = get_board_rect(image=img)
    board: Board = get_board_from_bounds(rect=board_rect)

    for x in range(0, 8):
        for y in range(0, 10):
            offset: OffsetCoord = OffsetCoord(col=x, row=y)
            print(f"{x=}, {y=}")
            h: Hex = qoffset_to_cube(-1, offset)
            for point in polygon_lines(board.layout, h):
                img[int(point.y), int(point.x), :] = [0, 255, 0]
    fromarray(img).show()


if __name__ == "__main__":
    # get_and_display_board()
    hwnd: Optional[int] = find_window_by_title("The Legend of Pirates Online [BETA]")
    client_region: Region = get_client_region(hwnd=hwnd)
    minigame_region: Region = get_minigame_region_from_client_region(client_region=client_region)
    board_region: Region = get_board_region_from_minigame_region(minigame_region=minigame_region)
    board: Board = get_board_from_bounds(board_region.rect)
    point: Point = hex_to_pixel(board.layout, board.grid[0, 0])
    mouse.move(point.x, point.y)
    board_region.show()
