"""Module containing logic for interacting with the potion brewing minigame's board."""
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import ClassVar
from typing import Literal
from typing import NamedTuple
from typing import Union

import cv2
import numpy as np
from PIL.Image import Image

from tlopo_toolkit.constants import REPO_FOLDER
from tlopo_toolkit.geometry import Hex
from tlopo_toolkit.geometry import Layout
from tlopo_toolkit.geometry import OffsetCoord
from tlopo_toolkit.geometry import Orientation
from tlopo_toolkit.geometry import Rect


class Color(NamedTuple):
    """Represents an RGB color."""
    red: int
    green: int
    blue: int


class Ingredient:
    """Class representing the different types of pieces in the potion brewing minigame's board."""
    EMPTY = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    ORANGE = 4
    GREY = 5
    PURPLE = 6




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

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def draw_bounding_box(img: np.ndarray, rect: Rect) -> np.ndarray:
    """Draws a bounding box on the given image."""
    new_img = img.copy()
    return cv2.rectangle(new_img, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (0, 255, 0), 2)


@dataclass(frozen=True)
class Board:
    """Class representing the potion brewing minigame's board geometry."""
    layout: Layout
    q_offset: Literal[-1, 1]

    grid: np.ndarray[tuple[int, int], Hex]


def get_board_from_image(img: Union[Image, np.ndarray]) -> Board:
    bgr: np.ndarray = _as_cv_img(img=img)


def _as_cv_img(img: Union[Image, np.ndarray]) -> np.ndarray:
    img_array: np.ndarray = np.array(img)
    bgr_img: np.ndarray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if len(img_array.shape) == 3 else img_array
    return bgr_img



def show(img: np.ndarray) -> None:
    """Shows the given image."""
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path: Path = REPO_FOLDER / "data" / "example.PNG"
    img = cv2.imread(str(img_path))
    for contour in get_contours(img):
        box: Rect = get_contour_bounding_box(contour)
        new_img = draw_bounding_box(img, contour)


    # imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    show(new_img)


