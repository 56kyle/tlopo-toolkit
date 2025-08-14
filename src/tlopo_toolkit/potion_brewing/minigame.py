"""Module containing logic for interacting with the potion brewing minigame."""

from typing import Optional
from typing import cast

import mouse
from window_input import Window

from tlopo_toolkit.geometry import Hex
from tlopo_toolkit.geometry import Point
from tlopo_toolkit.geometry import hex_to_pixel
from tlopo_toolkit.mouse import mouse_move
from tlopo_toolkit.potion_brewing.board import Board
from tlopo_toolkit.potion_brewing.board import get_board_from_window
from tlopo_toolkit.potion_brewing.piece import Piece
from tlopo_toolkit.potion_brewing.piece import PlacementPair
from tlopo_toolkit.util import find_window_by_title


def setup_minigame() -> None:
    pass


def get_initial_state(hwnd: int) -> None:
    window: Window = Window(hwnd=hwnd)
    board: Board = get_board_from_window(hwnd=hwnd)
    pieces: set[Piece] = set()
    current_pair: PlacementPair = _get_current_placement_pair(window=window, board=board)


def _get_current_placement_pair(window: Window, board: Board) -> PlacementPair:
    top_left_corner: Hex = cast(Hex, board.grid[0, 0])
    point: Point = hex_to_pixel(layout=board.layout, h=top_left_corner)
    x: int = int(point.x)
    y: int = int(point.y)
    print(point)
    print(window.pixel(x, y))
    mouse.move(x, y)
    # mouse_move(window.hwnd, int(point.x), int(point.y))


if __name__ == "__main__":
    hwnd: Optional[int] = find_window_by_title("The Legend of Pirates Online [BETA]")
    get_initial_state(hwnd=hwnd)
