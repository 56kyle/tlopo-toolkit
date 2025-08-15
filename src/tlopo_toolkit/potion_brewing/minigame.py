"""Module containing logic for interacting with the potion brewing minigame."""

from typing import Optional

from tlopo_toolkit.potion_brewing.board import Board
from tlopo_toolkit.potion_brewing.board import get_client_board_from_window
from tlopo_toolkit.potion_brewing.piece import PlacementPair
from tlopo_toolkit.util import find_window_by_title
from tlopo_toolkit.util import get_client_rect


def setup_minigame() -> None:
    pass


def get_initial_state(hwnd: int) -> None:
    """Gets the initial state of the minigame."""
    get_client_rect(hwnd=hwnd)
    client_board: Board = get_client_board_from_window(hwnd=hwnd)
    client_board.to_screen_relative()

    _get_current_placement_pair(hwnd=hwnd, board=client_board)


def _get_current_placement_pair(hwnd: int, board: Board) -> PlacementPair:
    pass


if __name__ == "__main__":
    hwnd: Optional[int] = find_window_by_title("The Legend of Pirates Online [BETA]")
    get_initial_state(hwnd=hwnd)
