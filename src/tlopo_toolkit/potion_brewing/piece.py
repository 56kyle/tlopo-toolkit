from dataclasses import dataclass
from typing_extensions import Self

from tlopo_toolkit.geometry import Hex
from tlopo_toolkit.potion_brewing.ingredient import Ingredient


@dataclass(frozen=True)
class Piece:
    """Class representing a piece in the potion brewing minigame."""

    ingredient: Ingredient
    hex: Hex


@dataclass(frozen=True)
class PlacementPair:
    """Class representing a pair of pieces in the potion brewing minigame."""

    left: Ingredient
    right: Ingredient

    def swap(self) -> Self:
        return PlacementPair(left=self.right, right=self.left)
