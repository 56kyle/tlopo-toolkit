"""Module containing info and logic related to ingredients in the potion brewing minigame."""

from dataclasses import dataclass
from enum import Enum

from shapely import Point
from typing_extensions import Self


class IngredientType(Enum):
    SCORPION = 1
    CRAB = 2
    ALLIGATOR = 3
    VOLCANO = 4
    PIRATE = 5
    CURSED = 6


@dataclass(frozen=True)
class Ingredient:
    type: IngredientType
    tier: int

    def upgrade(self) -> Self:
        return Ingredient(type=self.type, tier=self.tier + 1)


def get_ingredient_level_from_corner_colors(corners: tuple[Color]) -> int:
    """Returns the ingredient level from the given corners."""
    return sum(map(_is_corner_ingredient_level, corners))


def _is_corner_ingredient_level(corner: Point) -> bool:
    """Returns whether the given corner is a corner ingredient level."""
