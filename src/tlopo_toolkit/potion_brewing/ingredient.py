"""Module containing info and logic related to ingredients in the potion brewing minigame."""

from dataclasses import dataclass
from enum import Enum

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
