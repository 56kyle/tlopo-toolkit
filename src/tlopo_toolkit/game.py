"""Module containing logic for interacting with a particular instance of TLOPO."""
from dataclasses import dataclass

from window_input import Window


@dataclass
class Game:
    """Class representing a particular instance of TLOPO."""
    window: Window



