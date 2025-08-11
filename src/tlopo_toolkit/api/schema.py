"""Module containing schemas modeling the TLOPO API."""
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field



class Fleet(BaseModel):
    """Schema representing a single Fleet in the TLOPO Ocean API."""
    type: str
    state: str
    started: int
    ships_remaining: int = Field(alias="shipsRemaining")


class Invasion(BaseModel):
    """Schema representing a single Invasion in the TLOPO Ocean API."""
    started: int
    num_players: int = Field(alias="numPlayers")
    phase: int
    state: str
    location: str


class Ocean(BaseModel):
    """Schema representing a single Ocean in the TLOPO Ocean API."""
    available: bool
    name: str
    created: datetime
    invasion: Invasion
    fleet: Fleet
    is_low_latency: int = Field(alias="isLowLatency")
