"""Module containing schemas modeling the TLOPO API."""

from datetime import datetime

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


class LoginResponse(BaseModel):
    """Schema representing a single Login in the TLOPO Ocean API."""

    status: int
    message: str
    token: str
    gameserver: str
    dist: str
