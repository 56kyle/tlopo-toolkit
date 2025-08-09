"""Module containing logic for interacting with a particular instance of TLOPO."""

from dataclasses import dataclass

from keyring import get_password
from typing_extensions import Self
from window_input import Window


@dataclass(frozen=True)
class Credential:
    """Class representing a TLOPO account's credentials."""

    username: str
    password: str

    @classmethod
    def from_keyring(cls, service_name: str, username: str) -> Self:
        """Creates a Credential object from the keyring."""
        password: str = get_password(service_name, username)
        return cls(username=username, password=password)

    def __repr__(self) -> str:
        return f"Credential(username={self.username}, password=****)"


@dataclass(frozen=True)
class Pirate:
    """Class representing a TLOPO account's pirate."""

    name: str


@dataclass(frozen=True)
class Account:
    """Class representing a TLOPO account."""

    credential: Credential


@dataclass(frozen=True)
class Server:
    """Class representing a particular instance of TLOPO's game server."""

    name: str


@dataclass(frozen=True)
class Client:
    """Class representing a particular instance of TLOPO."""

    window: Window
    account: Account


@dataclass(frozen=True)
class Launcher:
    """Class representing a particular instance of TLOPO's launcher."""

    pid: int
    window: Window


@dataclass(frozen=True)
class Game:
    """Class representing a particular instance of TLOPO's game session."""

    client: Client
    server: Server
