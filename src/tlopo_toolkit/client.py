"""Module containing logic for interacting with a particular instance of TLOPO."""
from dataclasses import Field
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from keyring import get_password
from typing_extensions import Self

from tlopo_toolkit.application import Application
from tlopo_toolkit.config import Config, load_config
from tlopo_toolkit.process import Executable


config: Config = load_config()


@dataclass(frozen=True)
class Credential:
    """TLOPO account credentials."""
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
    """TLOPO pirate character."""
    name: str


@dataclass(frozen=True)
class Account:
    """TLOPO account."""
    credential: Credential


@dataclass(frozen=True)
class Server:
    """TLOPO game server."""
    name: str


@dataclass(frozen=True)
class Client(Application):
    """TLOPO client instance."""
    executable: ClassVar[Executable] = Executable(config.game_folder / "TLOPO.exe")

    account: Account


@dataclass(frozen=True)
class Launcher(Application):
    """TLOPO launcher instance."""

    executable: ClassVar[Executable] = Executable(config.game_folder / "Launcher.exe")


@dataclass(frozen=True)
class Game:
    """TLOPO game session."""
    client: Client
    server: Server

