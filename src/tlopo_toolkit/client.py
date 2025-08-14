"""Module containing logic for interacting with a particular instance of TLOPO."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from keyring import get_password
from typing_extensions import Self

from tlopo_toolkit.application import Application
from tlopo_toolkit.config import Config, load_config


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
class Client:
    """TLOPO client instance."""
    account: Account
    application

    def __init__(self, account: Account, executable_path: Optional[Path] = None):
        self.account = account
        self._application = Application(executable_path or config.game_folder / "TLOPO.exe")

    @property
    def application(self) -> Application:
        """Get the underlying application."""
        return self._application

    def spawn(self, command_args: list[str] = None):
        """Spawn the TLOPO client."""
        return self._application.spawn(command_args)

    def terminate(self, force: bool = False, timeout: float = 5.0):
        """Terminate the TLOPO client."""
        self._application.terminate(force, timeout)

    @property
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._application.is_running


class Launcher:
    """TLOPO launcher instance."""

    def __init__(self, executable_path: Optional[Path] = None):
        self._application = Application(executable_path or config.game_folder / "Launcher.exe")

    @property
    def application(self) -> Application:
        """Get the underlying application."""
        return self._application

    def spawn(self, command_args: list[str] = None):
        """Spawn the TLOPO launcher."""
        return self._application.spawn(command_args)

    def terminate(self, force: bool = False, timeout: float = 5.0):
        """Terminate the TLOPO launcher."""
        self._application.terminate(force, timeout)

    @property
    def is_running(self) -> bool:
        """Check if launcher is running."""
        return self._application.is_running


@dataclass(frozen=True)
class Game:
    """TLOPO game session."""
    client: Client
    server: Server
