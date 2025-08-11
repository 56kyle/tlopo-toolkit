"""Module containing logic for interacting with a particular instance of TLOPO."""
import multiprocessing
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from typing import Generator

from _win32typing import PyHANDLE
from keyring import get_password
from pywinctl import getAllWindows
from pywinctl._main import BaseWindow
from typing_extensions import Self

from tlopo_toolkit.application import Application
from tlopo_toolkit.config import Config
from tlopo_toolkit.config import load_config
from tlopo_toolkit.util import get_window_from_pid


config: Config = load_config()


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
class Client(Application):
    """Class representing a particular instance of TLOPO."""
    account: Account


@dataclass(frozen=True)
class Launcher(Application):
    """Class representing a particular instance of TLOPO's launcher."""



@dataclass(frozen=True)
class Game:
    """Class representing a particular instance of TLOPO's game session."""

    client: Client
    server: Server
