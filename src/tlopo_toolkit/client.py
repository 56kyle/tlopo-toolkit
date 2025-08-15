"""Module containing logic for interacting with a particular instance of TLOPO."""

import multiprocessing
import os
import time
from contextlib import contextmanager
from dataclasses import Field
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional
from typing import Generator
from urllib.parse import urlencode

import httpx
import keyring
import psutil
from keyring import get_password
from typing_extensions import Self

from tlopo_toolkit.api.schema import LoginResponse
from tlopo_toolkit.application import Application
from tlopo_toolkit.config import Config, load_config
from tlopo_toolkit.constants import REPO_FOLDER
from tlopo_toolkit.process import Executable
from tlopo_toolkit.process import find_all_processes
from tlopo_toolkit.process import find_process
from tlopo_toolkit.log import logger, _LOG_FILE_STEM


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

    @classmethod
    @contextmanager
    def login(cls, username: str) -> Generator[Self, None, None]:
        """Login to TLOPO and return a Client instance."""
        credential: Credential = Credential.from_keyring(config.keyring_service, username)
        cls.prepare_client_launch(credential)
        with cls.managed() as client:
            yield client

    @classmethod
    def prepare_client_launch(cls, credential: Credential) -> None:
        response: httpx.Response = httpx.post(
            config.api.login_url,
            data={
                "username": credential.username,
                "password": credential.password,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()
        login_response: LoginResponse = LoginResponse.model_validate(response.json())
        if login_response.status != 7:
            raise ValueError(f"Login failed: {login_response.message}")

        os.putenv("TLOPO_DIST", login_response.dist)
        os.putenv("TLOPO_GAMESERVER", login_response.gameserver)
        os.putenv("TLOPO_PLAYCOOKIE", login_response.token)
        os.putenv("FPS_BROWSER_APP_PROFILE_STRING", "Internet Explorer")
        os.putenv("FPS_BROWSER_USER_PROFILE_STRING", "Default")


@dataclass(frozen=True)
class Launcher(Application):
    """TLOPO launcher instance."""

    executable: ClassVar[Executable] = Executable(config.game_folder / "Launcher.exe")

    def login(self, credential: Credential) -> Client:
        """Login to TLOPO and return a Client instance."""


@dataclass(frozen=True)
class FakeLauncher:
    """Fake launcher instance for testing purposes."""

    @contextmanager
    def login(self, account_name: str) -> Generator[Client, None, None]:
        """Login to TLOPO and return a Client instance."""
        with Client.login(account_name) as client:
            yield client


@dataclass(frozen=True)
class Game:
    """TLOPO game session."""

    client: Client
    server: Server


def debug_logs(client: Optional[Client] = None) -> None:
    client: Optional[Client] = Client.find() if client is None else client
    process: Optional[psutil.Process] = find_process(Launcher.executable)
    launcher_diff_found: set[str] = set()
    client_diff_found: set[str] = set()
    new_log: Path = REPO_FOLDER / f"{_LOG_FILE_STEM}.json"
    logger.add(new_log)
    try:
        for i in range(100):
            baseline: set[str] = set(os.environ.keys())
            logger.info("Launcher ===============================================")
            for key, value in process.environ().items():
                if key not in baseline:
                    launcher_diff_found.add(key)
                    logger.info(f"{key} -> {value}")
            if client:
                logger.info("Client ===============================================")
                for key, value in client.process.environ().items():
                    if key not in baseline:
                        client_diff_found.add(key)
                        logger.info(f"{key} -> {value}")
            else:
                client: Optional[Client] = Client.find()
            time.sleep(10)
    finally:
        print(f"============== Launcher Diff ================")
        for key in launcher_diff_found:
            print(key)
        if client:
            print(f"============== Client Diff ================")
            for key in client_diff_found:
                print(key)


def login_pool(account: str):
    with launcher.login(account) as client:
        time.sleep(10)
        print(client.process.environ())


if __name__ == "__main__":
    account_str: str = keyring.get_password(config.keyring_service, config.keyring_accounts_user)
    account_list: list[str] = account_str.split(", ") if ", " in account_str else [account_str]

    launcher: FakeLauncher = FakeLauncher()
    Client.prepare_client_launch(Credential.from_keyring(config.keyring_service, account_list[0]))
    client: Client = Client.spawn()
    client.window.maximize()
