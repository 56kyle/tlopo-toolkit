"""Module containing configuration logic used throughout the tlopo_toolkit package."""

from typing import ClassVar

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from tlopo_toolkit.constants import DEFAULT_PYDANTIC_SETTINGS_CONFIG


class APIConfig(BaseModel):
    """Config for the TLOPO api."""

    login_url: str = "https://api.tlopo.com/login/"


class Config(BaseSettings):
    """Base config for the tlopo_toolkit package."""

    model_config: ClassVar[SettingsConfigDict] = DEFAULT_PYDANTIC_SETTINGS_CONFIG

    game_folder: DirectoryPath = None
    keyring_service: str = "tlopo-toolkit"
    keyring_accounts_user: str = "tlopo-toolkit-accounts"
    api: APIConfig = APIConfig()


def load_config() -> Config:
    """Loads the config from the default location."""
    load_dotenv(".env")
    return Config()


if __name__ == "__main__":
    config: Config = load_config()
    print(config.game_folder)
