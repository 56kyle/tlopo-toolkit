"""Module containing configuration logic used throughout the tlopo_toolkit package."""
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from tlopo_toolkit.constants import DEFAULT_PYDANTIC_SETTINGS_CONFIG


class Config(BaseSettings):
    """Base config for the tlopo_toolkit package."""

    model_config: ClassVar[SettingsConfigDict] = DEFAULT_PYDANTIC_SETTINGS_CONFIG

    game_folder: DirectoryPath


def load_config() -> Config:
    """Loads the config from the default location."""
    load_dotenv(".env")
    return Config()
