"""Module containing configuration logic used throughout the tlopo_toolkit package."""

from typing import ClassVar

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from tlopo_toolkit.constants import DEFAULT_PYDANTIC_SETTINGS_CONFIG


class Config(BaseSettings):
    """Base config for the tlopo_toolkit package."""

    model_config: ClassVar[SettingsConfigDict] = DEFAULT_PYDANTIC_SETTINGS_CONFIG
