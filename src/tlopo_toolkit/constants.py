"""Module containing constants used throughout the tlopo-toolkit package."""

import os
from datetime import datetime
from datetime import timezone
from pathlib import Path

from platformdirs import user_config_path
from platformdirs import user_data_path
from platformdirs import user_log_path
from platformdirs import user_runtime_path
from pydantic import ConfigDict
from pydantic_settings import SettingsConfigDict


FILE_SAFE_DATETIME_FORMAT: str = "%Y-%m-%dT%H%M%S%z"

APP_NAME: str = "tlopo-toolkit"
APP_AUTHOR: str = "56kyle"
APP_START_TIME: datetime = datetime.now(tz=timezone.utc)
APP_PROCESS_ID: int = os.getpid()
APP_RUNTIME_ID: str = f"{APP_START_TIME.strftime(FILE_SAFE_DATETIME_FORMAT)}-{APP_PROCESS_ID}"

USER_CONFIG_FOLDER: Path = user_config_path(appname=APP_NAME, appauthor=APP_AUTHOR, ensure_exists=True)
USER_LOG_FOLDER: Path = user_log_path(appname=APP_NAME, appauthor=APP_AUTHOR, ensure_exists=True)
USER_DATA_FOLDER: Path = user_data_path(appname=APP_NAME, appauthor=APP_AUTHOR, ensure_exists=True)

__USER_RUNTIME_FOLDER: Path = user_runtime_path(appname=APP_NAME, appauthor=APP_AUTHOR, ensure_exists=True)
USER_RUNTIME_FOLDER: Path = __USER_RUNTIME_FOLDER / APP_RUNTIME_ID
USER_RUNTIME_FOLDER.mkdir(parents=True, exist_ok=True)

DEFAULT_PYDANTIC_CONFIG: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
DEFAULT_FROZEN_PYDANTIC_CONFIG: ConfigDict = ConfigDict(arbitrary_types_allowed=True, frozen=True)
DEFAULT_PYDANTIC_SETTINGS_CONFIG: SettingsConfigDict = SettingsConfigDict(
    env_prefix="TLOPO_TOOLKIT", env_nested_delimiter="__", arbitrary_types_allowed=True, extra="allow"
)

REPO_FOLDER: Path = Path(__file__).resolve().parent.parent.parent
