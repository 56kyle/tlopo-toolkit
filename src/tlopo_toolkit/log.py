"""Module responsible for setting up the custom logging used throughout the tlopo_toolkit package."""

from pathlib import Path

from loguru import logger

from tlopo_toolkit.constants import APP_RUNTIME_ID
from tlopo_toolkit.constants import USER_LOG_FOLDER


_LOG_FILE_STEM: str = f"log-{APP_RUNTIME_ID}"

_USER_LOG_FILE: Path = Path(USER_LOG_FOLDER, _LOG_FILE_STEM).with_suffix(".json")

logger.add(sink=_USER_LOG_FILE, serialize=True, retention="1 Month")
